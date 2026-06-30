"""Structural tests for the single_step_decode fused dispatch.

These tests run WITHOUT JAX (pure-Python AST analysis) so they can execute
in any environment. They guard the key structural properties of the
single_step_decode implementation:

1. ``single_step_decode`` in decode_loop.py fuses 4 dispatches into 1 jit
   (model_fn + select + compute_logits_fn + sample_fn) with NO while_loop.
2. The select is inlined as ``hidden_states[logits_indices]`` (not a
   separate ``_select_from_array_fn`` dispatch).
3. ``enable_single_step_decode`` config flag is read in tpu_runner.py.
4. ``_execute_single_step_decode`` branch exists in ``_execute_model``.
5. ``_precompile_single_step_decode`` exists in compilation_manager.py.
6. Validation in tpu_platform.py (no PP, no pooling, mutually exclusive
   with continue_decode, compatible with async_scheduling).
"""

import ast
import os
import unittest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DECODE_LOOP_PATH = os.path.join(_REPO_ROOT, "tpu_inference", "runner",
                                 "decode_loop.py")
_TPU_RUNNER_PATH = os.path.join(_REPO_ROOT, "tpu_inference", "runner",
                                "tpu_runner.py")
_COMPILATION_MANAGER_PATH = os.path.join(_REPO_ROOT, "tpu_inference",
                                          "runner",
                                          "compilation_manager.py")
_TPU_PLATFORM_PATH = os.path.join(_REPO_ROOT, "tpu_inference", "platforms",
                                  "tpu_platform.py")


def _parse(path):
    with open(path, "r") as f:
        return ast.parse(f.read())


def _find_func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
    return None


def _find_method(cls_node, name):
    for node in cls_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
    return None


def _find_class(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _call_func_name(node):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
    return None


class TestSingleStepDecodeFunction(unittest.TestCase):
    """Verify the core single_step_decode function in decode_loop.py."""

    @classmethod
    def setUpClass(cls):
        cls.tree = _parse(_DECODE_LOOP_PATH)

    def test_function_exists(self):
        func = _find_func(self.tree, "single_step_decode")
        self.assertIsNotNone(func, "single_step_decode must exist in "
                                      "decode_loop.py")

    def test_is_jitted(self):
        func = _find_func(self.tree, "single_step_decode")
        self.assertIsNotNone(func)
        # Check decorator is jax.jit (possibly via functools.partial)
        self.assertTrue(len(func.decorator_list) > 0,
                        "single_step_decode must be decorated with jax.jit")
        dec = func.decorator_list[0]
        if isinstance(dec, ast.Call):
            func_name = _call_func_name(dec)
            self.assertIn(func_name, ("partial", "jit"),
                          "decorator must be functools.partial(jax.jit, ...)")

    def test_static_argnames_include_model_fns(self):
        """model_fn, compute_logits_fn, sample_fn, mesh must be static."""
        func = _find_func(self.tree, "single_step_decode")
        self.assertIsNotNone(func)
        dec = func.decorator_list[0]
        src = ast.unparse(dec)
        for name in ("model_fn", "compute_logits_fn", "sample_fn", "mesh",
                     "layer_name_to_kvcache_index", "is_first_rank",
                     "is_last_rank"):
            self.assertIn(name, src,
                          f"static_argnames must include {name}")

    def test_donates_kv_caches(self):
        func = _find_func(self.tree, "single_step_decode")
        self.assertIsNotNone(func)
        dec = func.decorator_list[0]
        src = ast.unparse(dec)
        self.assertIn("kv_caches", src,
                      "kv_caches must be in donate_argnames")

    def test_has_compiler_options(self):
        """compiler_options must be present (hoisted from inner jits)."""
        func = _find_func(self.tree, "single_step_decode")
        self.assertIsNotNone(func)
        dec = func.decorator_list[0]
        src = ast.unparse(dec)
        self.assertIn("compiler_options", src,
                      "compiler_options must be on the top-level jit")

    def test_fuses_model_fn_call(self):
        """model_fn must be called inside the function body."""
        func = _find_func(self.tree, "single_step_decode")
        self.assertIsNotNone(func)
        src = ast.unparse(func)
        self.assertIn("model_fn(", src,
                      "single_step_decode must call model_fn")

    def test_fuses_compute_logits_fn_call(self):
        func = _find_func(self.tree, "single_step_decode")
        self.assertIsNotNone(func)
        src = ast.unparse(func)
        self.assertIn("compute_logits_fn(", src,
                      "single_step_decode must call compute_logits_fn")

    def test_fuses_sample_fn_call(self):
        func = _find_func(self.tree, "single_step_decode")
        self.assertIsNotNone(func)
        src = ast.unparse(func)
        self.assertIn("sample_fn(", src,
                      "single_step_decode must call sample_fn")

    def test_inlined_select_not_separate_dispatch(self):
        """The select must NOT happen inside the fused jit — no
        _select_from_array_fn call and no logits_indices indexing.
        Token extraction happens on the HOST after device_get (like
        continue_decode's proven approach)."""
        func = _find_func(self.tree, "single_step_decode")
        self.assertIsNotNone(func)
        # Check only the body (skip docstring) for _select_from_array calls.
        body = func.body
        if body and isinstance(body[0], ast.Expr) and isinstance(
                body[0].value, ast.Constant):
            body = body[1:]  # skip docstring
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Call):
                name = _call_func_name(node)
                self.assertNotEqual(
                    name, "_select_from_array_fn",
                    "single_step_decode must NOT call _select_from_array_fn")
        # logits_indices must NOT be a parameter (select is on host).
        body_src = ast.unparse(ast.Module(body=body, type_ignores=[]))
        self.assertNotIn("logits_indices", body_src,
                         "logits_indices must NOT be used inside the fused "
                         "jit — token extraction happens on the host")

    def test_no_while_loop(self):
        """Unlike _decode_core, single_step_decode must NOT use a
        while_loop (that's the overhead source in continue_decode)."""
        func = _find_func(self.tree, "single_step_decode")
        self.assertIsNotNone(func)
        # Check only the body (skip docstring) for while_loop calls.
        body = func.body
        if body and isinstance(body[0], ast.Expr) and isinstance(
                body[0].value, ast.Constant):
            body = body[1:]  # skip docstring
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Call):
                name = _call_func_name(node)
                self.assertNotEqual(
                    name, "while_loop",
                    "single_step_decode must NOT use lax.while_loop")

    def test_returns_kv_caches_next_tokens_expert_indices(self):
        """Must return (kv_caches, next_tokens, expert_indices)."""
        func = _find_func(self.tree, "single_step_decode")
        self.assertIsNotNone(func)
        # Find the return statement.
        return_node = None
        for node in ast.walk(func):
            if isinstance(node, ast.Return):
                return_node = node
                break
        self.assertIsNotNone(return_node, "must have a return statement")
        # ast.unparse may add parentheses around tuple returns.
        ret_src = ast.unparse(return_node)
        self.assertIn("kv_caches", ret_src,
                      "return must include kv_caches")
        self.assertIn("next_tokens", ret_src,
                      "return must include next_tokens")
        self.assertIn("expert_indices", ret_src,
                      "return must include expert_indices")

    def test_no_update_loop_state(self):
        """single_step_decode must NOT call _update_loop_state (EOS handling
        is done by the scheduler, not on-device)."""
        func = _find_func(self.tree, "single_step_decode")
        self.assertIsNotNone(func)
        src = ast.unparse(func)
        self.assertNotIn("_update_loop_state", src,
                         "single_step_decode must NOT call "
                         "_update_loop_state (scheduler handles EOS)")


class TestTpuRunnerWiring(unittest.TestCase):
    """Verify tpu_runner.py wires single_step_decode correctly."""

    @classmethod
    def setUpClass(cls):
        cls.tree = _parse(_TPU_RUNNER_PATH)

    def test_config_flag_read(self):
        src = ast.unparse(self.tree)
        self.assertIn("enable_single_step_decode", src,
                      "enable_single_step_decode must be read from "
                      "additional_config")

    def test_execute_single_step_decode_method_exists(self):
        cls_node = _find_class(self.tree, "TPUModelRunner")
        self.assertIsNotNone(cls_node)
        method = _find_method(cls_node, "_execute_single_step_decode")
        self.assertIsNotNone(method,
                             "TPUModelRunner must have "
                             "_execute_single_step_decode method")

    def test_branch_in_execute_model(self):
        """_execute_model must branch to _execute_single_step_decode when
        is_decode_only and enable_single_step_decode."""
        cls_node = _find_class(self.tree, "TPUModelRunner")
        self.assertIsNotNone(cls_node)
        method = _find_method(cls_node, "_execute_model")
        self.assertIsNotNone(method)
        src = ast.unparse(method)
        self.assertIn("enable_single_step_decode", src,
                      "_execute_model must check enable_single_step_decode")
        self.assertIn("_execute_single_step_decode", src,
                      "_execute_model must call "
                      "_execute_single_step_decode")

    def test_imports_single_step_decode(self):
        src = ast.unparse(self.tree)
        self.assertIn("single_step_decode", src,
                      "tpu_runner.py must import single_step_decode")

    def test_stashes_in_continue_decode_output(self):
        """_execute_single_step_decode must stash output in
        self._continue_decode_output (so sample_tokens bypasses
        _sample_from_logits)."""
        cls_node = _find_class(self.tree, "TPUModelRunner")
        method = _find_method(cls_node, "_execute_single_step_decode")
        self.assertIsNotNone(method)
        src = ast.unparse(method)
        self.assertIn("_continue_decode_output", src,
                      "must stash output in _continue_decode_output")


class TestCompilationManagerPrecompile(unittest.TestCase):
    """Verify compilation_manager.py has precompile for single_step_decode."""

    @classmethod
    def setUpClass(cls):
        cls.tree = _parse(_COMPILATION_MANAGER_PATH)

    def test_precompile_method_exists(self):
        cls_node = _find_class(self.tree, "CompilationManager")
        self.assertIsNotNone(cls_node)
        method = _find_method(cls_node, "_precompile_single_step_decode")
        self.assertIsNotNone(method,
                             "CompilationManager must have "
                             "_precompile_single_step_decode method")

    def test_gate_in_capture_model(self):
        cls_node = _find_class(self.tree, "CompilationManager")
        method = _find_method(cls_node, "capture_model")
        self.assertIsNotNone(method)
        src = ast.unparse(method)
        self.assertIn("enable_single_step_decode", src,
                      "capture_model must gate on enable_single_step_decode")
        self.assertIn("_precompile_single_step_decode", src,
                      "capture_model must call _precompile_single_step_decode")

    def test_imports_single_step_decode(self):
        src = ast.unparse(self.tree)
        self.assertIn("single_step_decode", src,
                      "compilation_manager.py must import single_step_decode")


class TestTpuPlatformValidation(unittest.TestCase):
    """Verify tpu_platform.py validates single_step_decode constraints."""

    @classmethod
    def setUpClass(cls):
        cls.tree = _parse(_TPU_PLATFORM_PATH)

    def test_reads_enable_single_step_decode(self):
        src = ast.unparse(self.tree)
        self.assertIn("enable_single_step_decode", src,
                      "tpu_platform.py must read enable_single_step_decode "
                      "from additional_config")

    def test_validates_no_pipeline_parallel(self):
        src = ast.unparse(self.tree)
        # Must check pipeline_parallel_size > 1 for single_step_decode
        self.assertTrue(
            "single_step_decode" in src
            and "pipeline_parallel_size" in src,
            "must validate no pipeline parallelism for single_step_decode")

    def test_validates_no_pooling(self):
        src = ast.unparse(self.tree)
        self.assertTrue(
            "single_step_decode" in src
            and ("pooling" in src.lower()),
            "must validate no pooling for single_step_decode")

    def test_mutually_exclusive_with_continue_decode(self):
        src = ast.unparse(self.tree)
        self.assertTrue(
            "single_step_decode" in src
            and "continue_decode" in src
            and "mutually" in src.lower(),
            "must validate single_step_decode is mutually exclusive with "
            "continue_decode")

    def test_compatible_with_async_scheduling(self):
        """single_step_decode must NOT be mutually exclusive with
        async_scheduling (unlike continue_decode)."""
        # Read raw file content (ast.unparse strips comments, and the
        # compatibility note is in a comment).
        with open(_TPU_PLATFORM_PATH, "r") as f:
            raw_src = f.read()
        # Find the single_step_decode validation block.
        ssd_idx = raw_src.find("if enable_single_step_decode:")
        self.assertGreater(ssd_idx, -1,
                           "must have 'if enable_single_step_decode:' block")
        # Get the block from the if to the next method definition.
        ssd_block = raw_src[ssd_idx:]
        # The block should NOT raise on async_scheduling.
        self.assertNotIn(
            "not supported with async", ssd_block.lower(),
            "single_step_decode must NOT reject async_scheduling "
            "(unlike continue_decode)")
        # Should have a comment about compatibility.
        self.assertIn("compatible", ssd_block.lower(),
                      "single_step_decode should document compatibility "
                      "with async_scheduling")


if __name__ == "__main__":
    unittest.main()
