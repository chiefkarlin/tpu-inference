"""Structural tests for the single_step_prefill fused dispatch.

These tests run WITHOUT JAX (pure-Python AST analysis) so they can execute
in any environment. They guard the key structural properties of the
single_step_prefill implementation:

1. ``enable_single_step_prefill`` config flag is read in tpu_runner.py.
2. ``_execute_model`` branches to ``_execute_single_step_prefill`` for
   non-decode-only batches when the flag is set.
3. ``_execute_single_step_prefill`` reuses ``single_step_decode`` (same
   jitted function — prefill and decode share the fused body).
4. ``_execute_single_step_prefill`` implements the async protocol.
5. ``_execute_single_step_prefill`` does host extraction via logits_indices.
6. ``_precompile_single_step_prefill`` exists in compilation_manager.py and
   iterates ``num_tokens_paddings`` (not ``num_reqs_paddings``).
7. Validation in tpu_platform.py (no PP, no pooling, no speculative decoding,
   scheduler patch applied).
"""

import ast
import os
import unittest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


class TestTpuRunnerWiring(unittest.TestCase):
    """Verify tpu_runner.py wires single_step_prefill correctly."""

    @classmethod
    def setUpClass(cls):
        cls.tree = _parse(_TPU_RUNNER_PATH)

    def test_config_flag_read(self):
        src = ast.unparse(self.tree)
        self.assertIn("enable_single_step_prefill", src,
                      "enable_single_step_prefill must be read from "
                      "additional_config")

    def test_execute_single_step_prefill_method_exists(self):
        cls_node = _find_class(self.tree, "TPUModelRunner")
        self.assertIsNotNone(cls_node)
        method = _find_method(cls_node, "_execute_single_step_prefill")
        self.assertIsNotNone(method,
                             "TPUModelRunner must have "
                             "_execute_single_step_prefill method")

    def test_branch_in_execute_model(self):
        """_execute_model must branch to _execute_single_step_prefill when
        not is_decode_only and enable_single_step_prefill."""
        cls_node = _find_class(self.tree, "TPUModelRunner")
        self.assertIsNotNone(cls_node)
        method = _find_method(cls_node, "_execute_model")
        self.assertIsNotNone(method)
        src = ast.unparse(method)
        self.assertIn("enable_single_step_prefill", src,
                      "_execute_model must check enable_single_step_prefill")
        self.assertIn("_execute_single_step_prefill", src,
                      "_execute_model must call "
                      "_execute_single_step_prefill")

    def test_reuses_single_step_decode(self):
        """_execute_single_step_prefill must call single_step_decode (reuses
        the same jitted function — prefill and decode share the fused body)."""
        cls_node = _find_class(self.tree, "TPUModelRunner")
        method = _find_method(cls_node, "_execute_single_step_prefill")
        self.assertIsNotNone(method)
        src = ast.unparse(method)
        self.assertIn("single_step_decode", src,
                      "_execute_single_step_prefill must call "
                      "single_step_decode (the shared jitted function)")

    def test_stashes_in_continue_decode_output(self):
        """_execute_single_step_prefill must stash output in
        self._continue_decode_output (so sample_tokens bypasses
        _sample_from_logits)."""
        cls_node = _find_class(self.tree, "TPUModelRunner")
        method = _find_method(cls_node, "_execute_single_step_prefill")
        self.assertIsNotNone(method)
        src = ast.unparse(method)
        self.assertIn("_continue_decode_output", src,
                      "must stash in _continue_decode_output so "
                      "sample_tokens bypasses _sample_from_logits")

    def test_implements_async_protocol(self):
        """_execute_single_step_prefill must implement the async protocol
        when async_scheduling is True (same as single_step_decode)."""
        cls_node = _find_class(self.tree, "TPUModelRunner")
        method = _find_method(cls_node, "_execute_single_step_prefill")
        self.assertIsNotNone(method)
        src = ast.unparse(method)
        for pattern in [
                "async_scheduling",
                "_pre_async_results",
                "_modify_prev_results",
                "_update_placeholder",
                "copy_to_host_async",
                "AsyncPreResults",
                "AsyncTPUModelRunnerOutput",
        ]:
            self.assertIn(pattern, src,
                          f"_execute_single_step_prefill must reference "
                          f"'{pattern}' for async scheduling support")

    def test_host_extraction_via_logits_indices(self):
        """_execute_single_step_prefill must extract tokens on host via
        logits_indices (NOT inside the fused jit — avoids GSPMD issues)."""
        cls_node = _find_class(self.tree, "TPUModelRunner")
        method = _find_method(cls_node, "_execute_single_step_prefill")
        self.assertIsNotNone(method)
        src = ast.unparse(method)
        self.assertIn("logits_indices", src,
                      "must use logits_indices for host extraction")
        self.assertIn("device_get", src,
                      "must use device_get for host extraction")

    def test_forbids_multimodal(self):
        """_execute_single_step_prefill must raise for multimodal models."""
        cls_node = _find_class(self.tree, "TPUModelRunner")
        method = _find_method(cls_node, "_execute_single_step_prefill")
        self.assertIsNotNone(method)
        src = ast.unparse(method)
        self.assertIn("is_multimodal_model", src,
                      "must check is_multimodal_model and raise")

    def test_forbids_prompt_logprobs(self):
        """_execute_single_step_prefill must raise for prompt_logprobs."""
        cls_node = _find_class(self.tree, "TPUModelRunner")
        method = _find_method(cls_node, "_execute_single_step_prefill")
        self.assertIsNotNone(method)
        src = ast.unparse(method)
        self.assertIn("num_prompt_logprobs", src,
                      "must check num_prompt_logprobs and raise")


class TestCompilationManagerPrecompile(unittest.TestCase):
    """Verify compilation_manager.py has precompile for single_step_prefill."""

    @classmethod
    def setUpClass(cls):
        cls.tree = _parse(_COMPILATION_MANAGER_PATH)

    def test_precompile_method_exists(self):
        func = _find_func(self.tree, "_precompile_single_step_prefill")
        self.assertIsNotNone(func,
                             "compilation_manager.py must have "
                             "_precompile_single_step_prefill method")

    def test_precompile_called_in_capture_model(self):
        """capture_model must call _precompile_single_step_prefill when
        enable_single_step_prefill is True."""
        src = ast.unparse(self.tree)
        self.assertIn("enable_single_step_prefill", src,
                      "capture_model must check enable_single_step_prefill")
        self.assertIn("_precompile_single_step_prefill", src,
                      "capture_model must call _precompile_single_step_prefill")

    def test_iterates_num_tokens_paddings(self):
        """_precompile_single_step_prefill must iterate
        num_tokens_paddings (NOT num_reqs_paddings — prefill has larger
        token batches)."""
        func = _find_func(self.tree, "_precompile_single_step_prefill")
        self.assertIsNotNone(func)
        src = ast.unparse(func)
        self.assertIn("num_tokens_paddings", src,
                      "must iterate num_tokens_paddings for prefill shapes")
        # Verify the for-loop iterates num_tokens_paddings (not just
        # mentions it in the docstring).  Look for the actual loop.
        self.assertIn("for num_tokens in self.runner.num_tokens_paddings",
                      src,
                      "must have a for-loop iterating num_tokens_paddings")

    def test_reuses_single_step_decode(self):
        """_precompile_single_step_prefill must compile single_step_decode
        (the shared jitted function), not a separate function."""
        func = _find_func(self.tree, "_precompile_single_step_prefill")
        self.assertIsNotNone(func)
        src = ast.unparse(func)
        self.assertIn("single_step_decode", src,
                      "must compile single_step_decode (shared function)")

    def test_flush_called(self):
        """_precompile_single_step_prefill must be followed by
        _flush_compilations."""
        src = ast.unparse(self.tree)
        # Check that _flush_compilations appears after the prefill call
        idx_prefill = src.find("_precompile_single_step_prefill()")
        idx_flush = src.find("_flush_compilations()", idx_prefill)
        self.assertGreater(idx_flush, idx_prefill,
                           "_flush_compilations must be called after "
                           "_precompile_single_step_prefill")


class TestTpuPlatformValidation(unittest.TestCase):
    """Verify tpu_platform.py validates single_step_prefill correctly."""

    @classmethod
    def setUpClass(cls):
        cls.tree = _parse(_TPU_PLATFORM_PATH)

    def test_reads_enable_single_step_prefill(self):
        src = ast.unparse(self.tree)
        self.assertIn("enable_single_step_prefill", src,
                      "tpu_platform.py must read enable_single_step_prefill "
                      "from additional_config")

    def test_validates_no_pipeline_parallel(self):
        src = ast.unparse(self.tree)
        # Must have a validation block for single_step_prefill
        idx = src.find("enable_single_step_prefill")
        # Find the validation block after the config read
        validation_start = src.find("if enable_single_step_prefill:", idx)
        self.assertGreater(validation_start, -1,
                           "must have 'if enable_single_step_prefill:' block")
        block = src[validation_start:validation_start + 500]
        self.assertIn("pipeline_parallel", block,
                      "must validate no pipeline parallelism")

    def test_validates_no_pooling(self):
        src = ast.unparse(self.tree)
        idx = src.find("if enable_single_step_prefill:")
        block = src[idx:idx + 500]
        self.assertIn("pooling", block,
                      "must validate no pooling models")

    def test_validates_no_speculative(self):
        src = ast.unparse(self.tree)
        idx = src.find("if enable_single_step_prefill:")
        block = src[idx:idx + 500]
        self.assertIn("speculative", block,
                      "must validate no speculative decoding")

    def test_applies_scheduler_patch(self):
        """single_step_prefill must apply patch_vllm_scheduler_for_continue_decode
        (same as single_step_decode)."""
        src = ast.unparse(self.tree)
        idx = src.find("if enable_single_step_prefill:")
        block = src[idx:idx + 1000]
        self.assertIn("patch_vllm_scheduler_for_continue_decode", block,
                      "must apply the scheduler patch")


if __name__ == "__main__":
    unittest.main()
