# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure-Python structural tests for the Cohere2-MoE wiring.

These tests inspect the *source* of the Cohere2-MoE model and attention
modules via the :mod:`ast` module so they run **without** a JAX runtime.
They guard the highest-risk correctness invariants of the port:

* the **parallel** residual block (single shared norm, no
  ``post_attention_layernorm``);
* the per-layer **schedule** (``sliding_attention`` / ``full_attention``
  strings, sliding-window injection);
* the **MoE** routing overrides (``sigmoid`` / no renormalisation);
* the **sliding-window** forwarding into the RPA v3 kernel;
* the **model-loader** registration.

Runtime behavioural tests (forward-pass logic, weight shapes) live in
``test_cohere2_moe.py`` and require a JAX installation.
"""

import ast
import os
import unittest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))

_COHERE2_MOE_PATH = os.path.join(_REPO_ROOT,
                                 "tpu_inference", "models", "jax",
                                 "cohere2_moe.py")
_COHERE2_ATTN_PATH = os.path.join(
    _REPO_ROOT, "tpu_inference", "layers", "jax", "attention",
    "cohere2_attention.py")
_MODEL_LOADER_PATH = os.path.join(_REPO_ROOT, "tpu_inference", "models",
                                  "common", "model_loader.py")


def _parse(path):
    with open(path) as f:
        return ast.parse(f.read(), filename=path)


def _find_class(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found in source")


def _find_method(cls_node, name):
    for node in cls_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef
                             )) and node.name == name:
            return node
    raise AssertionError(f"method {name!r} not found in {cls_node.name!r}")


def _is_self_attr(node, attr):
    """True if *node* is ``self.<attr>``."""
    return (isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self" and node.attr == attr)


def _call_func_name(node):
    """Return a readable name for a Call's func, or None."""
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        if isinstance(f.value, ast.Name):
            return f"{f.value.id}.{f.attr}"
    return None


class TestParallelBlock(unittest.TestCase):
    """The parallel residual block is the #1 correctness risk — verify it."""

    @classmethod
    def setUpClass(cls):
        cls.tree = _parse(_COHERE2_MOE_PATH)
        cls.layer_cls = _find_class(cls.tree, "Cohere2DecoderLayer")

    def test_no_post_attention_layernorm(self):
        """Cohere2DecoderLayer must NOT create a post_attention_layernorm.

        The parallel block shares a single ``input_layernorm``; a
        ``post_attention_layernorm`` would indicate an accidental copy of
        the sequential deepseek/llama pattern.
        """
        for node in ast.walk(self.layer_cls):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if _is_self_attr(tgt, "post_attention_layernorm"):
                        self.fail(
                            "Cohere2DecoderLayer creates "
                            "post_attention_layernorm — expected parallel "
                            "block with only input_layernorm")

    def test_input_layernorm_called_once(self):
        """The shared norm must be invoked exactly once in __call__."""
        call_method = _find_method(self.layer_cls, "__call__")
        norm_calls = [
            n for n in ast.walk(call_method)
            if isinstance(n, ast.Call) and _is_self_attr(n.func,
                                                         "input_layernorm")
        ]
        self.assertEqual(
            len(norm_calls), 1,
            f"Expected exactly 1 input_layernorm call in parallel block, "
            f"got {len(norm_calls)}")

    def test_parallel_residual_three_way_add(self):
        """``hidden = x + attn_output + mlp_output`` (nested Add BinOp)."""
        call_method = _find_method(self.layer_cls, "__call__")
        # Find any BinOp(Add, BinOp(Add, _, _), _) — a three-way sum.
        found = False
        for node in ast.walk(call_method):
            if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
                    and isinstance(node.left, ast.BinOp)
                    and isinstance(node.left.op, ast.Add)):
                # Collect the three leaf names.
                leaves = []
                for leaf in (node.left.left, node.left.right, node.right):
                    if isinstance(leaf, ast.Name):
                        leaves.append(leaf.id)
                if set(leaves) >= {"x", "attn_output", "mlp_output"}:
                    found = True
                    break
        self.assertTrue(
            found,
            "Parallel residual `x + attn_output + mlp_output` not found "
            "in Cohere2DecoderLayer.__call__")

    def test_attn_and_mlp_share_norm_output(self):
        """Both self_attn and mlp must receive the norm output variable ``h``.

        In a sequential block the MLP would receive the attention *output*;
        in the parallel block both receive the normalised input ``h``.
        """
        call_method = _find_method(self.layer_cls, "__call__")
        # Collect all calls of the form self.self_attn(...) and self.mlp(...)
        # and inspect their first positional arg is Name('h').
        attn_args, mlp_args = [], []
        for node in ast.walk(call_method):
            if not isinstance(node, ast.Call):
                continue
            fname = _call_func_name(node)
            if fname == "self.self_attn":
                attn_args.extend(node.args)
            elif fname == "self.mlp":
                mlp_args.extend(node.args)
        self.assertTrue(attn_args, "self.self_attn not called in __call__")
        self.assertTrue(mlp_args, "self.mlp not called in __call__")
        self.assertIsInstance(
            attn_args[0], ast.Name,
            "self_attn first arg should be a variable (the norm output)")
        self.assertIsInstance(
            mlp_args[0], ast.Name,
            "mlp first arg should be a variable (the norm output)")
        self.assertEqual(
            attn_args[0].id, mlp_args[0].id,
            "self_attn and mlp must receive the SAME normalised input "
            "(parallel block); got different variables")


class TestScheduleResolution(unittest.TestCase):
    """Verify the per-layer schedule strings and sliding-window injection."""

    @classmethod
    def setUpClass(cls):
        cls.tree = _parse(_COHERE2_MOE_PATH)

    def test_layer_type_string_constants(self):
        """The model must compare against the exact layer-type strings."""
        # Collect all string constants in the source and check membership
        # (robust to quote-style differences from ast.unparse).
        strings = {
            n.value for n in ast.walk(self.tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        self.assertIn("sliding_attention", strings)
        self.assertIn("full_attention", strings)

    def test_sliding_window_injected_per_layer(self):
        """Cohere2MoeModel.__call__ must set md.sliding_window per layer."""
        model_cls = _find_class(self.tree, "Cohere2MoeModel")
        call_method = _find_method(model_cls, "__call__")
        # Look for an assignment to <something>.sliding_window inside __call__.
        found = False
        for node in ast.walk(call_method):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Attribute
                                  ) and tgt.attr == "sliding_window":
                        found = True
        self.assertTrue(
            found,
            "Cohere2MoeModel.__call__ must inject sliding_window into the "
            "attention metadata per layer (gpt_oss pattern)")

    def test_force_rope_logic_present(self):
        """The force_rope rule for the dense layer 0 must be present."""
        model_src = ast.unparse(self.tree)
        self.assertIn("force_rope", model_src)
        self.assertIn("prefix_dense_sliding_window_pattern", model_src)

    def test_use_attention_rope_passed_to_attn(self):
        """use_attention_rope must be forwarded to self_attn in __call__."""
        layer_cls = _find_class(self.tree, "Cohere2DecoderLayer")
        call_method = _find_method(layer_cls, "__call__")
        found = False
        for node in ast.walk(call_method):
            if isinstance(node, ast.Call) and _call_func_name(
                    node) == "self.self_attn":
                for kw in node.keywords:
                    if kw.arg == "use_attention_rope":
                        found = True
        self.assertTrue(
            found,
            "self_attn must be called with use_attention_rope= keyword")


class TestMoEConfig(unittest.TestCase):
    """Verify the sigmoid / no-renorm MoE routing overrides."""

    @classmethod
    def setUpClass(cls):
        cls.tree = _parse(_COHERE2_MOE_PATH)

    def test_sigmoid_scoring_no_renorm(self):
        """JaxMoE must be constructed with scoring_func=sigmoid,
        renormalize=False."""
        moe_block_cls = _find_class(self.tree, "Cohere2MoeSparseMoeBlock")
        init_method = _find_method(moe_block_cls, "__init__")
        kwargs = {}
        for node in ast.walk(init_method):
            if isinstance(node, ast.Call) and _call_func_name(
                    node) == "JaxMoE":
                for kw in node.keywords:
                    if kw.arg in ("scoring_func", "renormalize"):
                        # ast.Constant.value gives the raw Python value,
                        # independent of source quote style.
                        if isinstance(kw.value, ast.Constant):
                            kwargs[kw.arg] = kw.value.value
        self.assertEqual(kwargs.get("scoring_func"), "sigmoid",
                         f"expected scoring_func='sigmoid', got {kwargs}")
        self.assertEqual(kwargs.get("renormalize"), False,
                         f"expected renormalize=False, got {kwargs}")

    def test_router_is_jaxlinear_gate(self):
        """The router must be a JaxLinear named ``gate`` (mlp.gate in HF)."""
        moe_block_cls = _find_class(self.tree, "Cohere2MoeSparseMoeBlock")
        init_method = _find_method(moe_block_cls, "__init__")
        found_gate = False
        for node in ast.walk(init_method):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if _is_self_attr(tgt, "gate") and isinstance(
                            node.value, ast.Call) and _call_func_name(
                                node.value) == "JaxLinear":
                        found_gate = True
        self.assertTrue(found_gate, "self.gate must be a JaxLinear (router)")

    def test_experts_prefix(self):
        """JaxMoE must use prefix '...experts' so per-expert names resolve."""
        moe_block_cls = _find_class(self.tree, "Cohere2MoeSparseMoeBlock")
        init_method = _find_method(moe_block_cls, "__init__")
        for node in ast.walk(init_method):
            if isinstance(node, ast.Call) and _call_func_name(
                    node) == "JaxMoE":
                for kw in node.keywords:
                    if kw.arg == "prefix":
                        self.assertIn("experts", ast.unparse(kw.value))
                        return
        self.fail("JaxMoE prefix not found")

    def test_no_shared_expert(self):
        """NMC has num_shared_experts=0 — no shared expert must be built."""
        moe_block_cls = _find_class(self.tree, "Cohere2MoeSparseMoeBlock")
        src = ast.unparse(moe_block_cls)
        self.assertIn("shared_expert", src)

    def test_expert_intermediate_uses_config_intermediate_size(self):
        """REGRESSION: Cohere2 has NO ``moe_intermediate_size`` field.

        The expert intermediate size is ``config.intermediate_size`` (768).
        Using ``config.moe_intermediate_size`` (the Qwen3MoE name) crashes
        at construction with AttributeError. Guard against reintroduction.
        """
        moe_block_cls = _find_class(self.tree, "Cohere2MoeSparseMoeBlock")
        init_method = _find_method(moe_block_cls, "__init__")
        src = ast.unparse(init_method)
        self.assertNotIn(
            "moe_intermediate_size", src,
            "Cohere2 has no 'moe_intermediate_size' config field — use "
            "config.intermediate_size for the expert intermediate (768)")
        self.assertIn("intermediate_size_moe=config.intermediate_size", src,
                      "expert intermediate must be config.intermediate_size")


class TestDenseLayer0Intermediate(unittest.TestCase):
    """REGRESSION: the dense layer 0 must use prefix_dense_intermediate_size.

    Cohere2 names the *expert* intermediate ``intermediate_size`` (768) and
    the *dense* layer-0 intermediate ``prefix_dense_intermediate_size``
    (3072). Passing ``config.intermediate_size`` to the dense MLP yields a
    768-wide FFN that mismatches the 3072-wide checkpoint weight.
    """

    @classmethod
    def setUpClass(cls):
        cls.tree = _parse(_COHERE2_MOE_PATH)

    def test_dense_mlp_uses_prefix_dense_intermediate_size(self):
        layer_cls = _find_class(self.tree, "Cohere2DecoderLayer")
        init_method = _find_method(layer_cls, "__init__")
        src = ast.unparse(init_method)
        self.assertIn("prefix_dense_intermediate_size", src,
                      "dense layer 0 must read prefix_dense_intermediate_size")
        self.assertNotIn(
            "moe_intermediate_size", src,
            "no moe_intermediate_size field exists on Cohere2 config")

    def test_dense_mlp_intermediate_not_bare_config_intermediate(self):
        """The dense MLP must NOT pass config.intermediate_size directly
        (that is the expert size 768, not the dense size 3072)."""
        layer_cls = _find_class(self.tree, "Cohere2DecoderLayer")
        init_method = _find_method(layer_cls, "__init__")
        # Find the Cohere2MLP(...) call and check intermediate_size kwarg.
        for node in ast.walk(init_method):
            if isinstance(node, ast.Call) and _call_func_name(
                    node) == "Cohere2MLP":
                for kw in node.keywords:
                    if kw.arg == "intermediate_size":
                        val_src = ast.unparse(kw.value)
                        self.assertNotEqual(
                            val_src, "config.intermediate_size",
                            "dense layer 0 intermediate_size must be "
                            "prefix_dense_intermediate_size (3072), not "
                            "config.intermediate_size (768 = experts)")
                        return
        self.fail("Cohere2MLP(...) call not found in Cohere2DecoderLayer.__init__")

    def test_dense_detection_uses_first_k_dense_replace(self):
        """The dense layer 0 is identified via ``first_k_dense_replace``.

        ``layer_types`` cannot distinguish the dense layer 0 from the other
        ``full_attention`` MoE layers (layers 4, 8, ... are also
        ``full_attention``), so ``first_k_dense_replace`` is the only
        reliable signal. On the actual serving path (generic
        PretrainedConfig stores all kwargs as attributes) this returns 1.
        """
        layer_cls = _find_class(self.tree, "Cohere2DecoderLayer")
        init_method = _find_method(layer_cls, "__init__")
        src = ast.unparse(init_method)
        self.assertIn("first_k_dense_replace", src,
                      "dense detection must use first_k_dense_replace")
        self.assertIn("is_dense = layer_idx < first_k_dense_replace", src,
                      "is_dense must be computed from first_k_dense_replace")


class TestTiedEmbeddingsDefault(unittest.TestCase):
    """REGRESSION (minor #4): tie_word_embeddings default must be True.

    Both PretrainedConfig and Cohere2MoeConfig default
    ``tie_word_embeddings=True``. NMC's config.json does not set it
    explicitly. A False fallback would build a separate lm_head and then
    fail to load ``lm_head.weight`` (absent in the tied checkpoint).
    """

    @classmethod
    def setUpClass(cls):
        cls.tree = _parse(_COHERE2_MOE_PATH)

    def test_tie_default_is_true(self):
        causal_cls = _find_class(self.tree, "Cohere2MoeForCausalLM")
        init_method = _find_method(causal_cls, "__init__")
        found = False
        for node in ast.walk(init_method):
            if isinstance(node, ast.Call) and _call_func_name(
                    node) == "getattr":
                # getattr(hf_config, "tie_word_embeddings", <default>)
                if (len(node.args) >= 3 and isinstance(node.args[1], ast.Constant)
                        and node.args[1].value == "tie_word_embeddings"):
                    default = node.args[2]
                    self.assertIsInstance(
                        default, ast.Constant,
                        "tie_word_embeddings default must be a literal")
                    self.assertEqual(
                        default.value, True,
                        "tie_word_embeddings getattr default must be True "
                        "(PretrainedConfig/Cohere2MoeConfig default; NMC "
                        "relies on it)")
                    found = True
        self.assertTrue(found, "getattr(hf_config, 'tie_word_embeddings', ...) "
                        "not found in Cohere2MoeForCausalLM.__init__")


class TestTiedEmbeddings(unittest.TestCase):
    """Verify tied lm_head wiring (compute_logits uses embed.decode)."""

    @classmethod
    def setUpClass(cls):
        cls.tree = _parse(_COHERE2_MOE_PATH)

    def test_compute_logits_uses_embed_decode(self):
        """compute_logits must call embed_tokens.decode when tied."""
        causal_cls = _find_class(self.tree, "Cohere2MoeForCausalLM")
        method = _find_method(causal_cls, "compute_logits")
        src = ast.unparse(method)
        self.assertIn("embed_tokens.decode", src,
                      "compute_logits must use embed_tokens.decode for tied head")

    def test_logit_scale_applied(self):
        """logit_scale must be applied to the logits."""
        causal_cls = _find_class(self.tree, "Cohere2MoeForCausalLM")
        method = _find_method(causal_cls, "compute_logits")
        src = ast.unparse(method)
        self.assertIn("logit_scale", src)

    def test_lm_head_skipped_when_tied(self):
        """__init__ must conditionally skip lm_head when tied."""
        causal_cls = _find_class(self.tree, "Cohere2MoeForCausalLM")
        init_method = _find_method(causal_cls, "__init__")
        src = ast.unparse(init_method)
        self.assertIn("tie_word_embeddings", src)
        self.assertIn("JaxLmHead", src)


class TestCohere2AttentionWiring(unittest.TestCase):
    """Verify sliding-window forwarding and decode_block_sizes lever."""

    @classmethod
    def setUpClass(cls):
        cls.tree = _parse(_COHERE2_ATTN_PATH)

    def test_sliding_window_forwarded_to_kernel(self):
        """attention() must pass sliding_window= to ragged_paged_attention."""
        attn_cls = _find_class(self.tree, "Cohere2Attention")
        method = _find_method(attn_cls, "attention")
        found = False
        for node in ast.walk(method):
            if isinstance(node, ast.Call) and _call_func_name(
                    node) == "ragged_paged_attention":
                for kw in node.keywords:
                    if kw.arg == "sliding_window":
                        found = True
        self.assertTrue(
            found,
            "ragged_paged_attention must be called with sliding_window= "
            "(the K-1 fix)")

    def test_sliding_window_read_from_metadata(self):
        """sliding_window must be read via getattr(md, ...)."""
        attn_cls = _find_class(self.tree, "Cohere2Attention")
        method = _find_method(attn_cls, "attention")
        found = False
        for node in ast.walk(method):
            if (isinstance(node, ast.Call) and _call_func_name(node)
                    == "getattr"):
                # getattr(md, "sliding_window", ...)
                if node.args and isinstance(
                        node.args[1], ast.Constant
                ) and node.args[1].value == "sliding_window":
                    found = True
        self.assertTrue(
            found,
            "sliding_window must be read via getattr(md, 'sliding_window', ...)")

    def test_decode_block_sizes_present(self):
        """decode_block_sizes field with NMC-tuned default must exist."""
        attn_cls = _find_class(self.tree, "Cohere2Attention")
        src = ast.unparse(attn_cls)
        self.assertIn("decode_block_sizes", src)
        self.assertIn("4096", src)

    def test_d_block_sizes_passed_to_kernel(self):
        """d_block_sizes kwarg must reach ragged_paged_attention."""
        attn_cls = _find_class(self.tree, "Cohere2Attention")
        method = _find_method(attn_cls, "attention")
        found = False
        for node in ast.walk(method):
            if isinstance(node, ast.Call) and _call_func_name(
                    node) == "ragged_paged_attention":
                for kw in node.keywords:
                    if kw.arg == "d_block_sizes":
                        found = True
        self.assertTrue(found, "d_block_sizes must be passed to the kernel")

    def test_rope_interleaved_default(self):
        """rope_input_ordering default must be 'interleaved'."""
        attn_cls = _find_class(self.tree, "Cohere2Attention")
        # The default appears as a keyword/value on the class body.
        strings = {
            n.value for n in ast.walk(attn_cls)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        self.assertIn("interleaved", strings)

    def test_jaxeinsum_projections(self):
        """Projections must use JaxEinsum (not base Attention create_param)
        so HF weight names match under JaxAutoWeightsLoader."""
        attn_cls = _find_class(self.tree, "Cohere2Attention")
        post_init = _find_method(attn_cls, "__post_init__")
        einsum_count = 0
        for node in ast.walk(post_init):
            if isinstance(node, ast.Call) and _call_func_name(
                    node) == "JaxEinsum":
                einsum_count += 1
        self.assertEqual(
            einsum_count, 4,
            f"Expected 4 JaxEinsum projections (q/k/v/o), got {einsum_count}")


class TestModelLoaderRegistration(unittest.TestCase):
    """Verify the model is registered in the model loader."""

    def test_registry_contains_cohere2(self):
        with open(_MODEL_LOADER_PATH) as f:
            src = f.read()
        self.assertIn(
            "from tpu_inference.models.jax.cohere2_moe import Cohere2MoeForCausalLM",
            src, "Cohere2MoeForCausalLM import missing from model_loader.py")
        self.assertIn(
            '_MODEL_REGISTRY["Cohere2MoeForCausalLM"] = Cohere2MoeForCausalLM',
            src,
            "Cohere2MoeForCausalLM not registered in _MODEL_REGISTRY")

    def test_not_in_preferred_architectures(self):
        """NMC must NOT be in _VLLM_PREFERRED_ARCHITECTURES (forces
        flax_nnx impl)."""
        with open(_MODEL_LOADER_PATH) as f:
            src = f.read()
        # The preferred-architectures list should not mention Cohere2.
        # Find the list literal and check.
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name
                                  ) and tgt.id == "_VLLM_PREFERRED_ARCHITECTURES":
                        values = ast.unparse(node.value)
                        self.assertNotIn(
                            "Cohere2", values,
                            "Cohere2MoeForCausalLM must not be in "
                            "_VLLM_PREFERRED_ARCHITECTURES (impl must be "
                            "flax_nnx)")


if __name__ == "__main__":
    unittest.main()
