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
"""Cohere2-MoE (North-Mini-Code) JAX model.

Architecture (``model_type=cohere2_moe`` / ``Cohere2MoeForCausalLM``):

* **49 layers** with a hybrid attention schedule
  ``layer_types = [full, sliding, sliding, sliding] * 12 + [full]``. The
  ``sliding_attention`` layers apply a 4096-token sliding window (forwarded
  to the RPA v3 kernel); the ``full_attention`` layers attend to the whole
  KV cache. The first layer is a *dense* MLP (``first_k_dense_replace=1``)
  and layers 1..48 are MoE (128 experts, top-8, ``intermediate_size=768``).
* **RoPE** (interleaved / NeoX ordering, ``rope_theta=50000``) is applied
  iff the layer is a sliding layer OR it is the dense layer 0 and
  ``prefix_dense_sliding_window_pattern == 1`` (the ``force_rope`` rule).
  Full-attention MoE layers therefore run **without** RoPE.
* **Parallel block** (Cohere/Nemotron style): a single shared
  ``input_layernorm`` feeds *both* the attention and the MLP sublayers and
  there is **no** ``post_attention_layernorm``:
  ``hidden = x + attn(ln(x)) + mlp(ln(x))``. All other JAX models in this
  repo use the sequential form, so this is implemented explicitly.
* **MoE routing**: ``expert_selection_fn=sigmoid``,
  ``norm_topk_prob=false`` → top-k on the raw logits, sigmoid over the
  selected logits, **no** renormalisation (raw sigmoid weights). This maps
  to ``JaxMoE(scoring_func="sigmoid", renormalize=False, hidden_act="silu")``.
* **Tied embeddings**: ``lm_head`` is tied to ``embed_tokens``;
  ``logit_scale=1.0``.

Weight loading is handled by the default :class:`JaxAutoWeightsLoader`
(via :class:`LoadableWithIterator`): ``JaxEinsum`` projection params alias
to ``*.weight`` and auto-match HF checkpoint keys; the ``JaxMoE`` per-expert
stacking is handled by ``JaxMoE._load_weights``; the router ``gate`` is a
:class:`JaxLinear` whose 2D weight auto-transposes. No custom weight loader
is required.
"""

from dataclasses import InitVar, dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig

from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.attention.cohere2_attention import Cohere2Attention
from tpu_inference.layers.jax.base import _init_fn as init_fn
from tpu_inference.layers.jax.base import sharded_initializer
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.layers import modeling_flax_utils
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear, JaxLmHead
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.moe.utils import (get_expert_parallelism,
                                                select_moe_backend)
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import LoadableWithIterator

logger = init_logger(__name__)


def _weight_init(random_init: bool):
    """Weight initializer: sharded for random init, uniform for checkpoint load."""
    return sharded_initializer if random_init else nnx.initializers.uniform()


@dataclass(kw_only=True)
class Cohere2MLP(JaxModule):
    """Dense gated SiLU feed-forward block for the dense layer (layer 0).

    Mirrors :class:`DeepseekV3MLP` but uses ``prefix``-aware :class:`JaxEinsum`
    projections so HF checkpoint keys (``mlp.gate_proj.weight`` etc.) match
    parameter names automatically under :class:`JaxAutoWeightsLoader`.

    Computes ``down(silu(gate(x)) * up(x))``.
    """

    dtype: jnp.dtype
    hidden_act: str
    hidden_size: int
    intermediate_size: int
    df_sharding: P = P()
    fd_sharding: P = P()
    activation_ffw_td: P = P()
    random_init: bool = False
    quant_config: Optional[QuantizationConfig] = None
    prefix: str = ""

    rngs: InitVar[nnx.Rngs]

    def __post_init__(self, rngs: nnx.Rngs):
        D = self.hidden_size
        F = self.intermediate_size
        weight_init = _weight_init(self.random_init)
        # JaxEinsum aliases self.kernel -> self.weight, so the parameter is
        # named "<prefix>.gate_proj.weight" matching HF checkpoints.
        self.gate_proj = JaxEinsum(
            einsum_str="TD,DF->TF",
            kernel_shape=(D, F),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.df_sharding),
            prefix=self.prefix + ".gate_proj",
        )
        self.up_proj = JaxEinsum(
            einsum_str="TD,DF->TF",
            kernel_shape=(D, F),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.df_sharding),
            prefix=self.prefix + ".up_proj",
        )
        self.down_proj = JaxEinsum(
            einsum_str="TF,FD->TD",
            kernel_shape=(F, D),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.fd_sharding),
            prefix=self.prefix + ".down_proj",
        )

    def __call__(self, x_TD):
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = lax.with_sharding_constraint(x_TD, self.activation_ffw_td)
        with jax.named_scope("wi_0"):
            gating_TF = self.gate_proj(x_TD)
            activated_gating_TF = modeling_flax_utils.ACT2FN[self.hidden_act](
                gating_TF)
        with jax.named_scope("wi_1"):
            up_proj_TF = self.up_proj(x_TD)
        fuse_TF = activated_gating_TF * up_proj_TF
        with jax.named_scope("wo"):
            output_TD = self.down_proj(fuse_TF)
        return output_TD


class Cohere2MoeSparseMoeBlock(JaxModule):
    """MoE block: a sigmoid router (``mlp.gate``) + routed :class:`JaxMoE`.

    NMC uses ``expert_selection_fn=sigmoid`` with ``norm_topk_prob=false``:
    top-k is taken on the raw router logits, the selected logits are passed
    through ``sigmoid``, and the weights are **not** renormalised. This maps
    onto :class:`JaxMoE` via ``scoring_func="sigmoid"`` and
    ``renormalize=False`` (the defaults are ``softmax``/``True`` as used by
    qwen3_moe). ``num_shared_experts=0`` so no shared expert is built.
    """

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        dtype = vllm_config.model_config.dtype
        quant_config = vllm_config.quant_config

        # --- Sharding config (matches qwen3_moe) ---
        edf_sharding = (None, None, None)
        expert_axis_name = edf_sharding[0]
        num_expert_parallelism = get_expert_parallelism(expert_axis_name, mesh)
        use_ep = num_expert_parallelism > 1
        moe_backend = select_moe_backend(use_ep)

        # Router: mlp.gate (hidden_size -> num_experts). JaxLinear so the
        # weight auto-transposes 2D under JaxAutoWeightsLoader.
        self.gate = JaxLinear(
            config.hidden_size,
            config.num_experts,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rng,
            use_bias=False,
            quant_config=quant_config,
            prefix=prefix + ".gate",
        )
        self.gate.num_experts_per_tok = config.num_experts_per_tok

        # NMC has no shared experts.
        self.shared_expert = None

        self.enable_return_routed_experts = True
        self.experts = JaxMoE(
            dtype=dtype,
            num_local_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size_moe=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            rngs=rng,
            router=self.gate,
            num_experts_per_tok=config.num_experts_per_tok,
            mesh=mesh,
            activation_ffw_td=P(ShardingAxisName.MLP_DATA, None),
            activation_ffw_ted=P(ShardingAxisName.MLP_DATA, None, None),
            edf_sharding=P(None, ),
            efd_sharding=P(None, ),
            apply_expert_weight_before_computation=False,
            expert_axis_name=expert_axis_name,
            num_expert_parallelism=num_expert_parallelism,
            moe_backend=moe_backend,
            quant_config=quant_config,
            # NMC overrides: sigmoid routing + no top-k renormalisation.
            scoring_func="sigmoid",
            renormalize=False,
            enable_return_routed_experts=self.enable_return_routed_experts,
            prefix=prefix + ".experts",
        )

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, Optional[jax.Array]]:
        out, expert_ids = self.experts(x)
        if self.shared_expert is not None:
            out += self.shared_expert(x)
        return out, expert_ids


class Cohere2DecoderLayer(JaxModule):
    """A Cohere2 decoder layer using a **parallel** residual structure.

    Unlike the sequential blocks in deepseek/llama (which use
    ``input_layernorm`` + ``post_attention_layernorm``), Cohere2 shares a
    single ``input_layernorm`` across the attention and MLP sublayers and
    sums both residuals:

        h = input_layernorm(x)
        hidden = x + attn(h) + mlp(h)

    Per-layer attributes (fixed at construction, not mode-dependent):
      * ``sliding_window``: 4096 for ``sliding_attention`` layers, else None.
      * ``use_attention_rope``: True for sliding layers, or for the dense
        layer 0 when ``force_rope`` applies
        (``prefix_dense_sliding_window_pattern == 1``); False for
        full-attention MoE layers.
    """

    def __init__(self,
                 config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: QuantizationConfig,
                 layer_idx: int,
                 vllm_config: VllmConfig,
                 prefix: str = ""):
        hidden_size = config.hidden_size
        rms_norm_eps = config.rms_norm_eps
        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
        # The dense layer 0 forces RoPE when the prefix pattern is 1
        # (a full dense layer followed by sliding layers).
        prefix_dense_sliding_window_pattern = getattr(
            config, "prefix_dense_sliding_window_pattern", 0)
        layer_types = getattr(config, "layer_types", None)

        # --- Resolve the per-layer schedule (fixed for this layer) ---
        self.layer_idx = layer_idx
        self.is_dense = layer_idx < first_k_dense_replace
        if layer_types is not None and layer_idx < len(layer_types):
            layer_type = layer_types[layer_idx]
        else:
            # Fallback: treat as full attention if unspecified.
            layer_type = "full_attention"
        self.layer_type = layer_type
        self.sliding_window = (config.sliding_window
                               if layer_type == "sliding_attention" else None)
        # RoPE rule: sliding layers always use RoPE; the dense layer 0 uses
        # RoPE only when force_rope (prefix_dense_sliding_window_pattern==1).
        force_rope = (self.is_dense
                      and prefix_dense_sliding_window_pattern == 1)
        self.use_attention_rope = (self.sliding_window is not None) or force_rope

        # --- Shared input layernorm (parallel block: no post_attn_norm) ---
        self.input_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".input_layernorm",
        )

        # --- Attention ---
        self.self_attn = Cohere2Attention(
            hidden_size=hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim if hasattr(config, "head_dim") else (
                hidden_size // config.num_attention_heads),
            rope_theta=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
            dtype=dtype,
            mesh=mesh,
            kv_cache_dtype=kv_cache_dtype,
            rngs=rng,
            quant_config=quant_config,
            random_init=False,
            rope_input_ordering="interleaved",
            # Prefill chunking = sliding window (gemma4 pattern) so the
            # sliding window also reaches the prefill kernel path.
            attention_chunk_size=self.sliding_window,
            prefix=prefix + ".self_attn",
        )

        # --- MLP (dense for layer 0, MoE for the rest) ---
        if self.is_dense:
            self.mlp = Cohere2MLP(
                dtype=dtype,
                hidden_act=config.hidden_act,
                hidden_size=hidden_size,
                intermediate_size=config.intermediate_size,
                activation_ffw_td=P(ShardingAxisName.MLP_DATA, None),
                random_init=False,
                quant_config=quant_config,
                prefix=prefix + ".mlp",
                rngs=rng,
            )
        else:
            self.mlp = Cohere2MoeSparseMoeBlock(
                vllm_config=vllm_config,
                rng=rng,
                mesh=mesh,
                prefix=prefix + ".mlp",
            )

    def __call__(
        self,
        x: jax.Array,
        is_prefill: bool,
        kv_cache: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
        """Parallel residual forward.

        ``hidden = x + attn(input_layernorm(x)) + mlp(input_layernorm(x))``.
        Returns ``(updated_kv_cache, hidden, expert_ids_or_None)``.
        """
        # The shared normalized hidden state feeds BOTH sublayers.
        h = self.input_layernorm(x)

        kv_cache, attn_output = self.self_attn(
            h,
            is_prefill=is_prefill,
            kv_cache=kv_cache,
            attention_metadata=attention_metadata,
            use_attention_rope=self.use_attention_rope,
        )

        expert_ids = None
        mlp_output = self.mlp(h)
        if isinstance(mlp_output, tuple):
            mlp_output, expert_ids = mlp_output

        # Parallel residual: single add of both sublayer outputs.
        hidden = x + attn_output + mlp_output
        return kv_cache, hidden, expert_ids


class Cohere2MoeModel(JaxModule):
    """The Cohere2-MoE transformer body (embeddings + decoder layers + norm)."""

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "") -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        hidden_size = hf_config.hidden_size

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        # Embedding is created on the first rank, or also on the last rank
        # when the lm_head is tied (the tied head reads embed_tokens.decode).
        if self.is_first_rank or (hf_config.tie_word_embeddings
                                  and self.is_last_rank):
            self.embed_tokens = JaxEmbed(
                num_embeddings=vocab_size,
                features=hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            hf_config.num_hidden_layers,
            lambda layer_index: Cohere2DecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                quant_config=vllm_config.quant_config,
                layer_idx=layer_index,
                vllm_config=vllm_config,
                prefix=f"{prefix}.layers.{layer_index}",
            ))

        # Resolve the per-layer sliding-window schedule once for the body.
        # Used in __call__ to inject attention_metadata.sliding_window
        # per layer (gpt_oss L535-538 pattern) — required because
        # AttentionMetadata has no sliding_window field and
        # Cohere2Attention reads it via getattr.
        layer_types = getattr(hf_config, "layer_types", None)
        num_layers = hf_config.num_hidden_layers
        if layer_types is not None and len(layer_types) == num_layers:
            self.layer_types = list(layer_types)
        else:
            # Fallback: default Cohere2 schedule if config omits it.
            self.layer_types = (["full_attention", "sliding_attention",
                                 "sliding_attention", "sliding_attention"]
                                * (num_layers // 4 + 1))[:num_layers]
        self.sliding_window = getattr(hf_config, "sliding_window", 4096)

        if self.is_last_rank:
            self.norm = JaxRmsNorm(
                hidden_size,
                epsilon=hf_config.rms_norm_eps,
                dtype=dtype,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".norm",
            )
        else:
            self.norm = PPMissingLayer()

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
    ) -> Tuple[List[jax.Array], jax.Array] | Tuple[List[jax.Array], jax.Array,
                                                   jax.Array]:
        if self.is_first_rank:
            assert inputs_embeds is None
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            assert inputs_embeds is not None

        x = inputs_embeds
        new_kv_caches = []
        all_expert_ids = []
        # Decode path: is_prefill is False (matches gpt_oss L531 / llama4 L711).
        is_prefill = False

        # Iterate the full layer list (qwen3_moe / gpt_oss convention) so that
        # ``i`` indexes both ``kv_caches`` and ``self.layer_types`` by absolute
        # layer position. PPMissingLayer slots (pipeline-parallel placeholders)
        # pass their cache through untouched.
        for i, layer in enumerate(self.layers):
            if isinstance(layer, PPMissingLayer):
                new_kv_caches.append(kv_caches[i])
                continue

            # Inject the per-layer sliding window into the attention metadata
            # (dynamic attribute — AttentionMetadata has no such field). This
            # is the mechanism that makes Cohere2Attention's getattr() return
            # 4096 for sliding layers and None for full-attention layers.
            attention_metadata.sliding_window = (
                self.sliding_window
                if self.layer_types[i] == "sliding_attention" else None)

            kv_cache = kv_caches[i]
            kv_cache, x, expert_ids = layer(
                x,
                is_prefill=is_prefill,
                kv_cache=kv_cache,
                attention_metadata=attention_metadata,
            )
            if expert_ids is not None:
                all_expert_ids.append(expert_ids)
            new_kv_caches.append(kv_cache)

        if self.is_last_rank:
            x = self.norm(x)

        stacked_expert_ids = (jnp.stack(all_expert_ids, axis=0)
                              if all_expert_ids else None)
        return new_kv_caches, x, stacked_expert_ids


class Cohere2MoeForCausalLM(JaxModule, LoadableWithIterator):
    """Cohere2-MoE for causal LM with tied embeddings and ``logit_scale``."""

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        self.logit_scale = getattr(hf_config, "logit_scale", 1.0)
        self.tie_word_embeddings = getattr(hf_config, "tie_word_embeddings",
                                           False)

        self.model = Cohere2MoeModel(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix="model",
        )

        # Tied embeddings: do not build a separate lm_head. compute_logits
        # uses embed_tokens.decode (dot with weight.T). A non-tied model
        # would build JaxLmHead on the last rank.
        if not self.tie_word_embeddings:
            if self.model.is_last_rank:
                vocab_size = model_config.get_vocab_size()
                hidden_size = hf_config.hidden_size
                self.lm_head = JaxLmHead(
                    hidden_size=hidden_size,
                    vocab_size=vocab_size,
                    dtype=model_config.dtype,
                    param_dtype=model_config.dtype,
                    rngs=rng,
                    prefix="lm_head",
                )
            else:
                self.lm_head = PPMissingLayer()

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: JaxIntermediateTensors | None = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors,
               List[jax.Array], Optional[jax.Array]]:
        if not is_first_rank:
            assert intermediate_tensors is not None
            inputs_embeds = intermediate_tensors["hidden_states"]
        kv_caches, x, expert_indices = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
        )

        if not is_last_rank:
            x = JaxIntermediateTensors(tensors={"hidden_states": x}, )

        return kv_caches, x, [], expert_indices

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self, "lm_head"):
            logits = self.lm_head(hidden_states)
        else:
            # Tied embeddings: decode via the embedding weight.
            assert isinstance(self.model.embed_tokens, JaxEmbed)
            logits = self.model.embed_tokens.decode(hidden_states)
        # logit_scale is 1.0 for NMC; apply for correctness/generality.
        return logits * self.logit_scale

    def load_weights(self, weights) -> set:
        # LoadableWithIterator.load_weights builds a JaxAutoWeightsLoader:
        #   - skip_prefixes=["lm_head"] when there is no lm_head attr
        #     (i.e. tied embeddings) so the tied lm_head key is skipped.
        # JaxEinsum projection params alias to *.weight and auto-match HF
        # keys (with the correct reshape/permute set by name substring).
        # JaxMoE.load_weights handles per-expert stacking into the fused
        # 3-D kernels (gate/up_EDF, down_EFD). The router (mlp.gate) is a
        # JaxLinear whose 2D weight auto-transposes.
        return super().load_weights(weights)
