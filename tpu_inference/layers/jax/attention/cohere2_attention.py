# Copyright 2025 Google LLC
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
"""Cohere2 attention.

Cohere2 / North-Mini-Code uses GQA with ``head_dim=128`` and applies a
*sliding window* (4096) on a subset of layers (the ``sliding_attention``
layers) while leaving the ``full_attention`` layers unwindowed. The base
``Attention.attention()`` calls :func:`ragged_paged_attention` without
forwarding ``sliding_window``, so a window set on the attention metadata
would be silently dropped.

This module is a **standalone** attention (not a subclass of the base
``Attention``) that uses :class:`JaxEinsum` for the Q/K/V/O projections.
This is required for weight loading: ``JaxAutoWeightsLoader`` matches HF
checkpoint keys (``self_attn.q_proj.weight``) to parameter names, and
``JaxEinsum`` aliases its kernel to ``self.weight`` so the names match
automatically. The base ``Attention`` class uses ``create_param`` which
produces ``kernel_q_proj_DNH``-style names that do **not** match HF keys
and would require a custom weight loader.

The :meth:`attention` method forwards ``sliding_window`` (read as a
dynamic attribute on the ``AttentionMetadata`` — mirroring the gpt_oss
per-layer pattern) to the regular v3 ``ragged_paged_attention`` kernel.
NMC uses ``head_dim=128`` so the *regular* v3 kernel (not the hd64
variant) is correct.

RoPE (interleaved / NeoX ordering) and the per-layer
``use_attention_rope`` toggle are handled in :meth:`__call__`; the module
is constructed with ``rope_input_ordering="interleaved"`` and the caller
passes ``use_attention_rope`` per layer.
"""

from dataclasses import InitVar, dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Sharding
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_inference import utils
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    ragged_paged_attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.base import _init_fn as init_fn
from tpu_inference.layers.jax.base import sharded_initializer
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.layers.jax.rope_interface import apply_rope

KVCache = Tuple[jax.Array, jax.Array]


def _weight_init(random_init: bool):
    return sharded_initializer if random_init else nnx.initializers.uniform()


@dataclass(kw_only=True)
class Cohere2Attention(JaxModule):
    """Cohere2 attention with sliding-window support for the RPA v3 kernel.

    GQA attention (no bias, ``head_dim=128``) using :class:`JaxEinsum`
    projections so that HF checkpoint keys (``q_proj.weight`` etc.) match
    parameter names automatically under :class:`JaxAutoWeightsLoader`.

    ``sliding_window`` is read off the attention metadata (a dynamic
    attribute set per layer by the model, following the gpt_oss pattern)
    and forwarded to :func:`ragged_paged_attention`. Construct with
    ``rope_input_ordering="interleaved"``; the caller controls RoPE via
    ``use_attention_rope``.

    Decode block sizes are exposed as a class-level tuning lever
    (``decode_block_sizes``). NMC's decode is VMEM-borderline on the
    default ``bkv_sz=8192`` because the v3 kernel double-buffers the KV
    fetch. ``decode_block_sizes=(1, 4096, 1, 2048)`` halves the dominant
    DMA buffer (one full sliding-window fetch, two compute passes) while
    keeping every element a multiple of the page size. Leave ``None`` to
    use the kernel defaults. Tuple order is ``(bq_sz, bkv_sz, bq_csz,
    bkv_csz)``; see ``ragged_paged_attention`` validation.

    Prefill tuning: ``prefill_chunk_size`` (default 4096 = NMC
    ``sliding_window``) enables the dedicated PREFILL kernel launch with a
    static ``q_len``, which XLA optimizes more aggressively than the
    dynamic MIXED path. ``mixed_block_sizes`` and ``prefill_block_sizes``
    default to ``None`` (kernel auto-tuned via ``get_default_block_sizes``);
    override with explicit tuples to sweep.
    """

    # Core configuration
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    rope_scaling: Optional[dict]
    dtype: jnp.dtype
    mesh: Mesh
    kv_cache_dtype: str

    # Sharding — Q/KV head dims sharded on the "model" (TP) axis so that
    # GQA weights and activations are partitioned correctly under TP>1.
    # Without this, K/V tensors retain the full num_key_value_heads (e.g. 4)
    # while the kv_cache is TP-sharded to 1 head/device, causing a shape
    # mismatch in the RPA v3 kernel's static_validate_inputs.
    # Pattern follows llama4.py (Llama4Attention construction).
    dnh_sharding: Sharding = (None, ShardingAxisName.MODEL, None)
    dkh_sharding: Sharding = (None, ShardingAxisName.MODEL, None)
    nhd_sharding: Sharding = (ShardingAxisName.MODEL, None, None)

    activation_q_td: P = P(ShardingAxisName.ATTN_DATA)
    query_tnh: P = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.MODEL, None)
    keyvalue_skh: P = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.MODEL, None)
    attn_o_tnh: P = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.MODEL, None)

    rngs: InitVar[nnx.Rngs]

    random_init: bool = False
    attention_chunk_size: Optional[int] = None
    rope_input_ordering: str = "interleaved"

    # Decode block sizes (bq_sz, bkv_sz, bq_csz, bkv_csz) or None for kernel
    # defaults. Tuned for NMC: one sliding-window fetch (4096) split into two
    # compute passes (bkv_csz=2048) — halves the double-buffered KV footprint.
    decode_block_sizes: Optional[Tuple[int, int, int, int]] = (1, 4096, 1,
                                                               2048)

    # Mixed-case block sizes (bq_sz, bkv_sz, bq_csz, bkv_csz) or None for
    # kernel auto-tuned defaults. The mixed kernel handles queries with
    # dynamic q_len (neither pure-decode nor pure-prefill).
    #
    # Tuned candidate 1 (aggressive): (512, 4096, 256, 1024).
    # - 2x query fetch (bq_sz=512 vs heuristic 256) → fewer query blocks,
    #   better MXU utilization for compute-bound long prefill.
    # - 2x KV fetch (bkv_sz=4096 vs heuristic 2048) → single KV pass for
    #   4096-token prefill (eliminates outer KV loop overhead).
    # - 2x compute chunks (bq_csz=256, bkv_csz=1024) → 4x larger matmuls
    #   (256x1024x128 vs 128x512x128) → better MXU pipeline fill.
    # Roofline: at q_len=4096, AI≈2048 FLOPs/byte → compute-bound → larger
    # matmuls help. At q_len=128, AI≈64 → memory-bound → larger bkv_sz helps
    # HBM bandwidth.
    mixed_block_sizes: Optional[Tuple[int, int, int, int]] = (512, 4096, 256,
                                                               1024)

    # Prefill-case block sizes (bq_sz, bkv_sz, bq_csz, bkv_csz) or None for
    # kernel auto-tuned defaults. Only used when ``prefill_chunk_size`` is
    # set (enabling the dedicated PREFILL launch with a static q_len).
    prefill_block_sizes: Optional[Tuple[int, int, int, int]] = None

    # Chunk size for the dedicated PREFILL kernel launch. When set (non-None),
    # the RPA v3 dispatch runs a prefill-specific kernel with
    # ``static_q_len=chunk_prefill_size`` in addition to the decode and mixed
    # kernels. The static q_len enables XLA to optimize the prefill path more
    # aggressively than the dynamic mixed path. Set to NMC's ``sliding_window``
    # (4096) so a full sliding-window prefill is one launch. Set to ``None``
    # to disable the dedicated prefill path (all prefill falls through to the
    # mixed kernel).
    prefill_chunk_size: Optional[int] = None

    # Quantization scales (per-tensor; 1.0 when unquantized)
    _q_scale: float = 1.0
    _k_scale: float = 1.0
    _v_scale: float = 1.0

    quant_config: Optional[QuantizationConfig] = None
    prefix: str = ""

    kv_cache_quantized_dtype = None

    def __post_init__(self, rngs: nnx.Rngs):
        """Initializes the Q/K/V/O projection weights via JaxEinsum."""
        N = self.num_attention_heads
        K = self.num_key_value_heads
        D = self.hidden_size
        H = self.head_dim
        weight_init = _weight_init(self.random_init)

        self.q_proj = JaxEinsum(
            einsum_str="TD,DNH->TNH",
            kernel_shape=(D, N, H),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.dnh_sharding),
            prefix=self.prefix + ".q_proj",
        )
        self.k_proj = JaxEinsum(
            einsum_str="SD,DKH->SKH",
            kernel_shape=(D, K, H),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.dkh_sharding),
            prefix=self.prefix + ".k_proj",
        )
        self.v_proj = JaxEinsum(
            einsum_str="SD,DKH->SKH",
            kernel_shape=(D, K, H),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.dkh_sharding),
            prefix=self.prefix + ".v_proj",
        )
        self.o_proj = JaxEinsum(
            einsum_str="TNH,NHD->TD",
            kernel_shape=(N, H, D),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.nhd_sharding),
            prefix=self.prefix + ".o_proj",
        )

        if self.kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                self.kv_cache_dtype)

    def __call__(
        self,
        x: jax.Array,
        is_prefill: bool,
        kv_cache: KVCache,
        attention_metadata: AttentionMetadata,
        use_attention_rope: bool = True,
    ) -> Tuple[KVCache, jax.Array]:
        """Forward pass: Q/K/V projections, RoPE, attention, output projection.

        Args:
            x: Input tensor of shape ``(seq_len, d_model)``.
            is_prefill: Whether the mode is prefill (accepted for API
                consistency; the RPA kernel handles both modes).
            kv_cache: The key-value cache.
            attention_metadata: Attention metadata (``sliding_window`` is
                read as a dynamic attribute).
            use_attention_rope: Whether to apply RoPE to Q and K.

        Returns:
            (updated_kv_cache, attention_output_TD)
        """
        md = attention_metadata
        x_SD = jnp.asarray(x, self.dtype)
        x_q_TD = lax.with_sharding_constraint(x, self.activation_q_td)
        H = self.head_dim

        with jax.named_scope("q_proj"):
            q_TNH = self.q_proj(x_q_TD)
            if use_attention_rope:
                q_TNH = apply_rope(q_TNH, md.input_positions, H,
                                   self.rope_theta, self.rope_scaling,
                                   self.rope_input_ordering)
            q_TNH = lax.with_sharding_constraint(q_TNH, self.query_tnh)

        with jax.named_scope("k_proj"):
            k_SKH = self.k_proj(x_SD)
            if use_attention_rope:
                k_SKH = apply_rope(k_SKH, md.input_positions, H,
                                   self.rope_theta, self.rope_scaling,
                                   self.rope_input_ordering)
            k_SKH = lax.with_sharding_constraint(k_SKH, self.keyvalue_skh)

        with jax.named_scope("v_proj"):
            v_SKH = self.v_proj(x_SD)

        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            # TODO: Enable w8a8 when VREG spill issue is resolved.
            # q_scale = self._q_scale
            k_scale = self._k_scale
            v_scale = self._v_scale
            k_SKH, v_SKH = quantize_kv(self.kv_cache_quantized_dtype, k_SKH,
                                        v_SKH, k_scale, v_scale)

        with jax.named_scope("attn_op"):
            new_kv_cache, outputs_TNH = self.attention(
                is_prefill,
                kv_cache,
                q_TNH,
                k_SKH,
                v_SKH,
                attention_metadata,
                self.mesh,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        with jax.named_scope("o_proj"):
            o_TD = self.o_proj(outputs_TNH)
        return new_kv_cache, o_TD

    def attention(
        self,
        is_prefill: bool,
        kv_cache: KVCache,
        q_TNH: jax.Array,
        k_SKH: jax.Array,
        v_SKH: jax.Array,
        attention_metadata: AttentionMetadata,
        mesh: Mesh,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ) -> Tuple[KVCache, jax.Array]:
        """Scaled dot-product attention forwarding ``sliding_window``.

        Mirrors the base ``Attention.attention()`` implementation but adds
        ``sliding_window`` (read via :func:`getattr` with a ``None`` default
        so full-attention layers fall back to unwindowed attention) and the
        full set of v3 block-size / prefill-chunk tuning kwargs to the kernel
        call: ``d_block_sizes`` (decode), ``m_block_sizes`` (mixed),
        ``p_block_sizes`` (prefill), and ``chunk_prefill_size`` (enables the
        dedicated PREFILL launch with a static ``q_len``).
        """
        md = attention_metadata
        sliding_window = getattr(md, "sliding_window", None)

        kv_cache_spec = P(ShardingAxisName.ATTN_DATA, None, "model")
        in_specs = (
            self.query_tnh,  # q
            self.keyvalue_skh,  # k
            self.keyvalue_skh,  # v
            kv_cache_spec,  # kv_cache
            P(ShardingAxisName.ATTN_DATA),  # md.seq_lens
            P(ShardingAxisName.ATTN_DATA),  # md.block_tables
            P(ShardingAxisName.ATTN_DATA),  # md.query_start_loc
            P(ShardingAxisName.ATTN_DATA),  # md.request_distribution
        )

        out_specs = (self.attn_o_tnh, kv_cache_spec)

        def _ragged_paged_attention(*args):
            return ragged_paged_attention(
                *args,
                sm_scale=q_TNH.shape[-1]**-0.5,
                sliding_window=sliding_window,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                d_block_sizes=self.decode_block_sizes,
                m_block_sizes=self.mixed_block_sizes,
                p_block_sizes=self.prefill_block_sizes,
                chunk_prefill_size=self.prefill_chunk_size,
            )

        output_TNH, kv_cache = jax.jit(
            jax.shard_map(
                _ragged_paged_attention,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            ))(
                q_TNH,
                k_SKH,
                v_SKH,
                kv_cache,
                md.seq_lens,
                md.block_tables,
                md.query_start_loc,
                md.request_distribution,
            )
        return kv_cache, output_TNH
