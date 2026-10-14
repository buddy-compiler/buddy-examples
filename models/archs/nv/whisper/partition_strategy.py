#!/usr/bin/env python3
# ===- partition_strategy.py - Whisper layer split strategy ---------------===//
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
#
# ===----------------------------------------------------------------------===//
#
# Vertical split for Whisper encoder/decoder graphs using the same
# PartitionedGraphDriver path as DeepSeek (see docs/LayerPartitioning.md).
#
# Boundary op: one ScaledDotProductFlashAttentionForCpuOp per attention block.
# Encoder (6 layers): stem + 6 attentions → 7 partitions.
# Decoder (6 layers × self/cross): embed + 12 attentions → 13 partitions.
#
# Audio chip homogeneous cores consume grouped partitions (see AUDIO_SLICE_GROUPS).
#
# ===----------------------------------------------------------------------===//

from buddy.compiler.graph import SplitStrategy
from buddy.compiler.graph.operation import ScaledDotProductFlashAttentionForCpuOp

# Maps audio chip slices → contiguous PartitionedGraphDriver partition indices.
AUDIO_SLICE_GROUPS = {
    "enc_0": [0, 1, 2],  # stem + encoder layers 0–1
    "enc_1": [3, 4],  # encoder layers 2–3
    "enc_2": [5, 6],  # encoder layers 4–5
    "dec_0": [0, 1, 2, 3, 4, 5, 6],  # embed + decoder layers 0–2
    "dec_1": [7, 8, 9, 10, 11, 12],  # decoder layers 3–5 + lm_head
}


def layer_split_strategy(kind: str) -> SplitStrategy:
    """Return Whisper split strategy for kind in {encoder, decoder}."""
    if kind in ("encoder", "prefill"):
        # prefill alias keeps import_model.layer_split_strategy(kind) usable.
        return SplitStrategy(
            name="whisper_encoder_attn_boundaries",
            parallel_num=1,
            ops_count=[],
            stage_boundary_op=ScaledDotProductFlashAttentionForCpuOp,
            stage_boundary_op_num=6,
        )
    if kind in ("decoder", "decode"):
        return SplitStrategy(
            name="whisper_decoder_attn_boundaries",
            parallel_num=1,
            ops_count=[],
            stage_boundary_op=ScaledDotProductFlashAttentionForCpuOp,
            stage_boundary_op_num=12,
        )
    raise ValueError(f"unknown Whisper layer split kind: {kind}")
