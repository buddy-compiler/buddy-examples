#!/usr/bin/env python3
# ===- import-whisper.py -------------------------------------------------------
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
# ===---------------------------------------------------------------------------
#
# AOT importer for Whisper (buddy-cli / .rax) plus optional stage/layer cut.
#
# Always writes (compat with single_forward runtime):
#   forward.mlir, subgraph0.mlir, arg0.data
#
# With --layer-partitioned (default ON):
#   sg_encoder/, sg_decoder/ whole stage graphs
#   layer_partitioned/encoder/*.mlir + decoder/*.mlir
#   layer_partitioned/partition_manifest.json  (includes audio slice groups)
#   sg_dap.mlir  (copy of Buddy DAP whisper preprocess)
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy
import torch
import torch.nn as nn
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver, PartitionedGraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import WhisperForConditionalGeneration

from partition_strategy import AUDIO_SLICE_GROUPS, layer_split_strategy

_CODEGEN_DIR = Path(__file__).resolve().parent
DAP_MLIR = None
for _root in [_CODEGEN_DIR, *_CODEGEN_DIR.parents]:
    cand = (
        _root
        / "compiler/thirdparty/buddy-mlir/frontend/Interfaces/lib/DAP-extend.mlir"
    )
    if cand.is_file():
        DAP_MLIR = cand
        break


class WhisperEncoderModule(nn.Module):
    def __init__(self, model: WhisperForConditionalGeneration):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_features=input_features).last_hidden_state


class WhisperDecoderModule(nn.Module):
    def __init__(self, model: WhisperForConditionalGeneration):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj = model.proj_out

    def forward(
        self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
        ).last_hidden_state
        return self.proj(hidden)


def die(msg: str) -> None:
    raise SystemExit(f"[import-whisper] ERROR: {msg}")


def patch_packed_sequence() -> None:
    try:
        import transformers.masking_utils as mu

        mu.find_packed_sequence_indices = lambda *a, **k: None
    except Exception as err:
        print(f"[import-whisper] packed-sequence patch skipped: {err}")


def import_graph(module: nn.Module, *example_args):
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )
    with torch.no_grad():
        graphs = compiler.importer(module, *example_args)
    if len(graphs) != 1:
        die(f"expected 1 graph, got {len(graphs)}")
    graph = graphs[0]
    params = compiler.imported_params[graph]
    graph.fuse_ops([simply_fuse])
    return graph, params


def write_whole_graph(graph, params, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    driver = GraphDriver(graph)
    if len(driver.subgraphs) < 1:
        die(f"{prefix}: GraphDriver produced no subgraphs")
    driver.subgraphs[0].lower_to_top_level_ir()
    with open(output_dir / "subgraph.mlir", "w") as f:
        print(driver.subgraphs[0]._imported_module, file=f)
    with open(output_dir / "forward.mlir", "w") as f:
        print(driver.construct_main_graph(True), file=f)
    if len(params) == 0:
        die(f"{prefix}: zero params")
    numpy.concatenate(
        [p.detach().cpu().numpy().reshape([-1]) for p in params]
    ).astype(numpy.float32).tofile(output_dir / "params.data")
    print(f"[import-whisper] Wrote stage {prefix} → {output_dir}")


def write_partitioned(
    graph, kind: str, partition_root: Path, expected: int
) -> list[str]:
    strategy = layer_split_strategy(kind)
    driver = PartitionedGraphDriver(graph, strategy)
    n = len(driver.subgraphs)
    if n != expected:
        die(f"{kind}: expected {expected} partitions, got {n}")
    out_dir = partition_root / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i, subgraph in enumerate(driver.subgraphs):
        try:
            subgraph.lower_to_top_level_ir()
        except Exception as err:
            die(
                f"{kind} partition {i}/{n} lower_to_top_level_ir failed: {err}. "
                "This is a PartitionedGraphDriver/tosa gap for this subgraph; "
                "do not paper over it in the chip tree."
            )
        name = f"subgraph0_{kind}{i}.mlir"
        path = out_dir / name
        with open(path, "w") as f:
            print(subgraph._imported_module, file=f)
        files.append(f"{kind}/{name}")
        print(f"[import-whisper] Written: layer_partitioned/{kind}/{name}")
    combined = driver.construct_combined_main_graph(True)
    with open(out_dir / f"forward_{kind}.mlir", "w") as f:
        print(combined, file=f)
    files.append(f"{kind}/forward_{kind}.mlir")
    print(f"[import-whisper] Written: layer_partitioned/{kind}/forward_{kind}.mlir")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Whisper model AOT importer")
    parser.add_argument("--output-dir", type=str, default="./")
    parser.add_argument("--spec", type=str, default=None)
    parser.add_argument(
        "--layer-partitioned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export encoder/decoder stage graphs + encoder PartitionedGraphDriver cuts",
    )
    parser.add_argument(
        "--partition-decoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also layer-partition the decoder. Default OFF: lowering child "
            "decoder partitions currently hits None attn_mask in tosa "
            "(PartitionedGraphDriver edge). Encoder partitions remain ON."
        ),
    )
    parser.add_argument(
        "--decoder-seq-len",
        type=int,
        default=16,
        help="Decoder example seq len for stage/partition import",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = os.environ.get("WHISPER_MODEL_PATH")
    if not model_path and args.spec:
        with open(args.spec) as f:
            model_path = json.load(f).get("hf_model_path")
    if not model_path:
        model_path = "openai/whisper-base"

    print(f"[import-whisper] Loading model from: {model_path}")
    patch_packed_sequence()
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.config.use_cache = False
    model.eval()

    input_features = torch.zeros((1, 80, 3000), dtype=torch.float32)
    decoder_input_ids = torch.zeros((1, 448), dtype=torch.long)
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )
    with torch.no_grad():
        graphs = dynamo_compiler.importer(
            model,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
        )
    if len(graphs) != 1:
        die(f"monolithic import expected 1 graph, got {len(graphs)}")
    graph = graphs[0]
    params = dynamo_compiler.imported_params[graph]
    graph.fuse_ops([simply_fuse])
    driver = GraphDriver(graph)
    driver.subgraphs[0].lower_to_top_level_ir()
    with open(output_dir / "subgraph0.mlir", "w") as f:
        print(driver.subgraphs[0]._imported_module, file=f)
    with open(output_dir / "forward.mlir", "w") as f:
        print(driver.construct_main_graph(True), file=f)
    numpy.concatenate(
        [p.detach().numpy().reshape([-1]) for p in params]
    ).tofile(output_dir / "arg0.data")
    print(
        f"[import-whisper] Wrote forward.mlir, subgraph0.mlir, arg0.data → {output_dir}"
    )

    if not args.layer_partitioned:
        return 0

    if DAP_MLIR is None or not DAP_MLIR.is_file():
        die(f"DAP-extend.mlir not found (searched from {_CODEGEN_DIR})")
    shutil.copyfile(DAP_MLIR, output_dir / "sg_dap.mlir")
    print(f"[import-whisper] Wrote sg_dap.mlir from {DAP_MLIR}")

    mel = torch.zeros((1, 80, 3000), dtype=torch.float32)
    enc_states = torch.zeros((1, 1500, 512), dtype=torch.float32)
    dec_ids = torch.zeros((1, args.decoder_seq_len), dtype=torch.long)

    enc_graph, enc_params = import_graph(WhisperEncoderModule(model), mel)
    dec_graph, dec_params = import_graph(
        WhisperDecoderModule(model), dec_ids, enc_states
    )

    write_whole_graph(enc_graph, enc_params, output_dir / "sg_encoder", "sg_encoder")
    write_whole_graph(dec_graph, dec_params, output_dir / "sg_decoder", "sg_decoder")

    partition_root = output_dir / "layer_partitioned"
    if partition_root.exists():
        shutil.rmtree(partition_root)
    enc_files = write_partitioned(enc_graph, "encoder", partition_root, expected=7)

    dec_files: list[str] = []
    decoder_partition_status = "stage_only"
    if args.partition_decoder:
        dec_files = write_partitioned(
            dec_graph, "decoder", partition_root, expected=13
        )
        decoder_partition_status = "layer_partitioned"
    else:
        print(
            "[import-whisper] WARNING: decoder layer partition skipped "
            "(--partition-decoder). Homogeneous dec_0/dec_1 share sg_decoder "
            "until PartitionedGraphDriver+tosa None attn_mask is fixed."
        )

    manifest = {
        "model_family": "whisper",
        "encoder_subgraphs": 7,
        "decoder_subgraphs": (
            13 if decoder_partition_status == "layer_partitioned" else 0
        ),
        "decoder_partition_status": decoder_partition_status,
        "encoder_files": enc_files,
        "decoder_files": dec_files,
        "audio_slice_groups": AUDIO_SLICE_GROUPS,
        "sg_dap": "sg_dap.mlir",
        "sg_encoder": "sg_encoder/",
        "sg_decoder": "sg_decoder/",
        "policy": "Core-designers consume partitions; do not re-cut",
    }
    with open(partition_root / "partition_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(
        "[import-whisper] Layer partitioned export complete: "
        f"{manifest['encoder_subgraphs']} encoder partitions; "
        f"decoder={decoder_partition_status}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
