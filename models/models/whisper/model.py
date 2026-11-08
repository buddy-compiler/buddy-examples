# ===- model.py ----------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# Whisper-tiny logits of one full decoder window.
#
# ===---------------------------------------------------------------------------

from pathlib import Path

import torch
import transformers.masking_utils as masking_utils
from transformers import WhisperForConditionalGeneration

checkpoint = Path(__file__).resolve().parent / "output/whisper"


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features, decoder_input_ids):
        return self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        ).logits


def load():
    """Whisper-tiny returning the decoder logits, ready to import."""
    # The decoder positions form one sequence: keep the all-zero packed-sequence
    # mask that tracing would compute out of the graph.
    masking_utils.find_packed_sequence_indices = lambda *args, **kwargs: None
    return Logits(WhisperForConditionalGeneration.from_pretrained(checkpoint)).eval()


def inputs():
    """Example inputs of the graph: 30 s of log-mel features and 448 tokens."""
    return (
        torch.zeros((1, 80, 3000), dtype=torch.float32),
        torch.zeros((1, 448), dtype=torch.int64),
    )
