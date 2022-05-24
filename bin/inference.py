# Copyright (c) 2020, Soohwan Kim. All rights reserved.
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

import argparse
import torch
import torch.nn as nn
import numpy as np
import torchaudio
import os
from torch import Tensor

from postprocess import remove_repetition

from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kospeech.models import (
    SpeechTransformer,
    Jasper,
    DeepSpeech2,
    ListenAttendSpell,
    Conformer,
)

def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm'):
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)


parser = argparse.ArgumentParser(description='KoSpeech')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--audio_path', type=str, required=True)
parser.add_argument('--output_unit', type=str, required=False, default='character')
parser.add_argument('--sp_model_path', type=str, required=False, default=None)
parser.add_argument('--device', type=str, required=False, default='cpu')
opt = parser.parse_args()

extension = os.path.splitext(opt.audio_path)[1][1:]

feature = parse_audio(opt.audio_path, del_silence=True, audio_extension=extension)
input_length = torch.LongTensor([len(feature)])

print(f"{parser.output_unit}으로 진행합니다.")

if parser.output_unit == 'subword':
    vocab = KsponSpeechVocabulary(vocab_path="", output_unit='subword', sp_model_path=parser.sp_model_path)
else:
    try:
        vocab = KsponSpeechVocabulary('/opt/ml/input/kospeech/vocab/aihub_labels.csv') # 학습한 csv 파일
    except:
        print("학습된 csv 파일이 없어 기본 csv파일로 진행합니다.")
        vocab = KsponSpeechVocabulary('data/vocab/aihub_character_vocabs.csv')

model = torch.load(opt.model_path, map_location=lambda storage, loc: storage).to(opt.device)
if isinstance(model, nn.DataParallel):
    model = model.module
model.eval()

if isinstance(model, ListenAttendSpell):
    model.encoder.device = opt.device
    model.decoder.device = opt.device

    y_hats = model.recognize(feature.unsqueeze(0), input_length)
elif isinstance(model, DeepSpeech2):
    model.device = opt.device
    feature = feature.unsqueeze(0).to(torch.device("cuda"))
    y_hats = model.recognize(feature, input_length)
elif isinstance(model, SpeechTransformer) or isinstance(model, Jasper) or isinstance(model, Conformer):
    model.device = opt.device
    feature = feature.unsqueeze(0).to(torch.device("cuda"))
    y_hats = model.recognize(feature, input_length)

sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
print(remove_repetition(sentence))
