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
from torch import Tensor
import os

from pydub import AudioSegment

from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kospeech.models import (
    SpeechTransformer,
    Jasper,
    DeepSpeech2,
    ListenAttendSpell,
    Conformer,
)

from postprocess import remove_repetition


def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
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

def save_uploaded_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.getbuffer())
    return os.path.join(directory, file.name)


import streamlit as st
# # from confirm_button_hack import cache_on_button_press


# # SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


def main():
    st.title("STT Model")

    st.header("음성 파일을 올려주세요.")
    with st.spinner("Uploading..."):
        uploaded_file = st.file_uploader("Upload a file", type=["pcm", "wav","flac", "m4a"])
        # st.success("Done!")
    if uploaded_file:
        if os.path.splitext(uploaded_file.name)[1] == '.m4a':
            # wav_filename = r"F:\20211210_151013.wav"
            wav_filename = f"./audio/{uploaded_file.name}.wav"
            track = AudioSegment.from_file(uploaded_file.name,  format= 'm4a')
            file_handle = track.export(wav_filename, format='wav')
            print(file_handle)
            # audio_path = f"./{wav_filename}"
        else:
            audio_path = save_uploaded_file("audio", uploaded_file)
        print(os.path.splitext(uploaded_file.name)[1])
        print(audio_path)
        model_path = "/opt/ml/input/kospeech/bin/outputs/2022-05-17/08-35-20/model.pt"
        device = 'cuda'
        
        with st.spinner("In progress..."):
            feature = parse_audio(audio_path, del_silence=True)
            input_length = torch.LongTensor([len(feature)])
            vocab = KsponSpeechVocabulary('/opt/ml/input/kospeech/data/vocab/aihub_character_vocabs.csv')
            model = torch.load(model_path, map_location=lambda storage, loc: storage).to(device)
            if isinstance(model, nn.DataParallel):
                model = model.module
            model.eval()

            if isinstance(model, ListenAttendSpell):
                model.encoder.device = device
                model.decoder.device = device

                y_hats = model.recognize(feature.unsqueeze(0), input_length)
            elif isinstance(model, DeepSpeech2):
                model.device = device
                feature = feature.unsqueeze(0).to(torch.device("cuda"))
                y_hats = model.recognize(feature, input_length)
            elif isinstance(model, SpeechTransformer) or isinstance(model, Jasper) or isinstance(model, Conformer):
                y_hats = model.recognize(feature.unsqueeze(0), input_length)

            sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())

            st.header("Text")
            answer = remove_repetition(sentence)
            print(answer == '')
            st.write(answer)
        if answer == '':
            st.error('결과가 없습니다. 잘못된 음성 파일인지 확인해주세요.')
        else:
            st.success("Done")

main()