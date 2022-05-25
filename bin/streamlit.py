import os
import time
import yaml
import warnings
import argparse
from tqdm import tqdm
import streamlit as st

import wave
import librosa
import soundfile
from pydub import AudioSegment

from asr_inference import Speech2Text
from espnet2.utils import config_argparse

warnings.filterwarnings("ignore")

ASR_TRAIN_CONFIG = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/config.yaml"
ASR_MODEL_FILE = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/valid.acc.ave_10best.pth"
CONFIG_FILE = "/opt/ml/input/espnet-asr/conf/decode_asr.yaml"

st.set_page_config(layout="wide")

def downsampling(audio_file, sampling_rate=16000):
    audio, rate = librosa.load(audio_file, sr=sampling_rate)
    return audio, rate

def save_uploaded_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.getbuffer())
    return os.path.join(directory, file.name)

def main():
    st.title("STT")

    st.header("음성 파일을 올려주세요.")
    with st.spinner("wait"):
        uploaded_file = st.file_uploader("Upload a file", type=["pcm", "wav", "flac", "m4a"])

    if uploaded_file:
        audio_file = save_uploaded_file("audio", uploaded_file)

        audio, rate = downsampling(audio_file)

        start_time = time.time()
        print("JOB START!!!")

        with open(CONFIG_FILE) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        speech2text = Speech2Text(
            asr_train_config=ASR_TRAIN_CONFIG, 
            asr_model_file=ASR_MODEL_FILE, 
            device='cuda',
            dtype='float32',
            **config
            )

        result = speech2text(audio)

        st.write(result[0][0])
        st.write(result[0][1])

        print(f"Total time: {time.time() - start_time}")
        print("JOB DONE!!!")


if __name__ == "__main__":
    main()