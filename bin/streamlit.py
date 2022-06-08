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
        # st.write(result[0][1])

        print(f"Total time: {time.time() - start_time}")
        print("JOB DONE!!!")
    

    # 유튜브 추출
    youtube_link = st.text_input("유튜브 링크를 입력해주세요")

    if st.button('버튼을 클릭하세요'):
        # youtube link 유효성 체크
        if "youtu.be" not in youtube_link and "www.youtube.com" not in youtube_link:
            st.error("유효한 유튜브 링크를 넣어주세요.")
            return
        # 유효성 검사 완료
        # else:    
        # 상세 주소 가져오기
        specific_link = youtube_link.split('/')[-1]
        # 링크 그대로 가져왔을 때 후처리
        if "?v=" in specific_link:
            specific_link = specific_link.split('?v=')[1].split('&')[0]
        st.write(specific_link)

        # youtube link를 통해 음성 가져오기(recog_youtube.sh 이용)
        with st.spinner("유튜브에서 음성을 추출하고 있습니다."):
            # 폴더 없으면 만들기
            if not os.path.exists(f'./download/{specific_link}/'):
                os.makedirs(specific_link)

            # 파일 없으면 가져와
            if not os.path.exists(f'./download/{specific_link}/{specific_link}.wav'):
                sh_result = os.popen(f'bash tools/recog_youtube.sh --url {specific_link} --download-dir download/{specific_link}')

                # 파일 만들어질때까지 spinner 빠져나가지 않기
                while True:
                    # 1초마다 확인
                    time.sleep(1)
                    
                    # 파일 있으면 while 탈출
                    if os.path.exists(f'./download/{specific_link}/{specific_link}.wav'):
                        break
                
            # 파일 가져오기
            with open(f'./download/{specific_link}/{specific_link}.wav', 'r') as f:      
                audio_file = f.name
            st.write(audio_file)

            # 파일 가져오기 2
            # audio_file = f'./download/{specific_link}/{specific_link}.wav'

        # if audio_file:
        # # 음성을 가져와서 STT로 추출해보기
        with st.spinner("음성을 STT로 반환하고 있습니다."):
            # downsampling
            audio, rate = downsampling(audio_file) # librosa.load라서 40초 걸림
            # scipy.io.wavfile
            # wavfile.read() -> 1.5초
            # numpy로 반환, 0, 32767 - Normalize시켜줘야 함.
            # dataset.py __items__() 참고


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

            print(f"Total time: {time.time() - start_time}")
            print("JOB DONE!!!")


if __name__ == "__main__":
    main()
