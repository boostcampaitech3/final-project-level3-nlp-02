import streamlit as st
import requests
import time
import os
import yaml
import librosa
import sys

from scipy.io import wavfile
from streamlit_player import st_player
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np

# 상위 디렉토리에서 dataset 가져오기
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataset import SplitOnSilenceDataset
from asr_inference import Speech2Text


BATCH_SIZE = 32
backend_address = "http://localhost:8001"
ASR_TRAIN_CONFIG = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/config.yaml"
ASR_MODEL_FILE = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/valid.acc.ave_10best.pth"
CONFIG_FILE = "/opt/ml/input/espnet-asr/conf/decode_asr.yaml"
# CONFIG_FILE = "/opt/ml/input/espnet-asr/conf/fast_decode_asr.yaml"
DOWNLOAD_FOLDER_PATH = "../../download/"

def downsampling(audio_file, sampling_rate=16000):
    audio, rate = librosa.load(audio_file, sr=sampling_rate)
    return audio, rate


def save_uploaded_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.getbuffer())
    return os.path.join(directory, file.name)


def collate_fn(batch):
    speech_dict = dict()
    speech_tensor = torch.tensor([])

    audio_max_len = 0
    for data in batch:
        audio_max_len = max(audio_max_len, len(data))

    for data in batch:
        zero_tensor = torch.zeros((1, audio_max_len - len(data)))
        data = torch.unsqueeze(data, 0)
        tensor_with_pad = torch.cat((data, zero_tensor), dim=1)
        speech_tensor = torch.cat((speech_tensor, tensor_with_pad), dim=0)
    
    speech_dict['speech'] = speech_tensor

    return speech_dict


def main():
    st.header("음성 파일을 올려주세요.")
    with st.spinner("wait"):
        uploaded_file = st.file_uploader("Upload a file", type=["pcm", "wav", "flac", "m4a"])

    if uploaded_file:
        audio_file = save_uploaded_file("audio", uploaded_file)

        audio, rate = downsampling(audio_file)
        st.write(audio)
        st.write(rate)
        st.write(max(audio))

        start_time = time.time()
        print("JOB START!!!")

        with open(CONFIG_FILE) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        with st.spinner("STT 작업을 진행하고 있습니다"):
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

    url = st.text_input("텍스트를 입력해주세요", type="default")
    # 텍스트 입력안내 문구
    if not url:
        st.write('유튜브 링크를 넣어주세요.')
        return

    data = {
        'url': url,
    }

    # 유튜브 음성파일을 가리키는 링크인지 확인하기.
    response = requests.post(
        url=f"{backend_address}/check_link",
        json=data
    )
    
    if response.status_code == 400:
        st.write('유튜브 링크 형식이 잘못되었습니다.')
        return
    specific_url = response.json()['url']

    # 유튜브 영상 삽입
    st_player(f"https://youtu.be/{specific_url}")

    data = {
        'url': specific_url,
    }

    # url에 맞는 유튜브 음성파일 가져오기
    response = requests.post(
        url=f"{backend_address}/set_voice",
        json=data
    )

    st.write(response)

    # 유튜브 음성파일 생성되었는지 확인
    with st.spinner("유튜브에서 음성을 추출하고 있습니다."):
     # 파일 만들어질때까지 spinner 빠져나가지 않기
        while True:
            # 1초마다 확인
            time.sleep(1)
            
            # 파일 있으면 while 탈출
            if os.path.exists(f'{DOWNLOAD_FOLDER_PATH}{specific_url}/{specific_url}.wav'):
                break

    # 음성 파일 STT 돌리기
    st.write("음성 추출이 완료되었습니다.")
    st.write("STT 작업이 진행중입니다.")
    # with st.spinner("STT 작업을 진행하고 있습니다"):
        # response = requests.post(
        #     url=f"{backend_address}/stt",
        #     json=data
        # )
        # response.raise_for_status() # ensure we notice bad responses
        # 파일 가져오기
        # with open(f'{DOWNLOAD_FOLDER_PATH}{specific_url}/{specific_url}.wav', 'r') as f:      
        #     audio_file = f.name

        # audio, rate = downsampling(audio_file)
        # st.write(specific_url)
        # downsampling
        # start_time = time.time()

        ### wavfile.read가 빠른데 단점이 있다.
        ### downsampling(librosa.load)는 출력값이 한 열로 나오고, 최댓값이 0.548정도이다.
        ### wavfile.read는 출력값이 두 열로 나오고, 최댓값이 36만 정도? 이다.
        ### 두 열로 나올때와, 최댓값을 어떻게 normalize시켜줘야할지 헷갈린다.
        ### normalize시키기 위한 연산을 할 때, 시간이 많이 들어 오히려 downsampling보다 많은 시간이 소요되는 것 같다.
        ### 이상한 출력이 나오는 원인인 것 같다.
        # fs, audio = wavfile.read(f"{DOWNLOAD_FOLDER_PATH}{specific_url}/{specific_url}.wav")
        # maximum = max(max(au) for au in audio)
        # for i in range(len(audio)):
        #     for j in range(len(audio[i])):
        #         audio[i][j] = audio[i][j] / (maximum * 2)
        # st.write(f"calc time: {time.time() - start_time}")
        # if fs != 16000:
        #     audio, rate = downsampling(audio)
        # st.write(f"downsampling time: {time.time() - start_time}")
        
    ## 우진님 파일 자르는거 도입, app.py 75번 째 줄부터

    # config file 설정
    with open(CONFIG_FILE) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    speech2text = Speech2Text(
        asr_train_config=ASR_TRAIN_CONFIG, 
        asr_model_file=ASR_MODEL_FILE, 
        device='cuda',
        dtype='float32',
        **config
    )
    
    dataset = SplitOnSilenceDataset(f'{DOWNLOAD_FOLDER_PATH}{specific_url}/{specific_url}.wav')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    start_time = time.time()
    print("JOB START!!!")
    talk_list = list()
    with st.spinner("STT 작업을 진행하고 있습니다"):
        for batch in loader:
            mid_time = time.time()
            batch = {k: v for k, v in batch.items() if not k.endswith("_lengths")}
            results = speech2text(**batch)
            temp_list = list()
            for bat in results:
                talk_list.append(bat[0])
                temp_list.append(bat[0])
            st.write(temp_list)

            # for result in results:
            #     with open(OUTPUT_TXT, "a") as f:
            #         f.write(result[0]+"\n")

            # with open(OUTPUT_TXT, "a") as f:
            #     f.write(f"Total time: {end_time - start_time}\n\n")

            mid_end_time = time.time()
            print(f"check time: {mid_end_time - mid_time}")

        end_time = time.time()
        print(f"Total time: {end_time - start_time}")
        print("JOB DONE!!!")
        print('!!!!', results, type(results))
    
    st.write(talk_list)
    st.write("STT 작업이 완료되었습니다.")
  

if __name__ == "__main__":
    main()

    # get_response = requests.post(
    #     url=f"{backend_address}/write",
    #     data=data,
    #     headers={"Content-Type": "application/json"}
    # )