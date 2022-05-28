import streamlit as st
import requests
import time
import os
import yaml
import librosa

from utils.utils import Speech2Text
from scipy.io import wavfile


backend_address = "http://localhost:8001"
ASR_TRAIN_CONFIG = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/config.yaml"
ASR_MODEL_FILE = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/valid.acc.ave_10best.pth"
CONFIG_FILE = "/opt/ml/input/espnet-asr/conf/decode_asr.yaml"
# CONFIG_FILE = "/opt/ml/input/espnet-asr/conf/fast_decode_asr.yaml"
DOWNLOAD_FOLDER_PATH = "../../download/"

def downsampling(audio_file, sampling_rate=16000):
    audio, rate = librosa.load(audio_file, sr=sampling_rate)
    return audio, rate

def main():
    st.write('반갑습니다.')
    response = requests.get(f"{backend_address}/")
    st.write(response)
    label = response.json()
    st.write(label)
    st.write('반갑습니다.~')

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
    
    ### 필요없을거같아서 주석처리
    # # 생성한 파일 가져오기
    # response = requests.post(
    #     url=f"{backend_address}/get_voice",
    #     json=data
    # )
    # st.write(response.json())
    # print(response.json())
    # 음성 파일 STT 돌리기
    st.write("음성 추출이 완료되었습니다.")
    st.write("STT 작업이 진행중입니다.")
    with st.spinner("STT 작업을 진행하고 있습니다"):
        # response = requests.post(
        #     url=f"{backend_address}/stt",
        #     json=data
        # )
        # response.raise_for_status() # ensure we notice bad responses
        # 파일 가져오기
        with open(f'{DOWNLOAD_FOLDER_PATH}{specific_url}/{specific_url}.wav', 'r') as f:      
            audio_file = f.name

        # downsampling
        fs, audio = wavfile.read(f"{DOWNLOAD_FOLDER_PATH}{specific_url}/{specific_url}.wav")
        if fs != 16000:
            audio, rate = downsampling(audio_file)

        print('####@@@@', audio.shape)

        ## 우진님 파일 자르는거 도입해야댐

        start_time = time.time()
        print("JOB START!!!")

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

        result = speech2text(audio)
        print(f"Total time: {time.time() - start_time}")
        print("JOB DONE!!!")
        print('!!!!', result, type(result))


    
    st.write("STT 작업이 완료되었습니다.")
    st.write(result)
  



if __name__ == "__main__":
    main()

    # get_response = requests.post(
    #     url=f"{backend_address}/write",
    #     data=data,
    #     headers={"Content-Type": "application/json"}
    # )