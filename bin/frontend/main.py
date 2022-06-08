from multiprocessing.spawn import prepare
import streamlit as st
import requests
import time
import re
import os
import yaml
import librosa
import sys
# import pickle

# from scipy.io import wavfile
from streamlit_player import st_player
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
# from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
# from konlpy.tag import Okt
from itertools import combinations

# 상위 디렉토리에서 dataset 가져오기
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataset import SplitOnSilenceDataset
from asr_inference import Speech2Text
from utils import collate_fn, processing, post_process, dell_loop, get_split


st.set_page_config(layout="wide")

# BATCH_SIZE = 32
BATCH_SIZE = 8
backend_address = "http://localhost:8001"
ASR_TRAIN_CONFIG = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/config.yaml"
ASR_MODEL_FILE = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/valid.acc.ave_10best.pth"
# CONFIG_FILE = "/opt/ml/input/espnet-asr/conf/decode_asr.yaml"
CONFIG_FILE = "/opt/ml/input/espnet-asr/conf/fast_decode_asr.yaml"
DOWNLOAD_FOLDER_PATH = "../../download/"

# model load
model_path='/opt/ml/input/espnet-asr/bin/postprocessing_model/'

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(model_path,
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
model = GPT2LMHeadModel.from_pretrained(model_path)


def downsampling(audio_file, sampling_rate=16000):
    audio, rate = librosa.load(audio_file, sr=sampling_rate)
    return audio, rate


def save_uploaded_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.getbuffer())
    return os.path.join(directory, file.name)


def change_bool_state_true():
    st.session_state.push_stop_button = True


#@st.cache
@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def load_model():
    #model_ = BartForConditionalGeneration.from_pretrained('./kobart_summary') # minjun 기본
    #model_ = BartForConditionalGeneration.from_pretrained('./kobart_summary2_v_0') # minjun 합친거로 학습
    # model_ = BartForConditionalGeneration.from_pretrained('../kobart_summary2_v_1') # minjun 합친거로 학습
    model_ = BartForConditionalGeneration.from_pretrained('../kobart_summary4') # younhye
    return model_


def main():
    # push button이 없으면 설정해줌
    if 'push_stop_button' not in st.session_state:
        st.session_state.push_stop_button = False

    # 전체 대사가 없으면 설정해줌
    if 'youtube_scripts' not in st.session_state:
        st.session_state.youtube_scripts = list()

    # stop_button을 눌러서 온 게 아니라면 초기화
    if st.session_state.push_stop_button == False:
        st.session_state.youtube_scripts = list()

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

        print(f"Total time: {time.time() - start_time}")
        print("JOB DONE!!!")

    url = st.text_input("유튜브 링크를 넣어주세요.", type="default")
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
    with st.spinner("유튜브에서 음성을 추출하고 있습니다."):
        # url에 맞는 유튜브 음성파일 가져오기
        response = requests.post(
            url=f"{backend_address}/set_voice",
            json=data
        )

    # 음성 파일 STT 돌리기
    st.write("음성 추출이 완료되었습니다.")
    st.write("STT 작업을 진행합니다.")
    if st.button(label="작업 중지하기", on_click=change_bool_state_true()):
        st.warning('작업을 중지합니다.')
        if st.button("다시 시작하기"):
            st.session_state.youtube_scripts = list()
            st.write('다시 시작합니다.')
        if st.session_state.youtube_scripts:
            for word in st.session_state.youtube_scripts:
                st.write(word[0], word[1])
            st.session_state.push_stop_button = False

        st.stop()
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
    
    st.write('데이터를 나누고 있습니다.')
    dataset = SplitOnSilenceDataset(f'{DOWNLOAD_FOLDER_PATH}{specific_url}/{specific_url}.wav')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    start_time = time.time()
    print("JOB START!!!")
    talk_list = list()
    results = list()
    with st.spinner("STT 작업을 진행하고 있습니다"):
        for batch in loader:
            timelines = batch.pop("timeline")
            mid_time = time.time()
            batch = {k: v for k, v in batch.items() if not k.endswith("_lengths")}
            results = speech2text(**batch)
            temp_list = list()
            temp_words = ''
            for timeline, bat in zip(timelines, results):
                pretty_time = f"{int(timeline)//60:02}:{int(timeline)%60:02}"
                # words = post_process(bat[0])

                # 한 음절이면 저장하고 continue
                if len(bat[0].split()) == 1:
                    temp_words = bat[0] + ' ' + temp_words
                    continue
                
                # 저장한 words가 있으면 앞에 붙여주기
                if temp_words.strip():
                    
                    words = temp_words + bat[0]
                    temp_words = ''
                else:
                    words = bat[0]


                words = post_process(model, koGPT2_TOKENIZER, words)
                talk_list.append([pretty_time, words])
                temp_list.append([pretty_time, words])

                st.session_state.youtube_scripts.append([pretty_time, words])
                
                # st.columns
                col1, col2 = st.columns([1, 1])
                # 시간대별로 확장
                with col1.expander(label=f"{pretty_time}", expanded=False):
                    st_player(f"https://youtu.be/{specific_url}&t={timeline}s")
                
                col2.write(words)

            mid_end_time = time.time()
            print(f"check time: {mid_end_time - mid_time}")

        end_time = time.time()
        print(f"Total time: {end_time - start_time}")
        print("JOB DONE!!!")
        print('!!!!', results, type(results))

    st.write(talk_list)
    
    st.write("STT 작업이 완료되었습니다.")

    temp_talk_list = [talk[1] for talk in talk_list]
    data = {
        'talk_list': temp_talk_list,
    }

    st.write("요약 작업이 진행중입니다.")
    # 유튜브 음성파일을 가리키는 링크인지 확인하기.
    response = requests.get(
        url=f"{backend_address}/summary",
        json=data
    )
    # print('####', response.json())
    outputs = response.json()['outputs']
    st.write(outputs)

    data = {
        'talk_list': temp_talk_list,
    }

    # print(type(data['talk_list']))
    # print('@@@@', data['talk_list'])

    response = requests.get(
        # url=f"{backend_address}/summary",
        url=f"{backend_address}/keyword",
        json=data
    )

    # print('####', response.json())
    # print(response)
    # print(response.json())
    results = response.json()['outputs']
    for result in results:
        st.write(','.join(map(str, result)))
    # st.write(response.json()['outputs'][0])
    # st.write(response.json()['outputs'][1])
    # st.write(response.json()['outputs'][2])
  

if __name__ == "__main__":
    main()

    # get_response = requests.post(
    #     url=f"{backend_address}/write",
    #     data=data,
    #     headers={"Content-Type": "application/json"}
    # )