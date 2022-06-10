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
# from ..mrc_utils.func import MRC
from PIL import Image

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


# @st.cache()
def verfity_link(url):
    data = {
        'url': url,
    }

    # 유튜브 음성파일을 가리키는 링크인지 확인하기.
    response = requests.post(
        url=f"{backend_address}/check_link",
        json=data
    )
    
    if response.status_code == 400:
        return False
    
    specific_url = response.json()['url']
    return specific_url


# @st.cache()
def download_voice(specific_url):
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
    return


# @st.cache(hash_funcs={torch.jit._script.RecursiveScriptModule : lambda _: None})
def divide_data(specific_url, CONFIG_FILE):
    ### config file 설정 ###
    with open(CONFIG_FILE) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    speech2text = Speech2Text(
        asr_train_config=ASR_TRAIN_CONFIG, 
        asr_model_file=ASR_MODEL_FILE, 
        device='cuda',
        dtype='float32',
        **config
    )
    ###

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
    
    st.write("STT 작업이 완료되었습니다.")
    return talk_list


# @st.cache()
def get_summary(talk_list):
    temp_talk_list = [talk[1] for talk in talk_list]
    data = {
        'talk_list': temp_talk_list,
    }

    # 유튜브 음성파일을 가리키는 링크인지 확인하기.
    response = requests.get(
        url=f"{backend_address}/summary", # cos 유사도로 끊기
        # url=f"{backend_address}/summary_before", # 1000자씩 끊기
        json=data
    )
    # print('####', response.json())
    outputs = response.json()['outputs']
    st.write(outputs)
    return temp_talk_list


# @st.cache()
def set_keyword(temp_talk_list):
    data = {
        'talk_list': temp_talk_list,
    }

    response = requests.get(
        url=f"{backend_address}/keyword",
        json=data
    )

    results = response.json()['outputs']
    unique_keywords = list()
    for result in results:
        unique_keywords.extend(result[0].split())
        # st.warning(' '.join(map(str, result)))
    
    str_keyword = ' '.join(map(str, list(set(unique_keywords))))
    st.warning(str_keyword)
    return


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

    ### 로고 보여주기 ###
    image = Image.open('professor_logo.PNG')
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.write()
    with col2:
        st.image(image)
    with col3:
        st.write()
    ###

    ### 음성파일 업로드 ###
    # st.header("음성 파일을 올려주세요.")
    # with st.spinner("wait"):
    #     uploaded_file = st.file_uploader("Upload a file", type=["pcm", "wav", "flac", "m4a"])

    # if uploaded_file:
    #     audio_file = save_uploaded_file("audio", uploaded_file)

    #     audio, rate = downsampling(audio_file)
        
    #     start_time = time.time()
    #     print("JOB START!!!")

    #     with open(CONFIG_FILE) as f:
    #         config = yaml.load(f, Loader=yaml.FullLoader)

    #     with st.spinner("STT 작업을 진행하고 있습니다"):
    #         speech2text = Speech2Text(
    #             asr_train_config=ASR_TRAIN_CONFIG, 
    #             asr_model_file=ASR_MODEL_FILE, 
    #             device='cuda',
    #             dtype='float32',
    #             **config
    #             )

    #         result = speech2text(audio)

    #         st.write(result[0][0])

    #     print(f"Total time: {time.time() - start_time}")
    #     print("JOB DONE!!!")
    ###

    ### STT 유형, 링크 입력받기
    st.subheader("1. STT 방법을 선택해주세요.")
    choice_STT_mode = ['빠르게 STT', '정확하게 STT']
    status = st.radio('', choice_STT_mode)
    st.write('')
    st.write('')
    if status == choice_STT_mode[0]:
        CONFIG_FILE = "/opt/ml/input/espnet-asr/conf/fast_decode_asr.yaml"
    elif status == choice_STT_mode[1]:
        CONFIG_FILE = "/opt/ml/input/espnet-asr/conf/decode_asr.yaml"

    st.subheader("2. 유튜브 링크를 넣어주세요.")
    url = st.text_input("", type="default", key='youtube_link')
    # 텍스트 입력안내 문구
    if not url:
        st.write('유튜브 링크를 넣어주세요.')
        return
    ###
    st.write('')
    st.write('')

    ### 유튜브 링크 체크 ###
    specific_url = verfity_link(url)
    if specific_url == False:
        st.write('유튜브 링크 형식이 잘못되었습니다.')
        return
    ###

    ### 유튜브 영상 삽입 ###
    download_voice(specific_url)
    ###

    ### 음성 파일 STT 돌리기 ###
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
    ###

    ### 데이터 나누기 ###
    talk_list = divide_data(specific_url, CONFIG_FILE)
    ###

    ### 요약하기 ###
    st.subheader("요약")
    with st.spinner('요약 작업을 진행하고 있어요'):
        temp_talk_list = get_summary(talk_list)
    ###

    ### 키워드 추출하기 ###
    st.subheader("Keywords")
    with st.spinner('키워드 추출을 진행하고 있어요'):
        set_keyword(temp_talk_list)
    ###

    ### MRC ###
    # st.title("무엇이든 물어보세요!")
    
    # query = st.text_input("질문을 입력해주세요")
    # context = ' '.join(map(str, temp_talk_list))

    # if query != '' and context != '':
    #     prediction = MRC(query, context)


    # if st.button("알려주세요!"):    
    #     st.subheader(f'정답은??! {prediction}'.format(prediction))
    ###

    ### Timeline MRC ###
    st.subheader('검색할 키워드를 입력하세요.')
    query = st.text_input('', key='query')

    if query:
        data = {
            'question': [query],
            'talk_list': talk_list,
        }
        print('#@!#@!', type(data), data)
        # print('!@#!@#', data.question)

        response = requests.get(
            url=f"{backend_address}/query",
            json=data
        )

        print('@@@@', response.json()['outputs'])
        st.write("관련 있는 타임라인은 다음과 같습니다.")
        for text_result in response.json()['outputs']:
            st.write(text_result[0], text_result[1])
        # st.write('####', response.json()['outputs'])
    ###
  

if __name__ == "__main__":
    main()

    # get_response = requests.post(
    #     url=f"{backend_address}/write",
    #     data=data,
    #     headers={"Content-Type": "application/json"}
    # )