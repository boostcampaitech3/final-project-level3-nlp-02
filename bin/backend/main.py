from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import uvicorn
import os
import torch
import sys

from pytube import YouTube
from moviepy.editor import *
from transformers import ElectraForPreTraining, ElectraTokenizer, pipeline
from transformers.models.bart import BartForConditionalGeneration
from kobart import get_kobart_tokenizer
import time

# 상위 디렉토리에서 가져오기
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import dell_loop, get_split, make_specific_link, summary_post_processing, get_tfidf_vec, get_similar
from key_bert import load_embeddings, get_candidates, dist_keywords, max_sum_sim

app = FastAPI()

DOWNLOAD_FOLDER_PATH = "../../download/"


def load_model():
    # model_ = BartForConditionalGeneration.from_pretrained('../kobart_summary') # minjun 기본
    # model_ = BartForConditionalGeneration.from_pretrained('../kobart_summary2_v_1') # minjun 합친거로 학습
    model_ = BartForConditionalGeneration.from_pretrained('../kobart_summary4') # younhye
    return model_

def load_postprocess_model():
    #koelectra 모델 - 이상한 토큰 찾아서 masking
    discriminator = ElectraForPreTraining.from_pretrained("monologg/koelectra-base-v3-discriminator")
    electra_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    # KoBigBird
    fill_model = pipeline(
        "fill-mask",
        model='monologg/kobigbird-bert-base',
        tokenizer=('monologg/kobigbird-bert-base', {'use_fast':True}),
        framework = 'pt',
        top_k  = 1
        )

    return discriminator, electra_tokenizer, fill_model


def get_keyword(text: str, top_n: int=10):
    candidates = get_candidates(text)
    doc_embedding, candidate_embeddings = load_embeddings(text, candidates)

    results = list()
    results.append([dist_keywords(doc_embedding, candidate_embeddings, candidates, top_n=top_n)])
    results.append([max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=top_n, nr_candidates=top_n*2)])
    return results


# "유튜브 링크" dict 데이터 유효성 체크해주는 pydantic
class Url_check(BaseModel):
    url: str


class list_check(BaseModel):
    talk_list: list


class dict_check(BaseModel):
    question: list
    talk_list: list


# youtube link를 검증합니다.
@app.post("/check_link", description="입력을 저장합니다.")
def check_link(origin_url: Url_check):
    # 받아온 url 검증, 유효하지 않으면 False 반환
    specific_url = make_specific_link(origin_url)

    # 유효하지 않으면 400에러, 유튜브 링크 오류 안내
    if specific_url == False:
        return JSONResponse(
            status_code=400,
            content={"message": "유효한 유튜브 링크가 아닙니다."}
        )

    # 유효하면 정상 링크 안내
    return JSONResponse(
        status_code=200,
        content={
                "message": "유효한 유튜브 링크입니다.", 
                "url": specific_url
            }
    )


# 음성 파일 저장해주는 함수 -> 반환 없음.
@app.post("/set_voice", description="음성 파일을 저장합니다.")
def set_voice(specific_url: Url_check):
    specific_url = specific_url.url
# try:
    # 유튜브 링크 기반으로 폴더에 저장하기(추후 S3 등 확장)
    
    # 폴더 없으면 만들기
    if not os.path.exists(f'{DOWNLOAD_FOLDER_PATH}{specific_url}/'):
        os.makedirs(f'{DOWNLOAD_FOLDER_PATH}{specific_url}')

    # 만든 파일이 빈 파일이면 에러가 나는 경우 발생 -> 파일 크기가 0인지 체크
    if os.path.exists(f'{DOWNLOAD_FOLDER_PATH}{specific_url}/{specific_url}.wav'):
        file_size = os.path.getsize(f'{DOWNLOAD_FOLDER_PATH}{specific_url}/{specific_url}.wav')
        if file_size < 1:
            # 빈파일 삭제
            os.remove(f'{DOWNLOAD_FOLDER_PATH}{specific_url}/{specific_url}.wav')

    # # 파일 없으면 가져오기
    if not os.path.exists(f'{DOWNLOAD_FOLDER_PATH}{specific_url}/{specific_url}.wav'):
        start_time = time.time()

        ## pytube 라이브러리
        # 유튜브 전용 인스턴스 생성
        yt = YouTube(f"https://www.youtube.com/watch?v={specific_url}")
        # 파일 다운로드 받기
        yt.streams.filter(file_extension="mp4").first().download(f"{DOWNLOAD_FOLDER_PATH}{specific_url}/")
        # 폴더 내에서 mp4 파일 이름 찾기
        path_dir = f"{DOWNLOAD_FOLDER_PATH}{specific_url}/"
        file_list = os.listdir(path_dir)

        file_name = ""
        for file in file_list:
            if os.path.splitext(file)[-1] == ".mp4":
                file_name = file
                break
        
        # mp4를 wav로 바꿔주기
        sound = AudioFileClip(f"{DOWNLOAD_FOLDER_PATH}{specific_url}/{file_name}")
        sound.write_audiofile(f"{DOWNLOAD_FOLDER_PATH}{specific_url}/{specific_url}.wav", 16000, 2, 2000, "pcm_s32le")

        # mp4 파일 삭제하기
        os.remove(f"{DOWNLOAD_FOLDER_PATH}{specific_url}/{file_name}")
        end_time = time.time()
        print(f"소요시간: {end_time - start_time}")

    return JSONResponse(
        status_code=200,
        content={
                "message": "음성 파일을 저장하고 있습니다.", 
        }
    )


# cos 유사도 계산하여 문장 구별
@app.get("/summary")
def get_summary(talk_list: list_check):
    talk_list = talk_list.talk_list
    model_summary = load_model()
    discriminator, electra_tokenizer, fill_model = load_postprocess_model()
    _ = model_summary.eval()
    tokenizer = get_kobart_tokenizer()

    split_text_list = get_split(talk_list, tokenizer.tokenize, n=5) # [[문단,0],[문단,21],[문단,46] ... ]
    outputs = ""
    for split_text in split_text_list:
        sp_text = ' '.join(split_text[0])
        input_ids = tokenizer.encode(sp_text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0) # 이 길이가 1024개 까지만 들어간다.
        input_ids = input_ids.split(1024, dim=-1)[0] # get_split에서 자르지만 넘어갈 경우

        output = model_summary.generate(input_ids, eos_token_id=1, max_length=200, num_beams=5) # eos_token_id=1, max_length=100, num_beams=5)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        output = dell_loop(output)
        output = summary_post_processing(generated_summary=output, discriminator=discriminator, tokenizer=electra_tokenizer, fill_model=fill_model, threshold=0.6)
        output = output.replace("'", "").replace(",", "") + "\n\n"
        outputs += output
    outputs = dell_loop(outputs)

    # 유효하면 정상 링크 안내
    return JSONResponse(
        status_code=200,
        content={
                "message": "완료", 
                "outputs": outputs
            }
    )

# 1000개씩 잘라서 보여줌
@app.get("/summary_before")
def get_summary2(talk_list: list_check):
    model_summary = load_model()
    discriminator, electra_tokenizer, fill_model = load_postprocess_model()
    _ = model_summary.eval()
    tokenizer = get_kobart_tokenizer()
    temp_talk_list = [talk[1] for talk in talk_list]
    text = ' '.join(map(str, temp_talk_list))

    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0) # 이 길이가 1024개 까지만 들어간다.
    input_ids_list = input_ids.split(1000, dim=-1) # .으로 나누는 것 필요? 245

    outputs = ""
    for inputs in input_ids_list:
        output = model_summary.generate(inputs, eos_token_id=1, max_length=300, num_beams=10) # eos_token_id=1, max_length=100, num_beams=5)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        output = dell_loop(output)
        ## 후처리
        output = summary_post_processing(generated_summary=output, discriminator=discriminator, tokenizer=electra_tokenizer, fill_model=fill_model, threshold=0.6)
        outputs += output
    outputs = dell_loop(outputs)

    # 유효하면 정상 링크 안내
    return JSONResponse(
        status_code=200,
        content={
                "message": "완료", 
                "outputs": outputs
            }
    )


@app.get("/keyword")
def extract_keyword(talk_list: list_check):
    talk = talk_list.talk_list
    talk = ' '.join(talk)
    outputs = get_keyword(talk)

    return JSONResponse(
        status_code=200,
        content={
                "message": "완료", 
                "outputs": outputs
            }
    )


@app.get("/query")
def question_answer(talk_dict: dict_check):
    question = talk_dict.question
    talk_list = talk_dict.talk_list
    tokenizer = get_kobart_tokenizer()
    
    tfidfv, tfidf_matrix, sec_text_list = get_tfidf_vec(talk_list, tokenizer.tokenize) # 로컬에 tfidf모델이랑 벡터 지정하는게 좋을듯
    results = get_similar(question, tfidfv, tfidf_matrix, sec_text_list, 5)

    return JSONResponse(
        status_code=200,
        content={
            "message": "완료",
            "outputs": results
        }
    )

 
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
