from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any

import uvicorn
import os
import time
import librosa
import yaml

from utils.utils import make_specific_link, Speech2Text

app = FastAPI()

ASR_TRAIN_CONFIG = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/config.yaml"
ASR_MODEL_FILE = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/valid.acc.ave_10best.pth"
CONFIG_FILE = "/opt/ml/input/espnet-asr/conf/decode_asr.yaml"

@app.get("/")
def hello_world():
    print('get, hello')
    return {"hello": "world"}

def downsampling(audio_file, sampling_rate=16000):
    audio, rate = librosa.load(audio_file, sr=sampling_rate)
    return audio, rate

# "유튜브 링크" dict 데이터 유효성 체크해주는 pydantic
class Url_check(BaseModel):
    url: str

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
    if not os.path.exists(f'../../download/{specific_url}/'):
        os.makedirs(f'../../download/{specific_url}')

    # # 파일 없으면 가져오기
    if not os.path.exists(f'../../download/{specific_url}/{specific_url}.wav'):
        # 일단 local에 저장.. 추후 S3 확장 등 고려하기
        sh_result = os.popen(f'bash ../../tools/recog_youtube.sh --url {specific_url} --download-dir ../../download/{specific_url}')

    return JSONResponse(
        status_code=200,
        content={
                "message": "음성 파일을 저장하고 있습니다.", 
        }
    )

### 필요없을거같아서 주석처리
# # 음성 파일 반환해주는 함수. -> 음성 파일 위치 반환
# @app.post("/get_voice", description="음성 파일을 반환합니다.")
# def get_voice(specific_url: Url_check):
#         specific_url = specific_url.url

#         # 파일 가져오기
#         with open(f'../../download/{specific_url}/{specific_url}.wav', 'r') as f:      
#             audio_file = f.name

#         return audio_file

# temp

# STT 실행해주는 함수 -> STT 결과 반환
@app.post("/stt", description="STT 작업을 수행합니다.")
def stt(specific_url: Url_check):
    # 임시방편
    # if temp:
    #     print(temp)
    #     return temp
    specific_url = specific_url.url

     # 파일 가져오기
    with open(f'../../download/{specific_url}/{specific_url}.wav', 'r') as f:      
        audio_file = f.name
    
    # downsampling
    audio, rate = downsampling(audio_file)

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
    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
