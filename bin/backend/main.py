from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import uvicorn
import os
# import time
import librosa
# import yaml
from pytube import YouTube
from moviepy.editor import *
import time

from utils.utils import make_specific_link

app = FastAPI()

# ASR_TRAIN_CONFIG = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/config.yaml"
# ASR_MODEL_FILE = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/valid.acc.ave_10best.pth"
# CONFIG_FILE = "/opt/ml/input/espnet-asr/conf/decode_asr.yaml"
DOWNLOAD_FOLDER_PATH = "../../download/"

@app.get("/")
def hello_world():
    print('get, hello')
    return {"hello": "world"}

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
    if not os.path.exists(f'{DOWNLOAD_FOLDER_PATH}{specific_url}/'):
        os.makedirs(f'{DOWNLOAD_FOLDER_PATH}{specific_url}')

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
            # print(file, file[-1], '00', os.path.splitext(file)[-1])
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
        ## youtube-dl 라이브러리
        # sh_result = os.popen(f'bash ../../tools/recog_youtube.sh --url {specific_url} --download-dir ../../download/{specific_url}')

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

# 이건 프론트에서 하자,,
# STT 실행해주는 함수 -> STT 결과 반환
# @app.post("/stt", description="STT 작업을 수행합니다.")
# def stt(specific_url: Url_check):
#     # 임시방편
#     # if temp:
#     #     print(temp)
#     #     return temp
#     specific_url = specific_url.url

#     # 파일 가져오기
#     with open(f'../../download/{specific_url}/{specific_url}.wav', 'r') as f:      
#         audio_file = f.name
    
#     # downsampling
#     audio, rate = downsampling(audio_file)

#     start_time = time.time()
#     print("JOB START!!!")

#     # config file 설정
#     with open(CONFIG_FILE) as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)

#     speech2text = Speech2Text(
#         asr_train_config=ASR_TRAIN_CONFIG, 
#         asr_model_file=ASR_MODEL_FILE, 
#         device='cuda',
#         dtype='float32',
#         **config
#     )

#     result = speech2text(audio)
#     print(f"Total time: {time.time() - start_time}")
#     print("JOB DONE!!!")
#     print('!!!!', result, type(result))

#     # 파일 직접 송수신하면 에러 발생
#     # Tensor is not JSON serailizable 오류 방지
#     # print('####', type(result))
    
#     return JSONResponse(
#         status_code=200,
#         content={
#                 "message": "음성 파일을 저장했습니다.", 
#         }
#     )
    
 
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
