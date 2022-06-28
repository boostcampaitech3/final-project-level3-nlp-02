# 최종 프로젝트

![youtube_professor](https://user-images.githubusercontent.com/76618935/172767342-7c220388-39c1-441e-ab2c-e402e81db769.png)

원하는 유튜브 영상을 STT를 통한 **text 추출** 및 **요약, 키워드 추출, MRC**를 지원하는 서비스입니다.

## Members

|                            강혜윤                            |                            고우진                            |                            김윤혜                            |                            윤주엽                            |                            이민준                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image](https://user-images.githubusercontent.com/48987027/172989959-38762e4e-a6d5-414c-b60e-171d9bc3fea4.png) | ![image](https://user-images.githubusercontent.com/48987027/172990102-1fca7411-6db7-4995-bacc-c656d128953c.png) | ![image](https://user-images.githubusercontent.com/48987027/172990193-a33b7b09-ce89-48bc-9718-ba9cdbcf5c8f.png) | ![image](https://user-images.githubusercontent.com/48987027/172989866-23cdebf1-cb29-451a-9451-c45d7442d378.png) | ![image](https://user-images.githubusercontent.com/48987027/172990006-2f60f1ee-475d-4ada-8926-9677c5463663.png) | <img src='https://avatars.githubusercontent.com/u/56633607?v=4' height=80 width=80px></img> |
| [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/Khyeyoon) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/woojjn) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/yoonene) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/AttractiveMinki) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/s1c5000) |

## 저장소 구조

```
.
├── bin
│   ├── frontend
│   │   ├── main.py
│   │   ├── frontend.sh
│   │   └── check_summary.py
│   │
│   ├── backend
│   │   └── main.py
│   │
│   ├── models
│   │   ├── postprocessing
│   │   └── kobart_summary2_v_1
│   │   
│   ├── asr_inference.py
│   ├── dataset.py
│   ├── key_bert.py
│   └── utils.py
│
└── conf
    ├── decode_asr.yaml
    ├── fast_decode_asr.yaml
    └── fast_decode_asr_ksponspeech.yaml

```


# How to use
## 0. Creating virtual environment

위 서비스를 제대로 실행하기 위해서는 front-end와 back-end을 서로 다른 가상환경에서 실행해야 합니다.

```bash
conda create -n frontend python=3.8.13
```

```bash
conda create -n backend python=3.8.13
```

## 1. Installation

### Front-end

```bash
conda activate frontend
```

```bash
(frontend)
pip install streamlit
pip install streamlit_player
pip install typeguard
pip install espnet
pip install espnet_model_zoo
pip install transformers
pip install pydub
```

### Back-end

```bash
conda activate backend
```

```bash
(backend)
pip install pytube
pip install moviepy
pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
pip install konlpy
pip install sentence_transformers
pip install fastapi
pip install uvicorn
pip install pydub
apt-get update && apt install default-jdk
```

## 2. Downloading pre-train models
STT 모델은 [hchung12/espnet-asr](https://github.com/hchung12/espnet-asr)을 이용했습니다.

**ESPnet pre-train model with ksponspeech**을 다운로드 받으려면 다음을 실행하세요.

```bash
(frontend) tools/download_mdl.sh
```


**Postprocessing pre-train model**을 다운로드 받으려면
[여기](https://drive.google.com/file/d/1VlImbs9qh3mwVZmPcVZj3DDMv_p9CFtK/view?usp=sharing)
를 클릭하세요.  
모델은 압축을 풀어 ~~/espnet-asr/bin 에 postprocessing_model이라는 이름으로 넣습니다  
모델 폴더 경로 : ~~/espnet-asr/bin/postprocessing_model

**kobart pre-train model**을 다운로드 받으려면
[여기](https://drive.google.com/file/d/1A_ZVu8DtL-3rmxUaGOKWl7AMtbktivG7/view?usp=sharing)
를 클릭하세요.  
모델은 압축을 풀어 ~~/espnet-asr/bin 에 kobart_summary(frontend/main.py에 있는 load_model() 경로에 맞춰서)이라는 이름으로 넣습니다  
모델 폴더 경로 : ~~/espnet-asr/bin/kobart_summary

혹은, 다음 사이트를 방문하여 원하는 텍스트 후처리 모델을 다운로드하세요.
https://plaid-raja-512.notion.site/4c07fd772e334dc9a3d7dbc4acd1bcce

# 실행방법  
shell의 cd 기능을 이용하여 다음 폴더로 이동한 뒤, 다음 명령어를 통해 실행시킵니다.  

## frontend  
~~/espnet-asr/bin/frontend  
sh frontend.sh  

## backend  
~~ /espnet-asr/bin/backend  
python main.py  


# 화면 예시

## youtube link로 STT 수행

![first](https://user-images.githubusercontent.com/70371239/173291581-88f35e8c-e770-4397-8c1a-b886feb75964.gif)


## 실시간 STT 수행

![second](https://user-images.githubusercontent.com/70371239/173291709-03e7001d-11d7-4db3-a522-762fdda0fc84.gif)


## 요약

![third](https://user-images.githubusercontent.com/70371239/173291770-660c477b-a5ef-4afe-8888-c838396deb02.gif)


## 키워드 검색

![fourth](https://user-images.githubusercontent.com/70371239/173291803-5e8ae983-676f-4d26-988d-d26747ae2bb0.gif)


# 프로젝트 Review

다음 링크에서 확인하실 수 있습니다.

[https://plaid-raja-512.notion.site/Review-53fbb44d11dd45dbbc9a0de79a25353f](https://plaid-raja-512.notion.site/Review-53fbb44d11dd45dbbc9a0de79a25353f)
