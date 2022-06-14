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

다음 링크에서 더 가독성있게 보실 수 있습니다.

[https://plaid-raja-512.notion.site/Wrap-Up-Report-311465e6ff2b43838024148b774d8b9f](https://www.notion.so/Wrap-Up-Report-311465e6ff2b43838024148b774d8b9f)

# 프로젝트 개요

### **NLP_2조_강의 음성 데이터 요약 및 키워드 추출**

아, 또 못들었네! 되감기..

여러분도 이런 경험이 있으신가요?

강의를 들으며 불편했던 경험을 바탕으로, 강의 음성 STT 추출 및 요약, 키워드 추출 사이트를 만들었습니다.

### 발표 ppt

[NLP_2조_강의 음성 데이터 요약 및 키워드 추출.pptx](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d9fca61d-c5c4-49f9-bc56-65168874a376/NLP_2조_강의_음성_데이터_요약_및_키워드_추출.pptx)

# **데모 페이지**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b80dc58a-1508-4411-8964-adf9f9025bb2/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/11cfe22a-b683-4882-8088-2581e647de13/Untitled.png)

# 프로젝트 팀 구성 및 역할

NLP-02조

소금빵

### **🔅 Member**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/87e3eec3-e744-435d-be34-44f2b065e4ff/Untitled.png)

| 강혜윤 | 고우진 | 김윤혜 | 윤주엽 | 이민준 |
| --- | --- | --- | --- | --- |
| https://avatars.githubusercontent.com/u/48987027?v=4  | https://avatars.githubusercontent.com/u/76618935?v=4 | https://avatars.githubusercontent.com/u/56261032?v=4 | https://avatars.githubusercontent.com/u/70371239?v=4 | https://avatars.githubusercontent.com/u/60881356?v=4 |
| Github | Github | Github | Github | Github |

| 강혜윤 | STT 논문 조사, STT 결과 후처리 | Github |
| --- | --- | --- |
| 고우진 | ESPnet 개선, STT 파이프라인 개선 | Github |
| 김윤혜 | 요약 모델, 요약 후처리 모델 | Github |
| 윤주엽 | github 협업, STT 속도 개선, Front-End, Back-End | Github |
| 이민준 | 요약 모델 파이프라인 개선, keyword 추출 모델 | Github |

# 프로젝트 수행 및 방법

## 1. EDA

### **[ STT ]**

KsponSpeech 평가 데이터 

**좌- waveform graph**

**우- spectogram graph**

![그림1.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/207a29a5-7d5c-49cc-9de8-64ab7c52e817/그림1.png)

![spectogram.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eb956a7e-3f0a-42a7-b1b5-e2003c800830/spectogram.png)

STT의 학습 데이터로 사용된 데이터셋인 KsponSpeech는 약 한 문장 정도의 짧은 데이터로 이루어져있다. 

우리가 다룰 강의 음성은 짧게는 10분에서 길게는 몇 시간까지, 매우 긴 음성 데이터였기에 KsponSpeech와 유사한 데이터를 만들어주기 위해 전처리를 해줄 필요가 있었다. 다음은 강의 음성을 약 6초 정도로 자른 데이터를 waveform graph로 plot한 그림이다.

![unnamed.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9de01ec4-8238-4071-9b2e-3aacad7ca2a5/unnamed.png)

강의 음성의 경우 문장과 문장 사이에 최소 500ms 이상의 침묵이 존재했고, 이를 기반으로 split하여 STT를 진행하였다.

### **[ Summarization ]**

- 데이터셋
    - 총 338,000 개
    - 문서요약 데이터셋, 논문요약 데이터셋, 도서요약 데이터셋
    - 학술적인 주제와 긴 길이의 데이터를 고려하여 데이터셋을 선정
        
        <논문요약 데이터셋 label 예시>
        
        > '<서동요>는 신라 제26대 진평왕(眞平王) 때 지었다는 4구체 향가로 그 설화(說話)와 함께『삼국유사(三國遺事)』권2「무왕조(武王條)」에 실려서 전하고 있다. 무엇보다 우리는 <서동요>를 단순히 신라시대의 노래로만 볼 것이 아니라 현재까지도 계속해서 생성되고 있는 다양한 성격을 지닌 텍스트라는 사실을 잊지 말아야 한다.’
        > 

- 문서요약 데이터셋 (약 34,000 개)
    - 길이 분포

![                               original_text length](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/48916c28-5f61-4c38-ad65-9d4f3eb048dd/Untitled.png)

                               original_text length

![                              summary_text length](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f72ac76b-b157-4b87-803c-2876cec085e2/Untitled.png)

                              summary_text length

문서요약 데이터셋은 뉴스, 판결문, 잡지 등에 대한 original_text와 summary_text로 구성되어 있다.  original_text와 summary_text의 최장 길이는 각각 1957, 498이었다. summary_text는 한 문장으로  요약한 결과를 나타낸다.

- 논문요약 데이터셋 (약 144,000 개)
    - 길이 분포

![                              original_text length](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/19c76ae7-3b66-404b-bb88-0cba796a23c3/Untitled.png)

                              original_text length

![                              summary_text length](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/36c921d7-0ede-458f-91d8-509995eccf10/Untitled.png)

                              summary_text length

논문요약 데이터셋은 인문학, 사회과학, 자연과학, 공학 등 다양한 학술분야의 논문에 대한 original_text와 summary_text로  구성되어 있다.   original_text와 summary_text의 최장 길이는 각각 4672, summary_text의 최장 길이는 1062 이었다. summary_text는 여러 문장으로  요약한 결과를 나타낸다.

- 도서요약 데이터셋 (약 160,000개)
    - 길이 분포

![                              original_text length](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/03236e3d-d046-4514-a3eb-61ce02b3165f/Untitled.png)

                              original_text length

![                              summary_text length](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/219b5317-0eed-4a9e-9e68-82972932f721/Untitled.png)

                              summary_text length

도서요약 데이터셋은 기술과학, 사회과학, 예술 등 다양한 학술분야의 도서에 대한 original_text와 summary_text로 구성되어 있다. original_text와 summary_text의 최장 길이는 각각 1043, 423이었다. summary_text는 여러 문장으로 요약한 결과를 나타낸다.

## 2. Model 선택

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/65515943-8d94-4776-a1c2-cb3949ac9667/Untitled.png)

### **[STT]**

  STT 모델을 찾아본 결과 Deep Speech 2, Conformer, ContextNet과 같은 모델이 존재했고, 이와 같은 모델들이 KoSpeech, OpenSpeech라는 오픈 소스를 통해 구현되어 있었다. 두 오픈 소스 모두 KsponSpeech 데이터를 기반으로 학습할 수 있게 구성되었기에 우리는 STT 학습을 위한 데이터로 KsponSpeech를 사용하였다.

**1. 데이터**

  KsponSpeech는 약 1000시간의 음성 데이터로 이루어져 있다. 이 음성 데이터는 총 5개로 나눠져있고, 서버 저장 공간으로 인해 한 사람당 두 개의 데이터셋을 다운로드하여 학습 데이터셋을 구성하였다. 

**2. 모델**

  KoSpeech, OpenSpeech의 경우, 각자의 환경에 따라 제대로 실행이 되지 않았다. 우리는 이를 해결하기 위해 각자가 발생한 에러들을 정리하여 github 이슈 게시판 및 notion에 게시하여, 해결책을 찾으면 답변을 달아서 해결하는 방법으로 이를 해결했다. 이를 통해 Deep Speech2 모델을 학습시켜본 결과 epoch 8, cer 0.41, loss 0.55의 점수를 얻었다. 이를 model을 inference 해본 결과 그렇게 좋은 성능을 내진 못했다. 모델의 문제라고 생각하여, 더 큰 모델인 CNN을 이용한 ContextNet 모델 및 CNN과 Transformer를 결합한 Conformer 모델을 이용하여 학습을 진행해본 결과 큰 성능 향상을 보이지 않았다. 두 오픈 소스 모두 pre-train 모델이 부재하여, 처음부터 학습을 진행해야했고 정말 사용가능할 정도의 성능을 내기 위해선 매우 긴 기간동안 이를 학습시켜야 할 것이라 예상되었기에 짧은 기간 동안 프로덕트 서빙까지 구현해야하는 시간 상 문제로 인해 STT 모델을 학습시키는 것을 포기했다. 이에 대한 대안책으로 추가적인 학습을 진행할 순 없지만 pre-train 모델을 제공하는 ESPnet을 이용하여 STT를 진행했다. ESPnet이 기존 STT 모델보다 성능이 좋긴했으나 아직 부족하다고 생각하여, STT 후처리를 통해 STT 정확도를 향상시키려 했다.

### [STT 후처리]

아래와 같이 부정확한 STT Model의 인식 결과를 개선하기 위해, STT 후처리 모델을 개발하였다.
모델은 KoGPT2를 이용하였고, 데이터셋은 AI Hub 한국어 강의 음성, 한국어 음성 데이터셋을 활용하여 총 21만개 문장으로 구성하였다.

- STT Model 부정확한 인식 결과
    - 발화의 크기가 작거나 속도가 일정하지 않은 경우, 문장의 끝을 정확하게 인식하지 못하거나 단어를 부적절하게 예측
    - 마침표, 쉼표와 같은 문장부호를 적절히 예측하지 못함
    - 부정확한 띄어쓰기

프로젝트는 유튜브 속 다양한 주제의 영상을 다루어야 하기 때문에 높임말, 평어(반말) 모두를 적절히 예측할 수 있어야 한다.
높임표현 학습을 위한 데이터셋으로 한국어 강의 음성 데이터셋을 선택하였고, 평어(반말) 학습을 위한 데이터셋으로  한국어 음성 데이터셋을 선택하였다.

데이터셋의 구성은 다음과 같다.

- Post Processing dataset
    - 데이터셋 개수 : 212,351개 문장
    - 데이터셋 형식 : csv 파일 (id, x, label)
    1. 실제 STT 모델 결과값을 활용하여 제작한 데이터셋 (17만개 문장)
        
        강의 데이터셋을 이용하였기 때문에 높임말 위주로 구성
        
        - dataset : [AI Hub 한국어 강의 음성](https://aihub.or.kr/aidata/30708)
            - x : STT 모델(ESPnet)이 예측한 음성인식 결과
            - label : 데이터셋 라벨
            - example
                
                ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/06e338ce-2ae1-441d-b923-680babd8fbe0/Untitled.png)
                
    2. 랜덤으로 노이즈를 추가하여 제작한 데이터셋 (5만개 문장)
        
        일상 대회 데이터셋을 이용하였기 때문에 평어 위주로 구성
        
        - dataset : [AI Hub 한국어 음성](https://aihub.or.kr/aidata/105)
            - x : 데이터셋 라벨에 랜덤으로 노이즈 추가 (문장부호 추가, 띄어쓰기 삭제/추가, 단일 글자 추가)
            - label : 데이터셋 라벨
            - example
                
                ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4f03a03e-eb51-4e4f-ad12-0504d1f50a6c/Untitled.png)
                

- 후처리 모델 결과
    
    "아녕하세여" 와 같이 발음은 비슷하지만 부적절한 예측값이 "안녕하세요.", 마침표와 함께 적절한 문법으로 후처리된 것을 확인할 수 있다.
    또한, 부적절한 문장부호의 정정, 띄어쓰기가 되어 있지 않은 문장의 띄어쓰기 보정 등도 적절히 수행되는 것을 확인할 수 있다.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a1e13890-1d39-4cf7-a8b3-9e9bf4be7611/Untitled.png)
    

### [****Summarization****]

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eb720f5d-9178-4ea8-948d-290a3a60ab13/Untitled.png)

Summarization의 개요이다.

크게 요약문을 생성하는 summarization 파트와 생성된 요약문을 후처리하여 더욱 자연스러운 요약문을 생성하는 Post-Processing 파트로 나뉜다.

STT 모델을 통해 생성된 텍스트를 finetuning된 KoBART 모델에 입력하여 1차 요약문을 생성하고, 이를 KoELCETRA 모델과 KoBigBird 모델을 통해 후처리하여 최종 요약문을 생성한다.

1. **📊 데이터**
    
    AI HUB의 문서요약 데이터셋, 논문요약 데이터셋, 도서요약 데이터셋을 활용하여 총 338,000 여개의 데이터를 학습하였다.
    
    다양한 학술 분야에 관한 데이터를 포함하고 길이가 긴 데이터를 포함하기 때문에 강의 영상을 다루는 프로젝트의 주제에 적합하다고 판단하였다.
    
2. **Pre-Processing**
    - 괄호, 다중 기호 제거
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/81e3ca7f-f9cd-4dee-b143-eed40b42caf1/Untitled.png)
        
        STT를 통해 생성된 텍스트에는 괄호와 그 내용이 있을 수 없고 다중 기호를 포함할 수 없기 때문에 이를 제거하였다. 
        이외에도 공백 제거, url 제거 등 기본적인 cleansing을 수행하였다.
        
    - 주로 한국어가 아닌 데이터로 구성된 데이터 제거
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4c007561-4838-4a66-b18d-86421c8ceae2/Untitled.png)
        
        한국어 STT 모델의 결과로 기호나 외국어 등 한국어가 아닌 문자가 많이 포함된 텍스트는 출력되지 않는다. 
        따라서 EDA를 통해 제거 기준 비율을 설정하여 한국어가 아닌 문자가 텍스트를 구성하는 비율이 15퍼센트 이상인 데이터를 제거하였다.
        
    
3. **요약 모델 학습**

![요약 모델 학습 실험 기록표 중 일부](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a7796699-ca96-436b-a6ed-9ce37b3e3761/Untitled.png)

요약 모델 학습 실험 기록표 중 일부

- Pretrained Model
SKT-AI에서 BART를 한국어로 학습시킨 KoBART model과 이를 문서요약 데이터만으로 학습시킨 pretrained model을 실험을 통해 비교하였다.  **문서요약 데이터셋을 fine-tuning 하지 않은 KoBART 모델**에 fine-tuning 했을 때 정성적 성능이 더 좋았다.
- 데이터셋
문서요약 데이터셋, 논문요약 데이터셋, 도서요약 데이터셋을 개별 학습시켰을 때와 여러 조합으로 학습시켰을 때의 성능을 실험을 통해 비교하였다. 데이터의 크기가 클수록 좋지만 문서요약 데이터셋의 경우 모두 한 문장으로 요약되고 논문이나 도서 등 학술 분야가 아닌 뉴스 데이터로 구성되어 있기 때문에 단순히 3가지 데이터셋을 모두 학습시키는 것이 좋지 않을 수 있다고 생각하였다. 실험 결과 **세** **가지 데이터셋을 모두 활용**하였을 때 정성적 성능이 가장 좋았다.
- 정제 여부
위의 두 실험을 통해 데이터셋과 pretrained model을 선정한 후 데이터셋 전처리 여부에 따른 성능을 실험을 통해 비교하였다. 지난 대회를 통해 전처리를 한다고 성능이 무조건 좋아지는 것이 아니라는 것을 알았기 때문에 해당 실험을 설계하였다. 실험 결과 괄호 제거, 주로 한국어가 아닌 데이터 제거 등 **전처리를 수행**하였을 때 정성적 성능이 더 좋았다.
- 길이 기준 전처리
EDA를 통해 길이 분포를 탐색한 후 box-plot의 길이 이상치 제거,  특정 길이 이하 데이터 제거 등 길이 기준 전처리 실험을 하였다. 그 결과 **전처리를 하지 않았을 때**가 가장 좋았다.
- epochs
epoch 별로 모델을 저장한 후 비교한 결과, epoch 3 이상일 때  val_loss가 높아졌고 정성적 평가에서도 **epoch 2** 가 가장 성능이 좋다고 판단하였다.

1. **문제 개선**
    
    서비스 과정 중 kobart에서 발생한 문제는 크게 3개가 생겼다.
    
    - 첫번째는 토큰화 시켰을때 일정길이 이상의 문서에 대해서는 요약하지 못하는것
    - 두번째는 generate시 문장이 반복적으로 나오는것
    - 세번째는 어색한 토큰이 생성되는것 이었다.
    
    **해결 방안**
    
    - 첫번째 문제의 해결방법으로 코사인 유사도 기반으로 긴 문서를 자르는 방법을 적용했다.
        
        문맥의 흐름을 파악하기 위해 하나의 문장마다 유사도를 구하지 않고 m개의 문장끼리 묶은뒤 유사도를 구했다.
        
        묶음을 TF-IDF로 벡터화 한뒤 cosine similarity를 계산하였다. 그 후, cosine similarity가 낮은 부분을 잘라 문단화 시켰다. 이 때 문단의 최소길이와 최대길이를 지정하여 최소길이와 최대길이 사이의 문장길이로 잘리도록 하였다.
        

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ec4879fb-18dd-4789-a29f-ce282cded7d1/Untitled.png)

- 두번째 문제의 해결방법으로 반복적으로 등장하는 어구를 지워주는 후처리를 적용했다.
    
    모델이 생성한 요약문서 중 반복적으로 등장하는 어구가 있었다. 반복적으로 나오는 어구는 바로 이어지는 특성이 있었다. 
    
    바로 이어서 나오는 중복 어구를 지우게 하였다.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7cfbb9ba-b736-4f15-b633-7af3492bcede/Untitled.png)
    

- 세번째 문제의 해결방법으로 **[Summarization 후처리]** 모델을 사용하였다.

### [****Summarization**** 후처리]

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ea371c47-1e56-4368-8115-fda1d3bb2c80/Untitled.png)

KoELECTRA-discriminator 모델을 통해 생성된 요약문에서 어색한 토큰을 찾아내 마스킹하였다. 
이를 KoBigBird 모델에 입력하여 마스크 토큰을 적절한 토큰으로 대체하였다.

- **요약 결과 예시**
    
    다음은 위와 같은 방식으로 학습한 요약 모델을 통해 [기사](https://biz.chosun.com/it-science/ict/2022/05/31/AEXRV3ZPN5BQRNGQ5G4M4FVOVA/?utm_source=naver&utm_medium=original&utm_campaign=biz)를 요약한 결과이다. 핵심 내용을 잘 요약한 것을 확인할 수 있다.
    
    > 구글은 지난달부터 플레이 스토어에서 외부 결제 페이지로 연결되는 아웃링크를 삽입한 앱의 업데이트를 금지하는 정책을 시행하고 있다. 이에 국내 온라인 동영상 서비스(OTT), 음악, 웹툰 등 콘텐츠는 수수료 인상에 따른 가격 인상을 단행하고 있다.
    > 

### [key word]

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5b3d2fe5-4e14-44ff-b36a-c3baad76981f/Untitled.png)

sentenceTransformer의 SBERT와 KoNLPy를 이용해 키워드를 추출했다.
우선 konlpy로 명사구를 뽑은 뒤 SBERT로 키워드를 추출한다.

키워드 추출 방식은 3가지 버전이 있다.
첫번째 버전은 단순하게 키워드와 문서의 코사인 유사도를 통해 문서를 대표하는 키워드를 찾는 방법 이다.
두번째 버전은 코사인 유사도로 키워드를 추출한 뒤 서로 유사성이 가장 낮은 키워드를 선택하는 Max sum similarity 방법이다.
세번째 버전은 중복을 최소화하고 결과의 다양성을 극대화하는 Maximal Marginal Relevance 방법이다.

## 3. Product Serving

### [실행시간 단축]

**SplitOnSilence**

음성 파일을 나누기 위해 pydub의 detect_nonsilent을 이용하여 SplitOnSilence를 구현하였다.

17분 음성 파일을 SplitOnSilence했을 때, STT 작업에 440초가 소요되었다.

동일한 파일을 1분 단위로 나눈 뒤 SplitOnSilence를 하도록 구현하였다. 30초가 소요되어, 기존 대비 약 15배 가량의 시간 단축 효과를 얻었다.

**STT 배치 병렬화**

기존 코드는 배치 사이즈가 1로 정해져있어 병렬화를 할 수 없었다. 

기존 코드를 개선하여, 배치 병렬화를 통해 17분 음성 기준 144초가 소요되던 작업을 120초로 단축시켜, 기존 대비 20% 이상의 시간 단축 효과를 얻었다.

**YouTube 영상에서 음성 추출**

영상에서 음성을 추출할 때, 기존 ESPnet에서 사용하던 youtube-dl 라이브러리를 사용하기 위해 shell 언어를 이해하여 사용해야 했다.

또한, 70~80kbps/s의 속도로 음성을 추출했으며, 추출된 음성의 sampling rate가 48000으로, 학습 데이터로 사용한 KSponSpeech dataset의 sampling rate인 16000과 달랐기 때문에, sampling rate도 16000으로 바꿔줘야 하기 때문에, 많은 시간이 소요되었다.

이를 개선하여, pytube란 라이브러리로, 매우 빠른 시간 내에 유튜브 영상을 mp4 형태로 다운로드하였다. moviepy란 library를 통해 mp4 영상을 wav로 변환하고, sampling rate를 16000으로 바꾸어, 10배 이상 처리 속도를 개선하였다. 변환이 완료된 mp4 파일은 삭제하였다.

또한, 사용자가 CONFIG_FILE로 fast_decode_asr, decode_asr 중 하나를 선택할 수 있도록 하여, 빠른 STT 작업/정확한 STT 작업 중 선택할 수 있도록 구현하였다.

### [프론트엔드]

**streamlit**

짧은 시간 내에 End-to-End Service를 구현하기 위해, 수업 시간에 배운 streamlit을 사용하였다.

python을 사용하여 간편하게 구현할 수 있었고, 기본적으로 제공해주는 기능들은 편리하게 사용할 수 있었다. response를 통해 백엔드와 손쉽게 소통하며 유튜브 영상 링크 검증, 파일 다운로드 등의 작업을 할 수 있었다. st.write, st.spinner, st.session_state 등을 통해서 원하는 기능을 빠른 시간 내에 구현하였다.

유튜브 영상을 띄워줄 때 streamlit_player라는 라이브러리의 st_player를 통해, 아주 손쉽게 영상을 보여줄 수 있어서 만족스러웠다. 실시간으로 STT된 텍스트를 Timeline과 함께 보여줄 수 있는 점도 아주 좋았다.

그렇지만, 이것저것 구현하면서 streamlit의 근본적인 한계를 많이 느꼈다. 버튼을 누르거나 문장을 입력하면 페이지 전체가 새로고침되는 특성때문에, 원하는 기능을 마음껏 구현할 수 없었다.

STT 작업이 완료된 뒤에, 키워드를 검색하면 그 키워드와 관련된 Timeline과 대본을 보여주는 기능을 구현할 때, 검색이 완료되면 페이지 전체가 새로고침되는 문제가 있었다. 새로고침 이후에 STT 작업을 처음부터 다시 진행했기 때문에, 사용자 입장에선 불편할 것으로 생각한다.

검색 이후 페이지가 전체적으로 새로고침되는 현상을 막기 위해, st.cache를 사용하면 된다는 글을 보았다. st.cache를 사용하기 위해 모든 기능을 함수화하는 시도를 하였다. 함수화엔 성공하여 가독성을 높였으나, st.cache의 hash_funcs 기능을 파악하고 사용하는 데에 어려움을 겪어, st.cache를 사용하진 못했다.

또한, 페이지의 UI를 꾸미며 st.columns를 사용하여 페이지를 나누었는데, st.columns 안에서 st.columns를 사용할 수 없는 문제가 있었다. BootStrap의 그리드 시스템을 사용했다면, col-12 안에서 col-3과 같이 손쉽게 나눌 수 있고 col-3 안에서도 페이지를 나눌 수 있는데, streamlit에선 st.columns 안에서 st.columns를 나누는 기능을 지원하지 않아 불편했다.

시간이 좀 더 많았다면, Vue.js나 React로 구현할 수 있었을 것이라는 아쉬움이 남았다.

### [백엔드]

**FastAPI**

짧은 시간 내에 End-to-End Service를 구현하기 위해, 수업 시간에 배운 FastAPI를 사용하였다.

python을 사용하여 간편하게 구현할 수 있었고, 기본적으로 제공해주는 기능들은 편리하게 사용할 수 있었다. @app.get(”링크 주소) 만을 통해서 HTTP Method인 GET을 구현할 수 있는 점은 아주 매력적이었다. 바로 Swagger를 볼 수 있는 점도 아주 좋았다. return JSONResponse를 통해 손쉽게 Response와, 앞에서 말한 Request를 구현할 수 있는 것도 장점이었다.

입력된 변수를 검증하기 위해 class를 만들어야 하는 점은 불편했다. Django에서도 검증하는 단계가 있던 것으로 기억하는데, FastAPI에선 똑같이 코드를 짜도 순서에 따라 검증에 실패할 수도, 성공할 수도 있다는 점은 마음에 들지 않았다.

팀원들이 구현한 여러 기능들을 함수화하였고, 이를 utils.py에 따로 넣어 프론트엔드와 백엔드에 적용하였다. 여러 AI 관련 모델을 웹 사이트에 적용해보는 경험을 해 본 점이 아주 만족스럽다.

검증 방식 외의 기능은, 전체적으로 만족스러웠다. Django를 사용하기 어려운 환경이거나, 간단하고 빠르게 웹 사이트를 구현할 때 FastAPI는 만족스러운 선택지 중 하나일 것 같다.

# 프로젝트 수행 결과

---

### ****youtube link로 STT 수행****

![https://user-images.githubusercontent.com/70371239/173291581-88f35e8c-e770-4397-8c1a-b886feb75964.gif](https://user-images.githubusercontent.com/70371239/173291581-88f35e8c-e770-4397-8c1a-b886feb75964.gif)

### ****실시간 STT 수행****

![https://user-images.githubusercontent.com/70371239/173291709-03e7001d-11d7-4db3-a522-762fdda0fc84.gif](https://user-images.githubusercontent.com/70371239/173291709-03e7001d-11d7-4db3-a522-762fdda0fc84.gif)

### ****요약****

![https://user-images.githubusercontent.com/70371239/173291770-660c477b-a5ef-4afe-8888-c838396deb02.gif](https://user-images.githubusercontent.com/70371239/173291770-660c477b-a5ef-4afe-8888-c838396deb02.gif)

### ****키워드 검색****

![https://user-images.githubusercontent.com/70371239/173291803-5e8ae983-676f-4d26-988d-d26747ae2bb0.gif](https://user-images.githubusercontent.com/70371239/173291803-5e8ae983-676f-4d26-988d-d26747ae2bb0.gif)

## 자체 평가 의견

- 잘 한 점들
    - 짧은 시간 안에 End-to-End Product Serving을 완료하였다.
    - 음성처리 부분을 같이 공부하여, 끝까지 프로젝트를 완수했다.
    
- 시도 했으나 잘 되지 않았던 것들
    - STT 모델을 처음부터 학습시켰으나 성능이 좋지 못했다.
    - KoSpeech, OpenSpeech를 이용해 STT 작업을 진행해보았으나, 만족스러운 성능을 얻지 못했다.
    - STT 후처리 과정에서 데이터셋을 30만개 이상으로 늘려서 학습시켜보았지만, 성능이 개선되지 않았다.

- 아쉬웠던 점들
    - 음성인식 기술을 다루는 것이 처음이라 음성인식 네트워크 동작방식을 이해하는데 오랜 시간이 걸렸다.
    - STT 성능이 프로젝트 결과물에 아주 큰 영향을 미쳤다.

- 프로젝트를 통해 배운 점 또는 시사점
    - AI를 이용한 음성인식의 전체적인 동작과정을 이해할 수 있었다.
    - AI 모델 구현부터 서비스 적용 단계까지 진행해보았다.
    

---

# 개인 회고

### 강혜윤_T3008

## **Ⅰ. 학습 목표**

### **[1] 팀 학습 목표**

아이디어 구상부터 프로덕트 서빙까지 엔지니어링 프로세스의 A to Z 경험하고, 문제해결 과정을 통해 성장하는 것이 목표였습니다.

### **[2] 개인 학습 목표**

인공지능 기술을 이용한 음성인식 기술을 이해하고, 음성인식 기술의 핵심인 STT 성능을 개선하는 것이 목표였습니다.

## **Ⅱ. 문제 해결을 위한 과정**

### [1] Project

프로젝트의 주제는 유튜브 속 영상의 음성 데이터를 이용한 타임라인, 요약, 키워드 추출 서비스입니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ed1610e1-8647-41d0-85a1-d30ff6821e32/Untitled.png)

### **[2] Overview of Project**

프로젝트의 전체적인 흐름은 다음과 같습니다.

1. 사용자로부터 유튜브 링크를 입력 받아 유튜브 영상 속 음성 데이터 저장
2.  STT 기술을 이용하여 텍스트 추출
3. 추출된 텍스트를 이용하여 타임라인, 요약, 키워드추출을 수행하여 사용자에게 제공

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a375f166-e797-4ab5-aade-c07159f838b2/Untitled.png)

### [3] Post Processing for STT

아래와 같이 부정확한 STT Model의 인식 결과를 개선하기 위해, STT 후처리 모델을 개발하였습니다.
모델은 KoGPT2를 이용하였고, 데이터셋은 AI Hub 한국어 강의 음성, 한국어 음성 데이터셋을 활용하여 총 21만개 문장으로 구성하였습니다.

- STT Model 부정확한 인식 결과
    - 발화의 크기가 작거나 속도가 일정하지 않은 경우, 문장의 끝을 정확하게 인식하지 못하거나 단어를 부적절하게 예측
    - 마침표, 쉼표와 같은 문장부호를 적절히 예측하지 못함
    - 부정확한 띄어쓰기

프로젝트는 유튜브 속 다양한 주제의 영상을 다루어야 하기 때문에 높임말, 평어(반말) 모두를 적절히 예측할 수 있어야 합니다.
높임표현 학습을 위한 데이터셋으로 한국어 강의 음성 데이터셋을 선택하였고, 평어(반말) 학습을 위한 데이터셋으로  한국어 음성 데이터셋을 선택하였습니다.

데이터셋의 구성은 다음과 같습니다.

- Post Processing dataset
    - 데이터셋 개수 : 212,351개 문장
    - 데이터셋 형식 : csv 파일 (id, x, label)
    1. 실제 STT 모델 결과값을 활용하여 제작한 데이터셋 (17만개 문장)
        
        강의 데이터셋을 이용하였기 때문에 높임말 위주로 구성
        
        - dataset : [AI Hub 한국어 강의 음성](https://aihub.or.kr/aidata/30708)
            - x : STT 모델(ESPnet)이 예측한 음성인식 결과
            - label : 데이터셋 라벨
            - example
                
                ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/06e338ce-2ae1-441d-b923-680babd8fbe0/Untitled.png)
                
    2. 랜덤으로 노이즈를 추가하여 제작한 데이터셋 (5만개 문장)
        
        일상 대회 데이터셋을 이용하였기 때문에 평어 위주로 구성
        
        - dataset : [AI Hub 한국어 음성](https://aihub.or.kr/aidata/105)
            - x : 데이터셋 라벨에 랜덤으로 노이즈 추가 (문장부호 추가, 띄어쓰기 삭제/추가, 단일 글자 추가)
            - label : 데이터셋 라벨
            - example
                
                ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4f03a03e-eb51-4e4f-ad12-0504d1f50a6c/Untitled.png)
                
        
        다양한 구조의 문장을 복원시킬 수 있도록 노이즈(문장부호 추가, 띄어쓰기 삭제/추가, 단일 글자 추가)를 추가한 데이터셋을 구성하였습니다.
        
        - 노이즈 추가 방법
            
            아래의 함수들을 랜덤한 확률로 적용시켜 임의의 노이즈를 추가하였습니다.
            
            ```python
            def remove_space(text):
                string=""
                arr=text.split()
                
                for s in arr:
                    rand_num=random.randrange(1,3)
                    if rand_num==1:
                        string+=s
                    else:
                        string+=' '+s
                        
                return string
            
            def add_character(text):
                rand_num=random.randrange(0,len(text))
                char_arr=['.',',','?','!']
                
                num=random.randrange(0,3)
                
                for _ in range(num):
                    num=random.randrange(0,4)
                    text = text[:rand_num]+char_arr[num]+text[rand_num:]
                    
                return text
            
            def add_noise(text):
                rand_num=random.randrange(0,len(text))
                char_arr=['아','가','그','어','오','으','이','디','음','우','거']
                
                num=random.randrange(0,3)
                
                for _ in range(num):
                    num=random.randrange(0,len(char_arr))
                    text = text[:rand_num]+char_arr[num]+text[rand_num:]
                    
                return text
            ```
            
        
- 후처리 모델 결과
    
    "아녕하세여" 와 같이 발음은 비슷하지만 부적절한 예측값이 "안녕하세요.", 마침표와 함께 적절한 문법으로 후처리된 것을 확인할 수 있습니다.
    또한, 부적절한 문장부호의 정정, 띄어쓰기가 되어 있지 않은 문장의 띄어쓰기 보정 등도 적절히 수행되는 것을 확인할 수 있습니다.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a1e13890-1d39-4cf7-a8b3-9e9bf4be7611/Untitled.png)
    

## **Ⅲ. 프로젝트를 진행하면서 느낀 점/아쉬운 점**

주어진 데이터로 정해진 문제를 해결하는 대회와 달리 서비스 개발을 위한 전과정을 경험할 수 있어서 좋았습니다. 기획한 아이디어를 실현하는 과정은 생각보다 많은 어려움이 존재했고, 정해진 기간내에 완료하기 위해 포기해야 하는 부분들도 많았습니다. 하지만, 기획한 서비스를 개발하고 배포하는 과정에서 많은 것을 배울 수 있었고, 인공지능 기반 음성인식을 기술을 다룰 수 있는 좋은 경험이었습니다. 처음으로 음성 관련 프로젝트를 수행하며 음성인식 과정을 이해하는 것이 어려웠고, 아직도 모르는 부분이 많아서 추가적으로 공부할 생각입니다.

끝나지 않을 것 같았던 5개월간의 부스트캠프를 최종 프로젝트를 통해 마무리지었습니다. 부스트캠프에서 얻은 경험들을 토대로 실력있는 머신러닝 엔지니어로 성장하기 위해 계속해서 나아갈 것입니다.

---

### 고우진_T3009

## **Ⅰ. 학습 목표**

  1차적인 목표는 Product-serving의 end-to-end를 구현하는 것이었다. 그리고 우리가 주제로 정한 STT task에 대해 기존의 서비스 대비 차별점을 부각시킬만한 서비스를 제공하는 프로젝트를 진행하는 것이 목표였다.

## **Ⅱ. 문제 해결을 위한 과정**

나는 최종 프로젝트에서 음성 데이터 처리 및 STT 모델의 성능 향상의 역할을 맡았다. 먼저 우리가 다루는 음성 데이터는 짧게는 약 10분에서 길게는 몇 시간이 되는 한국어 강의 음성이었기에, 이를 모델에 입력으로 주기 위해서는 이를 전처리할 필요가 있었다. 우리가 다룰 STT 모델은 KsponSpeech를 학습 데이터로 하여 학습된 모델이었고, 이 데이터는 약 한 문장 정도로 짧은 데이터였기에 매우 긴 음성 데이터를 나눠줄 필요가 있었다. 또한, STT 모델의 배치 병렬화를 통해 효율성을 증진시키기 위해서나, 입력이 너무 크면 OOM(Out Of Memory) 문제가 발생했기 때문에 음성 데이터를 한 문장 단위로 나눠주는 것은 필수적인 사항이었다.

**음성 데이터 처리**

음성 데이터를 분석해 본 결과 대부분 문장과 문장 사이에는 500ms 이상, 그리고 일정 dB 미만의 구간 즉, 침묵이 존재했다. Pydub 라이브러리에서 이러한 침묵을 기준으로 나눠주는 함수인 SplitOnSilence가 존재했고, 이를 통해 강의 음성 데이터를 한 문장으로 나눠주는 작업을 진행할 수 있었다. 그러나 위 함수를 통해 17분 분량의 강의 영상을 split한 결과 약 440초의 시간의 소요됐다. 나는 이 함수가 O(N^2) 이상의 시간 복잡도를 가지는 것을 확인했고, 이를 단축시키기 위해 파일을 먼저 1분 단위로 나눠준 후 나눠진 파일에 대해 SplitOnSilence를 진행하여 총 시간 소요가 30초로 약 15배 가까운 시간 단축 효과를 얻을 수 있었다.

**STT 모델 성능 향상**

STT 모델을 학습시키기 위해 kospeech와 openspeech라는 오픈 소스를 통해 학습시킨 결과 긴 시간 학습한 것에 비해 좋은 성능을 보여주지 못했다. 프로젝트 기간이 4주로 매우 짧은 기간 안에 목표한 바를 달성하기 위해서는 STT 모델을 처음부터 학습시키는 것은 현실적으로 불가능하다는 판단이 들었고, pre-train 모델이 존재하는 ESPnet이라는 모델을 사용하여 STT 모델로 이용하였다. ESPnet은 우리가 직접 처음부터 학습시킨 모델에 비해 좋은 성능을 보여줬지만, batch-size가 1로 고정되는 문제가 있었다. 이로 인해 17분 음성의 경우 약 144초의 시간이 소요됐다. 나는 이를 해결하기 위해 기존 코드에서 batch 병렬화를 진행할 수 있도록 수정했고, 그 결과 약 20%의 시간 효율성 향상을 얻을 수 있었다.

## **Ⅲ. 프로젝트를 진행하면서 느낀 점/아쉬운 점**

- STT 모델을 직접 학습시키지 못한 것이 아쉬움이 남는다.
- ESPnet의 배치 병렬화를 진행하여 시간 단축을 하는데는 성공했지만 그 결과가 그렇게 만족스럽지 못했다. 그 이유는 STT 작업 중 Beamsearch를 병렬화시키지 못했기 때문에 Beamsearch 과정에서 병목 현상이 일어났기 때문이다. Beamsearch 병렬화를 성공시키지 못한 게 아쉬움이 남는다.
- STT 후처리 모델을 Seq2Seq를 통해 구현하고 학습시켰지만 그렇게 성능이 좋지 않았다.

  

---

### 김윤혜_T3045

## **Ⅰ. 학습 목표**

지난 부스트캠프 활동 동안 배운 점을 활용하여 아이디어를 구상 및 구현하고 서비스를 배포하는 end-to-end 프로젝트를 완성하는 것이 목표이다.

## **Ⅱ. 문제 해결을 위한 과정**

## Summarization

맡은 역할: 요약 모델, 요약 결과 후처리

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eb720f5d-9178-4ea8-948d-290a3a60ab13/Untitled.png)

## [1] 요약 모델 학습 실험

![요약 모델 학습 실험 기록표 중 일부](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a7796699-ca96-436b-a6ed-9ce37b3e3761/Untitled.png)

요약 모델 학습 실험 기록표 중 일부

- Pretrained Model
SKT-AI에서 BART를 한국어로 학습시킨 KoBART model과 이를 문서요약 데이터만으로 학습시킨 pretrained model을 실험을 통해 비교하였다.  문서요약 데이터셋을 fine-tuning 하지 않은 KoBART 모델에 fine-tuning 했을 때 정성적 성능이 더 좋았다.
- 데이터셋
문서요약 데이터셋, 논문요약 데이터셋, 도서요약 데이터셋을 개별 학습시켰을 때와 여러 조합으로 학습시켰을 때의 성능을 실험을 통해 비교하였다. 데이터의 크기가 클수록 좋지만 문서요약 데이터셋의 경우 모두 한 문장으로 요약되고 논문이나 도서 등 학술 분야가 아닌 뉴스 데이터로 구성되어 있기 때문에 단순히 3가지 데이터셋을 모두 학습시키는 것이 좋지 않을 수 있다고 생각하였다. 실험 결과 3가지 데이터셋을 모두 활용하였을 때 정성적 성능이 가장 좋았다.
- 정제 여부
위의 두 실험을 통해 데이터셋과 pretrained model을 선정한 후 데이터셋 전처리 여부에 따른 성능을 실험을 통해 비교하였다. 지난 대회를 통해 전처리를 한다고 성능이 무조건 좋아지는 것이 아니라는 것을 알았기 때문에 해당 실험을 설계하였다. 실험 결과 괄호 제거, 주로 한국어가 아닌 데이터 제거 등 전처리를 수행하였을 때 정성적 성능이 더 좋았다.
- 길이 기준 전처리
EDA를 통해 길이 분포를 탐색한 후 box-plot의 길이 이상치 제거,  특정 길이 이하 데이터 제거 등 길이 기준 전처리 실험을 하였다. 그 결과 전처리를 하지 않았을 때가 가장 좋았다.
- epochs
epoch 별로 모델을 저장한 후 비교한 결과, epoch 3 이상일 때  val_loss가 높아졌고 정성적 평가에서도 epoch 2 가 가장 성능이 좋다고 판단하였다.

### [2] 요약 데이터셋 Pre-Processing

- 괄호, 다중 기호 제거
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/81e3ca7f-f9cd-4dee-b143-eed40b42caf1/Untitled.png)
    
    STT를 통해 생성된 텍스트에는 괄호와 그 내용이 있을 수 없고 다중 기호를 포함할 수 없기 때문에 이를 제거하였다. 
    이외에도 공백 제거, url 제거 등 기본적인 cleansing을 수행하였다.
    
- 주로 한국어가 아닌 데이터로 구성된 데이터 제거
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4c007561-4838-4a66-b18d-86421c8ceae2/Untitled.png)
    
    한국어 STT 모델의 결과로 기호나 외국어 등 한국어가 아닌 문자가 많이 포함된 텍스트는 출력되지 않는다. 
    따라서 EDA를 통해 제거 기준 비율을 설정하여 한국어가 아닌 문자가 텍스트를 구성하는 비율이 15퍼센트 이상인 데이터를 제거하였다.
    

## [3] 요약 결과 Post-Processing

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ea371c47-1e56-4368-8115-fda1d3bb2c80/Untitled.png)

fake token을 찾도록 학습된 KoELECTRA-discriminator 모델을 통해 생성된 요약문에서 어색한 토큰을 찾아내 마스킹하였다. 
이를 KoBigBird 모델에 입력하여 마스크 토큰을 적절한 토큰으로 대체하였다. (fill-mask)

### 요약 결과 예시

다음은 위와 같은 방식으로 학습한 요약 모델을 통해 [기사](https://biz.chosun.com/it-science/ict/2022/05/31/AEXRV3ZPN5BQRNGQ5G4M4FVOVA/?utm_source=naver&utm_medium=original&utm_campaign=biz)를 요약한 결과이다. 핵심 내용을 잘 요약한 것을 확인할 수 있다.

> 구글은 지난달부터 플레이 스토어에서 외부 결제 페이지로 연결되는 아웃링크를 삽입한 앱의 업데이트를 금지하는 정책을 시행하고 있다. 이에 국내 온라인 동영상 서비스(OTT), 음악, 웹툰 등 콘텐츠는 수수료 인상에 따른 가격 인상을 단행하고 있다.
> 

## **Ⅲ. 대회를 진행하면서 느낀 점/아쉬운 점**

- 배운 내용을 토대로 아이디어를 구상하고 이를 서비스 배포하는 end-to-end 프로젝트를 좋은 환경에서 좋은 사람들과 경험할 수 있어서 좋았고 감사했다.
- 문제 해결을 위한 팀원들의 창의적인 아이디어를 보고 문제 해결 방법에 대한 시야가 넓어졌다.
- 요약 모델의 최대 입력 크기가 512인 한계를 4096으로 늘리기 위해 KoBART의 attention layer를 Longformer layer로 변환 후 학습을 시도했는데 시간이 부족했다.
- 정량적인 평가 결과를 제공하지 못한 것이 아쉽다. 비록 프로젝트는 끝났지만 rouge score를 구현해 평가할 계획이다.
- STT 모델 선정 과정에서 시행착오를 겪느라 주어진 시간을 효율적으로 활용하지 못한 것 같아 아쉬웠다.

---

### 윤주엽_T3136

# **Ⅰ. 학습 목표**

## **[1] 팀 학습 목표**

github 환경에서 원활한 협력을 통해 좋은 성능의 STT 모델을 개발하고, 편리한 서비스를 제공하는 것이 목표였다.

## **[2] 개인 학습 목표**

음성처리 단계를 이해하고, 효율적으로 코드를 개선하며, End-to-End Service를 구현하는 것이 목표였다.

## Ⅱ. 문제 해결을 위한 과정

**YouTube 영상에서 음성 추출**

영상에서 음성을 추출할 때, 기존 ESPnet에서 사용하던 youtube-dl 라이브러리를 사용하기 위해 shell 언어를 이해하여 사용해야 했다.

또한, 70~80kbps/s의 속도로 음성을 추출했으며, 추출된 음성의 sampling rate가 48000으로, 학습 데이터로 사용한 KSponSpeech dataset의 sampling rate인 16000과 달랐기 때문에, sampling rate도 16000으로 바꿔줘야 하기 때문에, 많은 시간이 소요되었다.

이를 개선하여, pytube란 라이브러리로, 매우 빠른 시간 내에 유튜브 영상을 mp4 형태로 다운로드하였다. moviepy란 library를 통해 mp4 영상을 wav로 변환하고, sampling rate를 16000으로 바꾸어, 10배 이상 처리 속도를 개선하였다. 변환이 완료된 mp4 파일은 삭제하였다.

또한, 사용자가 CONFIG_FILE로 fast_decode_asr, decode_asr 중 하나를 선택할 수 있도록 하여, 빠른 STT 작업/정확한 STT 작업 중 선택할 수 있도록 구현하였다.

### [프론트엔드]

**streamlit**

짧은 시간 내에 End-to-End Service를 구현하기 위해, 수업 시간에 배운 streamlit을 사용하였다.

python을 사용하여 간편하게 구현할 수 있었고, 기본적으로 제공해주는 기능들은 편리하게 사용할 수 있었다. response를 통해 백엔드와 손쉽게 소통하며 유튜브 영상 링크 검증, 파일 다운로드 등의 작업을 할 수 있었다. st.write, st.spinner, st.session_state 등을 통해서 원하는 기능을 빠른 시간 내에 구현하였다.

유튜브 영상을 띄워줄 때 streamlit_player라는 라이브러리의 st_player를 통해, 아주 손쉽게 영상을 보여줄 수 있어서 만족스러웠다. 실시간으로 STT된 텍스트를 Timeline과 함께 보여줄 수 있는 점도 아주 좋았다.

그렇지만, 이것저것 구현하면서 streamlit의 근본적인 한계를 많이 느꼈다. 버튼을 누르거나 문장을 입력하면 페이지 전체가 새로고침되는 특성때문에, 원하는 기능을 마음껏 구현할 수 없었다.

STT 작업이 완료된 뒤에, 키워드를 검색하면 그 키워드와 관련된 Timeline과 대본을 보여주는 기능을 구현할 때, 검색이 완료되면 페이지 전체가 새로고침되는 문제가 있었다. 새로고침 이후에 STT 작업을 처음부터 다시 진행했기 때문에, 사용자 입장에선 불편할 것으로 생각한다.

검색 이후 페이지가 전체적으로 새로고침되는 현상을 막기 위해, st.cache를 사용하면 된다는 글을 보았다. st.cache를 사용하기 위해 모든 기능을 함수화하는 시도를 하였다. 함수화엔 성공하여 가독성을 높였으나, st.cache의 hash_funcs 기능을 파악하고 사용하는 데에 어려움을 겪어, st.cache를 사용하진 못했다.

또한, 페이지의 UI를 꾸미며 st.columns를 사용하여 페이지를 나누었는데, st.columns 안에서 st.columns를 사용할 수 없는 문제가 있었다. BootStrap의 그리드 시스템을 사용했다면, col-12 안에서 col-3과 같이 손쉽게 나눌 수 있고 col-3 안에서도 페이지를 나눌 수 있는데, streamlit에선 st.columns 안에서 st.columns를 나누는 기능을 지원하지 않아 불편했다.

시간이 좀 더 많았다면, Vue.js나 React로 구현할 수 있었을 것이라는 아쉬움이 남았다.

### [백엔드]

**FastAPI**

짧은 시간 내에 End-to-End Service를 구현하기 위해, 수업 시간에 배운 FastAPI를 사용하였다.

python을 사용하여 간편하게 구현할 수 있었고, 기본적으로 제공해주는 기능들은 편리하게 사용할 수 있었다. @app.get(”링크 주소) 만을 통해서 HTTP Method인 GET을 구현할 수 있는 점은 아주 매력적이었다. 바로 Swagger를 볼 수 있는 점도 아주 좋았다. return JSONResponse를 통해 손쉽게 Response와, 앞에서 말한 Request를 구현할 수 있는 것도 장점이었다.

입력된 변수를 검증하기 위해 class를 만들어야 하는 점은 불편했다. Django에서도 검증하는 단계가 있던 것으로 기억하는데, FastAPI에선 똑같이 코드를 짜도 순서에 따라 검증에 실패할 수도, 성공할 수도 있다는 점은 마음에 들지 않았다.

팀원들이 구현한 여러 기능들을 함수화하였고, 이를 utils.py에 따로 넣어 프론트엔드와 백엔드에 적용하였다. 여러 AI 관련 모델을 웹 사이트에 적용해보는 경험을 해 본 점이 아주 만족스럽다.

검증 방식 외의 기능은, 전체적으로 만족스러웠다. Django를 사용하기 어려운 환경이거나, 간단하고 빠르게 웹 사이트를 구현할 때 FastAPI는 만족스러운 선택지 중 하나일 것 같다.

## Ⅲ. 대회를 진행하면서 느낀 점/아쉬운 점

편리한 streamlit, FastAPI 프레임워크를 알게 되었다.

streamlit의 한계점도 잘 알게 되었다.

시간을 더 여유있게 두어서, Vue.js로 프론트엔드를 만들었으면 어땠을까 하는 아쉬움이 남는다.

다양한 음성인식 오픈 소스를 다뤄본 점이 만족스럽다.

KoSpeech, OpenSpeech를 사용하기 위해 코드를 다소 고쳐야 해서 시간이 소요된 점은 아쉬웠다.

그렇지만, KsponSpeech Dataset으로 KoSpeech의 Deep Speech 2 모델을 직접 학습시킨 경험은 만족스러웠다.

대회 중간에 모델을 ESPnet으로 바꾸어, 절대적인 시간이 부족했던 것 같다.

CV, NLP도 어렵지만, 음성처리도 못지 않게 어렵다는 것을 알게 되었다.

---

### 이민준_T3145

## **Ⅰ**. 프로젝트 개요

### **NLP_2조_강의 음성 데이터 요약 및 키워드 추출**

아, 또 못들었네! 되감기..

여러분도 이런 경험이 있으신가요?

강의를 들으며 불편했던 경험을 바탕으로, 강의 음성 STT 추출 및 요약, 키워드 추출 사이트를 만들었습니다.

## Ⅱ. 프로젝트설명

영상 속 음성 데이터를 **텍스트로 변환**

변환된 텍스트를 활용하여 **타임라인**, **요약 서비스**, **키워드 추출** 제공

### **데모 페이지**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b80dc58a-1508-4411-8964-adf9f9025bb2/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/11cfe22a-b683-4882-8088-2581e647de13/Untitled.png)

### 모델 파이프라인

저희 서비스에 사용한 모델들과 파이프라인 입니다

음성데이터를 ESPnet을 통해 text데이터로 바꾸고 GPT2로 후처리를 해줍니다.

그 후 각각 요약을 위한 KoBART와 키워드 추출을 위한 SBERT로 들어갑니다.

KoBART로 나온 text는 Koelectra와 KoBigBird를 통해 후처리 됩니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/24a58ec0-022e-4520-af24-5fa2a45ec929/Untitled.png)

## Ⅲ. 문제 해결 과정

제가 맡은 부분은 Summarization과 keyword추출 이었습니다.

특히 Summarization의 서비스 과정 중 생기는 문제를 해결하는것에 집중했습니다.

### [****Summarization****]

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eb720f5d-9178-4ea8-948d-290a3a60ab13/Untitled.png)

Summarization의 개요입니다.

크게 요약문을 생성하는 summarization 파트와 생성된 요약문을 후처리하여 더욱 자연스러운 요약문을 생성하는 Post-Processing 파트로 나뉩니다.

STT 모델을 통해 생성된 텍스트를 finetuning된 KoBART 모델에 입력하여 1차 요약문을 생성하고, 이를 KoELCETRA 모델과 KoBigBird 모델을 통해 후처리하여 최종 요약문을 생성합니다.

1. **문제 개선**
    
    서비스 과정 중 kobart에서 발생한 문제는 크게 3개가 생겼습니다.
    
    - 첫번째는 토큰화 시켰을때 일정길이 이상의 문서에 대해서는 요약하지 못하는것
    - 두번째는 generate시 문장이 반복적으로 나오는것
    - 세번째는 어색한 토큰이 생성되는것 이었다.
    
    첫번째 문제의 해결방법으로 코사인 유사도 기반으로 긴 문서를 자르는 방법을 적용했습니다.
    
    우선 긴 문서를 m개의 문장으로 묶습니다. 하나의 문장으로 유사도를 구하지 않고 m개로 묶은 이유는 문맥의 흐름을 파악하기 위해서 입니다.
    
    이렇게 묶은 문서의 묶음을 TF-IDF를 적용해 vector화 합니다.
    
    그 후 이어지는 묶음 끼리 cosine similarity를 계산한 뒤 cosine similarity가 낮은 부분을 잘라 문단화 시킵니다.
    
    이 때 문단의 최소길이와 최대길이를 지정하여 최소길이와 최대길이 사이의 문장길이로 잘리도록 하였습니다.
    

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ec4879fb-18dd-4789-a29f-ce282cded7d1/Untitled.png)

두번째 문제의 해결방법으로 반복적으로 등장하는 어구를 지워주는 후처리를 적용했습니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/31528102-4534-4bb3-a258-1b244266ba73/Untitled.png)

모델이 생성한 요약문서 중 반복적으로 등장하는 어구가 있었습니다. 반복적으로 나오는 어구는 바로 이어지는 특성이 있었습니다. 

바로 이어서 나오는 중복 어구를 지우게 하였습니다.

### [key word]

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/02eba59e-2da3-4e9e-95ae-87b9503eaf10/Untitled.png)

sentenceTransformer의 SBERT와 KoNLPy를 이용해 키워드를 추출했습니다.
우선 konlpy로 명사구를 뽑은 뒤 SBERT로 키워드를 추출합니다.

키워드 추출 방식은 3가지 버전이 있습니다.
첫번째 버전은 단순하게 키워드와 문서의 코사인 유사도를 통해 문서를 대표하는 키워드를 찾는 방법 입니다.
두번째 버전은 코사인 유사도로 키워드를 추출한 뒤 서로 유사성이 가장 낮은 키워드를 선택하는 Max sum similarity 방법입니다.
세번째 버전은 중복을 최소화하고 결과의 다양성을 극대화하는 Maximal Marginal Relevance 방법 입니다.

### [질문과 관련이 높은 타임라인 제시]

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/aad12bd1-973d-42d2-8d0f-0ff6904a06a3/Untitled.png)

질문과 stt자막의 코사인 유사도를 구해 타임라인과 함께 제시했습니다.
