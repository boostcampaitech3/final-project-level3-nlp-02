# 최종 프로젝트

![youtube_professor](https://user-images.githubusercontent.com/76618935/172767342-7c220388-39c1-441e-ab2c-e402e81db769.png)

원하는 유튜브 영상을 STT를 통한 **text 추출** 및 **요약, 키워드 추출, MRC**를 지원하는 서비스입니다.


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


# 실행방법  
shell의 cd 기능을 이용하여 다음 폴더로 이동한 뒤, 다음 명령어를 통해 실행시킵니다.  
## frontend  
~~/espnet-asr/bin/frontend  
sh frontend.sh  

## backend  
~~ /espnet-asr/bin/backend  
python main.py  



# espnet-asr
*espnet-asr* is an End-to-end Automatic Speech Recognition (ASR) system using [ESPnet](https://github.com/espnet/espnet).

End-to-end ASR systems are reported to outperform conventional approaches.
However, it is not simple to train robust end-to-end ASR models and make recognition efficient.

In this project, we provide an easy-to-use inference code, pre-trained models, and training recipes to handle these problems.

The pre-trained models are tuned to achieve competitive performance for each dataset at the time of release, and an intuitive inference code is provided for easy evaluation.

## 1. Installation
To run the end-to-end ASR examples, you must install [PyTorch](https://pytorch.org/) and [ESPnet](https://github.com/espnet/espnet).
We recommend you to use virtual environment created by [conda](https://docs.conda.io/en/latest/miniconda.html).

```conda create -n ESPnet python=3.10.4```

```conda activate ESPnet```

```
conda install cudatoolkit
conda install cudnn
```


---

Install pytorch according to your **GPU** version (or **CPU**).

Detail [Here](https://pytorch.org/get-started/locally/)

**CPU**

```(ESPnet) conda install pytorch torchvision cpuonly -c pytorch```

**CUDA 10.2**

```(ESPnet) conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch```

**CUDA 11.3**

```(ESPnet) conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch```

---

Then, install ESPnet.

```(ESPnet) pip install espnet```


## 2. Downloading pre-trained models
You can download pre-trained models for Zeroth-Korean, ClovaCall, KSponSpeech and Librispeech datasets. You can check the performance of the pre-trained models [here](https://github.com/hchung12/espnet-asr/tree/master/recipes).

```(ESPnet) tools/download_mdl.sh```


