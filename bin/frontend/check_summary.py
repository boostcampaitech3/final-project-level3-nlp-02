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
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
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
    st.write("STT 작업이 완료되었습니다.")
    talk_list = ["""안녕하세요 입반 비디에서는 구글에서 발표한 어텐션 이제 올견이든 논문에서 다른 트랜스포머에 대해서 한번 알아볼까 합니다
    트랜스포머나는 기존 인코더디 코더를 발전시킨트 임러닝 모델이구요
    가장 큰 차이점은 알엔 엔을 사용하지 않는다는 데 있습니다
    트랜스포머는 기계 번역에 있어서 기존 알 앤 앤 기반인 코더디 코더보다 학습이 빠르고 성능이 좋아서 큰 관심을 이끌었구요
    논문에서 빨리 취한 이 시험 결과를 통해서 트랜스포머는 영어를 독일어와 프랑스어로 번역하는 항목에서
    학습 속도도 빨랐고 성능도 좋았다라는 것을 확인할 수 있습니다
    그런 데만입니다
    트랜스포머는 어떻게 더 빠르게 확습이 될 수 있었을까요
    이유는 바로 트랜스포머는 아레맨을 사용하지 않는다는데 크게 있으니
    트랜스포머를 한 단어로 표현하자면 벽렬화입니다
    게 말해서 트랜스포머는 일을 최대한 한 방에 처리합니다
    아래맹이 순차적으로 첫 번째로 입력된 단어부터 마지막 단어까지 계산하에서 입력된 단어들을 인코딩하는 반면 트랜스포머는 한 방에 이 과정을 처리하는 거죠
    말로 설명하는 거 없다 그림과 함께 한번 보도록 하겠습니다
    전통적인 아래는 기반 인 코더티코더는 입력값이 알러뷰일 경우에 아예부터 순차적으로 상태값을 계산하지요
    해서 러브도 계산하고 유도 계산하고 해서 최종적으로 이 상태 값을 문맥벡터로 사용하게 되는 거죠
    그거 저는 이 문맥 벡터를 기반으로 입력대 문장을 번역하게 됩니다
    자난 널 사랑해 이렇게 번역되게 되죠
    코다이 번역은 엔드 시그널이 나타날 때까지 아 계속돼서 이제 완료가 됐구요
    당근 보신 예제는 가장 전통적인 인코더 디코더 모델이구요 이게 가장 전통적인 인코더 디코더는 블랙벡터가 고정된 크기라서
    책과같이 기인 입력 문장 같은 경우에는
    정된 문래백터에 모든 정보를 저장하기에 힘들어서 번역 결과가 엉터리가 되는 경우가 많습니다
    이랑이 이유로 어 텐션 매커니즘을 사용한 조금 더 진보된 인코더 디코더가 출연하게 되는데요
    자 다음 슬라이드에서 보실 이 모델이 바로 트랜스포머 이전에 인코더 디코더라고 보시면 되겠습니다
    최초 아이강력 되고 스테이트 계산되고 스터디 상태가 계산됐구요 앱
    상태가 계산되고 스쿨 상태가 계산됩니다
    이렇게 아이 스터디 에스쿨 해서 이렇게 선택 값들이 개상이 된 다음에
    이 어텐션 활용 인코더티코더에 가장 큰 진보단점은 고정된 그게 문맥벡터를 이제 사용하지 않는다는 데 있습니다
    신 단어를 하나씩 번역할 때마다 동적으로 인코더 출력값에 어 텐션 메카니즘을 정해서 효율적으로 번역을 한다는 큰 장점이 있지요
    이 모델은 고정된 문백팩터 사용하지 않고 인코드에 모든 상태 값을 활용한다는 그 특징이 있어
    자 통적으로 인코더를 활용하기 때문에 김문장에 번역 성능이 이전 인코더 디코더 모델보다 더 나아졌어
    어텐션 맥카니즘은 기존 인코더 디코더의 성능을 상당히 강화시켰다고 큰 주목을 받았어
    하지만 여전히 아르넨세를 순차적으로 계산해서 늘다라는 단점이 존재했구요 성능도 더 높일 수 있는 가능성이 충분히 있었죠
    사람들은 아이 앤 앤을 대신에 빠르고 성능 좋은 방법을 찾아서 고민을 했고
    커텐션만으로도 입력 데이터에서 중요한 정보들을 찾아내서 단어를 인코딩 할 수 있지 않을까 생각을 하게 되었어
    그리고 그들은 성공했고 그 성공 결과를 다룬 논문이 바로 이 어텐션이죠 올 dd입니다
    아레나의 순차적인 계산은 트랜스포머에서 단순히 행렬 곱으로 한 번에 빵하고 철이 되게 넣었죠
    제가 트랜스포머는 한 번에 연산으로 모든 중요 정보를 각자나 인코딩 하게 됩니다
    이번 슬라이드에서 트랜스포머티코더의 연상과정은 기존에 어텐션 기반 인코더 디코더와 사문 닮아있음을 확인할 수 있어요
    짜 트랜스포머에서 가장 큰 특징은 누가 뭐라고 해도 아른 애는 성공적으로 인코더 디코드에서 제거했다는 것이죠
    코데이 번역과정은 기존 인코더 디코더 방법과 동일하게 자 스타트 싸인으로 시작해서 엔드 싸인까지 이렇게 번역을 하게 됩니다
    이대문에서 트랜스포머는 확실히 기존 인코더티 코대의 컨셉을 간직하고 있다는 것을 확인하고 싶네요
    제가 기존 인코더디코더의 주요 컨셉을 간직하되 아레네는 없애서 학습시간을 단축했고 어텐션뿐만 아니라 다음하는 스마트 함 기술들을 함께 제공하여서 성능도 올렸어요
    어텐션 및 그 왜 사용된 기술에 대해서는 앞으로 알아보도록 하겠습니다
    자 자연어 처리에서 문장을 처릴 때 실제 단어의 위치의 순서는 상당히 중요해요 아르맨이 자연어 처리에 상당히 마녀 활용되는 이유도 바로 이 아르맨이 단어의 위치와 순서 정보를 잘 활용하는 데 있기 때문이죠
    그렇다면 이 아래랭이 없는 트랜스포머는 어떻게 단어의 윗침의 순서 정보를 활용할 수 있을까요
    정답은 바로 포지션을 인코딩입니다
    포지션은 인코딩이랑 인코더미 디코드 입력관마다 상대적인 위치 정보를 더 해주는 기술이에요
    이번 슬라이드에서 벡터를 작은 연속된 상자로 나타내야 보았는데요
    짜 간단한 빛으로 사용한 포지션화 인코딩 예제가 보여드리고 있습니
    슬라이딩이 제가 첫 번째 단어 아이게는 00 1을 더 해줬고 두 번째 단어 스터디에는 0 인령에서는 0 1 1
    그리고 네 번째 단화에는 일 영영을 더 해줬죠
    같은 방식으로 디코더에 입력 값에도 포지션한 인코딩을 적용해 줄 수 있습니다
    자 트랜스포머는 이렇게 제가 예접으로 보여드렸던 이케 비트의 그 포지션한 인코딩이 아닌 싸인과 코싸인 함수를 활용한 포지션한 인코딩을 사용합니다
    싸인 코싸인 함수를 사용함포지 상황 인코딩 크게 두 가지 장점이 있어요
    진짜 첫 번째는 항상 포츠나 인코딩이 값은 마이너스 1부터 1 사이에 값이 나온다는 거고 두 번째로는 모든 상대적인 포츠나 인코딩의 장점으로써 학습데이터든 가장 긴 문장보다도 더 긴 문장이 실제 운영 중에 들어와도 포츠나 인코딩이 에러 없이 상대적인 인코딩 값을 줄 수 있다는 데 있겠습니
    자 갓 단어에 워딩 배딩에 포지션 임배딩을 저해준 후에
    그다음 해줘야 될게 바로 이 셀프 어텐션 연상입니다 가장 중요한 거죠
    인코드에서 이뤄진 언 어텐션 연산을 셀프 어텐션이라고 해요
    일단은 이번 슬라이드에서는 코어리키 밸류라는 개념만 일단 알아두심 될 거 같습니다 다음 슬라이드로 넘어가도록 할게요
    이제 이 코리키 밸류는 더블큐 더블 케이 w핑이 행렬에 의해서 각각 상성되고 이 행렬들은 단순히 웨이트 매트릭스로 딥런인 모델 학습 과정을 통해서 최적화됩니다
    워드 인 배딩은 벡터이고 실제 한문장은 행렬이라고 말 수가 있겠죠
    행렬은 행렬과 곱할 수 있으므로 강 문장에 있는 단어들 코리키 밸류는 행렬 곱을 청해서 한 방에 빵 하고 구할 수가 있습니다
    자 이제 코리 키
    베일리만 있으면 슬프 어텐션을 수행할 수가 있어요
    이 코리 키 밸류는 벡터에 형태라는 거 꼭 기억해 주시긴 해 다람이
    현재의 단어는 쿠어리구요 어떤 단어와의 상관관계를 구할 때 이 쿠어리를 그 어떤 단어에
    흰값을 곱해줍니다
    이 코리 와킬 곱판값의 어텐션 스코라고 해요
    퍼리와 키가 둘 다 백 타임으로 둘을 다 프로덕트로 곱할 경우에는 그 결과는 숫자로 나오게 되겠죠 이 숫자가 높을수록 단어의 연관성이 높고 낮을수록 연관성이 낮다 라고 생각하셔도 될 거 같습니다
    어텐션 스쿨을 0부터 1까지의 확률 개념으로 바꾸기 위해서 소프트 맥스를 적용해줘요
    논문에선 소프트맥스를 적용하기 전에
    코아를 키벡터에 찬스로에 루트값으로 나눠졌는데요
    논문에 따라는 이 키벡터에 차원이 늘어날수록 다 프로젝트 계사시 값이 증대되는 문제를 보완하기 위해서 이런 조치를 취했다고 합니다
    소프트맥스의 결과값은 깃값에 해당하는 단어가 현재 단어에 어느 정도 연관성이 있는지를 나타냅니다
    예를 들어서 단어 아이는 자기 자신과 92 자 스터디 5 에세 2 그리 스쿨에 1프로 연관성이 있다고 생각을 하실 수가 있겠습니다
    각 퍼센테이지 각희 해당하는 밸류 곱해줘
    연관성이 높아 연관성 망은 컵 해준 결과야 연관성이 별로 없는 밸류값은 거의 힘해졌죠 자 눈으로 직접 확인하실 수가 있습니다
    최종적으로 어텐션이 착용돼에 힘이 태진 이 밸류들 모두 도해줘요
    저기 최종 팩터는 이제 단순히 단어 아이가 아닌
    문장 속에서의 단어아이가 지인 전체적인 의미를 진인 벡터라고 간주를 할 수 있게 됩니다
    단어 인 배딩은 팩터임으로 입력 문장 전체는 행렬로 표시할 수가 있겠죠
    키 페리 코리더 모든 향렬로 저장되어 있으니까
    모든 단어에 대한 이 모든 어텐년 어텐션 연사는 행렬 곱을 한 방에 빵하고 또 처리할 수가 있습니다
    아래는 사용했다면
    처음 단어부터 끝 단어까지 순차적으로 계산을 했었어야 했는데 말이죠
    이것이 바로 어텐션을 사용은 벽열처리에
    가장 큰 장점입니다
    사실 트랜스포머는 보다 더 많이 벽열처리를 적극 활용합니다
    트랜스포머는 예전에 예제에서 보셨던 이 어텐션 레이어 여덟개를
    병렬로 동시에 수행해
    지금 슬라이드는 간당에 슬기 어텐션 레이얼 동시에 수행하는 모습을 제가 그려보았는데요
    째로는 여덟 개입니다
    여러 개의 어텐션 네이어를 박렬 처리함으로써 얻는 이점은 무엇일까요
    여기 유명한 제 알람하의 예제를 제가 참조해봤는데요
    병열처리된 어텐션 레이어를 멀티 헤드 어텐션이라고 부릅니다
    그리고 멀티 헤드 어텐션은 예의제와 같은 기계 번역에 큰 텀을 줘요
    제가 문장이 상당히 애매해였던 애니말디든 크로스트 어 스트리픽 거직 이 워치 투 타이얼
    무엇이 이실 거야 여기 이과
    진짜 문장이 이렇게 상당히 모할 경우에
    짜 두 개 다는 병렬화된 어텐션이 서로 다르지만
    타고 연관성이 높은 단어에 포커 싸고 있는 모습을 포실수가 있어요 까 첫 번째 어텐션은 어디죠 애니몰에 상당히 포커스를 맞췄고 두 번째 어텐션은 스트리트에 어텐션 포커스를 맞췄죠
    진짜 사람에 문장은 모호할 때가 상당히 많고 한 게 어텐션으로 이 모호한 정보를 충분히 인코딩하기 어렵기 때문에
    멀티해도 어텐션을 사용해서 되도록 연관된 정보를 다른 관점에서
    집해서 이 점을 보완할 수 있습니다 이게 바로 멀티해도 어텐션이죠
    그리고 바로 이 모습이 인코드에 전반적인 구조입니다 포시다시피 단어를 워드인 배딩을 전하는 후에 포지셔널 인코딩을 적용하죠
    그리고 멀티 헤드 어텐션에 입력합니다
    멀티에드 어텐션을 통해서 출력된 여러 개 결과값들은 모두 이어붙여서
    또 다른 행렬과 곱해져 결국 최초 워드 인별인 거야
    동일한 차원에 갖는 벡터로 출력이 되게 되었습니다
    가카 캐릭턴 따로 따로 또 플릭 커네킨을 레어로 들어가서 임력과 동일한 사이즈에 100도를 떠다시 출력이 되고 오구요
    못 받아격이 중요한 거는 이 출력 백터의 차원에 크기가 입력 벡터와 동일하다는 데 또 있습니다 이거 기억해주세요
    자 워딩 배딩에 포지션을 인코딩을 더 해줬던 거 기억하시죠 딥러닝 모델을 학습하다 보면은 역전파에 의해서 이 포지션을 인코딩이 많이 선실될 수가 있어요
    일을 포안하기 위해서 레지디얼 커넥션으로 인력 된 값을 다시 한 번 더 해주는 것도 눈여겨봐야 될 점이니
    레지즈오 커넥션 뒤에는 레이언 노마일리지에이션 사용해서 학습의 효율을 또 증진시킵니다
    여기까지 가 바로 인코더 레이어입니다 하지만 한 가지 더 인코더 레이어에 입력 벡터와 출력 벡터에 청원의 크기가 같다는 거
    기억하시고 계시죠 이만한 잭슨 인코더 레이어를 여러 개 붙여서 또 사용할 수 있다는 말이에요
    트랜스포머에 인코더는 실제 이 인코더 레얼을 여섯 개 연속적으로 붙인 구조입니다
    중요한 점은 각각의 인코들 레이어는 서로의 모델 파람 있다고 즉 가정치를 공유하지 않고 따로 학습시켜요
    트랜스포머 인코드에 최종 출력 값은 바로 여기에 보이시는 여섯 번째 인코더 레이의 출력 값입니다
    자 이제 정말 인코드에 대한 정리가 끝났네요 이제 디코드에 대해서 한번 알아보도록 할게요
    제가 포시다시피티코더는 인코더와 상당히 유사하게 생겼어요 인코더와 같이 여섯 개 동일할 레어로 구성된 것도 확인할 수가 있구요
    코더는 기존 인코더 디코더의 장동 방식과 같이 최초 단어부터 끝 단어까지 순차적으로 이 단어를 출력함이
    디코더 역시 어텐션 벽렬처리를 적극 활용해
    디코드에서 현재까지 출력된 값들에 어텐션을 적용하고 또한 인코더 최종 출력값에도 어텐션이 적용이 됩니다
    보시는 거야 같이 작동이 되죠
    디코더를 통해 어떻게 영어를 한글로 번역하는지 그림을 통해서 알아볼 수가 있습니다
    가 찍은 보시는 그림은 인코드의 구조예요 이제 디코드의 구조 한번 비교를 해보도록 하겠습니다
    인코드에는 뭘 테더 텐션 필드 포워든 레이어 그리고 레지디오 카넥션이 있죠
    제가 지금 보시는 건 디코더 레이어에요 디코더도 인코더와 상당히 유상한 구조를 지니고 있지만 몇 가지 차이점이 있습니다
    첫째 첫 번째 멀티해든 어텐션 레이어는 마스크든 멀티해든 어텐션 레이 레이어라고 불려요
    마스크드라는 이름이 붙은 것은 단순히 디코들 레이어에서
    지금까지 출력된 갑들에만 어텐션을 적용하기 위해서 뿌쳐진 이름입니다 티코 등시에 아직 출력되지 않은 미래의 단어의 어텐션을 적용하면 안 되기 때문이죠
    제가 다음 단계는 멀티해드 어텐션 레이어입니다
    인코더처럼 키 밸류 커리로 연산을 하는데요
    인코더에 멀티헤드 어텐션 레어와 가장 큰 차이점은
    디코더에 멀티 헤드 어텐션은
    현재 디코데 인력가스
    커리로 사용하고 인코더에 최종 출력값을 키와 벨류로 사용한다는 데 있습니
    쉽게 설명하는 디코더의 현재 상태를 커리로 인코드에 질문하는 거구요 인코더 출력급에서 중요한 정보를 키와 벨류로 획득해서 디코더의 다음 단어의 가장 적합한 단어를 출력하는 과정이라고 보지면 될 거 같습니
    그리고 다음엔 인코더 마찬가지로 피드포어들 레이얼 정해서 최종 값을 벡터로 출력하게 되어있습니다
    지금까지 백터로만 출력 값을 얘기했는데요 그렇다면 이 백터를 어떻게 실제 단어로 출력할 수 있을까요
    실제 단어로 출력하기 위해서 디코도 최종 단에는 리미어 레이어와 수업 트 맥스 레이어가 존재합니다
    링어레이어는 소프트 맥스에 일력 값으로 들어갈 로짓을 생성하는 거구요 자 서프트 맥스는 모델이 알고 있는 모든 단어들에 대한 확률값을 출력하게 되고 가장 높은 확률을 진입 값이 바로 다음 단어가 되는 거겠죠
    자 재밌게도 트랜스포머는 최종 단계에도 레이블 스모딩이라는 기술을 사용해서 모델의 퍼포넛을 다시 한번 한 단계 업그레이드시킵니다
    보통 딥러닝 모델을 서프트 맥스를 학습할 경우에는 레이브를 원 핫 인코딩으로 전환을 해주는데요 이번 슬라이드에서 확인할 수 있었을 있듯이 이 트랜스포머는 원 핫 인코딩이 아닌 뭔가 일에는 가깝지만 일이 아니고 영애는 가깝지만 영이 아닌 값을 표현되어 있는 것을 눈으로 직접 보실 수가 있어요
    자 이 기술 레입을 스무딩이라고 합니다 양 또는 1이 아닌 정답은 1에 가까운 값 자 오답은 영향이 가까운 값으로 이렇게 살짝살짝 변화해주는 기술인데요 모델 학습시에 모델이 너무 학습대에 치중하여 학습하지 못하 아 못하도록 보완하는 기술입니다
    자 이것이 어떻게 학습에 도움이 되는지 궁금하실 수 있습니다
    자 학습데이터가 매우 깔끔하고 예측 갈 값이 확실한 경우엔 도움이 안 될 수도 있어요 하지만 레이블이 노이지한 경우 즉 같은 인력 값인데 다른 출력 값들이 학습데이터에 많을 경우 레이블 스무딩은 큰 도움이 됩니다
    왜냐하면 결국 학습 필요하는 것은 진짜 소프트맥스에 출력 학과 벡터를 전환된 레이블의 차이를 줄이는 것인데
    어 같은 데이터에 서로 상현 정답들이 원하 인코딩도 존재한다면
    모델 파라미터가 크게 커졌다가 작아졌다가 반복하고 확실히 원활하지 않겠죠
    실제 예제를 들어보자면 영어로 땡큐에 대한 학습 데이터가 있다고 볼게요 두 개의 땡큐가 있는데 첫 번째 레이블은 한국말로 고마우고 두 번째 레이블은 감사합니다라고 되어 있을 수도 있겠죠
    두 학습 데이터 모두 잘못된 게 아니에요 하지만은 원 학 인코딩을 이 두 레이블에 적용시 고마워와 감사합니단 완전히 다른 상의한 두 벡터가 되고 땡크에 대한 학습은 레이블이 상의한 이유로 원활이 진행되지 않을 수 있겠습니다
    이럴 경우 레이블 스모딩을 정용하면 고마워 와 감사합니다 원 한 인코딩보단 조금만 가까진 벡터가 되고 또한 서프트맥스 출약 각과 레이블의 차이 없이 조금은 줄어들어서 효율적인 학습을 기대할 수 있게 되는 거겠죠
    자 트랜스포머에 대한 이야기는 여기까지니다
    그리고 여기 제가 공부하는데 너무나 많은 도움을 준 논문과 블로그 여기 제가 담아봤습니다
    트랜스포머보다 더 깊게 이해하고 싶으신 분들은 여기에 내 프랑스 참조의식을 바랍니다
    비디오 시청 얼 응 언제나 감사드리고요 이 비디오가 트랜스포머에 대한 이의 큰 도움에 되길 바랍니다
    자 그럼 타운 비디에서 뵐게요 감사합니다"""]
    text = ''.join(map(str, talk_list))
    # print(text)
    talk_list = text.split('\n')
    model_summary = load_model()
    _ = model_summary.eval()
    print('####', len(talk_list), talk_list)
    st.title("KoBART 요약 Test")
    temp_talk_list = [talk.strip() for talk in talk_list]
    print(temp_talk_list)
    # text = ' '.join(map(str, temp_talk_list))
    # text = st.text_area("입력:")
    # if '.' not in text:
    #     text = re.sub(r'\n',r'. ', text)

    # data = {
    #     'talk_list': temp_talk_list,
    # }

    # # 유튜브 음성파일을 가리키는 링크인지 확인하기.
    # response = requests.get(
    #     # url=f"{backend_address}/summary",
    #     url=f"{backend_address}/summary_before",
    #     json=data
    # )

    # outputs = response.json()['outputs']
    # # print(outputs)
    # st.write(outputs)

    data = {
        'talk_list': temp_talk_list,
    }

    print(type(data['talk_list']))
    print('@@@@', data['talk_list'])

    response = requests.get(
        # url=f"{backend_address}/summary",
        url=f"{backend_address}/keyword",
        json=data
    )

    print('####', response.json())
    print(response)
    print(response.json())
    
    results = response.json()['outputs']
    for result in results:
        st.write(','.join(map(str, result)))



    # if text:
    #     text = text.replace('\n', ' ')
    #     split_text_list = get_split(text, tokenizer.tokenize, n=3) # [[문단,0],[문단,21],[문단,46] ... ]
    #     st.markdown("## KoBART 요약 결과")
    #     with st.spinner('processing..'):
    #         outputs = ""
    #         for split_text in split_text_list:
    #             sp_text = '. '.join(split_text[0])
    #             input_ids = tokenizer.encode(sp_text)
    #             input_ids = torch.tensor(input_ids)
    #             input_ids = input_ids.unsqueeze(0) # 이 길이가 1024개 까지만 들어간다.

    #             #st.write('본문')
    #             #st.write(sp_text)
    #             #st.write(f'input_shape : {input_ids.shape}')
    #             output = model_summary.generate(input_ids, eos_token_id=1, max_length=200, num_beams=5) # eos_token_id=1, max_length=100, num_beams=5)
    #             output = tokenizer.decode(output[0], skip_special_tokens=True)
    #             output = dell_loop(output)
    #             outputs += output
    #             #st.write('요약')
    #             st.write(output)


    #     st.markdown("### 요약의 요약")
    #     outputs = tokenizer.encode(outputs)
    #     outputs = torch.tensor(outputs)
    #     outputs = outputs.unsqueeze(0)
    #     outputs = outputs.split(1024, dim=-1)[0]
    #     output_ = model_summary.generate(outputs, eos_token_id=1, max_length=300, num_beams=5)
    #     output_ = tokenizer.decode(output_[0], skip_special_tokens=True)
    #     output_ = dell_loop(output_)
    #     st.write(output_)

    #     st.markdown("## keywords")
    #     # get_keyword(text, top_n=10)

  

if __name__ == "__main__":
    main()

    # get_response = requests.post(
    #     url=f"{backend_address}/write",
    #     data=data,
    #     headers={"Content-Type": "application/json"}
    # )