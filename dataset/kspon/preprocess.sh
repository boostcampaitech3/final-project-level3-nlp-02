# Author
# Soohwan Kim, Seyoung Bae, Cheolhwang Won, Soyoung Cho, Jeongwon Kwak

DATASET_PATH="../../../kspon_dataset/train" #"SET_YOUR_DATASET_PATH"
VOCAB_DEST="../../vocab"
OUTPUT_UNIT='character'                                          # you can set character / subword / grapheme
PREPROCESS_MODE='phonetic'                                       # phonetic : 칠 십 퍼센트,  spelling : 70%
VOCAB_SIZE=5000                                                  # if you use subword output unit, set vocab size

# echo "Pre-process KsponSpeech Dataset.."

# character를 사용할 때는 위의 주석을, subword를 사용할 때는 아래 주석을 풀어주세요.

### character ###
# python main.py \
# --dataset_path $DATASET_PATH \
# --vocab_dest $VOCAB_DEST \
# --vocab_size $VOCAB_SIZE \
# --output_unit 'character' \
# --preprocess_mode "spelling"

### subword ###
python main.py \
--dataset_path $DATASET_PATH \
--savepath '/opt/ml/input/kospeech/vocab' \
--vocab_dest $VOCAB_DEST \
--vocab_size $VOCAB_SIZE \
--output_unit 'subword' \
--preprocess_mode "spelling" \
--vocab_size 10000
