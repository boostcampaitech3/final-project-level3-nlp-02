# Author
# Soohwan Kim, Seyoung Bae, Cheolhwang Won, Soyoung Cho, Jeongwon Kwak

DATASET_PATH="../../../kspon_dataset/train" #"SET_YOUR_DATASET_PATH"
VOCAB_DEST="../../vocab"
OUTPUT_UNIT='character'                                          # you can set character / subword / grapheme
PREPROCESS_MODE='phonetic'                                       # phonetic : 칠 십 퍼센트,  spelling : 70%
VOCAB_SIZE=10000                                                  # if you use subword output unit, set vocab size

# echo "Pre-process KsponSpeech Dataset.."

#### subword ###
# python main.py \
# --dataset_path $DATASET_PATH \
# --vocab_dest $VOCAB_DEST \
# --output_unit $OUTPUT_UNIT \
# --preprocess_mode $PREPROCESS_MODE \
# --vocab_size $VOCAB_SIZE \
# --output_unit 'subword' \
# --preprocess_mode "spelling" \
# --savepath "../../vocab"

### character ###
python main.py \
--dataset_path $DATASET_PATH \
--vocab_dest $VOCAB_DEST \
--output_unit $OUTPUT_UNIT \
--preprocess_mode $PREPROCESS_MODE \
--vocab_size $VOCAB_SIZE \
--output_unit 'character' \
--preprocess_mode "spelling" \
--savepath "../../vocab"
