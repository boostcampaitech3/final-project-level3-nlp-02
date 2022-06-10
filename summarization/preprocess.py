import pandas as pd
import re
import argparse


def cleansing(text):
    """
    괄호와 내부 내용 제거, 다중 기호 제거, 공백 정규화 등 텍스트를 정제하는 함수

    Args:
        text: 원본 데이터

    Returns:
        cleaned_text: 정제한 데이터
    """
    text = re.sub(r"(ftp:\/\/|www\.|https?:\/\/){1}[a-zA-Z0-9u00a1-\uffff0-]{2,}\.[a-zA-Z0-9u00a1-\uffff0-]{2,}(\S*)", "", text) # url 제거
    text = re.sub(r"[^·가-힣\x00-\x7F]", " ", text)  # ·, 한글, ASCII코드 외 삭제
    text = re.sub(r"[\(\{\[][^)]*[\)\}\]]", "", text)  # 괄호(){}[]와 내부 내용 삭제
    text = re.sub(r"\s{2,}", " ", text)  # 공백 정규화
    text = re.sub(r"\\n", "", text)  # 줄바꿈 제거
    text = re.sub(r"[=-]|;", "", text)  # =, - 제거
    text = re.sub(r'!+', '!',text)  # 반복되는 기호 하나로
    text = re.sub(r'\?+', '?',text)
    text = re.sub(r',+', ',',text)
    text = re.sub(r'\.+', '.',text)
    text = re.sub("\\\"", '"', text)
    cleaned_text = re.sub("\\'", "'", text)

    return cleaned_text


def alien_ratio(text) :
    """
    한국어가 아닌 문자가 해당 데이터(text)를 차지하는 비율을 반환하는 함수

    Args:
        text: 데이터

    Returns:
        alien_ratio: 한국어가 아닌 문자가 데이터를 차지하는 비율
    """
    alien_len = sum([len(word) for word in re.findall(r"[^ㄱ-힣 ]+", text)])
    alien_ratio = alien_len/len(text)
    return alien_ratio


def main(args):
    # load datasets
    dpb_train = pd.read_csv(args.train_path)
    dpb_test = pd.read_csv(args.test_path)

    # cleansing
    train_origin = [cleansing(text) for text in dpb_train['origin']]
    train_summarize = [cleansing(text) for text in dpb_train['summarize']]

    test_origin = [cleansing(text) for text in dpb_test['origin']]
    test_summarize = [cleansing(text) for text in dpb_test['summarize']]

    train_dict = {'origin': train_origin, \
        'summarize': train_summarize}

    valid_dict = {'origin': test_origin, \
        'summarize': test_summarize}

    train_df = pd.DataFrame(train_dict)
    valid_df = pd.DataFrame(valid_dict)

    # remove alien data
    alien_bool = [idx for idx, text in enumerate(train_df['origin']) if alien_ratio(text) > 0.15]
    train_df = train_df.drop(alien_bool)

    train_df.to_csv(args.train_save_path)
    valid_df.to_csv(args.test_save_path)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Argparse Tutorial')
    parser.add_argument('--train_path', type=str, default="/opt/ml/input/final-project-level3-nlp-02/data/train_dpb.csv")
    parser.add_argument('--test_path', type=str, default="/opt/ml/input/final-project-level3-nlp-02/data/test_dpb.csv")
    parser.add_argument('--train_save_path', type=str, default="/opt/ml/input/final-project-level3-nlp-02/data/processed_train_dpb.csv")
    parser.add_argument('--test_save_path', type=str, default="/opt/ml/input/final-project-level3-nlp-02/data/preocessed_test_dpb.csv")

    args = parser.parse_args()
    main(args)