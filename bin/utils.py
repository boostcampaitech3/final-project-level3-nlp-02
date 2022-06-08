import os
import itertools
from requests import delete

import torch
import numpy as np
import re
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer




def audio2wav(audio_path, sampling_rate=16000):
    """
    음성 파일을 wav 파일로 변환하고 변환된 파일 경로를 return 합니다.
    """
    path, extension = os.path.splitext(audio_path)
    extension = extension[1:]

    audio_data = AudioSegment.from_file(audio_path, extension)
    audio_data = audio_data.set_frame_rate(sampling_rate)
    audio_data = audio_data.set_channels(1)
    output_path = f"{path}_new.wav"
    audio_data.export(output_path, format="wav")

    if audio_path == output_path:
        print(f"The sampling rate of input audio file is not {self.sampling_rate}, so it is converted and overwritten.")    
    else:
        print(f"The input audio file is not a wav file, so it is converted to 'wav' and saved in {output_path}.")

    return output_path


def split_on_silence(audio_segment, min_silence_len=1000, silence_thresh=-16, keep_silence=100,
                     seek_step=1):
    """
    Returns list of audio segments from splitting audio_segment on silent sections
    audio_segment - original pydub.AudioSegment() object
    min_silence_len - (in ms) minimum length of a silence to be used for
        a split. default: 1000ms
    silence_thresh - (in dBFS) anything quieter than this will be
        considered silence. default: -16dBFS
    keep_silence - (in ms or True/False) leave some silence at the beginning
        and end of the chunks. Keeps the sound from sounding like it
        is abruptly cut off.
        When the length of the silence is less than the keep_silence duration
        it is split evenly between the preceding and following non-silent
        segments.
        If True is specified, all the silence is kept, if False none is kept.
        default: 100ms
    seek_step - step size for interating over the segment in ms
    """

    # from the itertools documentation
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    if isinstance(keep_silence, bool):
        keep_silence = len(audio_segment) if keep_silence else 0

    output_ranges = [
        [ start - keep_silence, end + keep_silence ]
        for (start,end)
            in detect_nonsilent(audio_segment, min_silence_len, silence_thresh, seek_step)
    ]

    for range_i, range_ii in pairwise(output_ranges):
        last_end = range_i[1]
        next_start = range_ii[0]
        if next_start < last_end:
            range_i[1] = (last_end+next_start)//2
            range_ii[0] = range_i[1]

    return ([
        audio_segment[ max(start,0) : min(end,len(audio_segment)) ]
        for start,end in output_ranges
    ], np.array([start + keep_silence for start,end in output_ranges]))


def collate_fn(batch):
    speech_dict = dict()
    speech_tensor = torch.tensor([])
    timelines = []

    audio_max_len = 0
    for data, timeline in batch:
        audio_max_len = max(audio_max_len, len(data))

    for data, timeline in batch:
        zero_tensor = torch.zeros((1, audio_max_len - len(data)))
        data = torch.unsqueeze(data, 0)
        tensor_with_pad = torch.cat((data, zero_tensor), dim=1)
        speech_tensor = torch.cat((speech_tensor, tensor_with_pad), dim=0)
        timelines.append(timeline)
    
    speech_dict['speech'] = speech_tensor
    speech_dict['timeline'] = timelines

    return speech_dict


def processing(text):
    # print("전처리 전:",text)
    # 그게 (0.1프로)(영 점 일 프로) 가정의 아이들과 가정의 모습이야?
    new_arr = []
    p = re.compile(r'(([(]([\w]|[\s]|[가-힣]|[.]|[-])+[)][/][(]([\w]|[\s]|[가-힣]|[.]|[-])+[)])|([(]([\w]|[\s]|[가-힣]|[.]|[-])+[)][(]([\w]|[\s]|[가-힣]|[.]|[-])+[)]))')

    p3 = re.compile(r'(([0-9]+[가-힣]+)|([0-9]+))')
    p4 = re.compile(r'([a-zA-z]+)')
    arr = re.split(p, text)

    i = 0
    # 중복, None 제거
    while i < len(arr):
        token = arr[i]
        if p.match(token):
            new_arr.append(token)
            i = i + 7
        else:
            new_arr.append(token)
            i += 1
    
    result = []
    for token in new_arr:
        if p.match(token):
            if '/' in token:
                t1, t2 = token.split('/')
                t1, t2 = t1[1:-1], t2[1:-1]

                if p3.match(t1) or p4.match(t1):
                    # print(t1,"앞에 선택")
                    result.append(t2)
                else:
                    result.append(t1)
            else:
                t1, t2 = token.split(')(')
                t1, t2 = t1[1:], t2[:-1]
                if p3.match(t1) or p4.match(t1):
                    # print(t1,"앞에 선택")
                    result.append(t2)
                else:
                    result.append(t1)
        else:
            result.append(token)
                
    text = "".join(result)
    text = text.replace('o/', "")
    text = text.replace('n/', "")
    text = text.replace('b/', "")
    text = text.replace('/', "")
    text = text.replace('*', "")
    text = text.replace('+', "")
    text = text.replace('  ', " ")

    return text 


def post_process(model, koGPT2_TOKENIZER, text):
    Q_TKN = "<usr>"
    A_TKN = "<sys>"
    BOS = '</s>'
    EOS = '</s>'
    MASK = '<unused0>'
    SENT = '<unused1>'
    PAD = '<pad>'

    sent = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    text = processing(text)
    processed_text = ""


    with torch.no_grad():
        if text != "":
            while 1:
                input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + text + SENT + sent + A_TKN + processed_text)).unsqueeze(dim=0)
                input_ids = input_ids.to(device)
                pred = model(input_ids)
                pred = pred.logits
                if pred.shape[1] > 100:
                    print("error!!")
                    processed_text = text
                    break
                gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]
                if gen == EOS:
                    break

                processed_text += gen.replace("▁", " ")

    # <unk> 토큰이 발생하면 처리 하지 말고 원본 텍스트 반환하기
    if '<unk>' in processed_text:
        processed_text = text.replace("▁", " ")

    processed_text = processed_text.replace("l", "")
    
    return processed_text


# 중복 문장 지움, 이어서 나오는 중복 문장에 한함
def dell_loop(text):
    new_text = text[:]
    text_ = text.split()
    arr = [i for i in range(len(text_)+1)]
    #print(len(arr))
    can_list = [com for com in combinations(arr, 2) if com[1] - com[0] + 1 <= len(arr)-com[1]]
    #print(can_list)
    for can in can_list:
        string = text_[can[0]:can[1]]
        #print(string)
        stick = can[1]
        len_string = len(string)
        cnt = 0
        for i in range((len(arr) - can[1]) // (can[1] - can[0])):
            end_stick = stick + len_string
            if string != text_[stick:end_stick]:
                continue
            cnt += 1
            stick = end_stick
        if cnt != 0:
            new_text_ = text_[:can[1]] + text_[can[1] + len_string*cnt:]
            new_text = ' '.join(new_text_)
            break
    if text == new_text:
        return text
    else:
        return dell_loop(new_text)


def get_split(text, tokenize_fn, n=3):

    min_len, max_len = 200, 1024

    split_list = text.split('.') # 문서 .으로 나눠놓음
    split_list_tokenize = [tokenize_fn(string) for string in split_list] # split_list 토큰화 함
    if sum([len(sp_t) for sp_t in split_list_tokenize]) < min_len:
        return text

    # n개의 문장씩 문서를 묶음
    text_list = []
    n = 3
    for i in range(len(split_list) - (n - 1)):
        text_list.append('. '.join(split_list[i:i+n]))
    
    # 누적 하고싶다면 주석 풀기, 근데 결과 별로
    # split_list = text.split('.')
    # sum_text = ""
    # text_list = []
    # for t in split_list:
    #     sum_text += t
    #     text_list.append(sum_text)
    # print(len(split_list)) # 159
    # print(len(text_list)) # 157

    # tfidf로 각 묶음 들 벡터화
    tfidf_vectorizer=TfidfVectorizer(
        tokenizer=tokenize_fn, 
        ngram_range=(1, 2), 
        #max_features=50000,
        )
    tfidfv=tfidf_vectorizer.fit(text_list)
    tfidf_matrix = tfidfv.transform(text_list)
    #tfidf_matrix = tfidf_vectorizer.fit_transform(text_list)
    # print(tfidf_matrix.shape) # 157, 4084
    # print(tfidf_matrix[0].shape) # 1, 4084
    #print(tfidfv.vocabulary_)

    # 문장묶음의 유사도 층정
    similar = []
    for idx, (front, back) in enumerate(zip(tfidf_matrix[:-n], tfidf_matrix[n:])):
        similar.append([(cosine_similarity(front, back)).item(),idx])
    # print(similar)
    # print(len(similar)) # 154


    s_similar = sorted(similar, key=lambda x : x[0])
    sp = [0, len(split_list)] # splirt_point, 나누는 지점 [0, 159]
    done_split = []
    finished = []
    def make_split():
        nonlocal finished
        nonlocal sp
        nonlocal done_split
        _, point = s_similar.pop(0) # 가장 유사도 낮은 지점
        point += n # n개의 문장끼리 묶었으니 n 더해줘야 위치 맞음
        if point in done_split: # 나눠놓은 문장을 나누려고 하면 종료
            # print('문장안 접근', point)
            return

        sp_c = sp[:] # sp_copy
        sp_c.append(point) # point 넣고 [0, 159, 73]
        sp_c.sort() # 정렬  [0, 73, 159]
        stick = sp_c.index(point) # 기준이 되는 인덱스 가져옴, 1
        up_p = sp_c[stick-1] # 기준점 위쪽의 문서들 시작 위치 0
        down_p = sp_c[stick+1] # 기준점 아래쪽의 문서들 끝 위치 159
        # print("####up,down,point", up_p, down_p, point, sp_c)
        up = split_list_tokenize[up_p:point]
        down = split_list_tokenize[point: down_p]
        up_len = sum([len(u_t) for u_t in up])
        down_len = sum([len(d_t) for d_t in down])
        # print(up_len, down_len)
        if up_len < min_len or down_len < min_len: # min_len 보다 작으면 자르지 않음
            return
        
        sp = sp_c # point가 자를 수 있으니 sp_c를 sp로 할당
        if up_len <= max_len: # min ~ max 이면 기록
            done_split += [p for p in range(up_p,point)]
            #finished.append([up_p, point])
            finished.append([split_list[up_p:point], up_p])

        if down_len <= max_len: # min ~ max 이면 기록
            done_split += [p for p in range(point,down_p)]
            #finished.append([point,down_p])
            finished.append([split_list[point:down_p], point])

    all_done = [i for i in range(len(split_list))]
    while True:
        #print(finished)
        done_split = sorted(done_split)
        # print(len(done_split), len(all_done))
        if done_split == all_done:
            break
        make_split()

    finished.sort(key=lambda x : x[1])
    # print(finished)
    return finished


def make_specific_link(youtube_link):
    # youtube link 유효성 체크
    youtube_link = youtube_link.url
    if "youtu.be" not in youtube_link and "www.youtube.com" not in youtube_link:
        return False
    # 유효성 검사 완료
    # else:    
    # 상세 주소 가져오기
    specific_link = youtube_link.split('/')[-1]
    # 링크 그대로 가져왔을 때 후처리
    if "?v=" in specific_link:
        specific_link = specific_link.split('?v=')[1].split('&')[0]

    return specific_link

