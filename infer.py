import torch
import streamlit as st
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
import re
import os
import pickle
from konlpy.tag import Okt
#from sklearn.metrics.pairwise import cosine_similarity
from key_bert import load_embeddings, get_candidates, max_sum_sim, mmr, dist_keywords
#from sentence_transformers import SentenceTransformer
from itertools import combinations


def dell_loop(text):
    new_text = text[:]
    text_ = text.split()
    arr = [i for i in range(len(text_)+1)]
    #print(len(arr))
    can_list = [com for com in combinations(arr,2) if com[1] - com[0] + 1 <= len(arr)-com[1]]
    #print(can_list)
    for can in can_list:
        string = text_[can[0]:can[1]]
        #print(string)
        stick = can[1]
        len_string = len(string)
        cnt = 0
        for i in range((len(arr) - can[1]) // (can[1]-can[0])):
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


#@st.cache
@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
    return model

model = load_model()
tokenizer = get_kobart_tokenizer()
st.title("KoBART 요약 Test")
text = st.text_area("입력:")
#text = re.sub('\n','.\n', text)

st.markdown("## 원문")
st.write(f'len_text : {len(text)}')
st.write(text)

if text:
    text = text.replace('\n', '')
    st.markdown("## KoBART 요약 결과")
    with st.spinner('processing..'):
        input_ids = tokenizer.encode(text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0) # 이 길이가 1024 까지만 들어간다.
        st.write(f'input_shape : {input_ids.shape}')
        input_ids_list = input_ids.split(1000, dim=-1)

        outputs = ""
        for inputs in input_ids_list:
            st.write(inputs.shape)
            output = model.generate(inputs, eos_token_id=1, max_length=300, num_beams=5)
            output = tokenizer.decode(output[0], skip_special_tokens=True)
            output = dell_loop(output)
            outputs += output
            st.write(output)

    st.markdown("### 요약의 요약")
    outputs = tokenizer.encode(outputs)
    outputs = torch.tensor(outputs)
    outputs = outputs.unsqueeze(0)
    outputs = outputs.split(1024, dim=-1)[0]
    output_ = model.generate(outputs, eos_token_id=1, max_length=300, num_beams=5)
    output_ = tokenizer.decode(output_[0], skip_special_tokens=True)
    st.write(dell_loop(output_))

    st.markdown("## keywords")
    candidates = get_candidates(text)
    doc_embedding, candidate_embeddings = load_embeddings(text, candidates)
    top_n = 10
    st.write(dist_keywords(doc_embedding, candidate_embeddings, candidates, top_n=top_n))
    st.write(max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=top_n, nr_candidates=10))
    st.write(mmr(doc_embedding, candidate_embeddings, candidates, top_n=top_n, diversity=0.2))
