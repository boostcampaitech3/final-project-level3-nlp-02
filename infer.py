import torch
import streamlit as st
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
import re

#@st.cache
@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
    # tokenizer = get_kobart_tokenizer()
    return model

model = load_model()
tokenizer = get_kobart_tokenizer()
st.title("KoBART 요약 Test")
text = st.text_area("뉴스 입력:")
text = re.sub('\n','.\n', text)

st.markdown("## 뉴스 원문")
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
            output = model.generate(inputs, eos_token_id=1, max_length=512, num_beams=5)
            output = tokenizer.decode(output[0], skip_special_tokens=True)
            outputs += output
            st.write(output)

    outputs = tokenizer.encode(outputs)
    outputs = torch.tensor(outputs)
    outputs = outputs.unsqueeze(0)
    output_ = model.generate(outputs, eos_token_id=1, max_length=512, num_beams=5)
    output_ = tokenizer.decode(output_[0], skip_special_tokens=True)
    st.write(output_)