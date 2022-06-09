from mrc_utils.func import MRC
import streamlit as st



if __name__ == "__main__":

    st.title("무엇이든 물어보세요!")
    
    query = st.text_input("질문을 입력해주세요")
    context = st.text_input("텍스트를 입력해주세요")

    if query!='' and context!='':
        prediction=MRC(query,context)


    if st.button("알려주세요!"):    
        st.subheader(f'정답은??! {prediction}'.format(prediction))
        

    
