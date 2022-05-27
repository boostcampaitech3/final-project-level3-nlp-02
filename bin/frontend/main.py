import streamlit as st
import requests
import time
import os


backend_address = "http://localhost:8001"

def main():
    st.write('반갑습니다.')
    response = requests.get(f"{backend_address}/")
    st.write(response)
    label = response.json()
    st.write(label)
    st.write('반갑습니다.~')

    url = st.text_input("텍스트를 입력해주세요", type="default")
    # 텍스트 입력안내 문구
    if not url:
        st.write('유튜브 링크를 넣어주세요.')
        return

    data = {
        'url': url,
    }

    # 유튜브 음성파일을 가리키는 링크인지 확인하기.
    response = requests.post(
        url=f"{backend_address}/check_link",
        json=data
    )
    
    if response.status_code == 400:
        st.write('유튜브 링크 형식이 잘못되었습니다.')
        return
    specific_url = response.json()['url']

    data = {
        'url': specific_url,
    }

    # url에 맞는 유튜브 음성파일 가져오기
    response = requests.post(
        url=f"{backend_address}/set_voice",
        json=data
    )

    st.write(response)

    # 유튜브 음성파일 생성되었는지 확인
    with st.spinner("유튜브에서 음성을 추출하고 있습니다."):
     # 파일 만들어질때까지 spinner 빠져나가지 않기
        while True:
            # 1초마다 확인
            time.sleep(1)
            
            # 파일 있으면 while 탈출
            if os.path.exists(f'../../download/{specific_url}/{specific_url}.wav'):
                break
    
    ### 필요없을거같아서 주석처리
    # # 생성한 파일 가져오기
    # response = requests.post(
    #     url=f"{backend_address}/get_voice",
    #     json=data
    # )
    # st.write(response.json())
    # print(response.json())
    # 음성 파일 STT 돌리기
    st.write("음성 추출이 완료되었습니다.")
    st.write("STT 작업이 진행중입니다.")
    with st.spinner("STT 작업을 진행하고 있습니다"):
        response = requests.post(
            url=f"{backend_address}/stt",
            json=data
        )
    st.write("STT 작업이 완료되었습니다.")
    st.write(response)
    st.write(response.json())





    


    # get_response = requests.post(
    #     url=f"{backend_address}/write",
    #     data=data,
    #     headers={"Content-Type": "application/json"}
    # )


if __name__ == "__main__":
    main()
