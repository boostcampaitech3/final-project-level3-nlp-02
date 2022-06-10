import os
import argparse
from matplotlib.pyplot import yscale
import pandas as pd
import re

import re
import random

def remove_space(text):
    string=""
    arr=text.split()
    
    for s in arr:
        rand_num=random.randrange(1,3)
        if rand_num==1:
            string+=s
        else:
            string+=' '+s
            
    return string

def add_character(text):
    rand_num=random.randrange(0,len(text))
    char_arr=['.',',','?','!']
    
    num=random.randrange(0,3)
    
    for _ in range(num):
        num=random.randrange(0,4)
        text = text[:rand_num]+char_arr[num]+text[rand_num:]
        
    return text

def add_noise(text):
    rand_num=random.randrange(0,len(text))
    char_arr=['아','가','그','어','오','으','이','디','음','우','거']
    
    num=random.randrange(0,3)
    
    for _ in range(num):
        num=random.randrange(0,len(char_arr))
        text = text[:rand_num]+char_arr[num]+text[rand_num:]
        
    return text

def processing(text):

    # 그게 (0.1프로)(영 점 일 프로) 가정의 아이들과 가정의 모습이야?
    new_arr = []
    p = re.compile(r'(([(]([\w]|[\s]|[가-힣]|[.*,%-])+[)][/][(]([\w]|[\s]|[가-힣]|[.*,%-])+[)])|([(]([\w]|[\s]|[가-힣]|[.*,%-])+[)][(]([\w]|[\s]|[가-힣]|[.*,%-])+[)]))')

    p3 = re.compile(r'(([0-9]+[가-힣]+)|([0-9]+))')
    p4 = re.compile(r'([a-zA-z]+)')
    arr = re.split(p, text)

    i=0
    # 중복, None 제거
    while i<len(arr):
        token=arr[i]
        if p.match(token):
            new_arr.append(token)
            i=i+7
        else:
            new_arr.append(token)
            i+=1
    
    result=[]
    for token in new_arr:
        if p.match(token):
            if '/' in token:
                t1,t2=token.split('/')
                t1,t2=t1[1:-1],t2[1:-1]

                if p3.match(t1) or p4.match(t1):
                    # print(t1,"앞에 선택")
                    result.append(t2)
                else:
                    result.append(t1)
            else:
                t1,t2=token.split(')(')
                t1,t2=t1[1:],t2[:-1]
                if p3.match(t1) or p4.match(t1):
                    # print(t1,"앞에 선택")
                    result.append(t2)
                else:
                    result.append(t1)
        else:
            result.append(token)
                
    text="".join(result)
    text=text.replace('o/',"")
    text=text.replace('n/',"")
    text=text.replace('b/',"")
    text=text.replace('u/',"")
    text=text.replace('/',"")
    text=text.replace('*',"")
    text=text.replace('+',"")
    text=text.replace('l',"")
    text=text.replace('u',"")
    text=text.replace('  '," ")

    return text 


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--df_save_name', type=str, default="0608_dataset")
    parser.add_argument('--kspon_num', type=int, default=60000)
    parser.add_argument('--dataset_num', type=int, default=150000)
    parser.add_argument('--trn_path', type=str, default="/opt/ml/post_processing/post_dataset/kspon/train.trn")
    parser.add_argument('--kspon_path', type=str, default="/opt/ml/post_processing/post_dataset/kspon/processing")
    parser.add_argument('--noise_ratio', type=float, default=0.3)

    args = parser.parse_args()

    x_list=[]
    y_list=[]
    id_list=[]
    pattern = re.compile(r'[a-zA-Z]')
    n=0

    # ESPnet 예측값으로 만드는 데이터셋
    while True:
        root_path=input("데이터셋으로 만들 폴더의 절대 경로를 입력하세요 (없다면 None) :")
        if root_path=='None':
            print("샘플링 완료")
            break

        file_list = os.listdir(root_path)

        for folder in file_list:
            if n>args.dataset_num:
                break
            cnt=0
            arr=[]
            for file in sorted(os.listdir(os.path.join(root_path,folder))):
                if file[0]=='.':
                    arr=[]
                    cnt=0
                    continue
                f = open(os.path.join(root_path,folder,file), 'r')
                lines = f.readlines()
                if lines==[]:
                    print("빈문장 pass")
                    arr=[]
                    cnt=0
                    continue
                
                arr.append(lines[0])
                cnt+=1

                if cnt==2:

                    y,x=arr

                    y=processing(y)

                    if '(' in y:
                        print("버리기")
                        # 처리가 안됐으면 버리기
                        cnt=0
                        arr=[]
                        continue

                    rand_num=random.randrange(1,101)
                    if rand_num<100*args.noise_ratio:
                        # noise 추가
                        rand_num=random.randrange(1,101)
                        x = remove_space(x)
                        if rand_num<30:
                            x = add_character(x)
                            x = add_noise(x)

                    x_list.append(x)
                    y_list.append(y)
                    id_list.append(file)
                    arr=[]
                    cnt=0
                    n+=1

    # kspon dataset 추가
    path_list=os.listdir(args.kspon_path)
    path_list.sort()

    for D_dir in path_list:
        file_path = os.path.join(args.kspon_path, D_dir)
        file_id=file_path.split('_')[-1]

        if file_path.split("/")[-1][:2]=='GT':
            # print(args.kspon_path+'/'+file_path.split("/")[-1][3:])
            if file_path.split("/")[-1][3:] in path_list:
                f = open(file_path, 'r')
                y = f.readline()
                f.close()

                if pattern.match(y)==None:
                
                    y=processing(y).strip()
                    y_list.append(y)
                    id_list.append(file_id)

                    f = open(args.kspon_path+'/'+file_path.split("/")[-1][3:], 'r')
                    x = f.readline()
                    f.close()
                    x_list.append(x)

    # 규칙기반으로 만드는 가상의 데이터셋
    f = open(args.trn_path, 'r')
    line = f.readlines()

    cnt=0
    text_x,text_y="",""
    for idx, l in enumerate(line):

        if cnt==args.kspon_num:
            break
            
        text_y=l.split("::")[-1]
        text_y=processing(text_y).strip()
        if '(' in text_y:
            print("가상 데이터 버리기")
            continue

        if pattern.match(text_y)==None:
            # 인물,장소명과 같은 고유명사를 임의의 알파벳으로 표시하는 것이 문제
            rand_num=random.randrange(1,101)

            if rand_num<30:
                text_x=text_y
                # print("동일하게 설정")
            else:
                rand_num=random.randrange(1,101)
                text_x = remove_space(text_y)
                if rand_num<100*args.noise_ratio:
                    text_x = add_character(text_x)
                    text_x = add_noise(text_x)

            x_list.append(text_x)
            y_list.append(text_y)
            id_list.append("-")

            cnt+=1

    f.close()

    
    df = pd.DataFrame((zip(id_list, x_list, y_list)), columns = ['id', 'x', 'label'])
    df.to_csv('/opt/ml/post_processing/post_dataset/'+args.df_save_name+'.csv')