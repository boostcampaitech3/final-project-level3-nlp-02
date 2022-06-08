from torch.utils.data import Dataset
import torch
import re

class RE_Dataset(Dataset):
    def __init__(self, input_text, target_text):
        self.inputs = input_text
        self.labels = target_text

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.inputs.items()}
        item['label_ids'] = torch.tensor(self.labels['input_ids'][idx])

        return item

    def __len__(self):
        return len(self.labels)

class Seq2SeqDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        self.dataset = csv_file
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x=self.tokenizer.encode('<sos>'+self.dataset.iloc[idx]['x']+'<eos>',add_special_tokens=False)
        y=self.tokenizer.encode(self.dataset.iloc[idx]['label']+'<eos>',add_special_tokens=False)
        
        return x,y

def processing(text):
    # print("전처리 전:",text)
    new_arr = []
    p = re.compile(r'(([(]([\w]|[\s]|[가-힣]|[-])+[)][/][(]([\w]|[\s]|[가-힣]|[-])+[)])|([(]([\w]|[\s]|[가-힣]|[-])+[)][(]([\w]|[\s]|[가-힣]|[-])+[)]))')

    p3 = re.compile(r'(([0-9]+[가-힣]+)|([0-9]+))')
    p4 = re.compile(r'([a-zA-z]+)')
    arr = re.split(p, text)

    cnt=0
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
    text=text.replace('n/',"")
    text=text.replace('b/',"")
    text=text.replace(',',"")
    text=text.replace('/',"")
    # print("전처리 후:",text)
    return text

def tokenized_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    input_text_list = []
    label_text_list = []
    for input_text, label_text in zip(dataset['x'],dataset['label']):
        # dataset에 nan값이 존재!!! 문제
        if isinstance(input_text,str) and isinstance(label_text,str):
            input_temp='<sos>'+input_text+'<eos>'
            label_temp=label_text+'<eos>'
            input_text_list.append(input_temp)
            label_text_list.append(label_temp)

    input_tokenized_sentences = tokenizer(
        input_text_list,
        return_tensors="pt",
        padding='max_length',
        # truncation=False,
        max_length=256,
        add_special_tokens=False,
        )
        
    label_tokenized_sentences = tokenizer(
        label_text_list,
        return_tensors="pt",
        padding='max_length', # max_length 만큼 패딩하겠다!
        # truncation=False,
        max_length=256,
        add_special_tokens=False,
        )
    
    return input_tokenized_sentences,label_tokenized_sentences