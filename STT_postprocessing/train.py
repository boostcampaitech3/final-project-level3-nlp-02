import math
import numpy as np
import pandas as pd
import random
import re
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

import warnings
warnings.filterwarnings("ignore")
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=40):  # 데이터셋의 전처리를 해주는 부분
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.mask = MASK
        self.tokenizer = koGPT2_TOKENIZER

    def __len__(self):  # chatbotdata 의 길이를 리턴한다.
        return len(self._data)

    def __getitem__(self, idx):  # 로드한 챗봇 데이터를 차례차례 DataLoader로 넘겨주는 메서드
        turn = self._data.iloc[idx]
        q = turn["x"]  # 질문을 가져온다.
        #q = re.sub(r"([?.!,])", r" ", q)  # 구둣점들을 제거한다.

        a = turn["label"]  # 답변을 가져온다.
        #a = re.sub(r"([?.!,])", r" ", a)  # 구둣점들을 제거한다.

        q_toked = self.tokenizer.tokenize(self.q_token + str(q) + self.sent_token)
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + str(a) + self.eos)
        a_len = len(a_toked)

        #질문의 길이가 최대길이보다 크면
        if q_len > self.max_len:
            a_len = self.max_len - q_len        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:       #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_toked = q_toked[-(int(self.max_len / 2)) :]   #질문길이를 최대길이의 반으로 
                q_len = len(q_toked)
                a_len = self.max_len - q_len              #답변의 길이를 최대길이 - 질문길이
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        #질문의 길이 + 답변의 길이가 최대길이보다 크면
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:       #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_toked = q_toked[-(int(self.max_len / 2)) :]   #질문길이를 최대길이의 반으로 
                q_len = len(q_toked)
                a_len = self.max_len - q_len              #답변의 길이를 최대길이 - 질문길이
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        # 답변 labels = [mask, mask, ...., mask, ..., <bos>,..답변.. <eos>, <pad>....]
        labels = [self.mask,] * q_len + a_toked[1:]

        # mask = 질문길이 0 + 답변길이 1 + 나머지 0
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        # 답변 labels을 index 로 만든다.
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        # 최대길이만큼 PADDING
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        # 질문 + 답변을 index 로 만든다.    
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        # 최대길이만큼 PADDING
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        #질문+답변, 마스크, 답변
        return (token_ids, np.array(mask), labels_ids)
    
def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

def train(args, koGPT2_TOKENIZER, model, train_dataloader, test_dataloader, epoch, learning_rate, criterion, optimizer, scheduler, device, save_dir):
    
    model.to(device)
    global_val_loss = int(1e9)
    
    for epoch in range(epoch):
        model.train()
        print(f'EPOCH : {epoch}')
        batch_idx=1
        loss_sum = 0
        for samples in tqdm(train_dataloader):
            optimizer.zero_grad()
            token_ids, mask, label = samples
            token_ids = token_ids.to(device)
            mask = mask.to(device)
            label = label.to(device)

            out = model(token_ids)
            out = out.logits      #Returns a new tensor with the logit of the elements of input
            mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
            mask_out = torch.where(mask_3d == 1, out, args.Sneg * torch.ones_like(out))
            loss = criterion(mask_out.transpose(2, 1), label)
            # 평균 loss 만들기 avg_loss[0] / avg_loss[1] <- loss 정규화
            avg_loss = loss.sum() / mask.sum()
            loss_sum+=avg_loss
            avg_loss.backward()
            optimizer.step()

            if batch_idx%20 == 0:
                val_loss=validation(args, model, test_dataloader, criterion, device)
                scheduler.step(val_loss)

                if global_val_loss>val_loss:
                    global_val_loss=val_loss
                    print("[model save]")
                    koGPT2_TOKENIZER.save_pretrained('/opt/ml/post_processing/GPT/output/'+save_dir)
                    model.save_pretrained('/opt/ml/post_processing/GPT/output/'+save_dir)
                    
            print(f'idx : {batch_idx}, train avg_loss : {loss_sum/batch_idx}')

            if args.wandb=='True':
                wandb.log({"train_loss": loss_sum/batch_idx})
                
            batch_idx+=1
            
def validation(args, model, test_dataloader, criterion, device):
    
    model.to(device)
    global_loss = 0
    model.eval()
    
    print(f'Start validation..')
    with torch.no_grad():
        for samples in tqdm(test_dataloader):
            token_ids, mask, label = samples
            token_ids = token_ids.to(device)
            mask = mask.to(device)
            label = label.to(device)

            out = model(token_ids)
            out = out.logits      #Returns a new tensor with the logit of the elements of input
            mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
            mask_out = torch.where(mask_3d == 1, out, args.Sneg * torch.ones_like(out))
            loss = criterion(mask_out.transpose(2, 1), label)
            # 평균 loss 만들기 avg_loss[0] / avg_loss[1] <- loss 정규화
            avg_loss = loss.sum() / mask.sum()
            global_loss+=avg_loss

        print(f'validation avg_loss : {global_loss/len(test_dataloader)}')
        if args.wandb=='True':
            wandb.log({"validation_loss": global_loss/len(test_dataloader)})
        
    return global_loss/len(test_dataloader)
                

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (default: 1e-3)')
    parser.add_argument('--batch', type=int, default=64, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--n_epochs', type=int, default=10, help='criterion type (default: cross_entropy)')
    parser.add_argument('--dataset_path', type=str, default='/opt/ml/post_processing/post_dataset/0531dataset_v2.csv', help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--model_name', type=str, default='skt/kogpt2-base-v2', help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--report_name', type=str, default='', help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--Sneg', type=float, default=-1e18, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--save_dir', type=str, default="output", help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--wandb', type=str, default="True", help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--max_len', type=int, default=50, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--inference', type=str, default="False", help='learning rate scheduler deacy step (default: 20)')

    args = parser.parse_args()
    
    if args.wandb=='True':
        wandb.init(project="post_processing", entity='salt-bread',name=args.report_name)
    
    print(args)
    
    Q_TKN = "<usr>"
    A_TKN = "<sys>"
    BOS = '</s>'
    EOS = '</s>'
    MASK = '<unused0>'
    SENT = '<unused1>'
    PAD = '<pad>'

    koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(args.model_name,
                bos_token=BOS, eos_token=EOS, unk_token='<unk>',
                pad_token=PAD, mask_token=MASK) 
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    
    dataset = pd.read_csv(args.dataset_path)
    trainset, testset = train_test_split(dataset, test_size=0.1, shuffle=True, random_state=34)
    
    train_set = ChatbotDataset(trainset, args.max_len)
    test_set = ChatbotDataset(testset, args.max_len)
    
    train_dataloader = DataLoader(train_set, batch_size=args.batch, num_workers=2, shuffle=True, collate_fn=collate_batch,)
    test_dataloader = DataLoader(test_set, batch_size=args.batch, num_workers=2, shuffle=False, collate_fn=collate_batch,)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=4, verbose=True)
    
    
    Path('/opt/ml/post_processing/GPT/output/'+args.save_dir).mkdir(parents=True, exist_ok=True)
    if args.inference=='False':
        train(args, koGPT2_TOKENIZER, model, train_dataloader, test_dataloader, args.n_epochs, args.lr, criterion, optimizer, scheduler, device, args.save_dir)
    

    if args.inference=='True':
        sent = '0'

        model.to(device)
        with torch.no_grad():
            while 1:
                q = input("입력 > ").strip()
                if q == "quit":
                    break
                a = ""
                while 1:
                    # print(Q_TKN + q + SENT + sent + A_TKN + a)
                    input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + sent + A_TKN + a)).unsqueeze(dim=0)
                    input_ids = input_ids.to(device)
                    pred = model(input_ids)
                    pred = pred.logits
                    if pred.shape[1]>args.max_len:
                        print("error!!")
                        break
                    gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]
                    if gen == EOS:
                        break
                    a += gen.replace("▁", " ")
                print("후처리 결과 > {}".format(a.strip()))


from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

def inference(model_path,text):

    Q_TKN = "<usr>"
    A_TKN = "<sys>"
    BOS = '</s>'
    EOS = '</s>'
    MASK = '<unused0>'
    SENT = '<unused1>'
    PAD = '<pad>'

    koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(model_path,
                bos_token=BOS, eos_token=EOS, unk_token='<unk>',
                pad_token=PAD, mask_token=MASK) 
    model = GPT2LMHeadModel.from_pretrained(model_path)

    sent = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    processed_text=""

    with torch.no_grad():
        if text!="":
            while 1:
                print(Q_TKN + q + SENT + sent + A_TKN + a)
                input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + sent + A_TKN + a)).unsqueeze(dim=0)
                print(input_ids)
                input_ids = input_ids.to(device)
                pred = model(input_ids)
                pred = pred.logits
                print(pred.shape)
                gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]
                if gen == EOS:
                    break
                processed_text += gen.replace("▁", " ")
    
    return processed_text