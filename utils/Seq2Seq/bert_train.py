import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoModel, AutoTokenizer
import argparse
from functools import partial

import numpy as np

import random
import math
import time
from torch.optim import AdamW

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas
from tqdm import tqdm
from model import Decoder,BERT_LSTM
from dataset import Seq2SeqDataset,tokenized_dataset,RE_Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, AutoConfig
import wandb

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def train(model, train_loader, valid_loader, optimizer, scheduler, criterion, clip, save_name, forcing):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    best_valid_loss = float('inf')
    
    epoch_loss = 0
    i=0
    eval_step=20
    
    for batch in tqdm(train_loader):
        model.train()
        optimizer.zero_grad()
        
        output = model(batch, forcing)
        output = output.to(device)

        # vocab size
        output_dim = output.shape[-1]

        trg = batch['label_ids']
        trg = trg.to(device)

        output = output.view(-1, output_dim)
        trg = trg.view(-1)
        print("output shape",output.shape," | trg shape",trg.shape)

        loss = criterion(output, trg)
        epoch_loss += loss.item()
        loss.backward()
        
        optimizer.step()                           
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        wandb.log({"train_loss": loss})

        if i%eval_step==0:
            valid_loss = evaluate(model, valid_loader, criterion)
            scheduler.step(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f"/opt/ml/post_processing/model/{save_name}.bin")
                # torch.save(model.state_dict(), f'/opt/ml/post_processing/model/{save_name}.pt')

            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            wandb.log({"val_loss": valid_loss})
        
        i+=1
        
    return epoch_loss / len(train_loader)

def evaluate(model, iterator, criterion):
    model.eval()
    
    epoch_loss = 0

    with torch.no_grad():
        for batch in tqdm(iterator):
            output = model(batch, 0)
            output = output.to(device)

            # vocab size
            output_dim = output.shape[-1]

            trg = batch['label_ids']
            trg = trg.to(device)

            output = output.view(-1, output_dim)
            trg = trg.view(-1)
            # print("output shape",output.shape," | trg shape",trg.shape)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def collate_fn(batch, eos_token_id=0):
    # batch가 1 sample씩 들어있음
    max_len=0
    EOS_TOKEN = eos_token_id
    new_batch={'input_ids':[],'attention_mask':[],'token_type_ids':[],'label_ids':[]}

    # max_len 구하기
    for element in batch:
        input_id = element['input_ids']
        label_id = element['label_ids']

        input_cur_len = int((input_id==EOS_TOKEN).nonzero(as_tuple=True)[0][-1])
        label_cur_len = int((label_id==EOS_TOKEN).nonzero(as_tuple=True)[0][-1])
        cur_len = max(input_cur_len,label_cur_len)
        max_len = max(max_len,cur_len)

    # 최대 길이 기준으로 자르기
    for element in batch:
        for key,value in element.items():
            # index 문제 여기??
            new_batch[key].append(value[:max_len+1].numpy().tolist())

    # 파이썬 리스트 -> 텐서 변환
    for key,value in new_batch.items():
        new_batch[key]=torch.tensor(new_batch[key])

    return new_batch

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--input_dim', type=int, default=32000, help='random seed (default: 42)')
    parser.add_argument('--output_dim', type=int, default=32000, help='number of epochs to train (default: 1)')
    parser.add_argument('--enc_emb_dim', type=int, default=256, help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--dec_emb_dim', type=int, default=256, help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--hid_dim", type=int, default=512, help='resize size for image when training')
    parser.add_argument('--n_layers', type=int, default=2, help='input batch size for training (default: 64)')
    parser.add_argument('--enc_dropout', type=float, default=0.5, help='input batch size for validing (default: 1000)')
    parser.add_argument('--dec_dropout', type=float, default=0.5, help='model type (default: BaseModel)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--batch', type=int, default=32, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--n_epochs', type=int, default=10, help='criterion type (default: cross_entropy)')
    parser.add_argument('--clip', type=int, default=1, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--dataset_path', type=str, default='/opt/ml/post_processing/post_dataset/dataset.csv', help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--save_name', type=str, default='LSTM', help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--pretrain_model', type=str, default='', help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--forcing', type=float, default=1.0, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--model_name', type=str, default="klue/bert-base", help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--dropout', type=float, default=0.2, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--decoder_hidden', type=int, default=512, help='learning rate scheduler deacy step (default: 20)')

    args = parser.parse_args()
    
    wandb.init(project="post_processing", entity='salt-bread',name=args.save_name)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    num_added_token = tokenizer.add_special_tokens({"additional_special_tokens":['<sos>','<eos>']})   

    args.input_dim = len(tokenizer.vocab)
    args.output_dim = len(tokenizer.vocab)
    print(args)

    df=pandas.read_csv(args.dataset_path)

    trainset,testset = train_test_split(df, test_size=0.1, shuffle=True, random_state=34)

    input_tokenized_train, label_tokenized_train = tokenized_dataset(trainset, tokenizer)
    input_tokenized_valid, label_tokenized_valid = tokenized_dataset(testset, tokenizer)

    RE_train_dataset = RE_Dataset(input_tokenized_train, label_tokenized_train)
    RE_valid_dataset = RE_Dataset(input_tokenized_valid, label_tokenized_valid)

    # Q. 저번 대회랑 똑같이 불러오는데 왜 안되는건지,,??
    trainset=[]
    for i in range(len(input_tokenized_train['input_ids'])):
        # print(input_tokenized_train['input_ids'].shape)
        trainset.append(RE_train_dataset[i])

    valset=[]
    for i in range(len(input_tokenized_valid['input_ids'])):
        valset.append(RE_valid_dataset[i])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(args.model_name)

    BERT_encoder = AutoModel.from_pretrained(args.model_name, config=model_config)
    BERT_encoder.resize_token_embeddings(num_added_token + tokenizer.vocab_size)

    if args.model_name=='klue/bert-base':
        hidden_size=768
    elif args.model_name=='klue/roberta-large':
        hidden_size=1024

    model=BERT_LSTM(BERT_encoder, tokenizer, hidden_size=hidden_size, decoder_hidden=args.decoder_hidden, drop_rate=args.dropout, lstm_layer=args.n_layers, batch_size=args.batch, trg_vocab_size=len(tokenizer.vocab))

    ######### 모델 불러오기
    model.to(device)

    eos_token_id=tokenizer.encode('<eos>',add_special_tokens=False)[0]
    # drop_last 문제?? 그럼 해결해야 할듯
    train_loader = DataLoader(trainset, batch_size=args.batch, shuffle=True, drop_last = False, collate_fn=partial(collate_fn, eos_token_id=eos_token_id))
    valid_loader = DataLoader(valset, batch_size=args.batch, shuffle=False, drop_last = True, collate_fn=partial(collate_fn, eos_token_id=eos_token_id))

    optimizer = AdamW(model.parameters(), lr=args.lr)

    TRG_PAD_IDX = tokenizer.encode('<sos>')
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX[1])
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=5, verbose=True)

    f = open(f"/opt/ml/post_processing/model/{args.save_name}.txt", 'w')
    f.write(f'model_name:{args.model_name} | hidden_size:{hidden_size} | decoder_hidden:{args.decoder_hidden} | n_layers:{args.n_layers}')
    f.close()

    for epoch in range(args.n_epochs):
        
        start_time = time.time()
        
        train_loss = train(model, train_loader, valid_loader, optimizer, scheduler, criterion, args.clip, args.save_name, args.forcing)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        
    
