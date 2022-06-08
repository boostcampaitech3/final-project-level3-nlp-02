# train 방법

```
python bert_train.py --n_layers 1 --save_name BERT_LSTM_v3_H768 --decoder_hidden 768 --dataset_path {데이터셋 csv 파일 경로} --save_name BERT_LSTM_H768 --dropout 0.2 --batch 64 --n_epochs 10
```

# inference 방법

```
python bert_inference.py --pretrain_model {모델 bin파일이 저장된 경로} --n_layers 1 --decoder_hidden 768 --dataset_path {데이터셋 csv 파일 경로} --batch 64 --n_epochs 10
```
