# quartznet15x5 train 
# python ./openspeech_cli/hydra_train.py \
#     dataset=ksponspeech \
#     dataset.dataset_path=/opt/ml/input/kspon_dataset/KsponSpeech \
#     dataset.manifest_file_path=/opt/ml/project/transcripts.txt \
#     dataset.test_dataset_path=/opt/ml/input/kspon_dataset/KsponSpeech_eval \
#     dataset.test_manifest_dir=/opt/ml/input/kspon_dataset/KsponSpeech_scripts \
#     tokenizer=kspon_character \
#     model=quartznet15x5 \
#     audio=mfcc \
#     lr_scheduler=warmup_reduce_lr_on_plateau \
#     trainer=gpu \
#     criterion=ctc \
#     tokenizer.vocab_path=/opt/ml/project/aihub_character_vocabs.csv \
#     trainer.batch_size=8

# joint_ctc_conformer_lstm train 
# python ./openspeech_cli/hydra_train.py \
#     dataset=ksponspeech \
#     dataset.dataset_path=/opt/ml/input/kspon_dataset/KsponSpeech \
#     dataset.manifest_file_path=/opt/ml/project/transcripts.txt \
#     dataset.test_dataset_path=/opt/ml/input/kspon_dataset/KsponSpeech_eval \
#     dataset.test_manifest_dir=/opt/ml/input/kspon_dataset/KsponSpeech_scripts \
#     tokenizer=kspon_character \
#     model=joint_ctc_conformer_lstm \
#     audio=fbank \
#     lr_scheduler=warmup_reduce_lr_on_plateau \
#     trainer=gpu-fp16 \
#     criterion=cross_entropy \
#     tokenizer.vocab_path=/opt/ml/project/aihub_character_vocabs.csv \
#     trainer.batch_size=8

# conformer train 
# python ./openspeech_cli/hydra_train.py \
#     dataset=ksponspeech \
#     dataset.dataset_path=/opt/ml/input/kspon_dataset/KsponSpeech \
#     dataset.manifest_file_path=/opt/ml/project/transcripts.txt \
#     dataset.test_dataset_path=/opt/ml/input/kspon_dataset/KsponSpeech_eval \
#     dataset.test_manifest_dir=/opt/ml/input/kspon_dataset/KsponSpeech_scripts \
#     tokenizer=kspon_character \
#     model=conformer \
#     audio=fbank \
#     lr_scheduler=warmup_reduce_lr_on_plateau \
#     trainer=gpu-fp16 \
#     criterion=ctc \
#     tokenizer.vocab_path=/opt/ml/project/aihub_character_vocabs.csv \
#     trainer.batch_size=16

# deepspeech2 train 
# python ./openspeech_cli/hydra_train.py \
#     dataset=ksponspeech \
#     dataset.dataset_path=/opt/ml/input/kspon_dataset/KsponSpeech \
#     dataset.manifest_file_path=/opt/ml/project/transcripts.txt \
#     dataset.test_dataset_path=/opt/ml/input/kspon_dataset/KsponSpeech_eval \
#     dataset.test_manifest_dir=/opt/ml/input/kspon_dataset/KsponSpeech_scripts \
#     tokenizer=kspon_character \
#     model=deepspeech2 \
#     audio=melspectrogram \
#     lr_scheduler=warmup_reduce_lr_on_plateau \
#     trainer=gpu-fp16 \
#     criterion=ctc \
#     tokenizer.vocab_path=/opt/ml/project/aihub_character_vocabs.csv \
#     trainer.batch_size=128

# listen_attend_spell train
# python ./openspeech_cli/hydra_train.py \
#     dataset=ksponspeech \
#     dataset.dataset_path=/opt/ml/input/kspon_dataset/KsponSpeech \
#     dataset.manifest_file_path=/opt/ml/project/new_transcripts.txt \
#     dataset.test_dataset_path=/opt/ml/input/kspon_dataset/KsponSpeech_eval \
#     dataset.test_manifest_dir=/opt/ml/input/kspon_dataset/KsponSpeech_scripts \
#     tokenizer=kspon_character \
#     model=listen_attend_spell \
#     audio=melspectrogram \
#     lr_scheduler=warmup_reduce_lr_on_plateau \
#     trainer=gpu-fp16 \
#     criterion=cross_entropy \
#     tokenizer.vocab_path=/opt/ml/project/aihub_character_vocabs.csv

