import os
import time
import yaml
import warnings
import argparse
from tqdm import tqdm

import wave
import librosa
import soundfile

import torch
import numpy as np
from pydub import AudioSegment
from torch.utils.data.dataloader import DataLoader

from utils import collate_fn
from dataset import SplitOnSilenceDataset

from asr_inference import Speech2Text


warnings.filterwarnings("ignore")

ASR_TRAIN_CONFIG = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/config.yaml"
ASR_MODEL_FILE = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/valid.acc.ave_10best.pth"
CONFIG_FILE = "/opt/ml/input/espnet-asr/conf/fast_decode_asr_ksponspeech.yaml"

BATCH_SIZE = 8
AUDIO_PATH = "/opt/ml/input/sample_dataset/ksw.wav"
OUTPUT_TXT = "/opt/ml/input/result.txt"



def main():
    start_time = time.time()
    print("JOB START!!!")

    with open(CONFIG_FILE) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(OUTPUT_TXT, "a") as f:
        f.write(f"ASR_TRAIN_CONFIG: {ASR_TRAIN_CONFIG}\nASR_MODEL_FILE: {ASR_MODEL_FILE}\nCONFIG_FILE: {CONFIG_FILE}\n")

    speech2text = Speech2Text(
        asr_train_config=ASR_TRAIN_CONFIG, 
        asr_model_file=ASR_MODEL_FILE, 
        device='cuda',
        dtype='float32',
        **config, 
        )

    dataset = SplitOnSilenceDataset(AUDIO_PATH, sampling_rate=16000, min_silence_len=500, silence_thresh=-40)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for batch in loader:
        timelines = batch.pop("timeline")
        batch = {k: v for k, v in batch.items() if not k.endswith("_lengths")}
        results = speech2text(**batch)

        end_time = time.time()

        for result, timeline in zip(results, timelines):
            with open(OUTPUT_TXT, "a") as f:
                f.write(f"{int(timeline)//60:02}:{int(timeline)%60:02}  {result[0]}\n")

        with open(OUTPUT_TXT, "a") as f:
            f.write(f"Total time: {end_time - start_time}\n\n")

        print(f"Total time: {time.time() - start_time}")
    print("JOB DONE!!!")

if __name__ == "__main__":
    main()