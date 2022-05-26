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

from dataset import SplitOnSilenceDataset

from asr_inference import Speech2Text


warnings.filterwarnings("ignore")

ASR_TRAIN_CONFIG = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/config.yaml"
ASR_MODEL_FILE = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/valid.acc.ave_10best.pth"
CONFIG_FILE = "/opt/ml/input/espnet-asr/conf/fast_decode_asr_ksponspeech.yaml"

BATCH_SIZE = 32
AUDIO_PATH = "/opt/ml/input/sample_dataset/AAYN.wav"
OUTPUT_TXT = "/opt/ml/input/result.txt"


def collate_fn(batch):
    speech_dict = dict()
    speech_tensor = torch.tensor([])

    audio_max_len = 0
    for data in batch:
        audio_max_len = max(audio_max_len, len(data))

    for data in batch:
        zero_tensor = torch.zeros((1, audio_max_len - len(data)))
        data = torch.unsqueeze(data, 0)
        tensor_with_pad = torch.cat((data, zero_tensor), dim=1)
        speech_tensor = torch.cat((speech_tensor, tensor_with_pad), dim=0)
    
    speech_dict['speech'] = speech_tensor

    return speech_dict


def downsampling(audio_file, sampling_rate=16000):
    audio, rate = librosa.load(audio_file, sr=sampling_rate)
    return audio, rate


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
        **config
        )

    dataset = SplitOnSilenceDataset(AUDIO_PATH)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for batch in loader:
        batch = {k: v for k, v in batch.items() if not k.endswith("_lengths")}
        results = speech2text(**batch)

        end_time = time.time()

        for result in results:
            with open(OUTPUT_TXT, "a") as f:
                f.write(result[0]+"\n")

        with open(OUTPUT_TXT, "a") as f:
            f.write(f"Total time: {end_time - start_time}\n\n")

        print(f"Total time: {time.time() - start_time}")
    print("JOB DONE!!!")


    # audio_path = "/opt/ml/input/chunks"
    # audio_file = "/opt/ml/input/espnet-asr/evalset/ksponspeech/wavs/KsponSpeech_E00001.wav"

    # total_duration = 0

    # for file_name in tqdm(sorted(os.listdir(audio_path))):
    #     audio_file = os.path.join(audio_path, file_name)
    #     audio, rate = downsampling(audio_file, sampling_rate=16000)
    #     duration = len(audio)/rate

    #     result = speech2text(audio)

    #     with open(FILE_NAME, "a") as f:
    #         f.write(f"{int(total_duration)//60:02}:{int(total_duration)%60:02}  {result[0][0]}\n")

    #     total_duration += duration
        
    # end_time = time.time()

    # with open(FILE_NAME, "a") as f:
    #     f.write(f"Total time: {end_time - start_time}\n\n\n")

    # print(f"Total time: {time.time() - start_time}")
    # print("JOB DONE!!!")


if __name__ == "__main__":
    main()