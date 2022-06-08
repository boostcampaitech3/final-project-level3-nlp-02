import os
import time
import json
from tkinter import dialog
import yaml
import warnings
import argparse
from tqdm import tqdm

import wave
import librosa
import soundfile
from pydub import AudioSegment

from asr_inference import Speech2Text
from espnet2.utils import config_argparse

warnings.filterwarnings("ignore")

ASR_TRAIN_CONFIG = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/config.yaml"
ASR_MODEL_FILE = "/opt/ml/input/espnet-asr/.cache/espnet/e589f98042183b3a316257e170034e5e/exp/asr_train_asr_transformer2_ddp_raw_bpe/valid.acc.ave_10best.pth"
CONFIG_FILE = "/opt/ml/input/espnet-asr/conf/fast_decode_asr_ksponspeech.yaml"

def downsampling(audio_file, sampling_rate=16000):
    audio, rate = librosa.load(audio_file, sr=sampling_rate)
    return audio, rate

def main():
    start_time = time.time()
    print("JOB START!!!")

    with open(CONFIG_FILE) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    speech2text = Speech2Text(
        asr_train_config=ASR_TRAIN_CONFIG, 
        asr_model_file=ASR_MODEL_FILE, 
        device='cuda',
        dtype='float32',
        **config
        )

    dataset_path = "/opt/ml/input/korean_lecture/dataset"

    for D_dir in os.listdir(dataset_path):
        D_path = os.path.join(dataset_path, D_dir, "G02")

        audio_path_list = []
        text_path_list = []
        
        for S_dir in os.listdir(D_path):
            S_path = os.path.join(D_path, S_dir)

            json_path = os.path.join(S_path, S_dir) + ".json"

            assert os.path.isfile(json_path)


            with open(json_path, "r") as f:
                audio_info = json.load(f)
            
            dialogs = audio_info["dataSet"]["dialogs"]

            for dialog in dialogs:
                audioPath = dialog["audioPath"]
                textPath = dialog["textPath"]

                audio_path = os.path.join(S_path, audioPath.split("/")[-1])
                text_path = os.path.join(S_path, textPath.split("/")[-1])

                assert os.path.isfile(audio_path)
                assert os.path.isfile(text_path)

                audio_path_list.append(audio_path)
                text_path_list.append(text_path)

        
        for audio_path, text_path in tqdm(zip(audio_path_list, text_path_list)):
            audio, rate = downsampling(audio_path, sampling_rate=16000)

            train_text_path = "_train".join(os.path.splitext(text_path))

            if os.path.isfile(train_text_path):
                continue
            
            result = speech2text(audio)
            context = f"{result[0][0]}"


            with open(train_text_path, "w") as f:
                f.write(context)

    
    end_time = time.time()

    print(f"total time: {end_time - start_time}")
    print("JOB DONE!!!")
        
        

    # total_duration = 0

    # for file_name in tqdm(sorted(os.listdir(audio_path))):
    #     audio_file = os.path.join(audio_path, file_name)
    #     audio, rate = downsampling(audio_file, sampling_rate=16000)
    #     duration = len(audio)/rate

    #     result = speech2text(audio)

    #     total_duration += duration
        
    # end_time = time.time()

    # print(f"Total time: {time.time() - start_time}")
    # print("JOB DONE!!!")


if __name__ == "__main__":
    main()