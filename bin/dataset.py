import os
import time
import pytz
from tqdm import tqdm, trange
from datetime import datetime

import torch
import numpy as np
from torch.utils.data.dataset import Dataset

import librosa
from scipy.io import wavfile
from pydub import AudioSegment
from utils import audio2wav, split_on_silence
# from pydub.silence import split_on_silence, detect_nonsilent


class SplitOnSilenceDataset(Dataset):
    """
    pytorch Dataset class
    
    음성 데이터 경로를 입력으로 주면, 침묵을 기준으로 나뉘어진 음성 파일을 동일한 경로에 디렉토리로 저장합니다.
    디렉토리에 저장된 음성 파일 순으로 __getitem__을 return 합니다.
    """
    def __init__(self, audio_path, step_ms=60000, overlap_ms=10000,
                dtype=torch.float32, sampling_rate=16000,
                min_silence_len=500, silence_thresh=-40, keep_silence=200):
        """
        Parameters:
            audio_path (str): 나눌 음성 파일 원본 경로

            step_ms (in ms): 주어진 음성 파일을 몇 ms 간격으로 나눌지 결정합니다.

            overlap_ms (in ms): 나눠진 음성 파일을 얼마나 겹치게 할지 결정합니다. 

            dtype (torch.dtype): 데이터 타입을 결정합니다. (! fp16의 경우 overflow가 발생하는 이슈가 있습니다.)
            
            sampling_rate (int): 음성 데이터의 sampling_rate를 결정합니다.

            min_silence_len (in ms): minimum length of a silence to be used for
                            a split. default: 1000ms

            silence_thresh (in dBFS): anything quieter than this will be
                            considered silence. default: -16dBFS

            keep_silence (in ms or True/False): leave some silence at the beginning
                            and end of the chunks. Keeps the sound from sounding like it
                            is abruptly cut off.
                            When the length of the silence is less than the keep_silence duration
                            it is split evenly between the preceding and following non-silent
                            segments.
                            If True is specified, all the silence is kept, if False none is kept.
                            default: 100ms
        """

        super().__init__()
        
        self.audio_path = audio_path
        self.dtype = dtype
        self.sampling_rate = sampling_rate
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh
        self.keep_silence = keep_silence
        self.timeline_ms_list = []

        self.step_ms = step_ms
        self.overlap_ms = overlap_ms

        tz = pytz.timezone('Asia/Seoul')
        cur_time = datetime.now(tz)
        HMS = "00_00_00" #cur_time.strftime("%H_%M_%S")

        self.path, self.extension = os.path.splitext(self.audio_path)
        self.chunk_path = f"{self.path}_chunk_{HMS}"

        self.setup()


    def setup(self):
        """
        음성 파일을 침묵을 기준으로 나눠준 후 동일한 디렉토리에 저장합니다.

        만약 음성 파일이 wav 형식이 아닌 경우, wav파일 형식 그리고 지정한 smapling rate로 저장합니다.
        만약 음성 파일이 지정한 sampling rate이 아닌 경우 덮어씌어 저장합니다.

        이미 디렉토리가 있는 경우 이 과정을 생략합니다.
        """
        if self.extension != '.wav':
            self.audio_path = audio2wav(self.audio_path, sampling_rate=self.sampling_rate)
    
        sound_file = AudioSegment.from_wav(self.audio_path)

        if sound_file.frame_rate != self.sampling_rate:
            self.audio_path = self.audio2wav(self.audio_path, sampling_rate=self.sampling_rate)
            sound_file = AudioSegment.from_wav(self.audio_path)

        if not os.path.exists(self.chunk_path):
            os.mkdir(self.chunk_path)

            _audio_chunks_len = None

            print("Splitting...")

            for idx, i in enumerate(trange(0, len(sound_file), self.step_ms)):
                start_t = max(0, i - self.overlap_ms)
                end_t = min(i + self.step_ms + self.overlap_ms, len(sound_file))

                chunk = sound_file[start_t:end_t]

                if i == 0:
                    audio_chunk_start = 0
                else:
                    audio_chunk_start = 1

                audio_chunks, timelines = split_on_silence(chunk, min_silence_len=self.min_silence_len, silence_thresh=self.silence_thresh, keep_silence=self.keep_silence)

                audio_chunks = audio_chunks[audio_chunk_start:-1]
                timelines = timelines[audio_chunk_start:-1]
                
                audio_chunks_len = list(map(len, audio_chunks))

                if not _audio_chunks_len:
                    _audio_chunks_len = list(map(len, audio_chunks))
                
                else:
                    try:
                        cut_line = audio_chunks_len.index(_audio_chunks_len[-1])
                    except:
                        cut_line = -1
                    audio_chunks = audio_chunks[cut_line + 1:]
                    timelines = timelines[cut_line + 1:] + (self.step_ms) * idx - self.overlap_ms

                    _audio_chunks_len = list(map(len, audio_chunks))

                for j, (chunk, timeline) in enumerate(zip(audio_chunks, timelines)):
                    output_file = os.path.join(self.chunk_path, f"{timeline:0>8}.wav")
                    chunk.export(output_file, format="wav")
                
            print(f"Audio file is splited on silence, they are saved in {self.chunk_path}.")
            
        else:
            print(f"{self.chunk_path} already exists, so create dataset with files in the path.")

        self.audio_files = [os.path.join(self.chunk_path, audio_file) for audio_file in sorted(os.listdir(self.chunk_path))]
        self.timeline_ms_list = [int(os.path.splitext(os.path.split(audio_file)[1])[0]) for audio_file in self.audio_files]


    def __getitem__(self, index):
        file_name = self.audio_files[index]

        audio, rate = librosa.load(file_name, sr=self.sampling_rate)
        data = torch.tensor(audio, dtype=self.dtype)

        timeline = self.timeline_ms_list[index] / 1000

        assert rate == self.sampling_rate, (rate, self.sampling_rate)

        return data, timeline


    def __len__(self):
        return len(self.audio_files)

