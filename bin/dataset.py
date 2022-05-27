import os
import time
import pytz
from datetime import datetime
from requests import delete

import torch
import numpy as np
from torch.utils.data.dataset import Dataset

from scipy.io import wavfile
from pydub import AudioSegment
from pydub.silence import split_on_silence


class SplitOnSilenceDataset(Dataset):
    def __init__(self, audio_path, dtype=torch.float32, sampling_rate=16000, min_silence_len=500, silence_thresh=-40):
        super().__init__()
        
        self.audio_path = audio_path
        self.dtype = dtype
        self.sampling_rate = sampling_rate
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh

        tz = pytz.timezone('Asia/Seoul')
        cur_time = datetime.now(tz)
        HMS = "00_00_00" #cur_time.strftime("%H_%M_%S")

        self.path, self.extension = os.path.splitext(self.audio_path)
        self.chunk_path = f"{self.path}_chunk_{HMS}"

        self.setup()


    def setup(self):
        if self.extension != '.wav':
            self.audio_path = self.audio2wav(self.audio_path, sampling_rate=self.sampling_rate)
    
        sound_file = AudioSegment.from_wav(self.audio_path)

        if sound_file.frame_rate != self.sampling_rate:
            self.audio_path = self.audio2wav(self.audio_path, sampling_rate=self.sampling_rate)
            sound_file = AudioSegment.from_wav(self.audio_path)

        if not os.path.exists(self.chunk_path):
            print(f"Audio file is splited on silence, they are saved in {self.chunk_path}.")
            os.mkdir(self.chunk_path)

            audio_chunks = split_on_silence(sound_file, min_silence_len=self.min_silence_len, silence_thresh=self.silence_thresh)

            for i, chunk in enumerate(audio_chunks):
                output_file = os.path.join(self.chunk_path, f"chunk{i:05}.wav")
                chunk.export(output_file, format="wav")

        else:
            print(f"{self.chunk_path} already exists, so create dataset with files in the path.")

        self.audio_files = [os.path.join(self.chunk_path, f) for f in sorted(os.listdir(self.chunk_path))]


    def __getitem__(self, index):
        file_name = self.audio_files[index]
        rate, audio = wavfile.read(file_name)
        assert rate == self.sampling_rate, rate
        data = torch.tensor(audio/32767, dtype=self.dtype)
        return data


    def __len__(self):
        return len(self.audio_files)


    def audio2wav(self, audio_path, sampling_rate):
        path, extension = os.path.splitext(audio_path)
        extension = extension[1:]

        audio_data = AudioSegment.from_file(audio_path, extension)
        audio_data = audio_data.set_frame_rate(sampling_rate)
        audio_data = audio_data.set_channels(1)
        output_path = f"{path}.wav"
        audio_data.export(output_path, format="wav", delete=True)

        if audio_path == output_path:
            print(f"The sampling rate of input audio file is not 16000, so it is converted and overwritten.")    
        else:
            print(f"The input audio file is not a wav file, so it is converted to 'wav' and saved in {output_path}.")

        return output_path

