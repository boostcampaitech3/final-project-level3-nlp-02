import os
import itertools
from requests import delete

import torch
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent



def audio2wav(audio_path, sampling_rate=16000):
    """
    음성 파일을 wav 파일로 변환하고 변환된 파일 경로를 return 합니다.
    """
    path, extension = os.path.splitext(audio_path)
    extension = extension[1:]

    audio_data = AudioSegment.from_file(audio_path, extension)
    audio_data = audio_data.set_frame_rate(sampling_rate)
    audio_data = audio_data.set_channels(1)
    output_path = f"{path}_new.wav"
    audio_data.export(output_path, format="wav")

    if audio_path == output_path:
        print(f"The sampling rate of input audio file is not {self.sampling_rate}, so it is converted and overwritten.")    
    else:
        print(f"The input audio file is not a wav file, so it is converted to 'wav' and saved in {output_path}.")

    return output_path


def split_on_silence(audio_segment, min_silence_len=1000, silence_thresh=-16, keep_silence=100,
                     seek_step=1):
    """
    Returns list of audio segments from splitting audio_segment on silent sections
    audio_segment - original pydub.AudioSegment() object
    min_silence_len - (in ms) minimum length of a silence to be used for
        a split. default: 1000ms
    silence_thresh - (in dBFS) anything quieter than this will be
        considered silence. default: -16dBFS
    keep_silence - (in ms or True/False) leave some silence at the beginning
        and end of the chunks. Keeps the sound from sounding like it
        is abruptly cut off.
        When the length of the silence is less than the keep_silence duration
        it is split evenly between the preceding and following non-silent
        segments.
        If True is specified, all the silence is kept, if False none is kept.
        default: 100ms
    seek_step - step size for interating over the segment in ms
    """

    # from the itertools documentation
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    if isinstance(keep_silence, bool):
        keep_silence = len(audio_segment) if keep_silence else 0

    output_ranges = [
        [ start - keep_silence, end + keep_silence ]
        for (start,end)
            in detect_nonsilent(audio_segment, min_silence_len, silence_thresh, seek_step)
    ]

    for range_i, range_ii in pairwise(output_ranges):
        last_end = range_i[1]
        next_start = range_ii[0]
        if next_start < last_end:
            range_i[1] = (last_end+next_start)//2
            range_ii[0] = range_i[1]

    return ([
        audio_segment[ max(start,0) : min(end,len(audio_segment)) ]
        for start,end in output_ranges
    ], np.array([start + keep_silence for start,end in output_ranges]))


def collate_fn(batch):
    speech_dict = dict()
    speech_tensor = torch.tensor([])
    timelines = []

    audio_max_len = 0
    for data, timeline in batch:
        audio_max_len = max(audio_max_len, len(data))

    for data, timeline in batch:
        zero_tensor = torch.zeros((1, audio_max_len - len(data)))
        data = torch.unsqueeze(data, 0)
        tensor_with_pad = torch.cat((data, zero_tensor), dim=1)
        speech_tensor = torch.cat((speech_tensor, tensor_with_pad), dim=0)
        timelines.append(timeline)
    
    speech_dict['speech'] = speech_tensor
    speech_dict['timeline'] = timelines

    return speech_dict