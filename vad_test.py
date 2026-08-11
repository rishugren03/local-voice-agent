# vad_test.py
import torch
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio

model = load_silero_vad()
wav = read_audio('chunk.wav', sampling_rate=16000)  # reuse a chunk.wav from earlier if you have one
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
print(speech_timestamps)