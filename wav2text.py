import whisper
import torch
import ssl

ssl._create_default_https_context = ssl._create_unverified_context # ERROR to except : urlopen certification

# Transformer 학습 모델 기반 wav2text 모델 : whisper 샘플코드


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# model_path = r'./model/base.pt'
model = whisper.load_model('base',device=device)

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)

