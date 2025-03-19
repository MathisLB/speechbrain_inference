from speechbrain.inference.speaker import SpeakerRecognition
import torchaudio
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./pretrained_ecapa")
signal, fs = torchaudio.load('./LibriSpeech/dev-clean-2/1272/135031/1272-135031-0003.flac')
embedding = verification.encode_batch(signal)
embedding.shape
print(embedding)