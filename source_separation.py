import torchaudio
import IPython.display as ipd
from speechbrain.inference.separation import SepformerSeparation

s1, fs = torchaudio.load('./LibriSpeech/dev-clean-2/1272/135031/1272-135031-0003.flac')
s2, fs = torchaudio.load('./LibriSpeech/dev-clean-2/1462/170142/1462-170142-0001.flac')

# we resample because we will use a model trained on 8KHz data.
resampler = torchaudio.transforms.Resample(fs, 8000)
s1 = resampler(s1)
s2 = resampler(s2)
fs= 8000

min_len = min(s1.shape[-1], s2.shape[-1])
s1 = s1[:, :min_len]
s2 = s2[:, :min_len]
mix = s1 + s2

ipd.Audio(mix[0], rate=fs)

separator = SepformerSeparation.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir="./pretrained_sepformer")

est_sources = separator.separate_batch(mix)

est_sources = est_sources[0] # strip batch dimension

ipd.Audio(est_sources[0], rate=fs) # first source
ipd.Audio(est_sources[1], rate=fs) # second source