from speechbrain.inference.speaker import SpeakerRecognition
import torchaudio
import glob
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./pretrained_ecapa")

# Different files from the same speaker
file1 = './LibriSpeech/dev-clean-2/1272/135031/1272-135031-0000.flac' # Same speaker
file2 = './LibriSpeech/dev-clean-2/1272/141231/1272-141231-0004.flac' # Same speaker
file3 = './LibriSpeech/dev-clean-2/1462/170142/1462-170142-0000.flac'  # Different speaker

# Test with 2 files from the same speaker
score, prediction = verification.verify_files(file1, file2)
print(score, prediction)

# Test with 2 files from  different speakers
score, prediction = verification.verify_files(file1, file3)
print(score, prediction)

utterances = glob.glob("./LibriSpeech/dev-clean-2/**/*.flac", recursive=True)

np.random.shuffle(utterances)
utterances = utterances[:20]

embeddings = []
labels = []
for u in utterances:
    tmp, fs = torchaudio.load(u)
    e = verification.encode_batch(tmp)
    embeddings.append(e[0, 0].numpy())
    spk_label = Path(u).parent.parent.stem
    labels.append(spk_label)

embeddings = np.array(embeddings)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(embeddings)

fig, ax = plt.subplots()
ax.scatter(principalComponents[:, 0], principalComponents[:, 1])

for i, spkid in enumerate(labels):
    ax.annotate(spkid, (principalComponents[i, 0], principalComponents[i, 1]))
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Speaker Embeddings")
plt.show()