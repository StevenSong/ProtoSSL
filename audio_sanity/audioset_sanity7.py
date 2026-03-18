import torchaudio
from pathlib import Path

wav_dir = Path("/gpfs/data/bbj-lab/data/audioset/audioset/audioset_train/train_wav")

files = list(wav_dir.glob("*.wav"))[:10]

for f in files:
    info = torchaudio.info(str(f))
    print(f.name, info.sample_rate, info.num_frames)