import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils import get_spectrograms


class PostDataset(Dataset):
    """LJSpeech dataset."""

    def __init__(self,
                 csv_file,
                 root_dir,
                 sample_rate=None,
                 preemphasis=None,
                 n_fft=None,
                 hop_length=None,
                 win_length=None,
                 n_mel=None,
                 max_db=None,
                 ref_db=None,
                 use_preprocessed=False,
                 use_cache=True):
        """

        :param csv_file: path to the csv file with annotations
        :param root_dir: directory with all the wavs
        :param sample_rate:
        :param preemphasis:
        :param n_fft:
        :param hop_length:
        :param win_length:
        :param n_mel:
        :param max_db:
        :param ref_db:
        :param use_preprocessed: whether to load preprocessed data
        :param use_cache: whether to use cache
        """

        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.preemphasis = preemphasis
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel = n_mel
        self.max_db = max_db
        self.ref_db = ref_db
        self.use_preprocessed = use_preprocessed
        self.use_cache = use_cache
        self._cache = {}

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if self.use_cache and idx in self._cache:
            return self._cache[idx]

        wav_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0]) + '.wav'

        if self.use_preprocessed:
            mel = np.load(wav_name[:-4] + '.pt.npy')
            mag = np.load(wav_name[:-4] + '.mag.npy')
        else:
            mel, mag = get_spectrograms(
                path=wav_name,
                sr=self.sample_rate,
                preemphasis=self.preemphasis,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mel=self.n_mel,
                max_db=self.max_db,
                ref_db=self.ref_db
            )

        sample = {'mel': mel, 'mag': mag}

        if self.use_cache:
            self._cache[idx] = sample

        return sample
