import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from text import text_to_sequence
from utils import get_spectrograms


class LJDataset(Dataset):
    """LJSpeech dataset."""

    def __init__(self,
                 csv_file,
                 root_dir,
                 cleaners,
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
        :param cleaners:
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
        self.cleaners = cleaners
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

        text = self.landmarks_frame.ix[idx, 1]
        text = np.asarray(text_to_sequence(text, [self.cleaners]), dtype=np.int32)

        wav_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0]) + '.wav'

        if self.use_preprocessed:
            mel = np.load(wav_name[:-4] + '.pt.npy')
        else:
            mel, _ = get_spectrograms(
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

        mel_input = np.concatenate([np.zeros([1, self.n_mel], np.float32), mel[:-1, :]], axis=0)
        text_length = len(text)
        pos_text = np.arange(1, text_length + 1)
        pos_mel = np.arange(1, mel.shape[0] + 1)

        sample = {
            'text': text,
            'mel': mel,
            'text_length': text_length,
            'mel_input': mel_input,
            'pos_mel': pos_mel,
            'pos_text': pos_text
        }

        if self.use_cache:
            self._cache[idx] = sample

        return sample
