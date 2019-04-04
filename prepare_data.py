import os

from torch.utils.data import DataLoader
from tqdm import tqdm

import config as hp
from data import PrepareDataset

if __name__ == '__main__':
    dataset = PrepareDataset(
        csv_file=os.path.join(hp.data_path, 'metadata.csv'),
        root_dir=os.path.join(hp.data_path, 'wavs'),
        sample_rate=hp.sr,
        preemphasis=hp.preemphasis,
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        n_mel=hp.n_mel,
        max_db=hp.max_db,
        ref_db=hp.ref_db
    )
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=10)

    for _ in tqdm(dataloader):
        pass
