import os

from torch.utils.data import DataLoader
from tqdm import tqdm

import config as config
from data import PrepareDataset

if __name__ == '__main__':
    dataset = PrepareDataset(
        csv_file=os.path.join(config.data_path, 'metadata.csv'),
        root_dir=os.path.join(config.data_path, 'wavs'),
        sample_rate=config.sr,
        preemphasis=config.preemphasis,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mel=config.n_mel,
        max_db=config.max_db,
        ref_db=config.ref_db
    )
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=10)

    for _ in tqdm(dataloader):
        pass
