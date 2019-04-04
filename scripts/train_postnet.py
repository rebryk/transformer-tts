import os

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PostDataset
from data import collate_fn_postnet
from network import *
from utils import adjust_learning_rate

if __name__ == '__main__':
    dataset = PostDataset(
        csv_file=os.path.join(config.data_path, 'train.csv'),
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

    model = nn.DataParallel(ModelPostNet().cuda(), device_ids=config.device_ids)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    writer = SummaryWriter(log_dir='logs/postnet')
    global_step = 0

    for epoch in range(config.epochs):
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn_postnet,
            drop_last=True,
            num_workers=8
        )

        progress_bar = tqdm(dataloader)
        for i, data in enumerate(progress_bar):
            progress_bar.set_description(f'Processing at epoch {epoch}')

            global_step += 1
            if global_step < 400_000:
                adjust_learning_rate(optimizer, config.lr, global_step)

            mel, mag = data

            mel = mel.cuda()
            mag = mag.cuda()

            mag_pred = model.forward(mel)

            loss = nn.L1Loss()(mag_pred, mag)

            writer.add_scalars(
                'training_loss',
                {'loss': loss},
                global_step
            )

            optimizer.zero_grad()

            # Calculate gradients
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.)

            # Update weights
            optimizer.step()

            if global_step % config.save_step == 0:
                torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    os.path.join(config.checkpoint_path, f'checkpoint_postnet_{global_step}.pth.tar')
                )
