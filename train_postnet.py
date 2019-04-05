import argparse
import os

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PostDataset
from network import *
from utils import adjust_learning_rate
from utils import collate_fn_postnet


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, help='Path to logs', default='logs/postnet')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint', default='checkpoint')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=64)
    parser.add_argument('--n_gpu', type=int, help='Number of GPUs', default=1)
    parser.add_argument('--n_worker', type=int, help='Number of workers', default=16)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_config()

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

    model = nn.DataParallel(Model().cuda(), device_ids=list(range(args.n_gpu)))

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    writer = SummaryWriter(log_dir=args.log_path)
    global_step = 0

    for epoch in range(config.epochs):
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_postnet,
            drop_last=True,
            num_workers=args.n_worker
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
                if not os.path.exists(args.checkpoint_path):
                    os.makedirs(args.checkpoint_path)

                torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    os.path.join(args.checkpoint_path, f'checkpoint_postnet_{global_step}.pth.tar')
                )
