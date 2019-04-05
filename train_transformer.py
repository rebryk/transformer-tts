import argparse
import os

import torch
import torch.nn as nn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data import LJDataset
from network import Model
from utils import adjust_learning_rate
from utils import collate_fn_transformer


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, help='Path to logs', default='logs/transformer')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint', default='checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_config()

    dataset = LJDataset(
        csv_file=os.path.join(config.data_path, 'train.csv'),
        root_dir=os.path.join(config.data_path, 'wavs'),
        cleaners=config.cleaners,
        sample_rate=config.sr,
        preemphasis=config.preemphasis,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mel=config.n_mel,
        max_db=config.max_db,
        ref_db=config.ref_db
    )

    model = nn.DataParallel(Model().cuda(), device_ids=config.device_ids)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    writer = SummaryWriter(log_dir=args.log_path)
    global_step = 0

    for epoch in range(config.epochs):
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn_transformer,
            drop_last=True,
            num_workers=16
        )

        progress_bar = tqdm(dataloader)
        for i, data in enumerate(progress_bar):
            progress_bar.set_description(f'Processing at epoch {epoch}')

            global_step += 1
            if global_step < 400_000:
                adjust_learning_rate(optimizer, config.lr, global_step)

            character, mel, mel_input, pos_text, pos_mel, _ = data
            stop_tokens = torch.abs(pos_mel.ne(0).type(torch.float) - 1)

            character = character.cuda()
            mel = mel.cuda()
            mel_input = mel_input.cuda()
            pos_text = pos_text.cuda()
            pos_mel = pos_mel.cuda()

            mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = model.forward(
                character,
                mel_input,
                pos_text,
                pos_mel
            )

            mel_loss = nn.L1Loss()(mel_pred, mel)
            post_mel_loss = nn.L1Loss()(postnet_pred, mel)

            loss = mel_loss + post_mel_loss

            writer.add_scalars(
                'training_loss',
                {
                    'mel_loss': mel_loss,
                    'post_mel_loss': post_mel_loss,
                },
                global_step
            )

            writer.add_scalars(
                'alphas',
                {
                    'encoder_alpha': model.module.encoder.alpha.data,
                    'decoder_alpha': model.module.decoder.alpha.data,
                },
                global_step
            )

            if global_step % config.image_step == 1:
                for i, prob in enumerate(attn_probs):
                    for j in range(4):
                        x = vutils.make_grid(prob[j * 16] * 255)
                        writer.add_image(f'Attention_{global_step}_0', x, i * 4 + j)

                for i, prob in enumerate(attns_enc):
                    for j in range(4):
                        x = vutils.make_grid(prob[j * 16] * 255)
                        writer.add_image(f'Attention_enc_{global_step}_0', x, i * 4 + j)

                for i, prob in enumerate(attns_dec):
                    for j in range(4):
                        x = vutils.make_grid(prob[j * 16] * 255)
                        writer.add_image(f'Attention_dec_{global_step}_0', x, i * 4 + j)

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
                    os.path.join(args.checkpoint_path, f'checkpoint_transformer_{global_step}.pth.tar')
                )
