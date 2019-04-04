import os

import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from tqdm import tqdm

from network import *
from preprocess import get_dataset, DataLoader, collate_fn_transformer


def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def main():
    dataset = get_dataset()
    global_step = 0
    
    model = nn.DataParallel(Model().cuda())

    model.train()
    optimizer = t.optim.Adam(model.parameters(), lr=hp.lr)

    pos_weight = t.FloatTensor([5.]).cuda()
    writer = SummaryWriter()
    
    for epoch in range(hp.epochs):
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=hp.batch_size,
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
                adjust_learning_rate(optimizer, global_step)
                
            character, mel, mel_input, pos_text, pos_mel, _ = data
            stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1)
            
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

            if global_step % hp.image_step == 1:
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
                        writer.add_image(f'Attention_dec_{global_step}_0', x, i*4+j)
                
            optimizer.zero_grad()

            # Calculate gradients
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            
            # Update weights
            optimizer.step()

            if global_step % hp.save_step == 0:
                t.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    os.path.join(hp.checkpoint_path, f'checkpoint_transformer_{global_step}.pth.tar')
                )


if __name__ == '__main__':
    main()
