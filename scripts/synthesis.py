import argparse
from collections import OrderedDict

import numpy as np
import torch
from scipy.io.wavfile import write
from tqdm import tqdm

import config
from network import ModelPostNet, Model
from text import text_to_sequence
from utils import spectrogram2wav


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step1', type=int, help='Global step to restore checkpoint', default=172000)
    parser.add_argument('--restore_step2', type=int, help='Global step to restore checkpoint', default=100000)
    parser.add_argument('--max_len', type=int, help='Global step to restore checkpoint', default=400)
    return parser.parse_args()


def load_checkpoint(step, model_name='transformer'):
    state_dict = torch.load(f'./checkpoint/checkpoint_{model_name}_{step}.pth.tar')

    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict


def synthesis(text, args):
    model = Model()
    model_post = ModelPostNet()

    model.load_state_dict(load_checkpoint(args.restore_step1, 'transformer'))
    model_post.load_state_dict(load_checkpoint(args.restore_step2, 'postnet'))

    text = np.asarray(text_to_sequence(text, [config.cleaners]))
    text = torch.LongTensor(text).unsqueeze(0)
    text = text.cuda()
    mel_input = torch.zeros([1, 1, 80]).cuda()
    pos_text = torch.arange(1, text.size(1) + 1).unsqueeze(0)
    pos_text = pos_text.cuda()

    model = model.cuda()
    model_post = model_post.cuda()
    model.train(False)
    model_post.train(False)

    progress_bar = tqdm(range(args.max_len))
    with torch.no_grad():
        for _ in progress_bar:
            pos_mel = torch.arange(1, mel_input.size(1) + 1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = model.forward(text, mel_input, pos_text, pos_mel)
            mel_input = torch.cat([mel_input, postnet_pred[:, -1:, :]], dim=1)

        mag_pred = model_post.forward(postnet_pred)

    wav = spectrogram2wav(
        mag=mag_pred.squeeze(0).cpu().numpy(),
        power=config.power,
        preemphasis=config.preemphasis,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        max_db=config.max_db,
        ref_db=config.ref_db,
        n_iter=config.n_iter
    )
    write(config.sample_path + '/test.wav', config.sr, wav)


if __name__ == '__main__':
    args = get_config()
    synthesis('Transformer model is so fast!', args)
