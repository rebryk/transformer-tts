import collections
import copy

import librosa
import numpy as np
import torch
from scipy import signal


def get_spectrograms(path,
                     sr,
                     preemphasis,
                     n_fft,
                     hop_length,
                     win_length,
                     n_mel,
                     max_db,
                     ref_db):
    """
    Parse the wave file.

    :param path: the full path of a sound file
    :param sr: sample rate
    :param preemphasis:
    :param n_fft:
    :param hop_length:
    :param win_length:
    :param n_mel:
    :param max_db:
    :param ref_db:
    :return: returns normalized mel-spectrogram and linear spectrogram.
    """

    # Loading sound file
    y, sr = librosa.load(path, sr=sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # STFT
    linear = librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )

    # Magnitude spectrogram: (1 + n_fft // 2, T)
    mag = np.abs(linear)

    # Mel spectrogram: (n_mel, 1 + n_fft // 2)
    mel_basis = librosa.filters.mel(sr, n_fft, n_mel)
    mel = np.dot(mel_basis, mag)  # (n_mel, t)

    # Convert to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # Normalize
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mel)
    mag = mag.T.astype(np.float32)  # (T, 1 + n_fft // 2)

    return mel, mag


def spectrogram2wav(mag,
                    power,
                    preemphasis,
                    n_fft,
                    hop_length,
                    win_length,
                    max_db,
                    ref_db,
                    n_iter):
    """
    Generate wave file from linear magnitude spectrogram.

    :param mag: a numpy array of (T, 1 + n_fft // 2)
    :param power:
    :param preemphasis:
    :param n_fft:
    :param hop_length:
    :param win_length:
    :param max_db:
    :param ref_db:
    :param n_iter:
    :return wav
    """

    # Transpose
    mag = mag.T

    # De-noramlize
    mag = (np.clip(mag, 0, 1) * max_db) - max_db + ref_db

    # Convert to amplitude
    mag = np.power(10.0, mag * 0.05)

    # Wav reconstruction
    wav = griffin_lim(mag ** power, n_iter, n_fft, hop_length, win_length)

    # De-preemphasis
    wav = signal.lfilter([1], [1, - preemphasis], wav)

    # Trimming
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram, n_iter, n_fft, hop_length, win_length):
    """Applies Griffin-Lim's raw."""

    X_best = copy.deepcopy(spectrogram)
    for i in range(n_iter):
        X_t = invert_spectrogram(X_best, hop_length, win_length)
        est = librosa.stft(X_t, n_fft, hop_length, win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best, hop_length, win_length)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram, hop_length, win_length):
    """
    Applies inverse fft.

    :param spectrogram: spectogram (1 + n_fft // 2, t)
    :return: signal
    """
    return librosa.istft(spectrogram, hop_length, win_length, window='hann')


def get_positional_table(d_pos_vec, n_position=1024):
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i + 1

    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """Sinusoid position encoding table."""

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # Zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def guided_attention(N, T, g=0.2):
    """Guided attention. Refer to page 3 on the paper."""

    W = np.zeros((N, T), dtype=np.float32)
    for n_pos in range(W.shape[0]):
        for t_pos in range(W.shape[1]):
            W[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(T) - n_pos / float(N)) ** 2 / (2 * g * g))

    return W


def adjust_learning_rate(optimizer, lr, step_num, warmup_step=4000):
    lr = lr * warmup_step ** 0.5 * min(step_num * warmup_step ** -1.5, step_num ** -0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def collate_fn_transformer(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):
        text = [d['text'] for d in batch]
        mel = [d['mel'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_text = [d['pos_text'] for d in batch]

        text = [i for i, _ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        mel = [i for i, _ in sorted(zip(mel, text_length), key=lambda x: x[1], reverse=True)]
        mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
        pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        pos_mel = [i for i, _ in sorted(zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
        text_length = sorted(text_length, reverse=True)
        # PAD sequences with largest length of the batch
        text = _prepare_data(text).astype(np.int32)
        mel = _pad_mel(mel)
        mel_input = _pad_mel(mel_input)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)

        return torch.LongTensor(text), torch.FloatTensor(mel), torch.FloatTensor(mel_input), \
               torch.LongTensor(pos_text), torch.LongTensor(pos_mel), torch.LongTensor(text_length)

    raise TypeError(f'Batch must contain tensors, numbers, dicts or lists; found {type(batch[0])}')


def collate_fn_postnet(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):
        mel = [d['mel'] for d in batch]
        mag = [d['mag'] for d in batch]

        # PAD sequences with largest length of the batch
        mel = _pad_mel(mel)
        mag = _pad_mel(mag)

        return torch.FloatTensor(mel), torch.FloatTensor(mag)

    raise TypeError(f'Batch must contain tensors, numbers, dicts or lists; found {type(batch[0])}')


def _pad_data(x, length):
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=0)


def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])


def _pad_mel(inputs):
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0, max_len - mel_len], [0, 0]], mode='constant', constant_values=0)

    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])
