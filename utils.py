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
