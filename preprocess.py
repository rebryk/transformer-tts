import collections

import numpy as np
import torch as t


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

        return t.LongTensor(text), t.FloatTensor(mel), t.FloatTensor(mel_input), \
               t.LongTensor(pos_text), t.LongTensor(pos_mel), t.LongTensor(text_length)

    raise TypeError(f'Batch must contain tensors, numbers, dicts or lists; found {type(batch[0])}')


def collate_fn_postnet(batch):
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):
        mel = [d['mel'] for d in batch]
        mag = [d['mag'] for d in batch]

        # PAD sequences with largest length of the batch
        mel = _pad_mel(mel)
        mag = _pad_mel(mag)

        return t.FloatTensor(mel), t.FloatTensor(mag)

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
