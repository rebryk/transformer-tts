n_mel = 80
n_fft = 2048
sr = 22050
preemphasis = 0.97
frame_shift = 0.0125  # seconds
frame_length = 0.05  # seconds
hop_length = int(sr * frame_shift)  # samples.
win_length = int(sr * frame_length)  # samples.
power = 1.2  # exponent for amplifying the predicted magnitude
min_level_db = -100
ref_level_db = 20
hidden_size = 256
embedding_size = 512
max_db = 100
ref_db = 20

n_iter = 60
# power = 1.5
outputs_per_step = 1

epochs = 1_000_000
lr = 0.001
save_step = 2000
image_step = 500
batch_size = 32

cleaners = 'english_cleaners'
data_path = '/data/LJSpeech'
sample_path = 'samples'

device_ids = [0]
