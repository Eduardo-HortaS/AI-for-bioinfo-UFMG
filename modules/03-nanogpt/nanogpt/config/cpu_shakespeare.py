# Character-level Shakespeare on a laptop CPU. These are the reduced settings
# from the "I only have a MacBook" section of the nanoGPT README, written out
# as a config file. Expect a validation loss of about 1.88 after 2000 steps.
# Run from the nanogpt folder: python train.py config/cpu_shakespeare.py

out_dir = 'out-shakespeare-char'
dataset = 'shakespeare_char'

eval_interval = 250
eval_iters = 20
log_interval = 100
always_save_checkpoint = False

# A small model: about 0.8M parameters.
n_layer = 4
n_head = 4
n_embd = 128
block_size = 64  # context length in characters
dropout = 0.0

batch_size = 12
gradient_accumulation_steps = 1
learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

device = 'cpu'
compile = False
# train.py otherwise picks float16 when no GPU is present.
dtype = 'float32'
