# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: 03-nanogpt
#     language: python
#     name: 03-nanogpt
# ---

# %% [markdown]
# # A tiny GPT, trained on Shakespeare and on proteins
#
# The large language models (LLMs) behind today's chat assistants are, at
# their core, the same kind of program you will train here: a transformer
# that reads a stretch of text and predicts what comes next. Theirs have
# hundreds of billions of parameters and read trillions of words; ours has
# 0.8 million parameters and reads one million characters of Shakespeare,
# then the protein sequences from module 01.
#
# The code is [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej
# Karpathy, a deliberately short implementation that can reproduce GPT-2. It
# lives in the `nanogpt` folder, unmodified apart from a few added files.
# Two files are worth reading in full once you have finished this lesson:
# `model.py` (the network, about 330 lines) and `train.py` (the training
# loop, about 340).
#
# The plan:
#
# 1. what a language model predicts, and how we measure it;
# 2. turning text into numbers;
# 3. the attention mechanism;
# 4. training on Shakespeare, and sampling from the result;
# 5. the same model on protein sequences;
# 6. what changes at scale, and what to be careful about.
#
# Training takes a few minutes per data set on a laptop CPU.

# %%
import inspect
import os
import pickle
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "nanogpt")
import model as nanogpt_model  # nanogpt/model.py

os.makedirs("out", exist_ok=True)


def run(script, *args):
    """Run a nanoGPT script inside the nanogpt folder, print its output and return it."""
    process = subprocess.Popen(
        [sys.executable, script, *args],
        cwd="nanogpt",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    lines = []
    for line in process.stdout:
        print(line, end="")
        lines.append(line)
    if process.wait() != 0:
        raise RuntimeError(f"{script} failed")
    return "".join(lines)


# %% [markdown]
# ## 1. Predicting the next token
#
# A language model reads a sequence of tokens (here, characters) and outputs
# a probability for every possible next token. That is all it does. Writing a
# paragraph is just this step repeated: pick a token from the predicted
# distribution, append it, predict again.
#
# Training adjusts the weights so that the token that actually came next gets
# a high probability. The loss is the cross-entropy: the average of
# `-ln(probability given to the correct next token)`, measured in nats. Two
# reference points make it readable:
#
# - Guessing uniformly among `V` possible tokens gives `ln(V)`. For the 65
#   characters in our Shakespeare text that is 4.17.
# - Knowing only how common each character is, and ignoring context
#   completely, does better. We will compute this "composition baseline" for
#   each data set below.
#
# Anything a model gains below the composition baseline, it has learnt from
# context. You will also see perplexity, `exp(loss)`: roughly, the number of
# equally likely options the model is still choosing between at each step.
#
# The chapter on n-gram language models in Jurafsky and Martin's free
# textbook, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/),
# covers language models, cross-entropy and perplexity from the beginning.

# %%
def composition_baseline(folder):
    """Cross-entropy on the validation set of predicting each token from its overall frequency alone."""
    train = np.fromfile(f"{folder}/train.bin", dtype=np.uint16)
    val = np.fromfile(f"{folder}/val.bin", dtype=np.uint16)
    p = np.bincount(train, minlength=val.max() + 1) / len(train)
    return -np.log(p[val]).mean()


# %% [markdown]
# ## 2. Tokenisation
#
# Networks work with numbers, so text is first split into tokens and each
# token mapped to an integer. The Shakespeare preparation script uses the
# simplest scheme: one token per character.

# %%
run("data/shakespeare_char/prepare.py")

with open("nanogpt/data/shakespeare_char/meta.pkl", "rb") as f:
    meta = pickle.load(f)
print(repr("".join(meta["itos"][i] for i in range(meta["vocab_size"]))))
print("To be, or not to be ->", [meta["stoi"][c] for c in "To be, or not to be"])

# %% [markdown]
# Production models use subword tokens instead, built by byte-pair encoding
# (BPE), which starts from bytes and repeatedly merges the most frequent
# adjacent pair. GPT-2 has 50,257 tokens, most of them whole words or common
# word pieces. Longer tokens mean shorter sequences for the same text, so the
# model sees more context for the same cost. On the other hand, the model
# never sees the letters inside a token, which is why LLMs can be oddly bad
# at spelling and counting characters.
#
# For proteins, the 20 amino acids are a natural character vocabulary, and
# it is what we will use. Most protein language models, including ESM-2 and
# ProGen, do the same; ProtGPT2 is an exception and uses BPE.
#
# Further reading: Sennrich et al., [Neural machine translation of rare
# words with subword units](https://arxiv.org/abs/1508.07909) (the paper
# that brought BPE to NLP), and Karpathy's
# [minbpe](https://github.com/karpathy/minbpe), a short, readable BPE
# implementation.
#
# ## 3. Attention
#
# The transformer, introduced by Vaswani et al. in
# [Attention is all you need](https://arxiv.org/abs/1706.03762), builds the
# representation of each position by mixing in information from other
# positions. Each position emits a query ("what am I looking for?") and a
# key ("what do I contain?"); the match between one position's query and
# another's key sets how much of that position's value is mixed in.
#
# In a model that generates text, a position may only look backwards:
# otherwise it could read the answer it is supposed to predict. This is the
# causal mask. Below, random attention scores for five positions are masked
# and turned into weights: each row sums to one, and everything above the
# diagonal (the future) is zero.

# %%
torch.manual_seed(0)
scores = torch.randn(5, 5)
mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))
weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
print(weights.numpy().round(2))

# %% [markdown]
# Here is the same thing in nanoGPT. `CausalSelfAttention.forward` does it
# for several heads at once, either with PyTorch's built-in
# `scaled_dot_product_attention` (`is_causal=True` applies the mask) or with
# the explicit version in the `else` branch, which matches the cell above.

# %%
print(inspect.getsource(nanogpt_model.CausalSelfAttention.forward))

# %% [markdown]
# Two consequences are worth knowing. Every position attends to every
# earlier position, so the cost grows with the square of the context length;
# that is why our context (`block_size`) is only 64 characters. And attention
# by itself ignores order, so position has to be added explicitly. nanoGPT
# learns one vector per position, as GPT-2 did; most current models use
# rotary position embeddings instead ([Su et al.](https://arxiv.org/abs/2104.09864)).
#
# For a gentler walk through the architecture, see
# [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
# and [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).
#
# ## 4. Training on Shakespeare
#
# `config/cpu_shakespeare.py` holds the settings: 4 transformer layers, 4
# attention heads, 128-dimensional embeddings, 2,000 steps of 12 sequences of
# 64 characters. Every 250 steps it estimates the loss on training and
# validation data.
#
# `train.py` prints a `FutureWarning` about `torch.cuda.amp.GradScaler`
# first. It is harmless: nanoGPT predates a renaming in PyTorch, and the
# scaler is switched off on CPU anyway.

# %%
run("train.py", "config/cpu_shakespeare.py")

# %% [markdown]
# The loss starts near `ln(65) = 4.17`, as expected from a network that knows
# nothing, and should end near 1.9.
#
# ### Sampling
#
# Generation draws each next character from the predicted distribution.
# The temperature rescales that distribution first: below 1 it sharpens it
# towards the most likely characters, above 1 it flattens it. `top_k` keeps
# only the `k` most likely candidates. Compare a low and a high temperature.

# %%
for temperature in ["0.5", "1.5"]:
    print(f"=== temperature {temperature} ===")
    run("sample.py", "--out_dir=out-shakespeare-char", "--device=cpu",
        "--num_samples=1", "--max_new_tokens=400", f"--temperature={temperature}")

# %% [markdown]
# At low temperature the text is more regular and more repetitive; at high
# temperature it becomes more varied and less coherent. The text looks like
# Shakespeare at a glance: speaker names in capitals, line breaks, archaic
# words. It does not mean anything. The model has learnt spelling, some
# vocabulary and the layout of a play, the things that 64 characters of
# context can reveal.
#
# The choice of sampling method matters more than you might think. See
# Holtzman et al., [The curious case of neural text
# degeneration](https://arxiv.org/abs/1904.09751), for why always choosing
# the most likely token produces dull, repetitive text, and for nucleus
# (top-p) sampling.
#
# ## 5. Proteins
#
# Now the same model and settings, trained on the module 01 proteins: one
# sequence per line, 21 tokens (20 amino acids and the newline that ends each
# protein). The validation proteins come from different UniRef50 clusters
# from the training proteins. With a random split, close relatives of
# validation proteins would sit in the training data, and the validation
# loss would partly measure memorisation.

# %%
run("data/protein_char/prepare.py")
run("train.py", "config/cpu_protein.py")

# %% [markdown]
# Now compare both models with the two reference points from section 1.

# %%
rows = [
    ("Shakespeare", "shakespeare_char", "out-shakespeare-char"),
    ("Proteins", "protein_char", "out-protein-char"),
]
print(f"{'':<12}{'uniform':>10}{'composition':>14}{'model':>10}")
for name, data, out in rows:
    with open(f"nanogpt/data/{data}/meta.pkl", "rb") as f:
        vocab_size = pickle.load(f)["vocab_size"]
    model_loss = float(torch.load(f"nanogpt/{out}/ckpt.pt", map_location="cpu")["best_val_loss"])
    baseline = composition_baseline(f"nanogpt/data/{data}")
    print(f"{name:<12}{np.log(vocab_size):>10.2f}{baseline:>14.2f}{model_loss:>10.2f}")

# %% [markdown]
# On Shakespeare the model ends far below the composition baseline: context
# tells you a lot about the next letter of an English word. On proteins it
# barely improves on composition. This is not a bug. Proteins look nearly
# random at the level of neighbouring residues: what determines which amino
# acid sits where is the fold and the function, which depend on contacts
# between residues that may be hundreds of positions apart, and on evolution
# across many related sequences. A 64-residue window and 4 million training
# tokens cannot capture that.
#
# ### Sampling proteins
#
# Each generated protein starts after a newline and ends at the next one.

# %%
output = run("sample.py", "--out_dir=out-protein-char", "--device=cpu",
             "--num_samples=1", "--max_new_tokens=3000", "--temperature=1.0")
generated = [line.strip() for line in output.splitlines()[1:]
             if line.strip() and set(line.strip()) <= set("ACDEFGHIKLMNPQRSTVWY")]
generated = generated[:-1]  # the last one was cut off by max_new_tokens

with open("nanogpt/data/protein_char/meta.pkl", "rb") as f:
    itos = pickle.load(f)["itos"]
real = "".join(itos[i] for i in np.fromfile("nanogpt/data/protein_char/val.bin", dtype=np.uint16)).split()

print(f"{len(generated)} complete generated proteins")
print(f"start with M: generated {np.mean([s[0] == 'M' for s in generated]):.0%}, real {np.mean([s[0] == 'M' for s in real]):.0%}")
print(f"median length: generated {int(np.median([len(s) for s in generated]))}, real {int(np.median([len(s) for s in real]))}")

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
gen_freq = np.array([sum(s.count(a) for s in generated) for a in amino_acids], dtype=float)
real_freq = np.array([sum(s.count(a) for s in real) for a in amino_acids], dtype=float)
fig, ax = plt.subplots(figsize=(9, 3))
x = np.arange(20)
ax.bar(x - 0.2, real_freq / real_freq.sum(), width=0.4, label="real (validation)")
ax.bar(x + 0.2, gen_freq / gen_freq.sum(), width=0.4, label="generated")
ax.set_xticks(x, list(amino_acids))
ax.set_ylabel("fraction of residues")
ax.legend()
fig.tight_layout()
fig.savefig("out/protein_composition.png", dpi=120)

# %% [markdown]
# The model has learnt what it could see: most generated proteins start with
# methionine, and the overall mix of amino acids is about right. Look closely
# at the sequences and you will probably find stretches dominated by one or
# two residues, such as serine, proline, glutamine or glutamate. Such
# low-complexity regions exist in real proteins, usually as disordered
# segments, and they are the easiest thing for a short-context model to
# imitate. Length is another giveaway: the model only ever sees 64 residues,
# so it has no way of knowing how long the protein it is writing has become.
#
# Nothing here tells us whether a generated sequence would fold or do
# anything. Output that looks plausible is not evidence that it is correct.
# That holds for proteins and for an LLM's confident answer to a question
# alike, and it is the most important practical lesson about these models.
#
# Real protein language models differ from ours mainly in scale, data and
# evaluation. Each paper below is free to read:
#
# - [ProtGPT2](https://www.nature.com/articles/s41467-022-32007-7)
#   (Ferruz et al., 2022): a GPT-2-sized model trained on UniRef50; its
#   generated sequences were checked with structure prediction.
# - [ProGen](https://arxiv.org/abs/2004.03497) (Madani et al.): generation
#   steered by tags for protein family and function, later tested in the lab.
# - [ESM-2 and ESMFold](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3)
#   (Lin et al.): a masked language model, the BERT family rather than GPT,
#   whose internal representations are good enough to predict 3D structure.
#
# ## 6. What changes at scale, and what to watch for
#
# **Scale.** Kaplan et al. ([2020](https://arxiv.org/abs/2001.08361)) found
# that loss falls predictably as parameters, data and compute grow together,
# and Hoffmann et al. ([2022](https://arxiv.org/abs/2203.15556), the
# "Chinchilla" paper) showed that for a fixed compute budget, many models
# had been too big for the data they were trained on. Their rule of thumb is
# about 20 training tokens per parameter. Our Shakespeare model saw about
# 1.5 million tokens for 0.8 million parameters.
#
# **Base models and assistants.** What we trained is a base model: it
# continues text. A chat assistant starts as a base model and is then
# trained further on examples of instructions and good answers, and on human
# preferences between answers. Ouyang et al.,
# [Training language models to follow instructions with human
# feedback](https://arxiv.org/abs/2203.02155), describes the method behind
# the first widely used assistants.
#
# **Over-fitting and memorisation.** A model with enough capacity can
# memorise its training text. Watch the gap between training and validation
# loss. Large models can reproduce rare passages from their training data
# word for word, which matters when those data are private.
#
# **Leakage.** If validation data resemble training data too closely,
# validation loss is optimistic. With text, that means duplicated documents;
# with proteins, homologues. It is the same problem as in module 01.
#
# **Compute.** Training cost grows with parameters × tokens, and attention
# cost with the square of the context length. That is why models of the kind
# people use daily are trained on thousands of GPUs, and why ours fits on a
# laptop.
#
# **Reproducibility.** Results depend on the random seed and, slightly, on
# the hardware and number of threads. nanoGPT fixes its seeds (1337), so a
# rerun on the same machine should give nearly the same numbers.
#
# **Data provenance.** A model can only reflect what it was trained on, and
# training data come with licences and biases. Ours are clean: Shakespeare is
# in the public domain and Swiss-Prot is released under CC BY 4.0. For
# large models, where the data come from is often the most important and
# least documented part.
#
# ## Further reading
#
# All free to read:
#
# - Andrej Karpathy, [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html),
#   a video course whose "Let's build GPT" episode writes the model in this
#   lesson line by line.
# - Zhang, Lipton, Li and Smola, [Dive into Deep Learning](https://d2l.ai),
#   chapters on attention and transformers, with runnable code.
# - Simon Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/),
#   the chapter on transformers, with clear figures.
# - Jurafsky and Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/),
#   the standard NLP textbook.
# - Hugging Face, [LLM Course](https://huggingface.co/learn/llm-course), for
#   using pretrained models in practice.
# - 3Blue1Brown, [Neural networks](https://www.3blue1brown.com/topics/neural-networks),
#   visual explanations of backpropagation, transformers and attention.
#
# ## Things to try
#
# 1. Train Shakespeare for longer (`--max_iters=5000 --lr_decay_iters=5000`)
#    and with a bigger model (`--n_layer=6 --n_embd=192`). Add these after
#    the config file in the `run("train.py", ...)` call. When does the gap
#    between training and validation loss open up?
# 2. Set `block_size` to 16 and to 128 for the protein model. Does more
#    context help, and what does it cost in time?
# 3. Sample proteins at temperatures 0.5 and 1.5. How do composition and
#    low-complexity stretches change?
# 4. Give the Shakespeare model a prompt: `--start="ROMEO:"`.
# 5. Read `model.py` from top to bottom. Find where the token and position
#    embeddings are added, and where the final layer turns each position's
#    vector into 65 scores.
