"""
Prepare protein sequences for character-level language modelling.

Reads the FASTA file from module 01 (or one given on the command line), with
headers of the form ">ACCESSION class CLUSTER_ID". Each sequence becomes one
line, so the vocabulary is the 20 amino acids plus the newline that ends every
protein. Train and validation are split by cluster: related sequences never
end up on both sides, which would make validation loss look better than it is.

Writes train.bin, val.bin and meta.pkl in the same format as shakespeare_char.
"""
import gzip
import os
import pickle
import random
import sys
from collections import defaultdict

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
default = os.path.join(here, '..', '..', '..', '..', '01-svm-localisation', 'data', 'localisation.fasta.gz')
fasta = sys.argv[1] if len(sys.argv) > 1 else default

by_cluster = defaultdict(list)
with gzip.open(fasta, 'rt') as f:
    for header, seq in zip(f, f):
        cluster = header.split()[2]
        by_cluster[cluster].append(seq.strip())

clusters = sorted(by_cluster)
random.Random(1337).shuffle(clusters)
n_val = len(clusters) // 10
splits = {
    'val': [s for c in clusters[:n_val] for s in by_cluster[c]],
    'train': [s for c in clusters[n_val:] for s in by_cluster[c]],
}

chars = sorted(set(''.join(splits['train'] + splits['val'])) | {'\n'})
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
print(f"vocab size: {len(chars)}")

for name, seqs in splits.items():
    text = ''.join(s + '\n' for s in seqs)
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    ids.tofile(os.path.join(here, f'{name}.bin'))
    print(f"{name} has {len(seqs):,} sequences, {len(ids):,} tokens")

with open(os.path.join(here, 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': len(chars), 'itos': itos, 'stoi': stoi}, f)
