# The same model as cpu_shakespeare.py, trained on protein sequences.
# Prepare the data first: python data/protein_char/prepare.py
# Run from the nanogpt folder: python train.py config/cpu_protein.py

exec(open('config/cpu_shakespeare.py').read())

out_dir = 'out-protein-char'
dataset = 'protein_char'
