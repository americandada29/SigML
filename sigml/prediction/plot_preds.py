import matplotlib.pyplot as plt
import pickle
import numpy as np
from sigml.utils.utils import build_data, plot_predictions

with open("dmft_input_atoms_sigs_efs.pkl","rb") as f:
    atoms, sig_texts_lines, efs = pickle.load(f)


sig_texts = []
for sig_text_lines in sig_texts_lines:
    sig_text = ''.join(sig_text_lines)
    sig_texts.append(sig_text)


dataset = build_data(atoms, sig_texts=sig_texts, efs=efs, radial_cutoff=3.0)

for o in range(5):
  plot_predictions(dataset, ind_restrict=list(range(0, 75)), orbital=o)



