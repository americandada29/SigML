import numpy as np
import matplotlib.pyplot as plt
import pickle


# data = np.loadtxt("FPS/temp_sig.dat")

# iws = data[:,0]
# sr = data[:,1]
# si = data[:,2]
# fig, axs = plt.subplots(2)
# axs[0].plot(iws, sr, marker="o")
# axs[1].plot(iws, si, marker="o")
# plt.show()


with open("atoms_predsigs.pkl","rb") as f:
    atoms, predsigs = pickle.load(f)

f = open("test.txt", "w")
for l in predsigs[0]:
    f.write(l)
f.close()

