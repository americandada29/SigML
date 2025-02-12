import pickle
import numpy as np
from nequip.ase.nequip_calculator import nequip_calculator
import matplotlib.pyplot as plt
from ase.io import read, write

#with open("cdse_testing.pkl","rb") as f:
#  atoms, energies, forces, stresses = pickle.load(f)

atoms = read('test.extxyz', index=":", format='extxyz')


calc = nequip_calculator("deployed_model/efsmodel.pth")

act_energies = []
pred_energies = []
act_forces = []
pred_forces = []
act_stress = []
pred_stress = []
Eerrors = []
Ferrors = []
Serrors = []


for i in range(len(atoms)):
  testatom = atoms[i].copy()
  testatom.calc = calc
  testenergy = atoms[i].get_potential_energy()/len(testatom)
  predenergy = testatom.get_potential_energy()/len(testatom)

  act_energies.append(testenergy)
  pred_energies.append(predenergy)
  print(i, predenergy, testenergy)
  Eerrors.append(np.abs(predenergy-testenergy))

  #predforce = testatom.get_forces()
  #actforce = atoms[i].get_forces()
  #for j in range(len(predforce)):
  #  act_forces.append(np.linalg.norm(actforce[j]))
  #  pred_forces.append(np.linalg.norm(predforce[j]))
  #  Ferrors.append(np.abs(np.linalg.norm(actforce[j]) - np.linalg.norm(predforce[j])))

  #act_stress.append(atoms[i].get_stress())
  #pred_stress.append(testatom.get_stress())
  #Serrors.append(np.abs(act_stress[-1] - pred_stress[-1]))

  if i == 100:
      break

Eerrors = np.array(Eerrors)
Ferrors = np.array(Ferrors)
Serrors = np.array(Serrors)

print("Energy MAE:", np.mean(Eerrors))

#print(np.mean(Eerrors), np.mean(Ferrors))
#print(np.mean(Serrors, axis=0))

#fig, axs = plt.subplots(4,2)

#x = np.linspace(np.amin(act_energies), np.amax(act_energies), 100)
#xf = np.linspace(np.amin(pred_forces), np.amax(pred_forces), 100)
#axs[0, 0].scatter(act_energies, pred_energies, c='red')
#axs[0, 0].plot(x,x, linestyle='--', c='black')
#axs[0, 1].scatter(act_forces, pred_forces, c='red')
#axs[0, 1].plot(xf,xf, linestyle='--', c='black')



x = np.linspace(np.amin(act_energies), np.amax(act_energies), 100)
plt.scatter(act_energies, pred_energies)
plt.plot(x, x, linestyle='--', c='black')


plt.show()
exit()



act_stress = np.array(act_stress)
pred_stress = np.array(pred_stress)

count1 = 0
count2 = 0
stress_names = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']

for i in range(6):
    xs = np.linspace(np.amin(act_stress[:,i]), np.amax(act_stress[:,i]), 100)
    axs[1 + count2, count1].scatter(act_stress[:,i], pred_stress[:,i])
    axs[1 + count2, count1].plot(xs, xs,  linestyle='--', c='black')
    axs[1 + count2, count1].set_title(stress_names[i])

    count2 += 1
    if count2 == 3:
        count2 = 0
        count1 += 1






plt.tight_layout()
plt.show()

