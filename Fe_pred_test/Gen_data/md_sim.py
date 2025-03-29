# from pymatgen.core import Lattice, Structure
import matgl
from matgl.ext.ase import MolecularDynamics, Relaxer
import warnings
# from pymatgen.io.ase import AseAtomsAdaptor as AAA
from matgl.ext.ase import M3GNetCalculator, MolecularDynamics, Relaxer
from matgl.apps.pes import Potential
from matgl.ext.ase import PESCalculator
import numpy as np
import matplotlib.pyplot as plt
from ase import units
# from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
# from pymatgen.analysis.local_env import CrystalNN
from ase.io import read, write
from ase.md.nptberendsen import NPTBerendsen
from ase.build.supercells import make_supercell
import warnings
from scipy.optimize import curve_fit
from ase.io.trajectory import Trajectory
from ase.filters import FrechetCellFilter
from ase.md.andersen import Andersen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution as MBD
warnings.filterwarnings('ignore')

def fit_func(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def fit_func_deriv(x, a, b, c, d, e):
    return 4*a*x**3 + 3*b*x**2 + 2*c*x + d


# data = np.loadtxt("Red_Points_Data.csv")
# popt, _ = curve_fit(fit_func, data[:,1]/7.6, data[:,0])
# print(popt)

# samp_x = np.linspace(np.amin(data[:,1]/7.6), np.amax(data[:,1]/7.6), 100)

# plt.plot(data[:,1]/7.6, data[:,0], marker="o", c="black")
# plt.plot(samp_x, fit_func(samp_x, *popt), c="blue")

# plt.yscale("log")
# plt.ylim(1, 10**5)
# plt.xlim(0.5, 4)
# plt.show()
# exit()



def calc_compressibility(pressure=300):
    data = np.loadtxt("Red_Points_Data.csv") 
    popt, _ = curve_fit(fit_func, data[:,0], data[:,1])
    pho = fit_func(pressure, *popt)
    cmptb = (1/pho)*fit_func_deriv(pressure, *popt)
    return cmptb/units.GPa

def calc_NPT(fe):
    pressure = 300*units.GPa

    fe1 = read("POSCAR", format="vasp")
    N = 1
    fe1 = make_supercell(fe1, P = np.diag([2,N,N]))
    fe1.set_cell(fe1.get_cell()[:]*(0.6**(1/3)), scale_atoms=True)
    fe1.calc = calc
    fe = FrechetCellFilter(fe1)
    K = calc_compressibility()
    dyn = NPTBerendsen(fe, timestep=5 * units.fs, temperature_K=5500,
                    taut=100* units.fs, pressure_au=300*units.GPa,
                    taup=1000 * units.fs, compressibility_au=K)

    def printenergy(a=fe):  # store a reference to atoms in the definition.
        """Function to print the potential, kinetic and total energy."""
        epot = a.get_potential_energy() / len(a)
        ekin = a.get_kinetic_energy() / len(a)
        print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
            'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

    traj = Trajectory('i3m3.traj', 'w', fe)
    dyn.attach(traj.write, interval=1)
    dyn.attach(printenergy, interval=10)
    printenergy()
    dyn.run(600)

def calc_NVT(atom, vratio, temp, calc, traj_path="test.traj", timestep=5*units.fs):

    fe = atom.copy()
    fe.set_cell(fe.get_cell()[:]*(vratio**(1/3)), scale_atoms=True)
    fe.calc = calc
    MBD(fe, temperature_K=temp)
    dyn = Andersen(fe, timestep=timestep, temperature_K=temp, andersen_prob=1e-3, trajectory=traj_path)

    def printenergy(a=fe):  # store a reference to atoms in the definition.
        volume = a.get_temperature()
        print("Vol:", volume)
    
    dyn.attach(printenergy, interval=10)
    printenergy()
    dyn.run(600)

def scan_volumes():
    model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    calc = PESCalculator(model)
    pressure = 300*units.GPa

    fe = read("POSCAR", format="vasp")
    N = 1
    fe = make_supercell(fe, P = np.diag([2,N,N]))

    vratios = np.linspace(0.55,0.7,20)
    for vratio in vratios:
        calc_NVT(fe, vratio, 5500, calc, traj_path="vol_scan_md/" + str(vratio) + "_dyn.traj")

def sim_certain_vol():
    model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    calc = PESCalculator(model)
    pressure = 300*units.GPa

    fs = ['194','225','229']
    vratios = [0.8, 0.65, 0.73]

    for i, f in enumerate(fs):
        fe = read("../Poscars/" + f, format='vasp')
        # N = 1
        # fe = make_supercell(fe, P = np.diag([2,N,N]))
        calc_NVT(fe, vratios[i], 9000, calc, traj_path="MD_65P/" + f + "_dyn.traj")


sim_certain_vol()
