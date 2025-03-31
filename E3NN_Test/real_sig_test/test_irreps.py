from e3nn import o3
from e3nn.o3 import Irreps



rot = o3.rand_matrix()
print(rot)
irreps_sh = Irreps.spherical_harmonics(lmax=2)
print(irreps_sh)
D = irreps_sh.D_from_matrix(rot)

print(D)