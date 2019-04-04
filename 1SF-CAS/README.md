# sf-ip-ea
Stand-alone SF-IP/EA code. In progress. Run on the command line using Python:
```
$ python input.py
```
Output is written to stdout. Right now it is compatible with Psi4:
```
import psi4
import sf
from sf import sf_psi4
n2_7 = psi4.core.Molecule.create_molecule_from_string("""
0 7
N 0 0 0 
N 0 0 1.5 
symmetry c1
""")

options = {"basis": "cc-pvdz", 'diis_start': 20, 'e_convergence': 1e-10, 'd_convergence': 1e-10}

# doing a 1SF-IP (remove 2 alpha, add 1 beta)
e = sf_psi4( 2, 1, n2_7, conf_space="", add_opts=options)
```
If you have the integrals, you can also run using do_sf_cas().
