# sf-ip-ea
Stand-alone SF-IP/EA code. In progress. Run on the command line using Python:
```
$ python input.py
```
Output is written to stdout. A sample input file can be found below.
```
import os
import psi4
import numpy as np
import sf
from sf import fock_ci
import numpy as np

# Psi4 Molecule object
n2_7 = psi4.core.Molecule.create_molecule_from_string("""
0 7
N 0 0 0 
N 0 0 2.5 
symmetry c1
""")

# Psi4 options
options = {"BASIS": "cc-pvdz", 'scf_type': 'pk', 'guess': 'gwh'}
# Fock CI options
sf_options = {'RETURN_VECTS': True, 'NUM_ROOTS': 2}

# do a 1SF
wfn = fock_ci( 1, 1, n2_7, conf_space="p", ref_opts=options, sf_opts=sf_options)
```
The object returned is a wfn_sf object. Any additional information needed can then be extracted from this object (ex. S**2 values or NumPy arrays of the eigenvectors). For example, you could get an array of the eigenvalues using `wfn.e`.
The primary function call is `fock_ci`, which is defined as follows:
```
fock_ci(delta_a, delta_b, mol, conf_space="", ref_opts="", sf_opts="")
delta_a -- Desired number of alpha electrons to remove (int)
delta_b -- Desired number of beta electrons to add (int)
mol -- Molecule object.
conf_space -- Desired configuration space.
                  "" CAS
                  "h" 1 hole excitation
                  "p" 1 particle excitation
                  "h,p" 1 hole + 1 particle excitation
ref_opts -- Options for the reference program (dict)
                      See relevant ref code (ex. do_sf_psi4) for details
sf_opts -- Additional options for stand-alone SF code (dict)
                      sf_diag_method -- Diagonalization method to use.
                          "RSP" Direct (deprecated)
                          "LANCZOS" Use NumPy's Lanczos
                          "DAVIDSON" Use our Davidson
                      num_roots -- Number of roots to solve for.
                      guess_type -- Type of guess vector to use
                          "CAS" Do CAS first and use that as an initial guess.
                          "RANDOM" Random orthonormal basis
                          "READ" Read guess from a NumPy file (TODO)
                      integral_type  Which integrals to use (DF or FULL)
                           "FULL" Use full integrals (no density fitting)
                           "DF" Use density fit integrals
```
# Important Parameters
### Number of Spin-Flips/IP/EA ###
The first and second values define the number of alpha electrons added and number of beta electrons removed, respectively. This allows you to 
control the number of spin-flips and the number of IP/EA.
### Excitation Level ###
The `conf_space` parameter allows you to add single hole and/or particle excitations. The default is CAS-nSF (`""`). Hole excitations are added by specifying `conf_space="h"` and particle excitations are specified with `conf_space="p"`.
Some example function calls for various methods can be found below.
```
# CAS-IP
wfn = fock_ci( 1, 0, n2_7, conf_space="", ref_opts=options, sf_opts=sf_options)
# CAS-1SF
wfn = fock_ci( 1, 1, n2_7, conf_space="", ref_opts=options, sf_opts=sf_options)
# CAS-1SF-EA
wfn = fock_ci( 1, 2, n2_7, conf_space="", ref_opts=options, sf_opts=sf_options)
# RAS(h)-1SF
wfn = fock_ci( 2, 1, n2_7, conf_space="h", ref_opts=options, sf_opts=sf_options)
# RAS(h,p)-1SF
wfn = fock_ci( 2, 1, n2_7, conf_space="hp", ref_opts=options, sf_opts=sf_options)
```

