Reference Options: REF_OPTS
===========================

The ``ref_opts`` keyword allows the user to pass in additional options for 
the program running the reference SCF portion of the calculation. These are 
set uniquely for each program (Psi4, PySCF, etc). It is passed like follows::

    psi4_options = {"basis": "cc-pvtz", "reference": "rohf", "guess": "gwh"}
    wfn = fock_ci(1, 1, mol, ref_opts=psi4_options)

See the documentation for each package for details.

Additional Options: SF_OPTS
===========================

The ``sf_opts`` keyword holds a dictionary that defines all other options for 
the ``fock_ci`` method. It can be used to select the diagonalization method, 
the CI guess type, which types of integrals to use, and more. These options 
are passed as an optional argument to ``fock_ci`` as follows::

    sf_options = {'sf_diag_method': 'lanczos', 'num_roots': 5}
    wfn = fock_ci(1, 1, mol, sf_opts=sf_options)

Number of Roots
---------------

By default, the program solves for only one root. To solve for multiple 
roots, use the ``NUM_ROOTS`` keyword::

    sf_options = {'num_roots': 5}
    wfn = fock_ci(1, 1, mol, sf_opts=sf_options)

Passing In A Reference
----------------------

At times, the user may want to use their own pre-converged ROHF reference 
wavefunction, rather than allowing ``fock_ci`` to do the SCF steps itself. 
(This is helpful in cases where orbitals need to be localized, for example, or 
when one ROHF solution can be used for multiple calculations.) In such a 
circumstance, the user may use the ``READ_PSI4_WFN`` keyword and the 
``PSI4_WFN`` keyword like so::

    # converge Psi4 ROHF wfn
    options = {"basis": "cc-pvtz", "reference": "rohf"}
    psi4.set_options(options)
    e, psi4_wfn = psi4.energy('scf', molecule=mol, return_wfn=True)

    # do SF portion
    sf_options = {'READ_PSI4_WFN': True, 'PSI4_WFN': psi4_wfn}
    wfn = fock_ci(1, 1, mol, ref_opts=options, sf_opts=sf_options)

Diagonalization Methods
-----------------------

Multiple options are available for diagonalization. To select one, :

    * ``RSP``: Full diagonalization (deprecated)
    * ``LANCZOS``: Use NumPy's built-in Lanczos solver (default)
    * ``DAVIDSON``: Use our Davidson (needs testing)

Currently, ``LANCZOS`` is the default, due to a lack of rigorous testing in 
the Davidson implementation. In the future, Davidson will be the default.

In addition, one can pass in ``DO_MATRIX`` to obtain the full Hamiltonian 
without diagonalization steps. This is stored in ``sf_wfn.H`` and can be 
accessed after a ``fock_ci`` call::

    sf_options = {'SF_DIAG_METHOD': 'DO_MATRIX'}
    wfn = fock_ci( 1, 1, n2_7, ref_opts=options, sf_opts=sf_options)
    print(wfn.H)

The guess for the diagonalization is set using ``GUESS_TYPE``. Currently, a 
random orthonormal basis is the default. In the future we will implement a 
rigorous CAS guess and an option to read guess vectors from a NumPy file.

Integral Type
-------------

The ``INTEGRAL_TYPE`` keyword allows the user to choose whether to use 
density-fitted or full integrals:
    * ``DF``: Use density-fitted integrals
    * ``FULL``: Use full integrals

Note that the auxiliary basis is chosen based on the Psi4 settings.



