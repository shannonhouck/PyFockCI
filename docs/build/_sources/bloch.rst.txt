Bloch Effective Hamiltonian Analysis
====================================

A Bloch effective Hamiltonian can be built in order to extract information
about coupling in complexes with more than two centers (for example in some 
mixed-valent complexes, or lanthanide dimers in fullerene cages). A
sample input file for such a situation is shown below::

    import psi4
    import sf_ip_ea
    from sf_ip_ea import bloch

    mol = psi4.core.Molecule.from_string("""
    0 7
    O 0 0 0
    O 2 0 0
    O 4 0 0
    symmetry c1
    """)

    options = {"BASIS": "sto-3g"}
    sf_options = {'NUM_ROOTS': 3}

    wfn = sf_ip_ea.fock_ci( 1, 1, mol, ref_opts=options, sf_opts=sf_options)
    n_sites = 3
    H = bloch.do_bloch(wfn, n_sites)

Note that Bloch is only supported for the Psi4 interface right now, and that 
the program only supports 1SF calculations at the moment.

Defining Sites
--------------

By default, each orbital in the CAS space is treated as its own site. 
This behavior can be overridden by specifying either of the following 
keywords:

    * site_list: A list of atom indexes, where each atom corresponds to 
      one site. The numbering here follows the ordering of atoms in the Psi4 
      Molecule object and starts at 0. For example::
          H = bloch.do_bloch(wfn, n_sites, site_list=[0,1,2])

    * site_list_orbs: A list of lists of orbital indexes, with each sub-list 
      counted as its own site. For example::
          H = bloch.do_bloch(wfn, n_sites, 
                             site_list_orbs=[[10,11,13],[12,14,15]],
                             skip_localization=True)
      Ordering follows Psi4 ordering, and orbital indexing starts at 1.
      In this case, the user should localize the orbitals themselves, 
      make the assignments by hand, and set ``skip_localization=True``
      in the function call.



