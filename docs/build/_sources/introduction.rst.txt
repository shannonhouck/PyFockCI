Installation
============

Clone the program from the GitHub repository::

    $ git clone https://github.com/shannonhouck/sf-ip-ea.git

Then naviate into the directory and use pip to install::

    $ cd sf-ip-ea/
    $ pip install -e .

You can now import this package into Python. Note that some dependencies 
may be required (ex. numpy, scipy, Psi4).

Running CAS-SF/IP/EA
====================

The RAS-SF-IP/EA code is run by calling the ``fock_ci`` function. A sample 
input file (using the Psi4 interface) can be found below::

    import psi4
    import sf_ip_ea

    # Psi4 Molecule object
    n2_7 = psi4.core.Molecule.from_string("""
    0 7
    N 0 0 0
    N 0 0 2.5
    symmetry c1
    """)

    # Psi4 options
    options = {"BASIS": "cc-pvdz"}
    # Fock CI options
    sf_options = {'NUM_ROOTS': 2}

    # do RAS(p)-1SF
    wfn = sf_ip_ea.fock_ci( 1, 1, n2_7, conf_space="p", ref_opts=options, 
                            sf_opts=sf_options)

A few important things to note:
    * The number of spin-flips and IP/EA is determined by the first two 
      numbers passed into ``fock_ci``. The first integer is the number of 
      alpha electrons to remove, relative to the reference wavefunction; 
      the second is the number of beta electrons to add. For more information 
      about method selection, see [TODO: add table link].
    * The object returned is a wfn_sf object. Any additional information 
      needed can then be extracted from this object (ex. Sz values, 
      information about RAS spaces, or eigenvectors). For example, you could 
      get an array of the eigenvalues using ``wfn.e``. For more information 
      about the wfn_sf object, see [TODO: link to wfn_sf page].

This input file can then be run from the command line::

    $ python input.py

For additional information about ``fock_ci``, see [TODO: add link]. 

Selecting nSF-IP/EA Method
--------------------------

As mentioned previously, the number of spin-flips and IP/EA is determined 
exclusively by the number of alpha electrons removed and beta electrons added. 
A table of currently-supported options is shown below.

+-------------+----------+----------+
| Scheme      | n_alpha  | n_beta   |
+=============+==========+==========+
| 1SF         | 1        | 1        |
+-------------+----------+----------+
| 2SF         | 2        | 2        |
+-------------+----------+----------+
| IP          | 1        | 0        |
+-------------+----------+----------+
| EA          | 0        | 1        |
+-------------+----------+----------+
| 1SF-IP      | 2        | 1        |
+-------------+----------+----------+
| 1SF-EA      | 1        | 2        |
+-------------+----------+----------+

Note that if you do not see your desired scheme here, you can use our 
slower but more general Psi4 version run using DETCI [TODO: add link].

Adding Hole and Particle Excitations
------------------------------------

Hole and particle excitations are often needed to provide orbital relaxation, 
particularly in the SF-IP/EA schemes (see [TODO: ref]). The following 
excitation schemes are currently implemented in our code:

+--------+-------+--------+--------+----------+
|        | CAS   | RAS(h) | RAS(p) | RAS(h,p) |
+--------+-------+--------+--------+----------+
| 1SF    | Y     | Y      | Y      | Y        |
+--------+-------+--------+--------+----------+
| 2SF    | Y     |        |        |          |
+--------+-------+--------+--------+----------+
| IP     | Y     | Y      | Y      |          |
+--------+-------+--------+--------+----------+
| EA     | Y     | Y      | Y      |          |
+--------+-------+--------+--------+----------+
| 1SF-IP | Y     | Y      |        |          |
+--------+-------+--------+--------+----------+
| 1SF-EA | Y     |        | Y      |          |
+--------+-------+--------+--------+----------+

In order to perform hole and particle excitations, use the ``conf_space`` 
keyword passed into ``fock_ci``. Schemes are called as follows:

+----------+---------------+
| Scheme   | conf_space    |
+----------+---------------+
| CAS      | ``""``        |
+----------+---------------+
| RAS(h)   | ``"h"``       |
+----------+---------------+
| RAS(p)   | ``"p"``       |
+----------+---------------+
| RAS(h,p) | ``"h,p"``     |
+----------+---------------+

So, for example, to do RAS(h)-1SF-IP, one would do::

    wfn = fock_ci( 2, 1, mol, conf_space="h")

If you do not see your desired scheme here, please revert to the Psi4 plugin 
version using DETCI [TODO: add link].





