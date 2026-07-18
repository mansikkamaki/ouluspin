#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""Extract the crystal-field Hamiltonian of a single J = 15/2 multiplet
(e.g. a Dy(III) complex) from an ORCA aniso file and decompose it into
irreducible tensor operators. This is essentially what the SINGLE_ANISO
module already does.

Input: an ORCA aniso file (containing the Hamiltonian and magnetic moment
operator matrices) given as the first command-line argument.

Run with the environment set up (see the README at the repository root):

    ouluspin-python one_site_crystal_field.py <file.anisofile>
"""

import sys
import time

from ouluspin import units as units_module
from ouluspin import pseudospin_operators
from ouluspin.qc import orca
from ouluspin.systems import electron_exchange_system

if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.exit("usage: ouluspin-python one_site_crystal_field.py "
                 "<file.anisofile>")
    aniso_file_name = sys.argv[1]

    tik = time.time()

    units = units_module.EnergyUnitSystem('wavenumber')
    print(units)

    calculation = orca.OrcaAnisoFile(aniso_file_name,units)
    print(calculation)

    # The ab initio states are reordered so that the energies are ascending
    # in the pseudospin projection ordering used by PseudoSpinBasis.
    reorder_list = [15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]

    hamiltonian_operator = pseudospin_operators\
        .GeneralOperatorMatrix(calculation.hamiltonian(),
                               diagonalize_operator_matrix=True,
                               translate_eigenvalues=True)
    magnetic_moment_operator = pseudospin_operators\
        .GeneralVectorOperatorMatrix(calculation.magnetic_moment(include_bohr_magneton=False),
                                     diagonalize_operator_matrix=True,
                                     translate_eigenvalues=False)

    # A single site with pseudospin S = 15/2 (pseudospins are given as
    # multiples of two).
    pseudospin_basis = pseudospin_operators.PseudoSpinBasis([15])
    print(pseudospin_basis)

    ab_initio_system = electron_exchange_system\
        .AbInitioElectronExchangeSystem(hamiltonian_operator,
                                        magnetic_moment_operator,
                                        pseudospin_basis,
                                        units,
                                        reorder_list=reorder_list,
                                        print_output=True)

    print(ab_initio_system.hamiltonian_tensor())

    # The magnetic moment operator is treated as a single mixed
    # Cartesian--spherical tensor instead of a list of three spherical
    # tensors.
    magnetic_moment_tensor = ab_initio_system.magnetic_moment_tensor()
    print(magnetic_moment_tensor.ITO_table(title="MAGNETIC MOMENT TENSOR",
                                           rank_threshold=1.0e-6,
                                           half_table=True))

    pseudospin_system = electron_exchange_system\
        .ElectronExchangeSystem([15],
                                [(ab_initio_system.hamiltonian_tensor(),0)],
                                [(magnetic_moment_tensor,0)],
                                units,
                                print_output=True)

    print(pseudospin_system.hamiltonian)

    tok = time.time()
    print("Total wall time: {0:18.3f}".format(tok-tik))
