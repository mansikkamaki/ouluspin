#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""This script is intended to study a Ln(II) complex with a 4f^n(6s/5d)^1
electronic configuration. The sciprt decomposes the Hamiltonian of a
coupled two-site system (pseudospins S = 15/2 and S = 1/2) into the crystal
fields of the individual sites and the intersite exchange operator.

Currently the calculation assumes a Dy(III) ion with J = 15/2 and a 6s/5d
spin with S = 1/2. These are given as multiples of two: 2J = 15 and 2S = 1.

Input: an ORCA aniso file (containing the Hamiltonian and magnetic moment
operator matrices of the coupled system) given as the first command-line
argument.

Run with the environment set up (see the README at the repository root):

    ouluspin-python two_site_crystal_field.py <file.anisofile>
"""

import sys
import time

from ouluspin import units as units_module
from ouluspin import pseudospin_operators
from ouluspin.qc import orca
from ouluspin.systems import electron_exchange_system

if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.exit("usage: ouluspin-python two_site_crystal_field.py "
                 "<file.anisofile>")
    aniso_file_name = sys.argv[1]

    tik = time.time()

    units = units_module.EnergyUnitSystem('wavenumber')
    print(units)

    calculation = orca.OrcaAnisoFile(aniso_file_name,units)
    print(calculation)

    hamiltonian_operator = pseudospin_operators\
        .GeneralOperatorMatrix(calculation.hamiltonian(),
                               diagonalize_operator_matrix=True,
                               translate_eigenvalues=True)
    magnetic_moment_operator = pseudospin_operators\
        .GeneralVectorOperatorMatrix(calculation.magnetic_moment(include_bohr_magneton=False),
                                     diagonalize_operator_matrix=True,
                                     translate_eigenvalues=False)

    # Two sites with pseudospins S = 15/2 and S = 1/2 (pseudospins are
    # given as multiples of two).
    pseudospin_basis = pseudospin_operators.PseudoSpinBasis([15,1])
    print(pseudospin_basis)

    # The ab initio states are reordered so that the energies are ascending
    # in the product-basis ordering used by PseudoSpinBasis.
    reorder_list = [31,29,30,27,28,25,26,23,24,21,22,19,20,17,18,
                    15,16,13,14,11,12,9,10,7,8,5,6,3,4,1,2,0]

    ab_initio_system = electron_exchange_system\
        .AbInitioElectronExchangeSystem(hamiltonian_operator,
                                        magnetic_moment_operator,
                                        pseudospin_basis,
                                        units,
                                        reorder_list=reorder_list,
                                        print_output=True)

    # Separate the total Hamiltonian tensor into the one-site crystal-field
    # tensors and the intersite exchange tensor.
    cf_tensors, exchange_tensor = ab_initio_system.hamiltonian_tensor()\
                                                  .separate_tensors()

    print(cf_tensors[0].ITO_table(symbol="B", title="Site 1 CF Operator",
                                  rank_threshold=0.01, half_table=True))
    print(cf_tensors[1].ITO_table(symbol="B", title="Site 2 CF Operator",
                                  rank_threshold=0.01, half_table=True))
    print(exchange_tensor.ITO_table(symbol="J", title="Exchange Operator",
                                    rank_threshold=0.01, half_table=True))

    tok = time.time()
    print("Total wall time: {0:18.3f}".format(tok-tik))
