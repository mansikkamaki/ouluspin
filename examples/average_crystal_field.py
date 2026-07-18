#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""This script is intented for the study of the crystal field (CF) of a lanthanide(III)
ion that is exchange-coupled to a radical ligand. Because the radical ligand has an
unparied electron, the lanthanide ion cannot be studied by a CASSCF calculation
without correlating the ligand electrons. This script provides an alternative way
where the ligand CF is constructed by averaging two CF calculations: one where
one electron has been removed from the ligand and one where one electron has been
added to the ligand. Thus, in both calculations the ligand is diamagnetic, and the
effect of the unpaired electron on the lanthanide is obtained by averaging the two
diamangetic sistuations. The two calculations will be referred to as cation (one
electron removed) and anion (one electron added).

This approach has been used, for example, in:

    N. Mavragani, D. Errulat, D. A. Gálico, A. A. Kitos, A. Mansikkamäki,
    M. Murugesu. Angew. Chem. Int. Ed. 2021, 60, 24206--24213.

Currently the calculation assumes a Dy(III) ion. In case of other ions, the variables
pseudospin (twice the Dy(III) J: 2*J = 2*15/2 = 15) and g-J (the Lande g-factor of
Dy(III)) need to be changed to those describing the correct ion.

Input: the SINGLE_ANISO datafiles (.aniso type files containing the
operator matrices, not the SINGLE_ANISO output files) of the cation and
anion calculations, given as command-line arguments.

Run with the environment set up (see the README at the repository root):

    ouluspin-python average_crystal_field.py <cation.anisofile> <anion.anisofile>
"""

import sys

import ouluspin

if __name__ == '__main__':

    if len(sys.argv) != 3:
        sys.exit("usage: ouluspin-python average_crystal_field.py "
                 "<cation.anisofile> <anion.anisofile>")
    cation_name = sys.argv[1]
    anion_name  = sys.argv[2]

    # A J = 15/2 multiplet (pseudospins are given as multiples of two) with
    # the Lande g-factor of Dy(III).
    pseudospin = 15
    g_J = 4.0/3.0

    units = ouluspin.EnergyUnitSystem('wavenumber')
    print(units)

    print('============================================')
    print('===  AVERAGED CRYSTAL-FIELD CALCULATION  ===')
    print('============================================')
    print()

    # The crystal field of the second calculation is rotated into the
    # principal magnetic frame of the first before the two are averaged;
    # the magnetic moment is constructed from the Lande g-factor, and the
    # averaged system is finally rotated into the principal magnetic frame
    # of its own ground doublet.
    system = ouluspin.AbInitioElectronExchangeSystem\
                     .from_average_aniso_data(cation_name,anion_name,
                                              pseudospin,g_J,units,
                                              print_output=True)

    average_cf = system.hamiltonian_tensor()
    print(average_cf)

    cf_operator = system.hamiltonian_operator()
    print(cf_operator)
    print(cf_operator.eigenvector_table())

    transition_magnetic_moments = system.static_transition_magnetic_moments()
    print(transition_magnetic_moments)

    # The doublet tabulation constructs each doublet with the full frame
    # bookkeeping of the system: the g-tensors and their principal magnetic
    # axes are reported in the input axis frame (the frame of the matrices
    # in the datafiles).
    print(system.pseudospin_doublet_table([(2*i,2*i+1) for i in range(0,8)]))
