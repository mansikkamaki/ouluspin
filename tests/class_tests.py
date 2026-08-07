#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run the internal test routines of all major classes in the library.

Every major class implements a run_tests class method that initiates the
class and runs a set of tests on the class constructor and the class
methods. This script calls all of these methods in turn, so that the
detailed test output originates from the classes themselves, and finishes
with a summary of which classes work correctly and which have been broken.

The script can be run directly from the command line (after sourcing the
setup.sh script at the project root, which sets up the module search path
and the runtime libraries):

    ouluspin-python tests/class_tests.py

The exit code is zero if all tests of all classes passed and one otherwise.

Notes:
  - AnisoCalculation is not tested directly as it is only intended to be
    inherited; its reading methods are tested through the run_tests methods
    of OpenMolcasCalculation and OrcaCalculation.
  - The Fortran routines have their own test suite in
    tests/fortran_tests.py.
"""

import sys

from ouluspin import units
from ouluspin import result_table
from ouluspin import result_plot
from ouluspin import tensors
from ouluspin import pseudospin_operators
from ouluspin import properties
from ouluspin import integration
from ouluspin import data_tables
from ouluspin.qc import molcas
from ouluspin.qc import orca
from ouluspin.qc import kuiva
from ouluspin.systems import electron_exchange_system


def main():
    """Run the run_tests class methods of all major classes and print a
    summary of the results. This function is used only internally by the
    library for testing purposes.
    """
    test_class_list = [
        units.EnergyUnitSystem,
        result_table.ResultTable,
        result_plot.ResultPlot,
        tensors.ChibotaruUngurSphericalTensor,
        tensors.IwaharaChibotaruSphericalTensor,
        tensors.MixedCartesianIwaharaChibotaruSphericalTensor,
        tensors.CartesianTensor,
        tensors.Rotation,
        pseudospin_operators.PseudoSpinBasis,
        pseudospin_operators.PseudoSpinOperator,
        pseudospin_operators.PseudoSpinVectorOperator,
        pseudospin_operators.GeneralOperatorMatrix,
        pseudospin_operators.GeneralVectorOperatorMatrix,
        integration.SimpleGrid,
        integration.LebedevLaikovGrid,
        integration.ZCWGrid,
        properties.IsothermalStaticMagnetization,
        properties.StaticMagneticSusceptibility,
        properties.StaticMagneticProperties,
        properties.StaticTransitionMagneticMoments,
        properties.PseudoSpinDoublet,
        electron_exchange_system.ElectronExchangeSystem,
        electron_exchange_system.AbInitioElectronExchangeSystem,
        data_tables.IonData,
        data_tables.MultipleIonData,
        molcas.OpenMolcasCalculation,
        orca.OrcaCalculation,
        orca.OrcaAnisoOutput,
        orca.OrcaAnisoFile,
        kuiva.KuivaPseudospinFile,
    ]

    print()
    print("    INTERNAL TESTS OF THE LIBRARY CLASSES")
    print()

    result_list = []
    for test_class in test_class_list:
        result = test_class.run_tests(print_output=True)
        result_list.append((test_class.__name__,result))

    print()
    print("    SUMMARY")
    print()
    print("      " + 50*"-")
    for class_name, result in result_list:
        if result:
            status = "ok"
        else:
            status = "FAILED"
        print("      {0:<42} {1}".format(class_name,status))
    print("      " + 50*"-")

    n_failed = sum(1 for _, result in result_list if not result)

    print()
    if n_failed == 0:
        print("    All {0} classes passed their tests.".format(len(result_list)))
    else:
        print("    {0} out of {1} classes FAILED their tests.".format(n_failed,
                                                                      len(result_list)))
    print()

    sys.exit(1 if n_failed > 0 else 0)


if __name__ == '__main__':
    main()
