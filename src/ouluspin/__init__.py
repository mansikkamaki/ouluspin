# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""OuluSpin -- a Python library for theoretical molecular magnetism.

OuluSpin provides routines for the construction of pseudospin Hamiltonians
and the calculation of various magnetic properties using these Hamiltonians,
following the pseudospin formalism of Chibotaru and co-workers. It also
contains utilities for extracting ab initio quantum-chemical data (operator
matrices) from the outputs of quantum-chemistry codes.

Subpackages and modules
    units                  Energy unit systems and physical constants.
    result_table           Structured representation of the result tables
    result_plot            Structured representation of the result plots
                           printed by the library.
    tensors                Spherical and Cartesian tensor structures
                           (Iwahara--Chibotaru, Chibotaru--Ungur, mixed and
                           Cartesian tensors, rotations).
    pseudospin_operators   Pseudospin bases and operators constructed from
                           the tensor structures.
    properties             Magnetic properties calculated from the operators
                           (magnetization, susceptibility, transition
                           moments, g-tensors).
    integration            Spherical integration grids (Lebedev--Laikov,
                           ZCW, simple spherical grids).
    qc                     Interfaces to quantum-chemistry codes
                           (OpenMolcas, ORCA, Kuiva).
    systems                Higher-level classes for specific types of
                           physical systems.

The most important classes are re-exported at the package top level, so that
they can be imported directly, e.g.

    from ouluspin import EnergyUnitSystem, PseudoSpinBasis

The computationally heavy routines are implemented in Fortran and compiled
into the extension module ouluspin._fortran.fortran_utils (built from the
sources in src/fortran; see the README). The package can be imported without
the compiled extension; an informative error is raised only when a routine
that needs it is actually called.
"""

__version__ = "1.1.1"

from ouluspin import units
from ouluspin import result_table
from ouluspin import result_plot
from ouluspin import tensors
from ouluspin import pseudospin_operators
from ouluspin import properties
from ouluspin import integration
from ouluspin import qc
from ouluspin import systems

from ouluspin.units import EnergyUnitSystem
from ouluspin.result_table import ResultTable
from ouluspin.result_plot import ResultPlot
from ouluspin.tensors import (
    ChibotaruUngurSphericalTensor,
    IwaharaChibotaruSphericalTensor,
    MixedCartesianIwaharaChibotaruSphericalTensor,
    CartesianTensor,
    Rotation,
)
from ouluspin.pseudospin_operators import (
    PseudoSpinBasis,
    PseudoSpinOperator,
    PseudoSpinVectorOperator,
    GeneralOperatorMatrix,
    GeneralVectorOperatorMatrix,
)
from ouluspin.properties import (
    IsothermalStaticMagnetization,
    StaticMagneticSusceptibility,
    StaticMagneticProperties,
    StaticTransitionMagneticMoments,
    PseudoSpinDoublet,
)
from ouluspin.integration import (
    SimpleGrid,
    LebedevLaikovGrid,
    ZCWGrid,
)
from ouluspin.systems.electron_exchange_system import (
    PseudoSpinSystem,
    ElectronExchangeSystem,
    AbInitioElectronExchangeSystem,
)
from ouluspin.data_tables import IonData, MultipleIonData
from ouluspin.qc.molcas import AnisoCalculation, OpenMolcasCalculation
from ouluspin.qc.orca import OrcaCalculation, OrcaAnisoOutput, OrcaAnisoFile
from ouluspin.qc.kuiva import KuivaPseudospinFile

__all__ = [
    "__version__",
    # Modules and subpackages.
    "units",
    "result_table",
    "result_plot",
    "tensors",
    "pseudospin_operators",
    "properties",
    "integration",
    "qc",
    "systems",
    # Units.
    "EnergyUnitSystem",
    # Result tables.
    "ResultTable",
    "ResultPlot",
    # Tensors.
    "ChibotaruUngurSphericalTensor",
    "IwaharaChibotaruSphericalTensor",
    "MixedCartesianIwaharaChibotaruSphericalTensor",
    "CartesianTensor",
    "Rotation",
    # Pseudospin operators.
    "PseudoSpinBasis",
    "PseudoSpinOperator",
    "PseudoSpinVectorOperator",
    "GeneralOperatorMatrix",
    "GeneralVectorOperatorMatrix",
    # Properties.
    "IsothermalStaticMagnetization",
    "StaticMagneticSusceptibility",
    "StaticMagneticProperties",
    "StaticTransitionMagneticMoments",
    "PseudoSpinDoublet",
    # Integration grids.
    "SimpleGrid",
    "LebedevLaikovGrid",
    "ZCWGrid",
    # Pseudospin systems.
    "PseudoSpinSystem",
    "ElectronExchangeSystem",
    "AbInitioElectronExchangeSystem",
    # Tabulated and calculated data of the ions.
    "IonData",
    "MultipleIonData",
    # Quantum-chemistry interfaces.
    "AnisoCalculation",
    "OpenMolcasCalculation",
    "OrcaCalculation",
    "OrcaAnisoOutput",
    "OrcaAnisoFile",
    "KuivaPseudospinFile",
]
