# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""Interfaces for extracting ab initio data (mostly operator matrices) from
the outputs of quantum-chemistry codes.

Modules
    molcas     OpenMolcas / SINGLE_ANISO outputs (AnisoCalculation,
               OpenMolcasCalculation).
    orca       ORCA outputs and aniso files (OrcaCalculation,
               OrcaAnisoOutput, OrcaAnisoFile).
"""

from ouluspin.qc import molcas
from ouluspin.qc import orca
