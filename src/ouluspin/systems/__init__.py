# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""Higher-level classes for specific types of physical systems treated with
pseudospin Hamiltonians.

Modules
    electron_exchange_system   Systems of exchange-coupled electronic
                               multiplets (ElectronExchangeSystem,
                               AbInitioElectronExchangeSystem), together
                               with their common base PseudoSpinSystem,
                               which carries the analysis that depends only
                               on the pseudospin operators of a system.
"""

from ouluspin.systems import electron_exchange_system
