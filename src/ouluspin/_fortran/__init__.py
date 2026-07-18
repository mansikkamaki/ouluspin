# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""Interface to the compiled Fortran extension module of OuluSpin.

The computationally heavy routines of the library are implemented in Fortran
and compiled with f2py into the extension module fortran_utils, which lives
in this directory (fortran_utils.so). The module is built from the sources
in src/fortran by the Makefile there; see the top-level README for build
instructions.

This package always imports successfully, so that the pure-Python parts of
OuluSpin can be used (and their documentation read) without the compiled
extension. If the extension is missing or cannot be loaded, an informative
RuntimeError is raised only when one of its routines is first accessed.

Usage (internal to the library):

    from ouluspin._fortran import fortran_utils as fu
    fu.matrix_utils.basis_transformation(...)
"""

try:
    from ouluspin._fortran import fortran_utils
    FORTRAN_AVAILABLE = True
except ImportError as exc:

    class _MissingFortranModule:
        """Placeholder that defers the import failure of the fortran_utils
        extension module until one of its routines is actually accessed, and
        then raises a RuntimeError with the original import error and build
        instructions.
        """

        def __init__(self, import_error):
            # Assign via __dict__ to bypass __getattr__-related surprises.
            self.__dict__["_import_error"] = import_error

        def __getattr__(self, name):
            raise RuntimeError(
                "The compiled Fortran extension module ouluspin._fortran"
                ".fortran_utils could not be imported, but the requested "
                "functionality needs it.\n"
                "Build it from the sources in src/fortran:\n"
                "    cd src/fortran && ./configure && make\n"
                "(see the README at the project root for details).\n"
                "The original import error was:\n"
                "    {0}".format(self.__dict__["_import_error"])
            )

    fortran_utils = _MissingFortranModule(exc)
    FORTRAN_AVAILABLE = False
