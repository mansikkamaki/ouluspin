# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""Internal debugging and testing helpers of the OuluSpin library.

This module contains matrix pretty-printing helpers and the test_check /
test_summary functions used by the run_tests class methods throughout the
library. It is internal to the library and not part of the public API.
"""

import sys

import numpy as np


def real_matrix_str(matrix,number_format="6.3f"):
    """Return a human-readable representation of a real square matrix."""
    if (not len(matrix.shape) == 2) or (not matrix.shape[0] == matrix.shape[1]):
        print("ERROR in ouluspin._debug.")
        print("ERROR: Not a square matrix.")
        print("Error termination.")
        sys.exit(1)

    dim = matrix.shape[0]

    tmp_str = ""

    for i in range(0,dim):
        for j in range(0,dim):
            tmp_str += (" {0:" + number_format + "}").format(matrix[i][j])
        tmp_str += "\n"
    tmp_str += "\n"

    return tmp_str


def complex_matrix_str(matrix,number_format="6.3f"):
    """Return a human-readable representation of the real and imaginary parts
    of a complex square matrix.
    """
    tmp_str  = "Real part:\n\n"
    tmp_str += real_matrix_str(matrix.real,number_format=number_format)
    tmp_str += "Imaginary part:\n\n"
    tmp_str += real_matrix_str(matrix.imag,number_format=number_format)

    return tmp_str


def test_check(class_name,test_name,condition,print_output=True):
    """Evaluate a single test condition used by the run_tests class methods,
    optionally print the outcome, and return the outcome as a boolean. This
    function is used only internally by the library for testing purposes.
    """
    condition = bool(condition)

    if print_output:
        if condition:
            status = "ok"
        else:
            status = "FAILED"
        print("      {0:<70} {1}".format(class_name + ": " + test_name,status))

    return condition


def test_summary(class_name,result_list,print_output=True):
    """Summarize a list of boolean test results produced by test_check, and
    return True if, and only if, all tests passed. This function is used only
    internally by the library for testing purposes.
    """
    n_tests  = len(result_list)
    n_failed = n_tests - sum(1 for result in result_list if result)

    all_passed = (n_failed == 0) and (n_tests > 0)

    if print_output:
        if all_passed:
            print("      {0}: all {1} tests passed".format(class_name,n_tests))
        else:
            print("      {0}: {1} out of {2} tests FAILED".format(class_name,n_failed,n_tests))
        print()

    return all_passed


def spin_matrices(pseudospin):
    """Construct matrix representations of the Cartesian spin operators
    S_x, S_y and S_z of a single spin in the basis |S,-S>, ..., |S,S>
    (ascending projection), which is the basis ordering used by
    PseudoSpinBasis. The pseudospin is given as a multiple of two. The
    matrices are returned as a list [S_x,S_y,S_z]. This function is used
    only internally by the library for testing purposes.
    """
    S   = pseudospin / 2.0
    dim = pseudospin + 1

    M_values = [-S + i for i in range(0,dim)]

    S_z = np.zeros((dim,dim), dtype=np.complex128)
    S_p = np.zeros((dim,dim), dtype=np.complex128)

    for i in range(0,dim):
        S_z[i][i] = M_values[i]
    for i in range(0,dim-1):
        S_p[i+1][i] = np.sqrt(S*(S + 1.0) - M_values[i]*(M_values[i] + 1.0))

    S_m = S_p.conj().T
    S_x = 0.5*(S_p + S_m)
    S_y = -0.5j*(S_p - S_m)

    return [S_x,S_y,S_z]


def operator_matrix_from_tensor(tensor,pseudospin_list):
    """Construct and return the matrix representation of an
    IwaharaChibotaruSphericalTensor in the pseudospin basis defined by
    pseudospin_list. The imports are carried out inside the function to
    avoid circular imports. This function is used only internally by the
    library for testing purposes.
    """
    from ouluspin import units, pseudospin_operators

    tmp_units = units.EnergyUnitSystem('wavenumber')
    basis     = pseudospin_operators.PseudoSpinBasis(pseudospin_list)

    operator = pseudospin_operators.PseudoSpinOperator(basis,[tensor],tmp_units,
                                                       diagonalize_operator_matrix=False,
                                                       store_operator_matrix=True)
    return operator.matrix
