#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test suite for the fortran_utils extension module.

The module is built from the Fortran sources in src/fortran by the Makefile
there (see src/fortran/README or the top-level README for build
instructions). This script exercises at least one routine from every Fortran
module and, wherever an analytic result is available, checks the returned
value against it. Routines whose result is not easily verified analytically
are smoke-tested: they are called with valid input and their output shape /
basic properties are checked.

Run (after building the module and sourcing setup.sh at the project root):

    ouluspin-python tests/fortran_tests.py
"""

import itertools
import math
import os
import random
import subprocess
import sys
from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np

from ouluspin._fortran import fortran_utils as fu

# High-precision decimals for the exact Clebsch-Gordan reference values used
# in the cg tests.
getcontext().prec = 60


# ---------------------------------------------------------------------------
# Tiny test harness.
# ---------------------------------------------------------------------------
_passed = 0
_failed = 0


def check(name, condition, detail=""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}   {detail}")


def close(a, b, tol=1e-9):
    return np.allclose(a, b, atol=tol, rtol=0.0)


# Pauli matrices (Hermitian) reused across several tests.
SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)


# ---------------------------------------------------------------------------
# num_utils : module parameters.
# ---------------------------------------------------------------------------
def test_num_utils():
    print("num_utils")
    check("pi", close(fu.num_utils.pi, math.pi, tol=1e-6))
    check("dp_one", close(fu.num_utils.dp_one, 1.0))
    check("imaginary_unit", close(fu.num_utils.imaginary_unit, 1j))


# ---------------------------------------------------------------------------
# math_utils.
# ---------------------------------------------------------------------------
def test_math_utils():
    print("math_utils")
    m = fu.math_utils
    check("factorial(5)", m.factorial(5) == 120)
    check("factorial(0)", m.factorial(0) == 1)
    check("real_close_to true", m.real_close_to(1.0, 1.0 + 1e-8))
    check("real_close_to false", not m.real_close_to(1.0, 2.0))
    check("real_is_zero", m.real_is_zero(1e-8) and not m.real_is_zero(1.0))
    check("complex_is_zero", m.complex_is_zero(0j) and not m.complex_is_zero(1 + 0j))
    # Angular momenta are stored as 2x their physical value; step of two.
    check("triangular_condition true", m.triangular_condition(2, 2, 4))
    check("triangular_condition false", not m.triangular_condition(2, 2, 1))
    check("generate_list_of_primes",
          list(m.generate_list_of_primes(5)) == [2, 3, 5, 7, 11])
    check("levi_civita 123", m.levi_civita(1, 2, 3) == 1)
    check("levi_civita 213", m.levi_civita(2, 1, 3) == -1)
    check("levi_civita repeated", m.levi_civita(1, 1, 2) == 0)
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11 -> reconstruct via power-of-primes.
    idx = m.largest_prime(12)
    pp = m.power_of_primes_factorial(12, idx)
    check("power_of_primes_factorial(12!)",
          close(m.evaluate_power_of_primes(pp), math.factorial(12)))
    # log_factorial returns real(real128) values, for which f2py cannot
    # generate valid wrappers; compile.sh must exclude it from the Python
    # interface. It is tested indirectly through the cg tests below.
    check("log_factorial excluded from interface",
          not hasattr(m, "log_factorial"))


# ---------------------------------------------------------------------------
# matrix_utils.
# ---------------------------------------------------------------------------
def test_matrix_utils():
    print("matrix_utils")
    mu = fu.matrix_utils

    # [Sx, Sy] = 2 i Sz for Pauli matrices.
    c = mu.commutator(SX, SY)
    check("commutator [Sx,Sy]=2iSz", close(c, 2j * SZ))

    # Matrix product against numpy.
    prod = mu.complex_matrix_product(SX, SY, 'N', 'N')
    check("complex_matrix_product", close(prod, SX @ SY))
    # With Hermitian conjugation of the first factor.
    prodH = mu.complex_matrix_product(SY, SY, 'C', 'N')
    check("complex_matrix_product conj", close(prodH, SY.conj().T @ SY))

    # Hermiticity check.
    check("check_hermicity true", bool(mu.check_hermicity(SX)))
    check("check_hermicity false", not bool(mu.check_hermicity(SX + 1j * SZ)))

    # sigma_x^2 + sigma_y^2 + sigma_z^2 = 3 I.
    vec = np.stack([SX, SY, SZ]).astype(np.complex128)
    sq = mu.squared_vector_operator(vec)
    check("squared_vector_operator = 3I", close(sq, 3 * np.eye(2)))

    # Diagonalization against numpy (Hermitian).
    A = np.array([[2.0, 1.0 - 1j], [1.0 + 1j, 3.0]], dtype=np.complex128)
    val, vecm = mu.external_diagonalize_complex_matrix(A)
    check("external_diagonalize eigenvalues",
          close(np.sort(val), np.sort(np.linalg.eigvalsh(A))))
    # Reconstruct A from its spectral decomposition.
    recon = mu.spectral_representation(val, vecm)
    check("spectral_representation reconstructs A", close(recon, A))

    # Basis transformation with a unitary C: C^H A C (direction 0).
    C = vecm  # eigenvectors are unitary
    B0 = mu.basis_transformation(C, A, 0)
    check("basis_transformation diagonalizes A",
          close(np.diag(B0).real, val) and close(B0 - np.diag(np.diag(B0)), 0))

    # Hadamard (element-wise) product.
    had = mu.hadamard_product(SX, SZ)
    check("hadamard_product", close(had, SX * SZ))

    # Phase matrix: arg of each element.
    pm = mu.phase_matrix(SY)
    check("phase_matrix", close(pm, np.angle(SY)))

    # Real symmetric diagonalization against numpy. The matrix argument has
    # intent(inout) and is replaced by the eigenvectors, so it must be
    # passed as a Fortran-ordered array.
    rng = np.random.default_rng(1)
    R = rng.standard_normal((6, 6))
    R = R + R.T
    Rf = np.asfortranarray(R)
    vals = mu.diagonalize_real_symmetric_matrix(Rf, True)
    check("diagonalize_real_symmetric eigenvalues",
          close(vals, np.linalg.eigvalsh(R)))
    check("diagonalize_real_symmetric eigenvectors",
          close(R @ Rf, Rf * vals[None, :]))


# ---------------------------------------------------------------------------
# angm_utils : Wigner functions.
# ---------------------------------------------------------------------------
def test_angm_utils():
    print("angm_utils")
    a = fu.angm_utils
    beta = 0.7
    # Analytic Wigner d for j = 1 (arguments are 2x the physical value).
    check("d^1_00 = cos b",
          close(a.wigner_small_d(2, 0, 0, beta), math.cos(beta)))
    check("d^1_11 = (1+cos b)/2",
          close(a.wigner_small_d(2, 2, 2, beta), (1 + math.cos(beta)) / 2))
    check("d^1_1,-1 = (1-cos b)/2",
          close(a.wigner_small_d(2, 2, -2, beta), (1 - math.cos(beta)) / 2))
    check("d^1_10 = -sin b / sqrt2",
          close(a.wigner_small_d(2, 2, 0, beta), -math.sin(beta) / math.sqrt(2)))
    # Wigner big D: for M1=M2=0 the phase factor is 1.
    D = a.wigner_big_d(2, 0, 0, 0.3, beta, 0.5)
    check("wigner_big_d M=0", close(D, math.cos(beta) + 0j))
    # coupled_basis is a stub, just confirm it is callable.
    Jout, Mout = a.coupled_basis(np.ones((4, 2), dtype=np.int32),
                                 np.zeros((4, 2), dtype=np.int32))
    check("coupled_basis callable", Jout.shape == (4, 2))

    # ---- wigner_small_d (log-factorial implementation) vs the original
    # power-of-primes implementation, for all (M1, M2) of several J and
    # several angles including the exact end points 0 and pi.
    maxdiff = 0.0
    for tJ in (1, 2, 3, 4, 7, 10, 16, 24):
        for angle in (0.0, 0.3, 1.1, 2.7, math.pi):
            for tM1 in range(-tJ, tJ + 1, 2):
                for tM2 in range(-tJ, tJ + 1, 2):
                    maxdiff = max(maxdiff,
                                  abs(a.wigner_small_d(tJ, tM1, tM2, angle)
                                      - a.old_wigner_small_d(tJ, tM1, tM2, angle)))
    check("wigner_small_d agrees with old_wigner_small_d (2j <= 24)",
          maxdiff < 1e-10, f"maxdiff = {maxdiff:.3e}")

    # d(0) is the identity and d(pi) is the exact anti-diagonal
    # (-1)^(j-m2) delta_{m1,-m2}.
    ok = True
    for tJ in (3, 8):
        for tM1 in range(-tJ, tJ + 1, 2):
            for tM2 in range(-tJ, tJ + 1, 2):
                ok = ok and close(a.wigner_small_d(tJ, tM1, tM2, 0.0),
                                  1.0 if tM1 == tM2 else 0.0)
                exact = (-1) ** ((tJ - tM2) // 2) if tM1 == -tM2 else 0.0
                ok = ok and close(a.wigner_small_d(tJ, tM1, tM2, math.pi), exact)
    check("wigner_small_d at 0 and pi", ok)

    # Symmetry relations d_{m1,m2} = (-1)^(m1-m2) d_{m2,m1} = d_{-m2,-m1}
    # at half-integer J.
    tJ = 39
    rng = random.Random(7)
    ok = True
    for _ in range(60):
        tM1 = rng.randrange(-tJ, tJ + 1, 2)
        tM2 = rng.randrange(-tJ, tJ + 1, 2)
        d12 = a.wigner_small_d(tJ, tM1, tM2, 1.234)
        d21 = a.wigner_small_d(tJ, tM2, tM1, 1.234)
        dmm = a.wigner_small_d(tJ, -tM2, -tM1, 1.234)
        ok = ok and close(d12, (-1) ** ((tM1 - tM2) // 2) * d21) \
            and close(d12, dmm)
    check("wigner_small_d symmetries (2j = 39)", ok)

    # Unitarity of the d matrix rows, sum_m2 d^2 = 1, up to very large
    # angular momenta (the requirement is at least 2j = 40, i.e. j = 20;
    # the Jacobi-recurrence implementation stays stable far beyond that).
    ok = True
    for tJ, tM1 in [(40, 40), (40, 0), (41, 1), (80, 20), (200, 0),
                    (400, 0), (1000, 100)]:
        total = sum(a.wigner_small_d(tJ, tM1, tM2, 0.83) ** 2
                    for tM2 in range(-tJ, tJ + 1, 2))
        ok = ok and close(total, 1.0)
    check("wigner_small_d unitarity up to 2j = 1000", ok)

    # Group property: D(b1) D(b2) = D(b1 + b2) as matrices (rotations about
    # the same axis compose additively).
    tJ = 10
    b1, b2 = 0.6, 1.05
    ms = range(-tJ, tJ + 1, 2)
    D1 = np.array([[a.wigner_small_d(tJ, u, v, b1) for v in ms] for u in ms])
    D2 = np.array([[a.wigner_small_d(tJ, u, v, b2) for v in ms] for u in ms])
    D12 = np.array([[a.wigner_small_d(tJ, u, v, b1 + b2) for v in ms] for u in ms])
    check("wigner_small_d group property", close(D1 @ D2, D12))

    # The real128 helpers must be excluded from the Python interface.
    check("triangle_coefficient_log excluded from interface",
          not hasattr(a, "triangle_coefficient_log"))
    check("wigner_6j_r128 excluded from interface",
          not hasattr(a, "wigner_6j_r128"))

    # ---- Wigner 6j symbol (all arguments are 2x the physical value).
    # Special value {a b c; 0 c b} = (-1)^(a+b+c) / sqrt([b][c]).
    ok = True
    for (a6, b6, c6) in [(2, 2, 2), (1, 1, 2), (3, 1, 2), (4, 2, 2),
                         (6, 4, 4), (5, 3, 2), (0, 0, 0)]:
        exact = (-1) ** ((a6 + b6 + c6) // 2) / math.sqrt((b6 + 1) * (c6 + 1))
        ok = ok and close(a.wigner_6j(a6, b6, c6, 0, c6, b6), exact)
    check("wigner_6j special values {a b c; 0 c b}", ok)

    # A triad violating the triangular condition gives zero.
    check("wigner_6j triangle violation", a.wigner_6j(2, 2, 6, 2, 2, 2) == 0.0)

    # Orthogonality: sum_x [x] {j1 j2 x; j3 j4 p} {j1 j2 x; j3 j4 q}
    #                = delta_pq / [p].
    j1, j2, j3, j4 = 3, 2, 3, 2
    ok = True
    for p in range(1, 6, 2):
        for q in range(1, 6, 2):
            total = 0.0
            for x in range(0, 8):
                total += (x + 1) * a.wigner_6j(j1, j2, x, j3, j4, p) \
                                 * a.wigner_6j(j1, j2, x, j3, j4, q)
            exact = (1.0 / (p + 1)) if p == q else 0.0
            ok = ok and close(total, exact)
    check("wigner_6j orthogonality", ok)

    # Contraction of four Clebsch-Gordan coefficients (recoupling of three
    # angular momenta):
    #   sum_{m1 m2 m3} <j1 m1 j2 m2|j12 m12> <j12 m12 j3 m3|J M>
    #                  <j2 m2 j3 m3|j23 m23> <j1 m1 j23 m23|J M>
    #   = (-1)^(j1+j2+j3+J) sqrt([j12][j23]) {j1 j2 j12; j3 J j23}.
    cg = fu.cg_utils.cg
    ok = True
    for (j1, j2, j3, J) in [(2, 2, 2, 2), (1, 2, 3, 2), (3, 1, 2, 4)]:
        for j12 in range(abs(j1 - j2), j1 + j2 + 1, 2):
            for j23 in range(abs(j2 - j3), j2 + j3 + 1, 2):
                M = J
                total = 0.0
                for m1 in range(-j1, j1 + 1, 2):
                    for m2 in range(-j2, j2 + 1, 2):
                        m3 = M - m1 - m2
                        if abs(m3) > j3:
                            continue
                        m12 = m1 + m2
                        m23 = m2 + m3
                        if abs(m12) > j12 or abs(m23) > j23:
                            continue
                        total += cg(j1, m1, j2, m2, j12, m12) \
                            * cg(j12, m12, j3, m3, J, M) \
                            * cg(j2, m2, j3, m3, j23, m23) \
                            * cg(j1, m1, j23, m23, J, M)
                exact = (-1) ** ((j1 + j2 + j3 + J) // 2) \
                    * math.sqrt((j12 + 1) * (j23 + 1)) \
                    * a.wigner_6j(j1, j2, j12, j3, J, j23)
                ok = ok and close(total, exact)
    check("wigner_6j against CG contraction", ok)

    # ---- Wigner 9j symbol.
    # A violated row or column triad gives zero.
    check("wigner_9j triangle violation",
          a.wigner_9j(2, 2, 6, 2, 2, 2, 2, 2, 2) == 0.0
          and a.wigner_9j(2, 2, 2, 2, 2, 2, 6, 2, 2) == 0.0)

    # Defining relation: the 9j symbol is the recoupling coefficient of four
    # angular momenta,
    #   <(j1 j2)J12 (j3 j4)J34; J M | (j1 j3)J13 (j2 j4)J24; J M>
    #   = sqrt([J12][J34][J13][J24]) {j1 j2 J12; j3 j4 J34; J13 J24 J},
    # evaluated here by explicit summation of Clebsch-Gordan products.
    def _recoupling_overlap(js, J12, J34, J13, J24, J):
        j1, j2, j3, j4 = js
        M = J
        total = 0.0
        for m1 in range(-j1, j1 + 1, 2):
            for m2 in range(-j2, j2 + 1, 2):
                for m3 in range(-j3, j3 + 1, 2):
                    m4 = M - m1 - m2 - m3
                    if abs(m4) > j4:
                        continue
                    if abs(m1 + m2) > J12 or abs(m3 + m4) > J34:
                        continue
                    if abs(m1 + m3) > J13 or abs(m2 + m4) > J24:
                        continue
                    total += cg(j1, m1, j2, m2, J12, m1 + m2) \
                        * cg(j3, m3, j4, m4, J34, m3 + m4) \
                        * cg(J12, m1 + m2, J34, m3 + m4, J, M) \
                        * cg(j1, m1, j3, m3, J13, m1 + m3) \
                        * cg(j2, m2, j4, m4, J24, m2 + m4) \
                        * cg(J13, m1 + m3, J24, m2 + m4, J, M)
        return total

    ok = True
    for js, J in [((1, 1, 1, 1), 2), ((2, 1, 2, 1), 2), ((2, 2, 2, 4), 2),
                  ((3, 2, 1, 2), 4)]:
        j1, j2, j3, j4 = js
        for J12 in range(abs(j1 - j2), j1 + j2 + 1, 2):
            for J34 in range(abs(j3 - j4), j3 + j4 + 1, 2):
                if not abs(J12 - J34) <= J <= J12 + J34:
                    continue
                for J13 in range(abs(j1 - j3), j1 + j3 + 1, 2):
                    for J24 in range(abs(j2 - j4), j2 + j4 + 1, 2):
                        if not abs(J13 - J24) <= J <= J13 + J24:
                            continue
                        lhs = _recoupling_overlap(js, J12, J34, J13, J24, J)
                        rhs = math.sqrt((J12 + 1) * (J34 + 1)
                                        * (J13 + 1) * (J24 + 1)) \
                            * a.wigner_9j(j1, j2, J12, j3, j4, J34,
                                          J13, J24, J)
                        ok = ok and close(lhs, rhs)
    check("wigner_9j against CG recoupling", ok)

    # Symmetries: invariance under transposition; an odd permutation of
    # rows multiplies the symbol by (-1)^(sum of all nine arguments).
    rng = random.Random(11)
    ok = True
    trials = 0
    while trials < 20:
        js = [rng.randrange(0, 9) for _ in range(4)] + [0] * 5
        j1, j2, j4, j5 = js[:4]
        j3 = rng.randrange(abs(j1 - j2), j1 + j2 + 1, 2) if j1 + j2 else 0
        j6 = rng.randrange(abs(j4 - j5), j4 + j5 + 1, 2) if j4 + j5 else 0
        j7 = rng.randrange(abs(j1 - j4), j1 + j4 + 1, 2) if j1 + j4 else 0
        j8 = rng.randrange(abs(j2 - j5), j2 + j5 + 1, 2) if j2 + j5 else 0
        j9s = [x for x in range(abs(j3 - j6), j3 + j6 + 1, 2)
               if abs(j7 - j8) <= x <= j7 + j8]
        if not j9s:
            continue
        j9 = rng.choice(j9s)
        trials += 1
        w = a.wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9)
        wt = a.wigner_9j(j1, j4, j7, j2, j5, j8, j3, j6, j9)
        wp = a.wigner_9j(j4, j5, j6, j1, j2, j3, j7, j8, j9)
        sign = (-1) ** ((j1 + j2 + j3 + j4 + j5 + j6 + j7 + j8 + j9) // 2)
        ok = ok and close(w, wt) and close(wp, sign * w)
    check("wigner_9j symmetries", ok)

    # Orthogonality (unitarity of the recoupling):
    #   sum_{J13 J24} [J13][J24][J12][J34] {9j}^2 = 1,
    # also probed at large angular momenta for numerical stability.
    ok = True
    for (j1, j2, j3, j4, J12, J34, J) in [(2, 2, 2, 2, 2, 2, 2),
                                          (3, 2, 1, 2, 3, 3, 4),
                                          (20, 20, 20, 20, 20, 20, 20),
                                          (41, 40, 39, 40, 41, 39, 30)]:
        total = 0.0
        for J13 in range(abs(j1 - j3), j1 + j3 + 1, 2):
            for J24 in range(abs(j2 - j4), j2 + j4 + 1, 2):
                total += (J13 + 1) * (J24 + 1) * (J12 + 1) * (J34 + 1) \
                    * a.wigner_9j(j1, j2, J12, j3, j4, J34, J13, J24, J) ** 2
        ok = ok and close(total, 1.0)
    check("wigner_9j orthogonality up to 2j = 41", ok)


# ---------------------------------------------------------------------------
# cg_utils : Clebsch-Gordan coefficients.
# ---------------------------------------------------------------------------
def exact_cg(tj1, tm1, tj2, tm2, tJ, tM):
    """Exact CG coefficient <j1 m1; j2 m2|J M> from Racah's formula using
    rational arithmetic; the arguments are doubled integers as in old_cg/cg.
    Returns a Decimal accurate to the decimal context precision."""
    if tm1 + tm2 != tM:
        return Decimal(0)
    if not (abs(tj1 - tj2) <= tJ <= tj1 + tj2) or (tj1 + tj2 + tJ) % 2:
        return Decimal(0)
    if abs(tm1) > tj1 or abs(tm2) > tj2 or abs(tM) > tJ:
        return Decimal(0)
    if (tj1 + tm1) % 2 or (tj2 + tm2) % 2 or (tJ + tM) % 2:
        return Decimal(0)
    fac = math.factorial
    a = (tj1 + tj2 - tJ) // 2
    b = (tj1 - tm1) // 2
    cc = (tj2 + tm2) // 2
    d = (tJ - tj2 + tm1) // 2
    e = (tJ - tj1 - tm2) // 2
    S = Fraction(0)
    for nu in range(max(0, -d, -e), min(a, b, cc) + 1):
        S += Fraction((-1) ** nu,
                      fac(nu) * fac(a - nu) * fac(b - nu) * fac(cc - nu)
                      * fac(d + nu) * fac(e + nu))
    if S == 0:
        return Decimal(0)
    pref = Fraction(
        (tJ + 1) * fac(a) * fac((tj1 - tj2 + tJ) // 2)
        * fac((tj2 - tj1 + tJ) // 2) * fac((tj1 + tm1) // 2) * fac(b)
        * fac(cc) * fac((tj2 - tm2) // 2) * fac((tJ + tM) // 2)
        * fac((tJ - tM) // 2), fac((tj1 + tj2 + tJ) // 2 + 1))
    val2 = pref * S * S
    mag = (Decimal(val2.numerator) / Decimal(val2.denominator)).sqrt()
    return mag if S > 0 else -mag


def random_valid_cg_args(two_j_max, rng):
    """Random valid (doubled) CG argument tuple with 2j up to two_j_max."""
    tj1 = rng.randint(0, two_j_max)
    tj2 = rng.randint(0, two_j_max)
    tJ = rng.choice(range(abs(tj1 - tj2), tj1 + tj2 + 1, 2))
    tm1 = rng.choice(range(-tj1, tj1 + 1, 2))
    lo, hi = max(-tj2, -tJ - tm1), min(tj2, tJ - tm1)
    if lo > hi:
        return None
    tm2 = rng.choice(range(lo, hi + 1, 2))
    return tj1, tm1, tj2, tm2, tJ, tm1 + tm2


def test_cg_utils():
    print("cg_utils")
    c = fu.cg_utils
    s2 = 1.0 / math.sqrt(2.0)
    # <1/2 1/2; 1/2 -1/2 | 1 0> = 1/sqrt(2)   (arguments are 2x physical).
    check("old_cg <1/2,1/2;1/2,-1/2|1,0>", close(c.old_cg(1, 1, 1, -1, 2, 0), s2))
    # <1/2 1/2; 1/2 1/2 | 1 1> = 1.
    check("old_cg <1/2,1/2;1/2,1/2|1,1>", close(c.old_cg(1, 1, 1, 1, 2, 2), 1.0))
    # <1 0; 1 0 | 2 0> = sqrt(2/3).
    check("old_cg <1,0;1,0|2,0>", close(c.old_cg(2, 0, 2, 0, 4, 0), math.sqrt(2 / 3)))
    # Projection non-conservation -> 0.
    check("old_cg selection rule", close(c.old_cg(1, 1, 1, 1, 2, 0), 0.0))
    # special_cg(J,0) corresponds to <J J; 0 0 | J J> = 1.
    check("special_cg(J,0)=1", close(c.special_cg(4, 0), 1.0))

    # --- cg: same analytic values as old_cg above.
    check("cg <1/2,1/2;1/2,-1/2|1,0>", close(c.cg(1, 1, 1, -1, 2, 0), s2))
    check("cg <1/2,1/2;1/2,1/2|1,1>", close(c.cg(1, 1, 1, 1, 2, 2), 1.0))
    check("cg <1,0;1,0|2,0>",
          close(c.cg(2, 0, 2, 0, 4, 0), math.sqrt(2 / 3)))

    # cg selection rules: projection, triangularity, |m| > j and a
    # mismatch in the integer/half-integer character of m and j.
    check("cg selection rules",
          c.cg(1, 1, 1, 1, 2, 0) == 0.0
          and c.cg(2, 0, 2, 0, 8, 0) == 0.0
          and c.cg(2, 4, 2, -4, 4, 0) == 0.0
          and c.cg(2, 1, 2, 1, 4, 2) == 0.0)

    # Exhaustive agreement with the exact rational value for all valid
    # arguments with 2j1, 2j2 <= 6.
    maxerr = 0.0
    for tj1, tj2 in itertools.product(range(7), repeat=2):
        for tJ in range(abs(tj1 - tj2), tj1 + tj2 + 1, 2):
            for tm1 in range(-tj1, tj1 + 1, 2):
                for tm2 in range(-tj2, tj2 + 1, 2):
                    if abs(tm1 + tm2) > tJ:
                        continue
                    got = c.cg(tj1, tm1, tj2, tm2, tJ, tm1 + tm2)
                    ref = float(exact_cg(tj1, tm1, tj2, tm2, tJ, tm1 + tm2))
                    maxerr = max(maxerr, abs(got - ref)
                                 / (abs(ref) if ref else 1.0))
    check("cg exhaustive vs exact (2j <= 6)", maxerr < 1e-14,
          f"max rel err {maxerr:.3e}")

    # Random agreement with the exact value and with old_cg in its trusted
    # region (all angular momenta <= 12, i.e. doubled arguments <= 24).
    rng = random.Random(1)
    maxerr = maxdiff = 0.0
    n = 0
    while n < 150:
        args = random_valid_cg_args(24, rng)
        if args is None:
            continue
        n += 1
        got, old, ref = c.cg(*args), c.old_cg(*args), float(exact_cg(*args))
        maxerr = max(maxerr, abs(got - ref) / (abs(ref) if ref else 1.0))
        if abs(ref) > 1e-8:  # old_cg returns noise instead of 0 at exact zeros
            maxdiff = max(maxdiff, abs(got - old) / abs(old))
    check("cg random vs exact (2j <= 24)", maxerr < 1e-14,
          f"max rel err {maxerr:.3e}")
    check("cg agrees with old_cg (2j <= 24)", maxdiff < 1e-10,
          f"max rel diff {maxdiff:.3e}")

    # Large angular momenta: random arguments at j up to 50 and small-m
    # arguments at j ~ 200 that force severe cancellation (the latter
    # exercise the internal recursion fallback of cg).
    maxerr = 0.0
    n = 0
    while n < 40:
        args = random_valid_cg_args(100, rng)
        if args is None:
            continue
        n += 1
        got, ref = c.cg(*args), float(exact_cg(*args))
        maxerr = max(maxerr, abs(got - ref) / (abs(ref) if ref else 1.0))
    check("cg random vs exact (2j <= 100)", maxerr < 1e-13,
          f"max rel err {maxerr:.3e}")

    maxerr = 0.0
    for args in [(400, 0, 400, 2, 246, 2), (390, -4, 400, 4, 146, 0),
                 (400, -6, 398, 6, 336, 0), (401, 1, 399, -1, 284, 0),
                 (743, 467, 591, -225, 780, 242)]:
        got, ref = c.cg(*args), float(exact_cg(*args))
        maxerr = max(maxerr, abs(got - ref) / (abs(ref) if ref else 1.0))
    check("cg cancellation fallback (j ~ 200-400)", maxerr < 1e-13,
          f"max rel err {maxerr:.3e}")

    # Orthogonality at j = 20: sum over m1 of <j1 m1 j2 M-m1|J M>^2 = 1.
    tj1 = tj2 = tJ = 40
    total = sum(c.cg(tj1, tm1, tj2, -tm1, tJ, 0) ** 2
                for tm1 in range(-tj1, tj1 + 1, 2))
    check("cg orthogonality (j = 20)", close(total, 1.0))

    # Agreement with special_cg, <J J; k 0|J J>, at large J.
    maxdiff = max(abs(c.cg(tJ, tJ, tk, 0, tJ, tJ) - c.special_cg(tJ, tk))
                  for tJ in (10, 40, 80) for tk in range(0, 21, 2))
    check("cg agrees with special_cg", maxdiff < 1e-13,
          f"max abs diff {maxdiff:.3e}")

    # cg_by_m_recursion (the internal fallback of cg, exposed as a
    # module function) must reproduce the exact values on its own. It
    # assumes valid arguments, so only such arguments are used here.
    maxerr = 0.0
    n = 0
    while n < 30:
        args = random_valid_cg_args(60, rng)
        if args is None:
            continue
        n += 1
        got, ref = c.cg_by_m_recursion(*args), float(exact_cg(*args))
        maxerr = max(maxerr, abs(got - ref) / (abs(ref) if ref else 1.0))
    check("cg_by_m_recursion vs exact (2j <= 60)", maxerr < 1e-13,
          f"max rel err {maxerr:.3e}")
    # The quadruple-precision recursion coefficient functions must be
    # excluded from the Python interface by compile.sh (as for
    # log_factorial in math_utils).
    check("cg_recursion_a/b/d excluded from interface",
          not any(hasattr(c, name) for name in
                  ("cg_recursion_a", "cg_recursion_b", "cg_recursion_d")))


# ---------------------------------------------------------------------------
# cg_utils : generalized sums of products of CG coefficients (cg_product).
# ---------------------------------------------------------------------------
def cg_product_reference(param_ranges, phase_vector, f, cg_map):
    """Straightforward Python reference implementation of cg_product: loop
    over the full rectangular parameter grid (doubled integers, step one),
    multiply the CG coefficients (via cg) and the weight f, apply the
    phase and accumulate. Used to validate the optimized Fortran routine."""
    c = fu.cg_utils
    total = 0.0
    ranges = [range(lo, hi + 1) for lo, hi in param_ranges]
    for p in itertools.product(*ranges):
        prod = 1.0
        for row in cg_map:
            prod *= c.cg(*(p[i - 1] for i in row))
            if prod == 0.0:
                break
        if prod == 0.0:
            continue
        fv = f(np.array(p, dtype=np.int32))
        if fv == 0.0:
            continue
        s = sum(ph * pi for ph, pi in zip(phase_vector, p))
        assert s % 2 == 0, "half-integer phase in a non-zero reference term"
        total += (1.0 if s % 4 == 0 else -1.0) * prod * fv
    return total


def one(p):
    """Constant unit weight for cg_product."""
    return 1.0


def test_cg_product():
    print("cg_utils.cg_product")
    c = fu.cg_utils
    i32 = np.int32

    # --- cg_is_allowed: the cheap selection-rule test used for pruning.
    check("cg_is_allowed accepts valid args",
          bool(c.cg_is_allowed(1, 1, 1, -1, 2, 0)))
    check("cg_is_allowed selection rules",
          not any(bool(c.cg_is_allowed(*args)) for args in
                  [(1, 1, 1, 1, 2, 0),    # projection non-conservation
                   (2, 0, 2, 0, 8, 0),    # triangular condition
                   (2, 0, 2, 0, 3, 0),    # integer character of j1+j2+J
                   (2, 4, 2, -4, 4, 0),   # |m| > j
                   (2, 1, 2, 1, 4, 2),    # m/j character mismatch
                   (-2, 0, 2, 0, 4, 0)])) # negative momentum

    # --- A single CG coefficient with all parameters constant.
    pr = np.array([[1, 1], [1, 1], [1, 1], [-1, -1], [2, 2], [0, 0]], dtype=i32)
    ph = np.zeros(6, dtype=i32)
    cm = np.array([[1, 2, 3, 4, 5, 6]], dtype=i32)
    check("single constant CG",
          close(c.cg_product(pr, ph, one, cm, False),
                1.0 / math.sqrt(2.0), tol=1e-14))

    # --- A constant CG coefficient violating a selection rule kills the
    # whole sum; f must then never be called.
    calls = []
    pr = np.array([[1, 1], [1, 1], [1, 1], [1, 1], [2, 2], [0, 0]], dtype=i32)
    val = c.cg_product(pr, ph, lambda p: calls.append(1) or 1.0, cm, False)
    check("vanishing constant CG short-circuits",
          val == 0.0 and not calls)

    # --- Orthogonality: sum_{m1,m2} <j1 m1 j2 m2|J M>^2 = 1. Two identical
    # coefficients through the same parameter mapping. Parameters: 1: j1,
    # 2: j2, 3: J, 4: M (constants), 5: m1, 6: m2 (full rectangular ranges).
    for tj1, tj2, tJ, tM in [(2, 2, 4, 0), (3, 1, 2, 2), (8, 6, 10, -2)]:
        pr = np.array([[tj1, tj1], [tj2, tj2], [tJ, tJ], [tM, tM],
                       [-tj1, tj1], [-tj2, tj2]], dtype=i32)
        ph = np.zeros(6, dtype=i32)
        cm = np.array([[1, 5, 2, 6, 3, 4], [1, 5, 2, 6, 3, 4]], dtype=i32)
        val = c.cg_product(pr, ph, one, cm, False)
        check(f"orthogonality 2j=({tj1},{tj2})", close(val, 1.0, tol=1e-13))
    # Different coupled states (J,M) vs (J',M') are orthogonal: parameters
    # 3,4 -> (J,M) of the first coefficient, 7,8 -> (J',M') of the second.
    pr = np.array([[2, 2], [2, 2], [4, 4], [0, 0],
                   [-2, 2], [-2, 2], [2, 2], [0, 0]], dtype=i32)
    ph = np.zeros(8, dtype=i32)
    cm = np.array([[1, 5, 2, 6, 3, 4], [1, 5, 2, 6, 7, 8]], dtype=i32)
    check("orthogonality J /= J'",
          close(c.cg_product(pr, ph, one, cm, False), 0.0, tol=1e-14))

    # --- Completeness: sum_{J,M} <j1 m1 j2 m2|J M><j1 m1' j2 m2'|J M> =
    # delta_{m1 m1'} delta_{m2 m2'}. J and M are summed over rectangular
    # ranges (J: 0...j1+j2, M: -(j1+j2)...j1+j2 in steps of one doubled
    # unit); the invalid combinations are pruned by the selection rules.
    tj1, tj2 = 4, 4
    for (tm1, tm2, tm1p, tm2p, expected) in [(2, -4, 2, -4, 1.0),
                                             (2, -4, 0, -2, 0.0)]:
        pr = np.array([[tj1, tj1], [tj2, tj2], [tm1, tm1], [tm2, tm2],
                       [tm1p, tm1p], [tm2p, tm2p],
                       [0, tj1 + tj2], [-(tj1 + tj2), tj1 + tj2]], dtype=i32)
        ph = np.zeros(8, dtype=i32)
        cm = np.array([[1, 3, 2, 4, 7, 8], [1, 5, 2, 6, 7, 8]], dtype=i32)
        val = c.cg_product(pr, ph, one, cm, False)
        check(f"completeness m'=({tm1p},{tm2p})",
              close(val, expected, tol=1e-13))

    # --- Phase factor: sum_m (-1)^(j-m) <j m; j -m|0 0> = sqrt(2j+1),
    # since <j m; j -m|0 0> = (-1)^(j-m)/sqrt(2j+1). Parameters: 1: j,
    # 2: m1, 3: m2, 4: the constant 0 reused for both J and M. The phase
    # exponent j - m is half-integer whenever the parity of m1 does not
    # match that of j, but those terms vanish by the selection rules, so
    # no error may be raised. Both integer and half-integer j are used.
    for tj in (4, 3):
        pr = np.array([[tj, tj], [-tj, tj], [-tj, tj], [0, 0]], dtype=i32)
        ph = np.array([1, -1, 0, 0], dtype=i32)
        cm = np.array([[1, 2, 1, 3, 4, 4]], dtype=i32)
        val = c.cg_product(pr, ph, one, cm, False)
        check(f"phase identity 2j={tj}", close(val, math.sqrt(tj + 1.0),
                                               tol=1e-13))

    # --- Weight function f: sum_m <j m; j -m|0 0>^2 m^2 = j(j+1)/3 (the
    # doubled m is p[1]; the physical value is half of it). Both leaf
    # evaluation orders must give the same result.
    def m_squared(p):
        return (0.5 * p[1]) ** 2

    tj = 4
    pr = np.array([[tj, tj], [-tj, tj], [-tj, tj], [0, 0]], dtype=i32)
    ph = np.zeros(4, dtype=i32)
    cm = np.array([[1, 2, 1, 3, 4, 4], [1, 2, 1, 3, 4, 4]], dtype=i32)
    expected = 0.5 * tj * (0.5 * tj + 1.0) / 3.0
    val_cg_first = c.cg_product(pr, ph, m_squared, cm, False)
    val_f_first = c.cg_product(pr, ph, m_squared, cm, True)
    check("f weight, CG first", close(val_cg_first, expected, tol=1e-13))
    check("f weight, f first", close(val_f_first, expected, tol=1e-13))

    # --- The f_first switch controls how often f is called: with
    # f_first=False f is only called for terms whose full CG product is
    # non-zero; with f_first=True it is called at every term whose hoisted
    # (outer-level) CG product is non-zero. The parameters are ordered so
    # that the coefficients live on the innermost level (the constant 0 is
    # parameter 2, the varying m1 and m2 are parameters 3 and 4), which
    # leaves the outer levels without coefficients: f is then called at
    # every grid point.
    def counting(counter):
        def g(p):
            counter[0] += 1
            return 1.0
        return g

    pr_leaf = np.array([[tj, tj], [0, 0], [-tj, tj], [-tj, tj]], dtype=i32)
    cm_leaf = np.array([[1, 3, 1, 4, 2, 2], [1, 3, 1, 4, 2, 2]], dtype=i32)
    n_cg_first, n_f_first = [0], [0]
    val_cg_first = c.cg_product(pr_leaf, ph, counting(n_cg_first), cm_leaf, False)
    val_f_first = c.cg_product(pr_leaf, ph, counting(n_f_first), cm_leaf, True)
    n_grid = (2 * tj + 1) ** 2
    n_valid = tj + 1  # m2 = -m1 with matching parity
    check("f call counts",
          n_cg_first[0] == n_valid and n_f_first[0] == n_grid,
          f"CG first {n_cg_first[0]} (expected {n_valid}), "
          f"f first {n_f_first[0]} (expected {n_grid})")
    check("f call order does not change the value",
          close(val_cg_first, 1.0, tol=1e-13)
          and close(val_f_first, 1.0, tol=1e-13))

    # --- Validation against the brute-force Python reference for
    # structures exercising several hoisting levels, phases and weights.
    # (a) Recoupling contraction of four CG coefficients with all
    # projections (m1, m2, m3, m12, m23) summed: proportional to a 6j
    # symbol, computed here by brute force. Parameters: 1: j1, 2: j2,
    # 3: j3, 4: j12, 5: j23, 6: J, 7: M (constants), 8: m1, 9: m2, 10: m3,
    # 11: m12, 12: m23.
    consts = [2, 2, 2, 4, 2, 2, 0]           # j1 j2 j3 j12 j23 J M (doubled)
    pr = np.array([[v, v] for v in consts]
                  + [[-2, 2]] * 3 + [[-4, 4], [-2, 2]], dtype=i32)
    ph = np.zeros(12, dtype=i32)
    cm = np.array([[1, 8, 2, 9, 4, 11],      # <j1 m1 j2 m2|j12 m12>
                   [4, 11, 3, 10, 6, 7],     # <j12 m12 j3 m3|J M>
                   [2, 9, 3, 10, 5, 12],     # <j2 m2 j3 m3|j23 m23>
                   [1, 8, 5, 12, 6, 7]], dtype=i32)  # <j1 m1 j23 m23|J M>
    got = c.cg_product(pr, ph, one, cm, False)
    ref = cg_product_reference(pr, ph, one, cm)
    check("recoupling vs reference", close(got, ref, tol=1e-13),
          f"got {got}, reference {ref}")
    check("recoupling non-trivial", abs(ref) > 1e-3)

    # (b) Coefficients living on different hoisting levels plus a phase
    # and a weight: C1 depends only on m1, C2 on m1 and m2. The phase
    # (-1)^(2*m1) alternates sign for odd doubled m1 and the weight is a
    # polynomial in the parameters. Parameters: 1: j1, 2: k, 3: 0, 4: j2,
    # 5: J, 6: M (constants), 7: m1, 8: m2.
    def weight(p):
        return 1.0 + 0.5 * p[6] + 0.25 * p[7] ** 2

    consts = [4, 4, 0, 2, 2, 0]              # j1 k 0 j2 J M (doubled)
    pr = np.array([[v, v] for v in consts] + [[-4, 4], [-2, 2]], dtype=i32)
    ph = np.array([0, 0, 0, 0, 0, 0, 2, 0], dtype=i32)
    cm = np.array([[1, 7, 2, 3, 1, 7],       # <j1 m1 k 0|j1 m1>
                   [1, 7, 4, 8, 5, 6]], dtype=i32)  # <j1 m1 j2 m2|J M>
    got = c.cg_product(pr, ph, weight, cm, False)
    ref = cg_product_reference(pr, ph, weight, cm)
    check("multi-level with phase and weight", close(got, ref, tol=1e-13),
          f"got {got}, reference {ref}")
    check("multi-level non-trivial", abs(ref) > 1e-3)

    # --- Numerical stability at large angular momenta (j = 40).
    tj = 80
    pr = np.array([[tj, tj], [-tj, tj], [-tj, tj], [0, 0]], dtype=i32)
    ph = np.zeros(4, dtype=i32)
    cm = np.array([[1, 2, 1, 3, 4, 4], [1, 2, 1, 3, 4, 4]], dtype=i32)
    val = c.cg_product(pr, ph, m_squared, cm, False)
    expected = 0.5 * tj * (0.5 * tj + 1.0) / 3.0
    check("large-j weighted sum (j = 40)",
          abs(val - expected) / expected < 1e-13,
          f"rel err {abs(val - expected) / expected:.3e}")
    tj1 = tj2 = 40
    pr = np.array([[tj1, tj1], [tj2, tj2], [2, 2], [0, 0],
                   [0, tj1 + tj2], [-(tj1 + tj2), tj1 + tj2]], dtype=i32)
    ph = np.zeros(6, dtype=i32)
    cm = np.array([[1, 3, 2, 4, 5, 6], [1, 3, 2, 4, 5, 6]], dtype=i32)
    check("large-j completeness (j = 20)",
          close(c.cg_product(pr, ph, one, cm, False), 1.0, tol=1e-13))

    # --- A half-integer phase exponent in a non-vanishing term must abort
    # with an error message. The Fortran error termination kills the
    # process, so this is checked in a subprocess: <j m; j -m|0 0> with
    # j = 1/2 and the phase (-1)^m, giving exponents of +-1/2 in the two
    # non-zero terms.
    code = (
        "import numpy as np\n"
        "from ouluspin._fortran import fortran_utils as fu\n"
        "pr = np.array([[1, 1], [-1, 1], [-1, 1], [0, 0]], dtype=np.int32)\n"
        "ph = np.array([0, 1, 0, 0], dtype=np.int32)\n"
        "cm = np.array([[1, 2, 1, 3, 4, 4]], dtype=np.int32)\n"
        "fu.cg_utils.cg_product(pr, ph, lambda p: 1.0, cm, False)\n"
    )
    env = dict(os.environ)
    src_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(fu.__file__))))
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, env=env)
    check("half-integer phase aborts",
          result.returncode != 0 and "ERROR" in result.stdout + result.stderr,
          f"returncode {result.returncode}")


# ---------------------------------------------------------------------------
# iwahara_utils.
# ---------------------------------------------------------------------------
def test_iwahara_utils():
    print("iwahara_utils")
    iw = fu.iwahara_utils
    # k = q = 0 with J1 == J2 is the identity operator element -> 1.
    check("iwahara_operator identity", close(iw.iwahara_operator(4, 2, 4, 2, 0, 0), 1.0))

    # Single-site rank-0 operator over a J = 1/2 (2x -> 1) doublet basis:
    # should give the identity matrix.
    J_array = np.array([[1], [1]], dtype=np.int32)   # J = 1/2 for both states
    M_array = np.array([[1], [-1]], dtype=np.int32)  # M = +1/2, -1/2
    k_vec = np.array([0], dtype=np.int32)
    q_vec = np.array([0], dtype=np.int32)
    op = iw.construct_iwahara_operator_matrix_for_one_rank(
        J_array, M_array, k_vec, q_vec, only_triangle=False)
    check("iwahara rank-0 matrix = I", close(op, np.eye(2)))

    # General constructor with a single rank and unit parameter reproduces it.
    k_array = np.array([[0]], dtype=np.int32)
    q_array = np.array([[0]], dtype=np.int32)
    params = np.array([1.0 + 0j], dtype=np.complex128)
    gen = iw.construct_general_iwahara_operator_matrix(
        J_array, M_array, k_array, q_array, params, only_triangle=False)
    check("general iwahara matrix = I", close(gen, np.eye(2)))


# ---------------------------------------------------------------------------
# statmech_utils.
# ---------------------------------------------------------------------------
def test_statmech_utils():
    print("statmech_utils")
    sm = fu.statmech_utils
    check("boltzmann_factor", close(sm.boltzmann_factor(2.0, 1.0, 1.0), math.exp(-0.5)))
    E = np.array([0.0, 1.0, 2.0])
    check("canonical_partition_function",
          close(sm.canonical_partition_function(1.5, E, 1.0),
                np.sum(np.exp(-E / 1.5))))

    # Expectation value with the eigenbasis == computational basis (C = I).
    eig = np.array([0.0, 1.0])
    A = np.diag([1.0, -1.0]).astype(np.complex128)
    C = np.eye(2, dtype=np.complex128)
    T = np.array([0.5, 1.0, 5.0])
    kB = 1.0
    val = sm.equilibrium_expectation_value(A, eig, C, T, kB, normalize=1)
    expected = (1.0 * np.exp(0.0) + (-1.0) * np.exp(-1.0 / T)) / \
               (np.exp(0.0) + np.exp(-1.0 / T))
    check("equilibrium_expectation_value", close(val, expected))


# ---------------------------------------------------------------------------
# powder_magnetization_utils.
# ---------------------------------------------------------------------------
def test_powder_magnetization_utils():
    print("powder_magnetization_utils")
    pm = fu.powder_magnetization_utils

    grid_v, grid_w = fu.grid_utils.generate_lebedev_laikov_grid(50)

    # Isotropic spin-1/2: H = 0, magnetic moment = Pauli matrices. The powder
    # magnetization must equal tanh(B / (kB T)).
    H = np.zeros((2, 2), dtype=np.complex128)
    # The Cartesian component of the magnetic moment operator is the last
    # index of the array, i.e. the operators are passed as (n, n, 3).
    moment = np.stack([SX, SY, SZ], axis=-1).astype(np.complex128)
    T = np.array([0.5, 1.0, 2.0, 10.0])
    B = 0.8
    kB = 1.0
    M = pm.powder_magnetization(H, moment, T, B, grid_v, grid_w, kB)
    check("powder_magnetization = tanh(B/kT)",
          close(M, np.tanh(B / (kB * T)), tol=1e-8))

    # diagonalize_hamiltonian: field along z gives eigenvalues ∓B (translated).
    val, vec = pm.diagonalize_hamiltonian(
        H, moment, np.array([0.0, 0.0, B]),
        compute_eigenvectors=1, translate_eigenvalues=1)
    check("diagonalize_hamiltonian eigenvalues",
          close(np.sort(val), np.array([0.0, 2 * B])))

    # free_rotation_magnetization must also run and stay in [-1, 1] here.
    Mfree = pm.free_rotation_magnetization(H, moment, T, B, grid_v, grid_w, kB)
    check("free_rotation_magnetization sane",
          np.all(np.abs(Mfree) <= 1.0 + 1e-9))

    # maximum_magnetization: for the isotropic case the max equals tanh(B/kT).
    maxM, maxvec = pm.maximum_magnetization(H, moment, T, B, grid_v, kB)
    check("maximum_magnetization = tanh(B/kT)",
          close(maxM, np.tanh(B / (kB * T)), tol=1e-2))

    # project_magnetic_moment: mu(n) = sum_j n_j mu_j.
    projected = pm.project_magnetic_moment(moment, np.array([0.0, 0.0, 1.0]))
    check("project_magnetic_moment along z", close(projected, SZ))

    projected = pm.project_magnetic_moment(moment, np.array([1.0, 2.0, 3.0]))
    check("project_magnetic_moment of a general direction",
          close(projected, SX + 2.0 * SY + 3.0 * SZ))

    # diagonalize_projected_hamiltonian: the same field as above, given as
    # the projected operator and the magnitude of the field.
    val2, vec2 = pm.diagonalize_projected_hamiltonian(
        H, SZ.astype(np.complex128), B,
        compute_eigenvectors=1, translate_eigenvalues=1)
    check("diagonalize_projected_hamiltonian eigenvalues",
          close(np.sort(val2), np.array([0.0, 2 * B])))


# ---------------------------------------------------------------------------
# pseudospin_utils.
# ---------------------------------------------------------------------------
def test_pseudospin_utils():
    print("pseudospin_utils")
    ps = fu.pseudospin_utils
    # Build a 4-level system; project onto the lowest 2 states.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    H = X + X.conj().T                      # Hermitian
    Y = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    mu = Y + Y.conj().T                     # Hermitian
    H = np.asarray(H, dtype=np.complex128)
    mu = np.asarray(mu, dtype=np.complex128)
    H_eff = ps.pseudospin_transformation(H, mu, 2)
    check("pseudospin_transformation shape", H_eff.shape == (2, 2))
    check("pseudospin_transformation Hermitian",
          close(H_eff, H_eff.conj().T, tol=1e-8))


# ---------------------------------------------------------------------------
# grid_utils : Lebedev-Laikov grids.
# ---------------------------------------------------------------------------
def test_grid_utils():
    print("grid_utils")
    g = fu.grid_utils
    for n in (6, 50, 194, 590):
        v, w = g.generate_lebedev_laikov_grid(n)
        ok = (v.shape == (n, 3) and w.shape == (n,)
              and close(w.sum(), 1.0)
              and close(np.linalg.norm(v, axis=1), np.ones(n)))
        check(f"lebedev grid n={n}", ok)
    # A spherical harmonic integrates to zero; the constant integrates to 1.
    v, w = g.generate_lebedev_laikov_grid(194)
    check("grid integrates constant", close(np.sum(w), 1.0))
    check("grid integrates z to 0", close(np.sum(w * v[:, 2]), 0.0, tol=1e-10))


# ---------------------------------------------------------------------------
# cfp_utils : coefficients of fractional parentage.
#
# In addition to checks against known analytic values and classification
# labels, the CFP magnitudes are verified against a completely independent
# brute-force construction in the Slater-determinant m-scheme: the
# antisymmetric term states of l^n and l^(n-1) are found as simultaneous
# eigenvectors of S^2 and L^2 in the determinant basis, the parent term is
# coupled to one additional electron with Clebsch-Gordan coefficients and
# the overlap with the l^n term state gives |cfp| * sqrt(n). This works for
# any term whose (S, L) is unique in its configuration (so that the
# m-scheme eigenstate is unambiguous).
# ---------------------------------------------------------------------------
class _MScheme:
    """Slater-determinant (m-scheme) model of the configuration l^n."""

    def __init__(self, l, n):
        self.l = l
        self.n = n
        # Spin-orbitals (ml, ms) with ms twice its physical value.
        self.orbs = [(ml, ms) for ml in range(-l, l + 1) for ms in (1, -1)]
        self.no = len(self.orbs)
        self.dets = list(itertools.combinations(range(self.no), n))
        self.index = {d: i for i, d in enumerate(self.dets)}
        self.dim = len(self.dets)

        pos = {o: i for i, o in enumerate(self.orbs)}
        lz = np.zeros((self.no, self.no))
        lp = np.zeros((self.no, self.no))
        sz = np.zeros((self.no, self.no))
        sp = np.zeros((self.no, self.no))
        for i, (ml, ms) in enumerate(self.orbs):
            lz[i, i] = ml
            sz[i, i] = 0.5 * ms
            if ml < l:
                lp[pos[(ml + 1, ms)], i] = math.sqrt(l * (l + 1) - ml * (ml + 1))
            if ms == -1:
                sp[pos[(ml, 1)], i] = 1.0
        one = np.eye(self.dim)
        self.Lz = self._onebody(lz)
        self.Sz = self._onebody(sz)
        Lp = self._onebody(lp)
        Sp = self._onebody(sp)
        self.Lm = Lp.T
        self.Sm = Sp.T
        self.L2 = Lp.T @ Lp + self.Lz @ (self.Lz + one)
        self.S2 = Sp.T @ Sp + self.Sz @ (self.Sz + one)

    def _onebody(self, c):
        """Matrix of the one-body operator sum_ab c[a,b] adag_a a_b."""
        M = np.zeros((self.dim, self.dim))
        for di, det in enumerate(self.dets):
            for pb, b in enumerate(det):
                sgn_b = (-1) ** pb
                rest = det[:pb] + det[pb + 1:]
                for a in np.nonzero(c[:, b])[0]:
                    if a in rest:
                        continue
                    na = sum(1 for x in rest if x < a)
                    nd = tuple(sorted(rest + (int(a),)))
                    M[self.index[nd], di] += c[a, b] * sgn_b * ((-1) ** na)
        return M

    def highest_weight_state(self, two_s, two_ll):
        """The unique term state with ML = L, MS = S (fails if not unique)."""
        S = two_s / 2.0
        L = two_ll / 2.0
        blk = [i for i in range(self.dim)
               if abs(2 * self.Lz[i, i] - two_ll) < 1e-9
               and abs(2 * self.Sz[i, i] - two_s) < 1e-9]
        A = (self.L2 - L * (L + 1) * np.eye(self.dim))[np.ix_(blk, blk)]
        B = (self.S2 - S * (S + 1) * np.eye(self.dim))[np.ix_(blk, blk)]
        K = A.T @ A + B.T @ B
        w, v = np.linalg.eigh(K)
        assert w[0] < 1e-9, "term not found in m-scheme"
        assert len(w) == 1 or w[1] > 1e-6, "term (S, L) is not unique"
        vec = np.zeros(self.dim)
        vec[blk] = v[:, 0]
        return vec

    def lowered_states(self, two_s, two_ll, hw_state):
        """All M-resolved states of the term, keyed by (2ML, 2MS), obtained
        from the highest-weight state with lowering operators (which fixes
        their relative phases in the Condon--Shortley convention)."""
        S = two_s / 2.0
        L = two_ll / 2.0
        states = {(two_ll, two_s): hw_state}
        for two_ms in range(two_s, -two_s - 2, -2):
            if two_ms < two_s:
                MS = two_ms / 2.0
                prev = states[(two_ll, two_ms + 2)]
                states[(two_ll, two_ms)] = self.Sm @ prev \
                    / math.sqrt(S * (S + 1) - MS * (MS + 1))
            for two_ml in range(two_ll - 2, -two_ll - 2, -2):
                ML = two_ml / 2.0
                prev = states[(two_ml + 2, two_ms)]
                states[(two_ml, two_ms)] = self.Lm @ prev \
                    / math.sqrt(L * (L + 1) - ML * (ML + 1))
        return states


def _adag_apply(ms_parent, ms_daughter, orbital, vec):
    """Apply the creation operator of the given spin-orbital to a state of
    the parent configuration, giving a state of the daughter configuration."""
    out = np.zeros(ms_daughter.dim)
    for di, det in enumerate(ms_parent.dets):
        c = vec[di]
        if c == 0.0 or orbital in det:
            continue
        na = sum(1 for x in det if x < orbital)
        nd = tuple(sorted(det + (orbital,)))
        out[ms_daughter.index[nd]] += c * ((-1) ** na)
    return out


def _cfp_terms_list(l2, n):
    """The terms of l^n as a list of (2S, 2L, v, w, 12g) tuples."""
    cu = fu.cfp_utils
    nt = cu.cfp_number_of_terms(l2, n)
    return list(zip(*[a.tolist() for a in cu.cfp_terms(l2, n, nt)]))


def _slater_cfp_magnitudes(l, daughter_two_s, daughter_two_ll,
                           ms_daughter, ms_parent, parent_terms):
    """Brute-force |cfp| of one daughter term with respect to every parent
    term (all terms must have unique (S, L) in their configurations).
    Returns a dict keyed by the parent (2S, 2L)."""
    cg = fu.cg_utils.cg
    n = ms_daughter.n
    alpha = ms_daughter.highest_weight_state(daughter_two_s, daughter_two_ll)
    orb_pos = {o: i for i, o in enumerate(ms_parent.orbs)}
    result = {}
    for (p_two_s, p_two_ll) in parent_terms:
        if abs(daughter_two_s - p_two_s) != 1 \
           or not (abs(p_two_ll - 2 * l) <= daughter_two_ll <= p_two_ll + 2 * l):
            result[(p_two_s, p_two_ll)] = 0.0
            continue
        hw = ms_parent.highest_weight_state(p_two_s, p_two_ll)
        pstates = ms_parent.lowered_states(p_two_s, p_two_ll, hw)
        psi = np.zeros(ms_daughter.dim)
        for two_ml in range(-2 * l, 2 * l + 1, 2):
            two_mlp = daughter_two_ll - two_ml
            if abs(two_mlp) > p_two_ll:
                continue
            for two_ms in (1, -1):
                two_msp = daughter_two_s - two_ms
                if abs(two_msp) > p_two_s:
                    continue
                cgl = cg(p_two_ll, two_mlp, 2 * l, two_ml,
                         daughter_two_ll, daughter_two_ll)
                cgs = cg(p_two_s, two_msp, 1, two_ms,
                         daughter_two_s, daughter_two_s)
                if cgl == 0.0 or cgs == 0.0:
                    continue
                orbital = orb_pos[(two_ml // 2, two_ms)]
                psi += cgl * cgs * _adag_apply(ms_parent, ms_daughter,
                                               orbital, pstates[(two_mlp, two_msp)])
        result[(p_two_s, p_two_ll)] = abs(float(alpha @ psi)) / math.sqrt(n)
    return result


def _threej(tj1, tm1, tj2, tm2, tj3, tm3):
    """3j symbol from the CG coefficient (doubled arguments)."""
    if tm1 + tm2 + tm3 != 0:
        return 0.0
    return (-1) ** ((tj1 - tj2 - tm3) // 2) / math.sqrt(tj3 + 1) \
        * fu.cg_utils.cg(tj1, tm1, tj2, tm2, tj3, -tm3)


def _mscheme_reduced_element(ms, l, term1, term2, tk, with_spin):
    """Independent m-scheme evaluation of the reduced matrix element
    <term1||T||term2> in the configuration of the _MScheme object ms, where
    T = sum_i u^(k)_0(i) for with_spin = False (tk = 2k) and
    T = sum_i s_z(i) u^(1)_0(i) (the (0, 0) double-tensor component of
    V^(11)) for with_spin = True. Both terms must have unique (S, L). The
    result carries the arbitrary relative phase of the two m-scheme term
    states, so only diagonal elements are sign-definite."""
    (tS1, tL1), (tS2, tL2) = term1, term2
    c = np.zeros((ms.no, ms.no))
    for a, (mla, msa) in enumerate(ms.orbs):
        for b, (mlb, msb) in enumerate(ms.orbs):
            if msa != msb:
                continue
            val = (-1) ** (l - mla) * _threej(2 * l, -2 * mla, tk, 0,
                                              2 * l, 2 * mlb)
            if with_spin:
                val *= 0.5 * msa
            c[a, b] = val
    op = ms._onebody(c)

    st1 = ms.lowered_states(tS1, tL1, ms.highest_weight_state(tS1, tL1))
    st2 = ms.lowered_states(tS2, tL2, ms.highest_weight_state(tS2, tL2))

    # Find a common (ML, MS) projection with a nonvanishing Wigner-Eckart
    # denominator and divide it out.
    for tms in range(min(tS1, tS2), -min(tS1, tS2) - 1, -2):
        for tml in range(min(tL1, tL2), -min(tL1, tL2) - 1, -2):
            denom = (-1) ** ((tL1 - tml) // 2) \
                * _threej(tL1, -tml, tk, 0, tL2, tml)
            if with_spin:
                denom *= (-1) ** ((tS1 - tms) // 2) \
                    * _threej(tS1, -tms, 2, 0, tS2, tms)
            # (For the spin-scalar U^k with S1 /= S2 the operator matrix
            # element vanishes identically, so the quotient correctly
            # reproduces the zero reduced element.)
            if abs(denom) > 0.05:
                val = float(st1[(tml, tms)] @ op @ st2[(tml, tms)])
                return val / denom
    raise RuntimeError("no usable projection for the m-scheme reduced element")


def test_cfp_utils():
    print("cfp_utils")
    cu = fu.cfp_utils

    # Build (and time) the complete tables of all four shells.
    import time
    t0 = time.time()
    for l2 in (0, 2, 4, 6):
        cu.cfp_build_tables(l2)
    print(f"  (all tables built in {time.time() - t0:.2f} s)")

    # Term counts of every configuration against the known values.
    known_counts = {
        (0, 1): 1, (0, 2): 1,
        (2, 1): 1, (2, 2): 3, (2, 3): 3, (2, 4): 3, (2, 5): 1, (2, 6): 1,
        (4, 1): 1, (4, 2): 5, (4, 3): 8, (4, 4): 16, (4, 5): 16,
        (4, 6): 16, (4, 7): 8, (4, 8): 5, (4, 9): 1, (4, 10): 1,
        (6, 1): 1, (6, 2): 7, (6, 3): 17, (6, 4): 47, (6, 5): 73,
        (6, 6): 119, (6, 7): 119, (6, 8): 119, (6, 9): 73, (6, 10): 47,
        (6, 11): 17, (6, 12): 7, (6, 13): 1, (6, 14): 1,
    }
    ok = all(cu.cfp_number_of_terms(l2, n) == c
             for (l2, n), c in known_counts.items())
    check("term counts of all configurations", ok)

    # Orthonormality: CFP rows of terms with the same (S, L) are
    # orthonormal, for every configuration of every shell.
    ok = True
    for l2 in (0, 2, 4, 6):
        for n in range(1, 2 * l2 + 3):
            T = _cfp_terms_list(l2, n)
            P = _cfp_terms_list(l2, n - 1)
            M = cu.cfp_full_table(l2, n, len(T), len(P))
            G = M @ M.T
            for i, ti in enumerate(T):
                for j, tj in enumerate(T):
                    if ti[:2] == tj[:2]:
                        ok = ok and close(G[i, j], 1.0 if i == j else 0.0)
    check("orthonormality of CFP rows", ok)

    # The trivial shells and configurations: a single parent gives cfp = 1.
    check("s^1 cfp", close(cu.cfp(0, 1, 1, 0, 1, 1, 0, 0, 0, 1), 1.0))
    check("s^2 cfp", close(cu.cfp(0, 2, 0, 0, 0, 1, 1, 0, 1, 1), 1.0))
    ok = True
    for l2 in (2, 4, 6):
        T = _cfp_terms_list(l2, 2)
        for t in T:
            ok = ok and close(cu.cfp(l2, 2, t[0], t[1], t[2], t[3],
                                     1, l2, 1, 1), 1.0)
    check("all l^2 cfps are +1", ok)

    # Pauli principle in l^2: only terms with even S + L occur.
    ok = True
    for l2 in (2, 4, 6):
        for t in _cfp_terms_list(l2, 2):
            ok = ok and ((t[0] + t[1]) // 2) % 2 == 0
    check("l^2 terms satisfy the Pauli principle", ok)

    # p^3 against the Nielson--Koster values: 4S: (3P) 1;
    # 2P: (3P) 1/2, (1S) 2/9, (1D) 5/18 (squared); 2D: (3P) 1/2, (1D) 1/2.
    p2 = {t[:2]: t for t in _cfp_terms_list(2, 2)}
    p3 = {t[:2]: t for t in _cfp_terms_list(2, 3)}

    def cfp_by_sl(l2, n, dt, pt):
        return cu.cfp(l2, n, dt[0], dt[1], dt[2], dt[3],
                      pt[0], pt[1], pt[2], pt[3])

    ok = close(cfp_by_sl(2, 3, p3[(3, 0)], p2[(2, 2)]), 1.0)
    ok = ok and close(cfp_by_sl(2, 3, p3[(1, 2)], p2[(2, 2)]) ** 2, 0.5)
    ok = ok and close(cfp_by_sl(2, 3, p3[(1, 2)], p2[(0, 0)]) ** 2, 2.0 / 9.0)
    ok = ok and close(cfp_by_sl(2, 3, p3[(1, 2)], p2[(0, 4)]) ** 2, 5.0 / 18.0)
    ok = ok and close(cfp_by_sl(2, 3, p3[(1, 4)], p2[(2, 2)]) ** 2, 0.5)
    ok = ok and close(cfp_by_sl(2, 3, p3[(1, 4)], p2[(0, 4)]) ** 2, 0.5)
    ok = ok and close(cfp_by_sl(2, 3, p3[(1, 4)], p2[(0, 0)]), 0.0)
    check("p^3 CFP magnitudes (Nielson-Koster)", ok)

    # Seniority labels: p^3 2P has v = 1; d^3 2D occurs with v = 1 and
    # v = 3; f^3 2F occurs with v = 1 and v = 3.
    check("p^3 seniorities", p3[(3, 0)][2] == 3 and p3[(1, 2)][2] == 1)
    d3_2d = sorted(t[2] for t in _cfp_terms_list(4, 3) if t[:2] == (1, 4))
    check("d^3 2D seniorities", d3_2d == [1, 3])
    f3_2f = sorted(t[2] for t in _cfp_terms_list(6, 3) if t[:2] == (1, 6))
    check("f^3 2F seniorities", f3_2f == [1, 3])

    # G2 Casimir labels (twelve times the eigenvalue) of the f^2 and f^3
    # terms against Racah's classification: (00) -> 0, (10) -> 6,
    # (11) -> 12, (20) -> 14, (21) -> 21.
    f2_expected = {(2, 2): [12], (2, 6): [6], (2, 10): [12], (0, 0): [0],
                   (0, 4): [14], (0, 8): [14], (0, 12): [14]}
    f2 = _cfp_terms_list(6, 2)
    ok = all(sorted(t[4] for t in f2 if t[:2] == k) == v
             for k, v in f2_expected.items())
    check("f^2 G2 Casimir labels", ok)

    f3_expected = {(3, 0): [0], (3, 4): [14], (3, 6): [6], (3, 8): [14],
                   (3, 12): [14], (1, 2): [12], (1, 4): [14, 21],
                   (1, 6): [6, 21], (1, 8): [14, 21], (1, 10): [12, 21],
                   (1, 12): [14], (1, 14): [21], (1, 16): [21]}
    f3 = _cfp_terms_list(6, 3)
    ok = all(sorted(t[4] for t in f3 if t[:2] == k) == v
             for k, v in f3_expected.items())
    check("f^3 G2 Casimir labels", ok)

    # Point queries against the full table.
    T = _cfp_terms_list(6, 3)
    P = _cfp_terms_list(6, 2)
    M = cu.cfp_full_table(6, 3, len(T), len(P))
    ok = True
    for i in (0, 5, 9, 16):
        for j in (0, 3, 6):
            ok = ok and close(cu.cfp(6, 3, *T[i][:4], *P[j][:4]), M[i, j])
    check("cfp() consistent with cfp_full_table()", ok)

    # Nearly filled shells: p^6 1S and f^14 1S have a single parent with
    # |cfp| = 1.
    check("p^6 cfp magnitude", close(abs(cu.cfp(2, 6, 0, 0, 0, 1, 1, 2, 1, 1)), 1.0))
    check("f^14 cfp magnitude", close(abs(cu.cfp(6, 14, 0, 0, 0, 1, 1, 6, 1, 1)), 1.0))

    # ---- Racah unit tensor operators U^(k) and V^(11) evaluated from the
    # CFPs.

    # One-electron values: (l||u^(k)||l) = 1 and (ls||v^(11)||ls) =
    # sqrt(3/2) by definition.
    ok = True
    for l2 in (2, 4, 6):
        for tk in range(2, 2 * l2 + 1, 2):
            ok = ok and close(cu.cfp_u_tensor_element(
                l2, 1, 1, l2, 1, 1, 1, l2, 1, 1, tk), 1.0)
        ok = ok and close(cu.cfp_v11_tensor_element(
            l2, 1, 1, l2, 1, 1, 1, l2, 1, 1), math.sqrt(1.5))
    check("U^k and V11 one-electron values", ok)

    # U^(1) is proportional to the total orbital angular momentum: the
    # matrix must be diagonal with elements
    # sqrt(L(L+1)(2L+1) / (l(l+1)(2l+1))).
    ok = True
    for l2, n in [(2, 3), (4, 3), (4, 5), (6, 3), (6, 6)]:
        T = _cfp_terms_list(l2, n)
        M = cu.cfp_u_tensor_matrix(l2, n, 2, len(T))
        l = l2 // 2
        ref = np.diag([math.sqrt((tl // 2) * (tl // 2 + 1) * (tl + 1)
                                 / (l * (l + 1) * (2 * l + 1)))
                       for (_, tl, _, _, _) in T])
        ok = ok and close(M, ref)
    check("U^1 proportional to L", ok)

    # The G2 Casimir labels stored in the term tables must be reproduced
    # from the U^(5) matrix through
    # g = [3 L(L+1)/84 + 11 (U5.U5)] / 4, (U5.U5)_ii = sum_z M_iz^2 / [L_i].
    ok = True
    for n in (2, 3, 4):
        T = _cfp_terms_list(6, n)
        M5 = cu.cfp_u_tensor_matrix(6, n, 10, len(T))
        for i, (ts, tl, tv, tw, tg) in enumerate(T):
            u5u5 = float((M5[i, :] ** 2).sum()) / (tl + 1)
            g12 = 12.0 * (3.0 * (tl // 2) * (tl // 2 + 1) / 84.0
                          + 11.0 * u5u5) / 4.0
            ok = ok and close(g12, tg, tol=1e-8)
    check("G2 Casimir labels reproduced from U^5 matrices", ok)

    # Symmetry relations of the reduced matrix elements:
    # (a||U^k||a') = (-1)^(L-L') (a'||U^k||a) and
    # (a||V11||a') = (-1)^(S-S'+L-L') (a'||V11||a).
    T = _cfp_terms_list(6, 3)
    M2 = cu.cfp_u_tensor_matrix(6, 3, 4, len(T))
    V = cu.cfp_v11_tensor_matrix(6, 3, len(T))
    ok = True
    for i, ti in enumerate(T):
        for j, tj in enumerate(T):
            ok = ok and close(M2[j, i],
                              (-1) ** ((ti[1] - tj[1]) // 2) * M2[i, j])
            ok = ok and close(V[j, i],
                              (-1) ** ((ti[0] - tj[0] + ti[1] - tj[1]) // 2)
                              * V[i, j])
    check("U^k and V11 symmetry relations", ok)

    # Selection rules: U^k is diagonal in S; V11 obeys |dS| <= 1 and
    # |dL| <= 1.
    ok = True
    for i, ti in enumerate(T):
        for j, tj in enumerate(T):
            if ti[0] != tj[0]:
                ok = ok and M2[i, j] == 0.0
            if abs(ti[0] - tj[0]) > 2 or abs(ti[1] - tj[1]) > 2:
                ok = ok and V[i, j] == 0.0
    check("U^k and V11 selection rules", ok)

    # Element routines consistent with the matrix routines.
    ok = True
    for i in (0, 4, 9):
        for j in (2, 9, 15):
            ok = ok and close(cu.cfp_u_tensor_element(
                6, 3, *T[i][:4], *T[j][:4], 4), M2[i, j])
            ok = ok and close(cu.cfp_v11_tensor_element(
                6, 3, *T[i][:4], *T[j][:4]), V[i, j])
    check("tensor element routines consistent with matrices", ok)

    # Brute-force verification of the CFP magnitudes in the Slater
    # determinant m-scheme for all terms with unique (S, L).
    for l, l2, terms_to_check in [
            (1, 2, [(3, 0), (1, 2), (1, 4)]),
            (2, 4, [(3, 6), (3, 2), (1, 10), (1, 8), (1, 6), (1, 2)]),
            (3, 6, [(3, 0), (3, 4), (3, 6), (3, 8), (3, 12),
                    (1, 2), (1, 12), (1, 14), (1, 16)]),
    ]:
        ms3 = _MScheme(l, 3)
        ms2 = _MScheme(l, 2)
        daughters = {t[:2]: t for t in _cfp_terms_list(l2, 3)}
        parents = {t[:2]: t for t in _cfp_terms_list(l2, 2)}
        ok = True
        for dsl in terms_to_check:
            ref = _slater_cfp_magnitudes(l, dsl[0], dsl[1], ms3, ms2,
                                         list(parents.keys()))
            for psl, mag in ref.items():
                val = cfp_by_sl(l2, 3, daughters[dsl], parents[psl])
                ok = ok and close(abs(val), mag)
        check(f"l = {l}: l^3 CFP magnitudes against m-scheme", ok)

    # Independent m-scheme verification of the U^(2) and V^(11) reduced
    # matrix elements for terms of p^3 and d^3 with unique (S, L). The
    # m-scheme term states carry arbitrary phases, so diagonal elements
    # are compared exactly and off-diagonal ones in absolute value.
    for l, l2, u_pairs, v_pairs in [
            (1, 2,
             [((1, 2), (1, 2)), ((1, 4), (1, 4)), ((1, 2), (1, 4)),
              ((3, 0), (1, 4))],
             [((1, 2), (1, 2)), ((1, 4), (1, 4)), ((3, 0), (1, 2)),
              ((1, 2), (1, 4))]),
            (2, 4,
             [((3, 6), (3, 6)), ((3, 6), (3, 2)), ((1, 10), (1, 10))],
             [((3, 6), (3, 6)), ((1, 10), (1, 8))]),
    ]:
        ms = _MScheme(l, 3)
        terms = {t[:2]: t for t in _cfp_terms_list(l2, 3)}
        ok = True
        for t1, t2 in u_pairs:
            ref = _mscheme_reduced_element(ms, l, t1, t2, 4, False)
            got = cu.cfp_u_tensor_element(l2, 3, *terms[t1][:4],
                                          *terms[t2][:4], 4)
            if t1 == t2:
                ok = ok and close(got, ref)
            else:
                ok = ok and close(abs(got), abs(ref))
        for t1, t2 in v_pairs:
            ref = _mscheme_reduced_element(ms, l, t1, t2, 2, True)
            got = cu.cfp_v11_tensor_element(l2, 3, *terms[t1][:4],
                                            *terms[t2][:4])
            if t1 == t2:
                ok = ok and close(got, ref)
            else:
                ok = ok and close(abs(got), abs(ref))
        check(f"l = {l}: U^2 and V11 against m-scheme", ok)


def main():
    tests = [
        test_num_utils,
        test_math_utils,
        test_matrix_utils,
        test_angm_utils,
        test_cg_utils,
        test_cg_product,
        test_iwahara_utils,
        test_cfp_utils,
        test_statmech_utils,
        test_powder_magnetization_utils,
        test_pseudospin_utils,
        test_grid_utils,
    ]
    for t in tests:
        t()
    print()
    print(f"{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
