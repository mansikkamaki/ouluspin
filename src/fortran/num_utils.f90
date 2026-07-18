! SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
! SPDX-License-Identifier: GPL-3.0-or-later

module num_utils
  
  ! Module containing default parameters to be used by other Fortran modules.
  ! This module MUST NOT depend on any other module.
  ! use, intrinsic :: iso_fortran_env
  
  implicit none (type,external)

  ! real64 and int64 kinds are defined here instead of using iso_fortran_env
  ! to keep things compatible with f2py. If this code is complied purely for
  ! Fortran use, these definitions can be removed and the use iso_fortran_env
  ! lines should be uncommented in all modules.
  integer, parameter          :: real64 = kind(1.d0)
  integer, parameter          :: int64  = selected_int_kind(16)

  ! Quadruple-precision kind used internally by numerically sensitive
  ! routines (e.g. log_factorial in math_utils and new_cg in cg_utils).
  ! Note that functions with real(real128) results cannot be meaningfully
  ! called through the f2py interface; they are for Fortran-internal use.
  integer, parameter          :: real128 = selected_real_kind(30)

  ! Here we define the number of prime numbers which are relevant in the
  ! power-of-primes expansions. Having too small a value here will lead to numerical
  ! problems and two large a value will require more cpu time/power. Setting the values
  ! here is not ideal and should be fixed in the future. This value is rather
  ! conservative.
  integer, parameter         :: relevant_primes = 100

  complex(real64)            :: imaginary_unit = cmplx(0.0,1.0,kind=real64)

  ! Note the d0 suffix: without it the literal would be a default (single
  ! precision) real and pi would only carry about seven significant digits.
  real(real64), parameter    :: pi = 3.1415926535897932384626433d0
  real(real64), parameter    :: dp_zero = 0.0, dp_one = 1.0
  real(real64), parameter    :: small_number = 1.0e-6, very_small_number = 1.0e-9, really_small_number = 1.0e-12

  ! Product of Avogadro's constant and Bohr magneton in cm^3 T mol^-1 (in cgsemu with teslas as field unit)
  ! real(real64), parameter    :: N_Amu_B = 0.5584939449309

  ! Parameters used by the coefficient-of-fractional-parentage tables in
  ! cfp_utils.
  !
  !   cfp_max_terms            The largest possible number of terms in an l^n
  !                            configuration with l <= 3. The f^6, f^7 and f^8
  !                            configurations have 119 terms, which is the
  !                            largest number that can occur.
  !   cfp_projector_tolerance  The eigenvalues of the antisymmetrizing
  !                            projector are exactly zero or one. Deviations
  !                            larger than this tolerance indicate an internal
  !                            inconsistency and produce a fatal error.
  !   cfp_degeneracy_tolerance Two eigenvalues of the G2 Casimir operator are
  !                            considered degenerate if they differ by less
  !                            than this tolerance.
  !   cfp_drop_tolerance       A pair-created state whose norm relative to its
  !                            progenitor is below this tolerance is considered
  !                            to vanish (this happens for configurations
  !                            beyond the half-filled shell, where some
  !                            seniorities are no longer allowed).
  integer, parameter          :: cfp_max_terms = 120
  real(real64), parameter     :: cfp_projector_tolerance  = 1.0e-8
  real(real64), parameter     :: cfp_degeneracy_tolerance = 1.0e-7
  real(real64), parameter     :: cfp_drop_tolerance       = 1.0e-8

end module num_utils
