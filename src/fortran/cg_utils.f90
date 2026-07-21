! SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
! SPDX-License-Identifier: GPL-3.0-or-later

module cg_utils
  
  use num_utils
  use math_utils

  implicit none (type,external)

contains
  function old_cg(j1,m1,j2,m2,J,M) result(coeff)
    ! Evaluate the Clebsch--Gordan (CG) coefficient <j1,m1,j2,m2|J,M> using
    ! the Condon--Shortley phase convention.
    !
    ! This is the original implementation based on power-of-primes factorial
    ! decompositions. It is retained for reference and testing; new code
    ! should use the cg function below, which takes exactly the same
    ! arguments, gives the same values, and remains accurate for arbitrarily
    ! large angular momenta.
    !
    ! The evaluation is based on eq. (3) in Section 8.2.1 in
    !
    !   D. A. Varshalovich, A. N. Moskalev, V. K. Khersonskii.
    !   Quantum Theory of Angular Momentum.
    !   1988, World Scientific, Singapore.
    !
    ! Evaluation of the actual factorials in construction of the coefficient C
    ! might lead to numerical instabilities for a specific combination of large
    ! and small arguments. Thus, one should be careful in using this function
    ! to evaluate CG coefficients with large arguments. So far, no problems have
    ! been encountered if all angular momenta are 12 or less.
    !
    implicit none (type,external)
    
    integer, intent(in)                :: j1,m1,j2,m2,J,M
    real(real64)                       :: coeff
    integer, allocatable, dimension(:) :: sqrt_term, C
    integer                            :: largest_factorial = 0
    integer                            :: max_index = 0
    real(real64)                       :: sum = 0.0, sqrt_float = 0.0
    integer                            :: nu = 0, nu_max = 0, nu_min = 0
    
    ! Check conservation of angular momentum projection.
    if (m1 + m2 /= M) then
       coeff = 0.0
       ! Check the triangular condition.
    else if (triangular_condition(j1,j2,J) .eqv. .false.) then
       coeff = 0.0
       ! Check that the projections are not larger than the momenta.
    else if (abs(m1) > j1 .or. abs(m2) > j2 .or. abs(M) > J) then
       coeff = 0.0
       ! Else evaluate the coefficient:
    else
       ! Find the largest factorial needed for the evaluation of the square root
       ! term.
       largest_factorial = max((j1 + j2 - J) / 2, (J + j1 - j2) / 2, &
            (j2 + J - j1) / 2, (j1 + m1) / 2, (j1 - m1) / 2, &
            (j2 + m2) / 2, (j2 - m2) / 2, (J + M) / 2, (J - M) / 2, &
            (j1 + j2 + J + 2) / 2)
       
       max_index = largest_prime(largest_factorial)
       
       allocate(sqrt_term(1:max_index))
       
       sqrt_term = power_of_primes_factorial((j1 + j2 - J) / 2, max_index) &
            + power_of_primes_factorial((J + j1 - j2) / 2, max_index) &
            + power_of_primes_factorial((j2 + J - j1) / 2, max_index) &
            + power_of_primes_factorial((j1 + m1) / 2, max_index) &
            + power_of_primes_factorial((j1 - m1) / 2, max_index) &
            + power_of_primes_factorial((j2 + m2) / 2, max_index) &
            + power_of_primes_factorial((j2 - m2) / 2, max_index) &
            + power_of_primes_factorial((J + M) / 2, max_index) &
            + power_of_primes_factorial((J - M) / 2, max_index) &
            - power_of_primes_factorial((j1 + j2 + J + 2) / 2, max_index)

       sqrt_float = evaluate_power_of_primes(sqrt_term,max_index)
       
       nu_max = min((j1+j2-J)/2, &
            (j1-m1)/2, &
            (j2+m2)/2)
       nu_min = max(0, -(J-j2+m1)/2, -(J-j1-m2)/2)
       
       sum = 0.0
       do nu = nu_min, nu_max
          ! Find the largest factorial needed for the evaluatin of the coefficient C.
          largest_factorial = max((j1 + j2 - J - 2*nu) / 2, &
               (j1 - m1 - 2*nu) / 2, &
               (j2 + m2 - 2*nu) / 2, &
               (J - j2 + m1 + 2*nu) / 2, &
               (J - j1 - m2 + 2*nu) / 2, &
               nu)

          max_index = largest_prime(largest_factorial)
          allocate(C(1:max_index))

          C = power_of_primes_factorial((j1 + j2 - J - 2*nu) / 2, max_index) &
               + power_of_primes_factorial((j1 - m1 - 2*nu) / 2, max_index) &
               + power_of_primes_factorial((j2 + m2 - 2*nu) / 2, max_index) &
               + power_of_primes_factorial((J - j2 + m1 + 2*nu) / 2, max_index) &
               + power_of_primes_factorial((J - j1 - m2 + 2*nu) / 2, max_index) &
               + power_of_primes_factorial(nu, max_index)

          sum = sum + (-1.0)**nu / evaluate_power_of_primes(C, max_index)

          deallocate(C)
       end do
       
       ! Note that J is already multiplied by two below and hence we are omitting the
       ! factor of two in 2J+1.
       coeff = sqrt((J + 1) * sqrt_float) * sum
    end if
  end function old_cg
  
    
  function cg(j1,m1,j2,m2,J,M) result(coeff)
    ! Evaluate the Clebsch--Gordan (CG) coefficient <j1,m1,j2,m2|J,M> using
    ! the Condon--Shortley phase convention. This is the default CG routine
    ! of the library (formerly named new_cg); it takes exactly the same
    ! arguments as the original old_cg function above (all angular momenta
    ! given as integers equal to TWICE their physical value, so that
    ! half-integer momenta can be represented) and returns the same
    ! coefficient.
    !
    ! The evaluation is based on eq. (3) in Section 8.2.1 in
    !
    !   D. A. Varshalovich, A. N. Moskalev, V. K. Khersonskii.
    !   Quantum Theory of Angular Momentum.
    !   1988, World Scientific, Singapore.
    !
    ! Numerical strategy: all factorials enter only through their logarithms,
    ! which are read from the cached quadruple-precision log(n!) table of the
    ! log_factorial function in math_utils (built on the first call and grown
    ! on demand), so no overflow or underflow can occur for any argument
    ! size. The alternating sum over nu
    ! suffers from catastrophic cancellation at large angular momenta; to
    ! control this the term magnitudes (which are unimodal in nu) are
    ! normalized to the largest term, and the sum is accumulated outward from
    ! that term using exact term-ratio recurrences in quadruple precision.
    ! The ~33 significant digits of quadruple precision absorb the
    ! cancellation, so the returned double-precision value stays accurate to
    ! machine precision up to angular momenta of about 100. In the rare cases
    ! where the detected cancellation exceeds even quadruple precision (which
    ! can only happen for angular momenta above ~100), the evaluation
    ! automatically falls back to the numerically stable three-term recursion
    ! in m1 (the cg_by_m_recursion function below), also carried out in
    ! quadruple precision, which handles arbitrarily large angular momenta. The result
    ! is accurate to double precision for arbitrary angular momenta
    ! (validated against exact rational arithmetic up to angular momenta of
    ! 1000). Note that in the fallback branch non-trivial zeros of the
    ! coefficient are returned as small values of the order of the
    ! quadruple-precision epsilon instead of exact zeros.
    !
    ! Unphysical argument combinations (projection non-conservation, broken
    ! triangular condition, |m| > j, or m and j of different half-integer
    ! character) return zero.
    !
    ! Note: because of the cached log-factorial table in math_utils the
    ! first call is marginally more expensive and concurrent first calls
    ! from multiple threads are not safe; all subsequent calls are pure
    ! lookups.
    !
    implicit none (type,external)

    integer, intent(in) :: j1,m1,j2,m2,J,M
    real(real64)        :: coeff

    integer       :: a, b, c, d, e
    integer       :: nu, nu_min, nu_max, nu_star
    integer       :: sign_factor
    real(real128) :: log_prefactor, log_peak_term, term, term_sum

    coeff = 0.0_real64

    ! Check conservation of the angular momentum projection.
    if (m1 + m2 /= M) return
    ! Check the triangular condition (this also enforces a consistent
    ! integer/half-integer character of j1, j2 and J).
    if (.not. triangular_condition(j1,j2,J)) return
    ! Check that the projections are not larger than the momenta.
    if (abs(m1) > j1 .or. abs(m2) > j2 .or. abs(M) > J) return
    ! Check that each m has the same integer/half-integer character as its j.
    if (mod(j1+m1,2) /= 0 .or. mod(j2+m2,2) /= 0 .or. mod(J+M,2) /= 0) return

    ! Non-doubled integer combinations appearing in the factorials of the
    ! summation over nu.
    a = (j1 + j2 - J) / 2
    b = (j1 - m1) / 2
    c = (j2 + m2) / 2
    d = (J - j2 + m1) / 2
    e = (J - j1 - m2) / 2

    ! Summation limits; for arguments that pass the checks above the sum is
    ! never empty, so this is purely defensive.
    nu_min = max(0, -d, -e)
    nu_max = min(a, b, c)
    if (nu_max < nu_min) return

    ! Logarithm of the square-root prefactor. Note that J is already twice
    ! the physical value, so J + 1 equals the physical 2J + 1.
    log_prefactor = 0.5_real128 * ( log(real(J + 1, real128)) &
         + log_factorial(a) &
         + log_factorial((j1 - j2 + J) / 2) &
         + log_factorial((j2 - j1 + J) / 2) &
         + log_factorial((j1 + m1) / 2) &
         + log_factorial(b) &
         + log_factorial(c) &
         + log_factorial((j2 - m2) / 2) &
         + log_factorial((J + M) / 2) &
         + log_factorial((J - M) / 2) &
         - log_factorial((j1 + j2 + J) / 2 + 1) )

    ! The magnitude of the nu-th term,
    !
    !   |t(nu)| = 1 / ( nu! (a-nu)! (b-nu)! (c-nu)! (d+nu)! (e+nu)! ),
    !
    ! is unimodal in nu because the magnitude of the ratio
    ! |t(nu+1)/t(nu)| = (a-nu)(b-nu)(c-nu) / ((nu+1)(d+nu+1)(e+nu+1)) is
    ! strictly decreasing in nu. Locate the peak term nu_star.
    nu_star = nu_min
    do nu = nu_min, nu_max - 1
       if (real(a-nu,real128) * real(b-nu,real128) * real(c-nu,real128) > &
            real(nu+1,real128) * real(d+nu+1,real128) * real(e+nu+1,real128)) then
          nu_star = nu + 1
       else
          exit
       end if
    end do

    ! Logarithm of the magnitude of the peak term.
    log_peak_term = -( log_factorial(nu_star) &
         + log_factorial(a - nu_star) &
         + log_factorial(b - nu_star) &
         + log_factorial(c - nu_star) &
         + log_factorial(d + nu_star) &
         + log_factorial(e + nu_star) )

    ! Accumulate the sum normalized to the peak term, term_sum =
    ! sum_nu t(nu)/t(nu_star), walking outward from nu_star with the signed
    ! term-ratio recurrence. Every relative term has magnitude <= 1, so the
    ! accumulation can never overflow.
    term_sum = 1.0_real128
    term = 1.0_real128
    do nu = nu_star, nu_max - 1
       term = -term * (real(a-nu,real128) * real(b-nu,real128) * real(c-nu,real128)) &
            / (real(nu+1,real128) * real(d+nu+1,real128) * real(e+nu+1,real128))
       term_sum = term_sum + term
    end do
    term = 1.0_real128
    do nu = nu_star - 1, nu_min, -1
       term = -term * (real(nu+1,real128) * real(d+nu+1,real128) * real(e+nu+1,real128)) &
            / (real(a-nu,real128) * real(b-nu,real128) * real(c-nu,real128))
       term_sum = term_sum + term
    end do

    if (term_sum == 0.0_real128) return

    ! The terms are normalized to the peak term, so |term_sum| << 1 signals
    ! cancellation of leading digits in the sum. If the cancellation is
    ! severe enough that the quadruple-precision sum can no longer guarantee
    ! full double precision in the result, fall back to the three-term
    ! recursion in m1, which is numerically stable for arbitrarily large
    ! angular momenta.
    if (abs(term_sum) < real(nu_max - nu_min + 1, real128)**2 * 2.0e-19_real128) then
       coeff = cg_by_m_recursion(j1,m1,j2,m2,J,M)
       return
    end if

    ! The sign of the coefficient is the sign of the peak term, (-1)^nu_star,
    ! times the sign of the normalized sum.
    sign_factor = 1
    if (mod(nu_star, 2) /= 0) sign_factor = -sign_factor
    if (term_sum < 0.0_real128) sign_factor = -sign_factor

    ! |coeff| <= 1 always holds, so the exponential cannot overflow.
    coeff = real(sign_factor, real64) &
         * real(exp(log_prefactor + log_peak_term + log(abs(term_sum))), real64)

  end function cg


  function cg_by_m_recursion(j1,m1,j2,m2,J,M) result(rec_coeff)
    ! Evaluate the CG coefficient <j1,m1,j2,m2|J,M> by the numerically
    ! stable three-term recursion in m1 at fixed j1, j2, J and M
    ! (Schulten--Gordon type recursion, cf. J. H. Luscombe and M. Luban,
    ! Phys. Rev. E 57 (1998) 7274). All angular momentum arguments are
    ! integers equal to twice their physical value, exactly as in old_cg and
    ! cg. The recursion relation follows from the matrix elements of
    ! J1.J2 = (J^2 - J1^2 - J2^2)/2 in the product and the coupled basis:
    !
    !   B(m1) C(m1+1) = D(m1) C(m1) - A(m1) C(m1-1),
    !
    ! where A, B and D are given by the functions cg_recursion_a,
    ! cg_recursion_b and cg_recursion_d below. The recursion is run upward
    ! from the smallest allowed m1 and downward from the largest allowed
    ! m1; each direction is numerically stable up to its classical turning
    ! point, where the two branches are matched. The composed solution is
    ! normalized with sum_m1 C(m1)^2 = 1 and its overall sign is fixed by
    ! the Condon--Shortley convention, which makes the coefficient with
    ! the largest allowed m1 positive.
    !
    ! This function is used internally by cg as its severe-
    ! cancellation fallback. It assumes that the arguments satisfy all the
    ! CG selection rules; unlike old_cg and cg it does not check them, and
    ! it must not be called with unphysical argument combinations.
    implicit none (type,external)

    integer, intent(in) :: j1,m1,j2,m2,J,M
    real(real64)        :: rec_coeff

    ! The linear recursion is homogeneous, so a growing branch can be
    ! rescaled at any time without changing the final normalized solution.
    real(real128), parameter :: rescale_limit  = 1.0e250_real128
    real(real128), parameter :: rescale_factor = 1.0e-250_real128

    integer :: tm_lo, tm_hi, n_val, i_target, i, tm
    integer :: i_lo_turn, i_hi_turn, i_match, w_lo, w_hi
    real(real128), allocatable, dimension(:) :: psi_f, psi_b
    real(real128) :: phi, branch_max_f, branch_max_b, best_product, node_product
    real(real128) :: norm

    ! Range of allowed m1 values (doubled integers) and the position of
    ! the requested coefficient within it.
    tm_lo = max(-j1, M - j2)
    tm_hi = min(j1, M + j2)
    n_val = (tm_hi - tm_lo) / 2 + 1
    i_target = (m1 - tm_lo) / 2 + 1

    ! With a single allowed m1 value the normalized coefficient is +1.
    if (n_val == 1) then
       rec_coeff = 1.0_real64
       return
    end if

    allocate(psi_f(1:n_val), psi_b(1:n_val))

    ! Forward (upward in m1) pass. A(tm_lo) vanishes identically, so the
    ! second value follows from the first without a third term. The first
    ! decrease in magnitude marks the lower classical turning point.
    psi_f(1) = 1.0_real128
    psi_f(2) = cg_recursion_d(tm_lo,j1,j2,J,M) * psi_f(1) / cg_recursion_b(tm_lo,j1,j2,M)
    i_lo_turn = 0
    if (abs(psi_f(2)) < abs(psi_f(1))) i_lo_turn = 1
    do i = 2, n_val - 1
       tm = tm_lo + 2*(i - 1)
       psi_f(i+1) = (cg_recursion_d(tm,j1,j2,J,M) * psi_f(i) - cg_recursion_a(tm,j1,j2,M) * psi_f(i-1)) / cg_recursion_b(tm,j1,j2,M)
       if (i_lo_turn == 0 .and. abs(psi_f(i+1)) < abs(psi_f(i))) i_lo_turn = i
       if (abs(psi_f(i+1)) > rescale_limit) then
          psi_f(1:i+1) = psi_f(1:i+1) * rescale_factor
       end if
    end do
    if (i_lo_turn == 0) i_lo_turn = n_val

    ! Backward (downward in m1) pass; B(tm_hi) vanishes identically. The
    ! first decrease in magnitude marks the upper classical turning point.
    psi_b(n_val) = 1.0_real128
    psi_b(n_val-1) = cg_recursion_d(tm_hi,j1,j2,J,M) * psi_b(n_val) / cg_recursion_a(tm_hi,j1,j2,M)
    i_hi_turn = 0
    if (abs(psi_b(n_val-1)) < abs(psi_b(n_val))) i_hi_turn = n_val
    do i = n_val - 1, 2, -1
       tm = tm_lo + 2*(i - 1)
       psi_b(i-1) = (cg_recursion_d(tm,j1,j2,J,M) * psi_b(i) - cg_recursion_b(tm,j1,j2,M) * psi_b(i+1)) / cg_recursion_a(tm,j1,j2,M)
       if (i_hi_turn == 0 .and. abs(psi_b(i-1)) < abs(psi_b(i))) i_hi_turn = i
       if (abs(psi_b(i-1)) > rescale_limit) then
          psi_b(i-1:n_val) = psi_b(i-1:n_val) * rescale_factor
       end if
    end do
    if (i_hi_turn == 0) i_hi_turn = 1

    ! Match the branches inside the classical region between the turning
    ! points, where both passes are accurate. The matching point must not
    ! fall near a node of the solution, where both branch values are pure
    ! rounding noise and would randomize the overall sign; therefore the
    ! index maximizing |psi_f(i)*psi_b(i)| (proportional to C(m1)^2 where
    ! both branches are accurate) is chosen within a window around the
    ! midpoint of the classical region. Each branch is normalized to its
    ! window maximum so that the products cannot overflow.
    i_match = min(max((i_lo_turn + i_hi_turn + 1) / 2, 1), n_val)
    w_lo = max(1, min(i_lo_turn, i_hi_turn), i_match - 25)
    w_hi = min(n_val, max(i_lo_turn, i_hi_turn), i_match + 25)
    branch_max_f = maxval(abs(psi_f(w_lo:w_hi)))
    branch_max_b = maxval(abs(psi_b(w_lo:w_hi)))
    if (branch_max_f > 0.0_real128 .and. branch_max_b > 0.0_real128) then
       best_product = -1.0_real128
       do i = w_lo, w_hi
          node_product = (abs(psi_f(i)) / branch_max_f) &
               * (abs(psi_b(i)) / branch_max_b)
          if (node_product > best_product) then
             best_product = node_product
             i_match = i
          end if
       end do
    end if
    ! Purely defensive: never match at an exact zero of either branch. The
    ! search cannot cycle because the branches are nonzero over the
    ! classical region.
    do while (psi_f(i_match) == 0.0_real128 .or. psi_b(i_match) == 0.0_real128)
       i_match = i_match + 1
       if (i_match > n_val) i_match = 1
    end do

    ! Both branches are scaled to unity at the matching point; the
    ! composed solution then has classical values of order one and
    ! decaying tails, so the normalization sum cannot overflow.
    norm = 0.0_real128
    do i = 1, i_match
       phi = psi_f(i) / psi_f(i_match)
       norm = norm + phi**2
    end do
    do i = i_match + 1, n_val
       phi = psi_b(i) / psi_b(i_match)
       norm = norm + phi**2
    end do

    if (i_target <= i_match) then
       phi = psi_f(i_target) / psi_f(i_match)
    else
       phi = psi_b(i_target) / psi_b(i_match)
    end if

    ! In the composed solution the coefficient with the largest m1 equals
    ! psi_b(tm_hi)/psi_b(i_match) = 1/psi_b(i_match) up to a positive
    ! rescaling, so the Condon--Shortley sign correction is the sign of
    ! psi_b(i_match).
    rec_coeff = real(sign(1.0_real128, psi_b(i_match)) * phi / sqrt(norm), real64)
  end function cg_by_m_recursion


  function cg_recursion_a(tm1,j1,j2,M) result(a_val)
    ! Coefficient A(m1) coupling C(m1) to C(m1-1) in the three-term
    ! recursion used by cg_by_m_recursion:
    !
    !   A(m1) = sqrt( (j1-m1+1)(j1+m1)(j2+M-m1+1)(j2-M+m1) )
    !
    ! All arguments are doubled integers; the physical values are used in
    ! the formula. Intended for Fortran-internal use: the real(real128)
    ! result cannot be meaningfully accessed through the f2py interface.
    implicit none (type,external)
    ! Force inlining into the recursion loops of cg_by_m_recursion;
    ! without it the out-of-line calls slow the recursion down.
    !DIR$ ATTRIBUTES FORCEINLINE :: cg_recursion_a

    integer, intent(in) :: tm1,j1,j2,M
    real(real128)            :: a_val
    real(real128)            :: xj1, xj2, xm1, xM

    xj1 = 0.5_real128 * j1
    xj2 = 0.5_real128 * j2
    xm1 = 0.5_real128 * tm1
    xM  = 0.5_real128 * M

    a_val = sqrt((xj1 - xm1 + 1.0_real128) * (xj1 + xm1) &
         * (xj2 + xM - xm1 + 1.0_real128) * (xj2 - xM + xm1))
  end function cg_recursion_a


  function cg_recursion_b(tm1,j1,j2,M) result(b_val)
    ! Coefficient B(m1) coupling C(m1) to C(m1+1) in the three-term
    ! recursion used by cg_by_m_recursion:
    !
    !   B(m1) = sqrt( (j1+m1+1)(j1-m1)(j2-M+m1+1)(j2+M-m1) )
    !
    ! All arguments are doubled integers; the physical values are used in
    ! the formula. Intended for Fortran-internal use: the real(real128)
    ! result cannot be meaningfully accessed through the f2py interface.
    implicit none (type,external)
    ! Force inlining into the recursion loops of cg_by_m_recursion;
    ! without it the out-of-line calls slow the recursion down.
    !DIR$ ATTRIBUTES FORCEINLINE :: cg_recursion_b

    integer, intent(in) :: tm1,j1,j2,M
    real(real128)            :: b_val
    real(real128)            :: xj1, xj2, xm1, xM

    xj1 = 0.5_real128 * j1
    xj2 = 0.5_real128 * j2
    xm1 = 0.5_real128 * tm1
    xM  = 0.5_real128 * M

    b_val = sqrt((xj1 + xm1 + 1.0_real128) * (xj1 - xm1) &
         * (xj2 - xM + xm1 + 1.0_real128) * (xj2 + xM - xm1))
  end function cg_recursion_b


  function cg_recursion_d(tm1,j1,j2,J,M) result(d_val)
    ! Diagonal coefficient D(m1) in the three-term recursion used by
    ! cg_by_m_recursion:
    !
    !   D(m1) = J(J+1) - j1(j1+1) - j2(j2+1) - 2 m1 (M - m1)
    !
    ! All arguments are doubled integers; the physical values are used in
    ! the formula. Intended for Fortran-internal use: the real(real128)
    ! result cannot be meaningfully accessed through the f2py interface.
    implicit none (type,external)
    ! Force inlining into the recursion loops of cg_by_m_recursion;
    ! without it the out-of-line calls slow the recursion down.
    !DIR$ ATTRIBUTES FORCEINLINE :: cg_recursion_d

    integer, intent(in) :: tm1,j1,j2,J,M
    real(real128)            :: d_val
    real(real128)            :: xj1, xj2, xJ, xm1, xM

    xj1 = 0.5_real128 * j1
    xj2 = 0.5_real128 * j2
    xJ  = 0.5_real128 * J
    xm1 = 0.5_real128 * tm1
    xM  = 0.5_real128 * M

    d_val = xJ * (xJ + 1.0_real128) - xj1 * (xj1 + 1.0_real128) &
         - xj2 * (xj2 + 1.0_real128) - 2.0_real128 * xm1 * (xM - xm1)
  end function cg_recursion_d


  function cg_is_allowed(j1,m1,j2,m2,J,M) result(allowed)
    ! Cheap O(1) integer test of the Clebsch--Gordan selection rules for the
    ! coefficient <j1,m1,j2,m2|J,M>. All arguments are integers equal to
    ! TWICE their physical value, exactly as in old_cg and cg. The function
    ! returns .true. if and only if the coefficient is not forced to vanish
    ! by one of the selection rules:
    !
    !   - conservation of the projection, m1 + m2 = M,
    !   - non-negativity of the angular momenta,
    !   - the triangular condition |j1 - j2| <= J <= j1 + j2 together with
    !     the requirement that j1 + j2 + J is an integer (an even doubled
    !     value),
    !   - |m| <= j for each momentum--projection pair,
    !   - identical integer/half-integer character of each m and its j.
    !
    ! A .true. result does not exclude a non-trivial (accidental) zero of
    ! the coefficient. The test duplicates the checks made at the entry of
    ! cg, but unlike a call to cg it involves no floating-point
    ! work at all, so it can be used to prune away vanishing terms in large
    ! summations over products of CG coefficients (see cg_product) at a
    ! negligible cost.
    implicit none (type,external)

    integer, intent(in) :: j1,m1,j2,m2,J,M
    logical             :: allowed

    allowed = .false.
    if (m1 + m2 /= M) return
    if (j1 < 0 .or. j2 < 0 .or. J < 0) return
    if (J < abs(j1 - j2) .or. J > j1 + j2) return
    if (mod(j1 + j2 + J, 2) /= 0) return
    if (abs(m1) > j1 .or. abs(m2) > j2 .or. abs(M) > J) return
    if (mod(j1 + m1, 2) /= 0 .or. mod(j2 + m2, 2) /= 0 .or. mod(J + M, 2) /= 0) return
    allowed = .true.
  end function cg_is_allowed


  function partial_cg_product(p, cg_map, cg_level, level, n_par, n_cg) result(prod)
    ! Product of all the CG coefficients of a cg_product summation that are
    ! assigned to the given hoisting level (see cg_product below for the
    ! meaning of the levels and of the cg_map argument). The array p holds
    ! the current values of the sum parameters (doubled integers) and
    ! cg_level holds the hoisting level of each coefficient as computed by
    ! cg_product. If no coefficient lives on the requested level the result
    ! is exactly 1. The selection rules of every coefficient on the level
    ! are first checked with the cheap integer test cg_is_allowed; if any
    ! coefficient vanishes by a selection rule the result is exactly 0 and
    ! no floating-point evaluation is performed at all. Only when all
    ! coefficients on the level pass the selection rules are they actually
    ! evaluated (with cg, which is accurate to double precision for
    ! arbitrarily large angular momenta).
    !
    ! This function is a helper of cg_product and of little use on its own.
    implicit none (type,external)

    integer, intent(in)                       :: n_par, n_cg, level
    integer, intent(in), dimension(n_par)     :: p
    integer, intent(in), dimension(n_cg,6)    :: cg_map
    integer, intent(in), dimension(n_cg)      :: cg_level
    real(real64)                              :: prod

    integer :: c

    prod = 0.0_real64

    ! First pass: pure integer selection-rule checks, no floating point.
    do c = 1, n_cg
       if (cg_level(c) /= level) cycle
       if (.not. cg_is_allowed(p(cg_map(c,1)), p(cg_map(c,2)), &
            p(cg_map(c,3)), p(cg_map(c,4)), p(cg_map(c,5)), p(cg_map(c,6)))) return
    end do

    ! Second pass: all coefficients on the level are non-zero by the
    ! selection rules; evaluate them.
    prod = 1.0_real64
    do c = 1, n_cg
       if (cg_level(c) /= level) cycle
       prod = prod * cg(p(cg_map(c,1)), p(cg_map(c,2)), &
            p(cg_map(c,3)), p(cg_map(c,4)), p(cg_map(c,5)), p(cg_map(c,6)))
    end do
  end function partial_cg_product


  subroutine cg_product(param_ranges, phase_vector, f, cg_map, f_first, total_sum, &
       n_par, n_cg)
    ! Evaluate a generalized sum of products of Clebsch--Gordan (CG)
    ! coefficients of the form
    !
    !   total_sum = sum_p  phase(p) * f(p) * C(1) * C(2) * ... * C(n_cg),
    !
    ! where p = (p_1, ..., p_{n_par}) is a set of sum parameters (angular
    ! momenta and angular momentum projections), phase(p) is a real phase
    ! factor of the type (-1)^{a + b + ...} built from the parameters, f is
    ! an arbitrary scalar function of the parameters supplied by the caller
    ! (in practice a Python function passed through the f2py call-back
    ! mechanism) and C(1), ..., C(n_cg) are CG coefficients whose six
    ! arguments are sum parameters. As everywhere in the library, all sum
    ! parameters are integers equal to TWICE their physical value, so that
    ! half-integer angular momenta can be represented exactly.
    !
    ! Arguments:
    !
    !   param_ranges  Integer array of shape (n_par, 2). The first index is
    !                 the parameter index. param_ranges(i,1) and
    !                 param_ranges(i,2) are the smallest and largest
    !                 (doubled) values taken by parameter i; the summation
    !                 runs over ALL integers between them in steps of ONE
    !                 doubled unit (i.e. physical steps of one half). A
    !                 parameter is held constant by setting its minimum and
    !                 maximum equal. Values of the wrong integer/half-integer
    !                 character are pruned away automatically by the CG
    !                 selection rules, so rectangular ranges can be used
    !                 even when the physically allowed range of one
    !                 parameter depends on the value of another (e.g. a
    !                 projection -j <= m <= j with j itself a sum parameter,
    !                 or an intermediate momentum bounded by triangular
    !                 conditions).
    !
    !   phase_vector  Integer array of length n_par determining the phase
    !                 factor. In terms of the PHYSICAL (halved) parameter
    !                 values a_i = p_i / 2 the phase of each term is
    !
    !                   (-1)^(phase_vector(1)*a_1 + phase_vector(2)*a_2 + ...)
    !
    !                 If the exponent is a half-integer in a term that is
    !                 otherwise non-zero (non-zero CG product and non-zero
    !                 f), the input is inconsistent — the phase would be
    !                 imaginary — and the routine terminates with an error
    !                 message (an f2py caller is killed with it; this
    !                 follows the general error convention of the library).
    !                 Half-integer exponents in terms that vanish anyway are
    !                 allowed and simply do not contribute.
    !
    !   f             The function f(p). Through the Python interface any
    !                 callable can be passed; it is called as f(p) with p
    !                 the integer array of the current (doubled) parameter
    !                 values and it must return a real number. From Python:
    !                 def f(p): return <float>. A constant weight is
    !                 obtained with e.g. (lambda p: 1.0).
    !
    !   cg_map        Integer array of shape (n_cg, 6). The first index is
    !                 the CG coefficient index. cg_map(c,1:6) are the
    !                 (one-based) parameter indices that supply the six
    !                 arguments (j1, m1, j2, m2, J, M) of coefficient c,
    !                 i.e. C(c) = <j1,m1,j2,m2|J,M> with j1 = p(cg_map(c,1))
    !                 and so on. The same parameter may appear in any number
    !                 of slots and coefficients.
    !
    !   total_sum     The value of the summation (intent(out); through the
    !                 Python interface it is the return value).
    !
    !   f_first       Logical switch selecting the evaluation order at each
    !                 term of the summation. If .true., f is evaluated
    !                 first and the innermost CG coefficients are only
    !                 evaluated when f is non-zero — advantageous when f is
    !                 cheap and vanishes often. If .false. (the recommended
    !                 default), the CG product is completed first and f is
    !                 only called for terms whose CG product is non-zero —
    !                 advantageous when f is expensive, which is always the
    !                 case for a Python call-back, since each call crosses
    !                 the Fortran/Python boundary. Note that the switch only
    !                 concerns the coefficients on the innermost loop level:
    !                 coefficients hoisted to outer levels (see below) are
    !                 always evaluated before f, since their cost is shared
    !                 by many terms and a zero among them prunes the inner
    !                 loops without f ever being called.
    !
    !   n_par, n_cg   The numbers of sum parameters and CG coefficients.
    !                 Hidden in the Python interface (inferred from the
    !                 array shapes).
    !
    ! Efficiency: the summation is organized as a loop nest in which
    ! parameter 1 is the outermost (slowest) and parameter n_par the
    ! innermost (fastest) loop. Two measures keep the cost close to the
    ! minimum possible:
    !
    !   (1) Loop hoisting. Each CG coefficient is assigned to the outermost
    !       loop level at which all its arguments are known (the largest
    !       index of any non-constant parameter it depends on) and is
    !       (re)evaluated only when that level advances; running partial
    !       products over the levels are kept in memory, so coefficients
    !       depending only on outer parameters are never recomputed in the
    !       inner loops. For maximum benefit the caller should order the
    !       parameters so that those appearing in many coefficients come
    !       first.
    !
    !   (2) Subtree pruning. Selection rules of every coefficient are
    !       checked with pure integer arithmetic (cg_is_allowed) before any
    !       floating-point evaluation, and whenever the partial product at
    !       some level is zero the entire inner loop nest below that level
    !       is skipped. Terms excluded by the selection rules therefore
    !       cost essentially nothing, which makes the rectangular parameter
    !       ranges affordable.
    !
    !   In addition, the shared log-factorial table used by cg is grown
    !   to its final size once, before the summation starts.
    !
    ! Numerical stability: the individual coefficients are accurate to
    ! double precision for arbitrarily large angular momenta (see cg)
    ! and the terms are accumulated in quadruple precision, so cancellation
    ! between terms does not degrade the double-precision result. The
    ! summation as a whole has been validated against exact rational
    ! arithmetic and analytic recoupling identities up to angular momenta
    ! of 40 (see test_fortran_utils.py).
    !
    ! Zero detection: terms are skipped by comparing against exact
    ! floating-point zeros. Selection-rule zeros of the CG coefficients are
    ! exact zeros, so they are always pruned; non-trivial (accidental)
    ! zeros may be represented by values of the order of the quadruple-
    ! precision epsilon (see cg) and are then simply accumulated, which
    ! is harmless. A user-supplied f must return an exact 0.0 for the
    ! pruning of vanishing f values to take effect.
    implicit none (type,external)

    integer, intent(in)                    :: n_par, n_cg
    integer, intent(in), dimension(n_par,2):: param_ranges
    integer, intent(in), dimension(n_par)  :: phase_vector
    integer, intent(in), dimension(n_cg,6) :: cg_map
    logical, intent(in)                    :: f_first
    external                               :: f
    real(real64)                           :: f
    real(real64), intent(out)              :: total_sum

    integer                          :: i, k, c, l, s
    integer, dimension(n_par)        :: p
    integer, dimension(n_cg)         :: cg_level
    real(real64), dimension(0:n_par) :: level_product
    real(real64)                     :: leaf_product, f_value, term
    real(real128)                    :: accumulator, table_warmup
    logical                          :: descending

    ! ---- Validation of the input. ----
    if (n_par < 1 .or. n_cg < 1) then
       write (*,*) 'ERROR in cg_utils.'
       write (*,*) 'ERROR: cg_product needs at least one sum parameter and one'
       write (*,*) '       CG coefficient.'
       write (*,*) 'Error termination.'
       error stop
    end if
    do i = 1, n_par
       if (param_ranges(i,1) > param_ranges(i,2)) then
          write (*,*) 'ERROR in cg_utils.'
          write (*,*) 'ERROR: In cg_product the minimum of a sum parameter range is'
          write (*,*) '       larger than its maximum.'
          write (*,*) 'Error termination.'
          error stop
       end if
    end do
    do c = 1, n_cg
       do k = 1, 6
          if (cg_map(c,k) < 1 .or. cg_map(c,k) > n_par) then
             write (*,*) 'ERROR in cg_utils.'
             write (*,*) 'ERROR: In cg_product the cg_map array refers to a sum'
             write (*,*) '       parameter index outside 1...n_parameters.'
             write (*,*) 'Error termination.'
             error stop
          end if
       end do
    end do

    ! ---- Grow the shared log-factorial table of math_utils to its final ----
    ! ---- size in one go. The largest argument cg can request is     ----
    ! ---- (j1 + j2 + J)/2 + 1 <= 3*max|p|/2 + 1.                         ----
    i = 0
    do l = 1, n_par
       i = max(i, abs(param_ranges(l,1)), abs(param_ranges(l,2)))
    end do
    table_warmup = log_factorial(3*i/2 + 2)

    ! ---- Hoisting level of each coefficient: the largest index of any   ----
    ! ---- non-constant parameter among its six arguments (0 if it only   ----
    ! ---- depends on constant parameters).                               ----
    do c = 1, n_cg
       cg_level(c) = 0
       do k = 1, 6
          i = cg_map(c,k)
          if (param_ranges(i,1) < param_ranges(i,2)) cg_level(c) = max(cg_level(c), i)
       end do
    end do

    ! ---- Constant coefficients (level 0) are evaluated exactly once. If ----
    ! ---- any of them vanishes the whole sum is zero.                    ----
    p = param_ranges(:,1)
    level_product(0) = partial_cg_product(p, cg_map, cg_level, 0, n_par, n_cg)
    total_sum = 0.0_real64
    if (level_product(0) == 0.0_real64) return

    ! ---- The loop nest over the parameters, implemented iteratively.    ----
    ! ---- "descending" means level l has just received a new value in    ----
    ! ---- p(l) and the levels below it must be (re)entered; otherwise    ----
    ! ---- p(l) is advanced, or, when its range is exhausted, control     ----
    ! ---- backtracks to level l-1.                                       ----
    accumulator = 0.0_real128
    l = 1
    p(1) = param_ranges(1,1)
    descending = .true.

    do
       if (descending) then
          if (l < n_par) then
             level_product(l) = level_product(l-1) &
                  * partial_cg_product(p, cg_map, cg_level, l, n_par, n_cg)
             if (level_product(l) == 0.0_real64) then
                ! Prune: the whole inner loop nest would only give zeros.
                descending = .false.
             else
                l = l + 1
                p(l) = param_ranges(l,1)
             end if
          else
             ! Innermost level: assemble and accumulate one term of the sum.
             term = 0.0_real64
             if (f_first) then
                f_value = f(p, n_par)
                if (f_value /= 0.0_real64) then
                   term = level_product(l-1) * f_value &
                        * partial_cg_product(p, cg_map, cg_level, l, n_par, n_cg)
                end if
             else
                leaf_product = level_product(l-1) &
                     * partial_cg_product(p, cg_map, cg_level, l, n_par, n_cg)
                if (leaf_product /= 0.0_real64) then
                   f_value = f(p, n_par)
                   term = leaf_product * f_value
                end if
             end if
             if (term /= 0.0_real64) then
                ! Twice the phase exponent; must be even in a non-zero term.
                s = 0
                do i = 1, n_par
                   s = s + phase_vector(i) * p(i)
                end do
                if (modulo(s,2) /= 0) then
                   write (*,*) 'ERROR in cg_utils.'
                   write (*,*) 'ERROR: In cg_product the phase exponent is a half-integer'
                   write (*,*) '       in a non-vanishing term of the summation, which'
                   write (*,*) '       would make the term imaginary. The phase vector is'
                   write (*,*) '       inconsistent with the summation structure.'
                   write (*,*) 'Error termination.'
                   error stop
                end if
                if (modulo(s,4) == 0) then
                   accumulator = accumulator + real(term, real128)
                else
                   accumulator = accumulator - real(term, real128)
                end if
             end if
             descending = .false.
          end if
       else
          if (p(l) < param_ranges(l,2)) then
             p(l) = p(l) + 1
             descending = .true.
          else
             l = l - 1
             if (l == 0) exit
          end if
       end if
    end do

    total_sum = real(accumulator, real64)
  end subroutine cg_product


  function special_cg(J,k) result(coeff)
    ! Evaluate the specific Clebsch--Gordan (CG) coeffient:
    !
    !     <J,J,k,0|J,J>
    !
    ! used the evaluation of Iwahara operator matrix elements. Use eq. (42) in Section
    ! 8.5.2 in
    !
    !   D. A. Varshalovich, A. N. Moskalev, V. K. Khersonskii.
    !   Quantum Theory of Angular Momentum.
    !   1988, World Scientific, Singapore.
    !

    integer, intent(in) :: J, k
    real(real64)        :: coeff

    integer :: largest_factorial = 0, max_index = 0
    
    integer, allocatable, dimension(:) :: sqrt_term

    ! Check the triangular condition.
    if (triangular_condition(J,k,J) .eqv. .false.) then
       coeff = dp_zero
    else
       largest_factorial = max(((2*J-k) / 2), ((2*J+k+2) / 2), ((2*J) / 2))
       
       max_index = largest_prime(largest_factorial)
       
       allocate(sqrt_term(1:max_index))

       ! We include the factor of 2J into the term. Since J is already multiplied
       ! by two, we will not multiply it futher. Because the 2J term is outside the
       ! square root expression, we need to raise it to second power inside the
       ! square root term; because of this we multiply the power-of-primes vector
       ! by two.
       sqrt_term = -power_of_primes_factorial((2*J-k) / 2, max_index) &
            - power_of_primes_factorial((2*J+k+2) / 2, max_index) &
            + 2*power_of_primes_factorial(J, max_index)
       
       coeff = sqrt((J+1.0d0) * evaluate_power_of_primes( sqrt_term,max_index))
    end if
  end function special_cg


end module cg_utils
