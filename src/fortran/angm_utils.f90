! SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
! SPDX-License-Identifier: GPL-3.0-or-later

module angm_utils
  
  use num_utils
  use math_utils
  use matrix_utils
  
  implicit none (type,external)
  
contains
  
  function wigner_small_d(J,M1,M2,angle) result(d_value)
    ! Evaluate a Wigner d matrix element
    !
    !     d^J_{M1,M2}(angle)
    !
    ! in the Condon--Shortley phase convention of Section 4.3 of
    !
    !   D. A. Varshalovich, A. N. Moskalev, V. K. Khersonskii.
    !   Quantum Theory of Angular Momentum.
    !   1988, World Scientific, Singapore.
    !
    ! (the same convention as old_wigner_small_d and wigner_big_d). The
    ! angular momentum arguments are given as TWICE their physical value,
    ! following the convention used throughout the library, so that
    ! half-integer angular momenta can be passed as integers.
    !
    ! Numerical strategy: instead of the alternating sum of equation (2) in
    ! Section 4.3.1, which suffers from catastrophic cancellation at large
    ! angular momenta, the element is built by the three-term recurrence in
    ! the angular momentum j at fixed (M1, M2). The recurrence follows from
    ! the Clebsch--Gordan series of the product D^j_{m',m} D^1_{0,0}:
    !
    !   cos(b) d^j_{m',m} = a+_j d^(j+1)_{m',m} + a0_j d^j_{m',m}
    !                     + a-_j d^(j-1)_{m',m} ,
    !
    !   a+_j = sqrt( (j-m'+1)(j+m'+1)(j-m+1)(j+m+1) ) / ((2j+1)(j+1)) ,
    !   a0_j = m' m / (j(j+1)) ,
    !   a-_j = sqrt( (j-m')(j+m')(j-m)(j+m) ) / (j(2j+1)) ,
    !
    ! solved for d^(j+1). The recurrence is seeded at j0 = max(|m'|, |m|),
    ! where the element has the exact single-sign closed form
    !
    !   d^j0_{m',m} = xi sqrt( (mu+nu)! / (mu! nu!) )
    !                 * sin^mu(b/2) cos^nu(b/2) ,
    !
    !   mu = |m-m'|,  nu = |m+m'|,
    !   xi = 1 for m >= m' and (-1)^(m'-m) for m < m'
    !
    ! (a-_j0 vanishes identically, so no second seed is needed). Recursing
    ! upward in j runs in the direction in which the element grows, which
    ! makes the recurrence numerically stable in every regime, including
    ! the classically forbidden region where the elements are exponentially
    ! small; a recurrence in the degree of the equivalent Jacobi polynomial
    ! is unstable exactly there. The seed prefactor is evaluated from the
    ! cached quadruple-precision log(n!) table of the log_factorial
    ! function in math_utils and the recurrence is carried out in
    ! quadruple precision. The routine is therefore both much more
    ! efficient than the power-of-primes approach of old_wigner_small_d and
    ! numerically stable up to extremely high angular momenta: it retains
    ! full double precision at least up to 2J = 1800 (validated against
    ! exact rational arithmetic and by unitarity sum rules; accuracy starts
    ! to degrade gradually around 2J ~ 2000).
    implicit none (type,external)

    integer,      intent(in) :: J, M1, M2
    real(real64), intent(in) :: angle
    real(real64)             :: d_value

    integer       :: mu, nu, two_j, two_j0
    real(real128) :: x, cos_half, sin_half, prefactor_log, xi
    real(real128) :: d_previous, d_current, d_new
    real(real128) :: a_plus, a_zero, a_minus

    mu = abs(M1 - M2) / 2
    nu = abs(M1 + M2) / 2
    two_j0 = max(abs(M1), abs(M2))

    if (M2 >= M1) then
       xi = 1.0_real128
    else
       xi = real((-1)**((M1 - M2)/2), real128)
    end if

    cos_half = cos(0.5_real128 * real(angle, real128))
    sin_half = sin(0.5_real128 * real(angle, real128))
    x = cos(real(angle, real128))

    ! The seed element at j0 = max(|m'|, |m|).
    prefactor_log = 0.5_real128 * ( log_factorial(mu + nu) &
         - log_factorial(mu) - log_factorial(nu) )
    d_current  = xi * exp(prefactor_log) * sin_half**mu * cos_half**nu
    d_previous = 0.0_real128

    ! Recurrence upward in j (in doubled units) from j0 to J. All
    ! coefficient formulas are rewritten in the doubled integers, e.g.
    ! (j - m' + 1) = (two_j - M1 + 2)/2.
    do two_j = two_j0, J - 2, 2
       a_plus = sqrt( real(two_j - M1 + 2, real128) &
            *         real(two_j + M1 + 2, real128) &
            *         real(two_j - M2 + 2, real128) &
            *         real(two_j + M2 + 2, real128) ) &
            / (2.0_real128 * real(two_j + 1, real128) * real(two_j + 2, real128))
       if (two_j == 0) then
          ! j = 0 implies m' = m = 0; both a0 and a- vanish.
          a_zero  = 0.0_real128
          a_minus = 0.0_real128
       else
          a_zero  = real(M1, real128) * real(M2, real128) &
               / (real(two_j, real128) * real(two_j + 2, real128))
          a_minus = sqrt( real(two_j - M1, real128) &
               *          real(two_j + M1, real128) &
               *          real(two_j - M2, real128) &
               *          real(two_j + M2, real128) ) &
               / (2.0_real128 * real(two_j, real128) * real(two_j + 1, real128))
       end if
       d_new      = ((x - a_zero) * d_current - a_minus * d_previous) / a_plus
       d_previous = d_current
       d_current  = d_new
    end do

    d_value = real(d_current, real64)
  end function wigner_small_d


  function old_wigner_small_d(J,M1,M2,angle) result(d_value)
    ! Evaluate a Wigner d matrix using equation (2) in Section 4.3.1 of
    !
    !   D. A. Varshalovich, A. N. Moskalev, V. K. Khersonskii.
    !   Quantum Theory of Angular Momentum.
    !   1988, World Scientific, Singapore.
    !
    ! This is the original implementation based on power-of-primes
    ! factorial decompositions. It is retained for reference and testing;
    ! new code should use the more efficient wigner_small_d above, which
    ! gives the same values.
    !
    ! This is not a very efficient implementation and it is assumed that this
    ! function will not be called repeatedly in a computationally intensive
    ! routine.
    implicit none (type,external)
    
    integer, intent(in)      :: J, M1, M2
    real(real64), intent(in) :: angle
    real(real64)             :: d_value
    
    integer :: largest_factorial = 0, max_index = 0
    integer :: phase = 0
    integer :: k = 0, k_max = 0, k_min = 0
    
    integer, allocatable, dimension(:) :: sqrt_term, sum_term
    
    real(real64) :: sum = 0.0, A = 0.0, B = 0.0
    
    
    k_max = min((J-M1)/2,(J-M2)/2)
    k_min = max(0, -(M1+M2)/2)
    
    largest_factorial = max((J+M1)/2, (J-M1)/2, (J+M2)/2, (J-M2)/2, &
         k_max, (J-M1)/2 - k_min, (J-M2)/2 - k_min, (M1+M2)/2 + k_max)
    max_index = largest_prime(largest_factorial)
    
    allocate(sqrt_term(1:max_index))
    allocate(sum_term(1:max_index))
    
    phase = (-1)**((J-M2) / 2)
    
    sqrt_term = power_of_primes_factorial((J+M1)/2, max_index) &
         + power_of_primes_factorial((J-M1)/2, max_index) &
         + power_of_primes_factorial((J+M2)/2, max_index) &
         + power_of_primes_factorial((J-M2)/2, max_index)
    
    sum = 0.0
    do k = k_min, k_max
       sum_term = power_of_primes_factorial(k, max_index) &
            + power_of_primes_factorial((J-M1-2*k)/2, max_index) &
            + power_of_primes_factorial((J-M2-2*k)/2, max_index) &
            + power_of_primes_factorial((M1+M2+2*k)/2, max_index)
       
       A = evaluate_power_of_primes(sqrt_term - sum_term, max_index)
       B = (cos(angle/2.0))**((M1 + M2) / 2 + 2*k) * (sin(angle/2.0))**(J - (M1 + M2) / 2 - 2*k)
       
       sum = sum + (-1)**k * A*B
    end do
    
    d_value = phase * 1.0 / sqrt(evaluate_power_of_primes(sqrt_term, max_index)) * sum

  end function old_wigner_small_d
    
    
  function wigner_big_d(J,M1,M2,alpha,beta,gamma) result(D_value)
    ! Evaluate a Wigner D matrix using equation (1) in Section 4.3.1 of
    !
    !   D. A. Varshalovich, A. N. Moskalev, V. K. Khersonskii.
    !   Quantum Theory of Angular Momentum.
    !   1988, World Scientific, Singapore.
    !
    implicit none (type,external)
    
    integer, intent(in)      :: J, M1, M2
    real(real64), intent(in) :: alpha, beta, gamma
    complex(real64)          :: D_value
    
    D_value = exp(-imaginary_unit * (real(M1,kind=real64)/2.0 * alpha &
         + real(M2,kind=real64)/2.0 * gamma)) * wigner_small_d(J,M1,M2,beta)
      
  end function wigner_big_d


  function wigner_6j(j1,j2,j3,l1,l2,l3) result(sixj)
    ! Evaluate the Wigner 6j symbol
    !
    !     { j1 j2 j3 }
    !     { l1 l2 l3 }
    !
    ! using the Racah formula, equation (1) in Section 9.2.1 of
    !
    !   D. A. Varshalovich, A. N. Moskalev, V. K. Khersonskii.
    !   Quantum Theory of Angular Momentum.
    !   1988, World Scientific, Singapore.
    !
    ! All six angular momentum arguments are given as TWICE their physical
    ! value, following the convention used throughout the library, so that
    ! half-integer angular momenta can be passed as integers. If any of the
    ! four triads (j1,j2,j3), (j1,l2,l3), (l1,j2,l3), (l1,l2,j3) violates the
    ! triangular condition, the symbol vanishes and zero is returned.
    !
    ! The sum is accumulated in quadruple precision using the tabulated
    ! log_factorial function of math_utils (see wigner_6j_r128), which keeps
    ! the result accurate to essentially full double precision for all
    ! arguments that can occur in atomic shell problems.
    implicit none (type,external)

    integer, intent(in) :: j1,j2,j3,l1,l2,l3
    real(real64)        :: sixj

    sixj = real(wigner_6j_r128(j1,j2,j3,l1,l2,l3), real64)
  end function wigner_6j


  function wigner_6j_r128(j1,j2,j3,l1,l2,l3) result(sixj)
    ! Quadruple-precision core of wigner_6j; see the documentation there.
    ! Keeping the full quadruple-precision value available internally allows
    ! contracted quantities (such as the 9j symbol in wigner_9j) to be
    ! accumulated without an intermediate rounding to double precision.
    !
    ! Note: this function is intended for Fortran-internal use. Its
    ! real(real128) result cannot be meaningfully accessed through the f2py
    ! interface.
    implicit none (type,external)

    integer, intent(in) :: j1,j2,j3,l1,l2,l3
    real(real128)       :: sixj

    integer       :: a1, a2, a3, a4, b1, b2, b3
    integer       :: t, t_min, t_max
    real(real128) :: delta_log, term

    sixj = 0.0_real128

    ! The symbol vanishes unless all four triads satisfy the triangular
    ! condition. The parity of the triads (integer perimeter) is also
    ! checked by triangular_condition because it proceeds in steps of two.
    if (.not. triangular_condition(j1,j2,j3)) return
    if (.not. triangular_condition(j1,l2,l3)) return
    if (.not. triangular_condition(l1,j2,l3)) return
    if (.not. triangular_condition(l1,l2,j3)) return

    ! Sums of the four triads (a) and of the three quadrangles (b). As the
    ! arguments are twice their physical values, all of these are integers.
    a1 = (j1 + j2 + j3) / 2
    a2 = (j1 + l2 + l3) / 2
    a3 = (l1 + j2 + l3) / 2
    a4 = (l1 + l2 + j3) / 2
    b1 = (j1 + j2 + l1 + l2) / 2
    b2 = (j2 + j3 + l2 + l3) / 2
    b3 = (j3 + j1 + l3 + l1) / 2

    ! Logarithm of the product of the four triangle coefficients
    ! Delta(j1,j2,j3), Delta(j1,l2,l3), Delta(l1,j2,l3) and Delta(l1,l2,j3),
    ! where Delta(a,b,c)^2 = (a+b-c)!(a-b+c)!(-a+b+c)!/(a+b+c+1)!.
    delta_log = 0.0_real128
    delta_log = delta_log + triangle_coefficient_log(j1,j2,j3)
    delta_log = delta_log + triangle_coefficient_log(j1,l2,l3)
    delta_log = delta_log + triangle_coefficient_log(l1,j2,l3)
    delta_log = delta_log + triangle_coefficient_log(l1,l2,j3)

    t_min = max(a1, a2, a3, a4)
    t_max = min(b1, b2, b3)

    do t = t_min, t_max
       term = log_factorial(t+1) &
            - log_factorial(t - a1) - log_factorial(t - a2) &
            - log_factorial(t - a3) - log_factorial(t - a4) &
            - log_factorial(b1 - t) - log_factorial(b2 - t) &
            - log_factorial(b3 - t)
       sixj = sixj + real((-1)**t, real128) * exp(delta_log + term)
    end do
  end function wigner_6j_r128


  function triangle_coefficient_log(a,b,c) result(log_delta)
    ! Return the logarithm of the triangle coefficient
    !
    !   Delta(a,b,c) = sqrt( (a+b-c)! (a-b+c)! (-a+b+c)! / (a+b+c+1)! )
    !
    ! in quadruple precision. The arguments are angular momenta given as
    ! TWICE their physical values and must satisfy the triangular condition;
    ! this is not checked here as the function is only intended as a helper
    ! for wigner_6j_r128.
    !
    ! Note: this function is intended for Fortran-internal use. Its
    ! real(real128) result cannot be meaningfully accessed through the f2py
    ! interface.
    implicit none (type,external)

    integer, intent(in) :: a, b, c
    real(real128)       :: log_delta

    log_delta = 0.5_real128 * ( &
           log_factorial((a + b - c) / 2) &
         + log_factorial((a - b + c) / 2) &
         + log_factorial((-a + b + c) / 2) &
         - log_factorial((a + b + c) / 2 + 1) )
  end function triangle_coefficient_log


  function wigner_9j(j1,j2,j3,j4,j5,j6,j7,j8,j9) result(ninej)
    ! Evaluate the Wigner 9j symbol
    !
    !     { j1 j2 j3 }
    !     { j4 j5 j6 }
    !     { j7 j8 j9 }
    !
    ! through its contraction over a product of three 6j symbols, equation
    ! (20) in Section 10.2.4 of
    !
    !   D. A. Varshalovich, A. N. Moskalev, V. K. Khersonskii.
    !   Quantum Theory of Angular Momentum.
    !   1988, World Scientific, Singapore.
    !
    !   {..} = sum_x (-1)^(2x) (2x+1) {j1 j2 j3; j6 j9 x}
    !                                 {j4 j5 j6; j2 x j8}
    !                                 {j7 j8 j9; x j1 j4} .
    !
    ! All nine angular momentum arguments are given as TWICE their physical
    ! value, following the convention used throughout the library, so that
    ! half-integer angular momenta can be passed as integers. If any row or
    ! column triad violates the triangular condition, the symbol vanishes
    ! and zero is returned.
    !
    ! The 6j symbols and the contraction sum are evaluated entirely in
    ! quadruple precision (wigner_6j_r128), so no accuracy is lost to the
    ! alternating sum before the final rounding to double precision.
    implicit none (type,external)

    integer, intent(in) :: j1,j2,j3,j4,j5,j6,j7,j8,j9
    real(real64)        :: ninej

    integer       :: x, x_min, x_max
    real(real128) :: total

    ninej = 0.0

    ! Row and column triads.
    if (.not. triangular_condition(j1,j2,j3)) return
    if (.not. triangular_condition(j4,j5,j6)) return
    if (.not. triangular_condition(j7,j8,j9)) return
    if (.not. triangular_condition(j1,j4,j7)) return
    if (.not. triangular_condition(j2,j5,j8)) return
    if (.not. triangular_condition(j3,j6,j9)) return

    x_min = max(abs(j1 - j9), abs(j2 - j6), abs(j4 - j8))
    x_max = min(j1 + j9, j2 + j6, j4 + j8)

    total = 0.0_real128
    do x = x_min, x_max, 2
       ! The phase (-1)^(2x) equals (-1)^x in the doubled-integer
       ! representation of x.
       total = total + real((-1)**x, real128) * real(x + 1, real128) &
            * wigner_6j_r128(j1, j2, j3, j6, j9, x) &
            * wigner_6j_r128(j4, j5, j6, j2, x, j8) &
            * wigner_6j_r128(j7, j8, j9, x, j1, j4)
    end do

    ninej = real(total, real64)
  end function wigner_9j


  subroutine coupled_basis(J_array,M_array,coupled_J_array,coupled_M_array,n_basis,n_sites)
    ! Construct a coupled set of angularmomentum eigenstates from the
    ! uncoupled set of eigenstates defined in the J_array and M_array
    ! arrays.
    !
    ! NOTE!!! This subroutine is an unimplemented stub. The output arrays
    ! are initialized to zero so that no uninitialized memory is returned,
    ! but the results are not meaningful.
    implicit none (type,external)

    integer,                             intent(in)  :: n_basis, n_sites
    integer, dimension(n_basis,n_sites), intent(in)  :: J_array, M_array
    integer, dimension(n_basis,n_sites), intent(out) :: coupled_J_array, coupled_M_array

    coupled_J_array = 0
    coupled_M_array = 0
  end subroutine coupled_basis


end module angm_utils
