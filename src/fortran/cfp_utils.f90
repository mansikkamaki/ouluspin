! SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
! SPDX-License-Identifier: GPL-3.0-or-later
!
! NOTE: This module has been written entirely by AI using
!       Claude Code (model Fable 5.0).

module cfp_utils

  ! ===========================================================================
  ! Coefficients of fractional parentage (CFPs) for s^n, p^n, d^n and f^n
  ! configurations,
  !
  !   ( l^(n-1) (alpha1 S1 L1) l; S L {| l^n alpha S L ) ,
  !
  ! i.e. the expansion coefficients of an antisymmetric n-electron term of the
  ! shell l^n in terms of the antisymmetric terms of l^(n-1) (the parents)
  ! coupled to one additional l electron. The tabulation of
  !
  !   C. W. Nielson, G. F. Koster. Spectroscopic Coefficients for the p^n,
  !   d^n, and f^n Configurations. 1963, MIT Press, Cambridge MA.
  !
  ! which follows the classification of
  !
  !   G. Racah. Phys. Rev. 1943, 63, 367 (Racah III) and
  !   G. Racah. Phys. Rev. 1949, 76, 1352 (Racah IV),
  !
  ! is the reference for the conventions used here (see the notes on phase
  ! conventions below).
  !
  ! Algorithm
  ! ---------
  ! The tables are built recursively from n = 1 up to the full shell
  ! n = 4l + 2. For every candidate (S, L) of l^n the matrix of the
  ! antisymmetrizing projector in the basis of the parent-coupled states
  ! |l^(n-1)(alpha1 S1 L1) l; S L> is
  !
  !   P = [ 1 - (n-1) T ] / n ,
  !
  ! where T is the matrix of the transposition of the last two electrons.
  ! T is evaluated from the CFPs of the previous level and 6j recoupling
  ! coefficients (the spectral method of Bayman and Lande, Nucl. Phys. 1966,
  ! 77, 1). P has eigenvalues exactly one (physical states) and zero
  ! (spurious states), and the components of the eigenvalue-one eigenvectors
  ! are the CFPs.
  !
  ! Within a degenerate (S, L) space the physical states are classified
  ! following Racah:
  !
  !   1. Terms with seniority v < n are constructed explicitly as the
  !      antisymmetrized image of the corresponding l^(n-2) term with an
  !      added electron pair coupled to 1S. Pair creation commutes with the
  !      seniority (R_(2l+1)) and G2 group structure, so these states inherit
  !      all classification labels from their l^(n-2) progenitors. This is
  !      exactly Racah's seniority-chain construction and also fixes the
  !      phases of these states (positive multiple of the pair-created
  !      state).
  !   2. The orthogonal complement holds the new terms with v = n. For the
  !      f shell, degenerate new terms are separated by diagonalizing
  !      Racah's G2 Casimir operator
  !
  !        G = [ 3 (U1 . U1) + 11 (U5 . U5) ] / 4 ,
  !
  !      whose eigenvalues g(u1,u2) = (u1^2 + u2^2 + u1 u2 + 5 u1 + 4 u2)/12
  !      identify the irreducible representation U = (u1 u2) of G2. For the
  !      s, p and d shells (S, L, v) is a complete classification and no
  !      further separation is needed.
  !
  ! Term labels
  ! -----------
  ! A term of l^n is labelled by (two_s, two_ll, seniority, w):
  !
  !   two_s     = 2S  (twice the total spin)
  !   two_ll    = 2L  (twice the total orbital angular momentum)
  !   seniority = v   (Racah's seniority quantum number)
  !   w         = ordinal (1, 2, ...) among all terms with the same (2S, 2L)
  !               of the configuration, counted in the canonical order
  !               defined below.
  !
  ! In addition, for the f shell the routines report twelve times the
  ! eigenvalue of the G2 Casimir operator (an integer), which identifies the
  ! representation U = (u1 u2) through 12 g = u1^2 + u2^2 + u1 u2 + 5 u1
  ! + 4 u2. For the s, p and d shells this label is reported as zero.
  !
  ! The canonical order of the terms of a configuration is: decreasing 2S,
  ! then increasing 2L, then increasing seniority, then increasing G2
  ! Casimir eigenvalue, and finally the order of construction (which for
  ! remaining degenerate pairs follows the order of the progenitor terms).
  !
  ! Phase conventions
  ! -----------------
  ! The CFPs of a term are defined up to the overall sign of the term's
  ! state. The signs are fixed as follows:
  !
  !   1. Terms with v < n: positive multiple of the antisymmetrized state
  !      |l^(n-2)(alpha v S L) (l^2 1S); S L> (Racah's convention).
  !   2. Terms with v = n: the CFP to the first parent (in the canonical
  !      order of the l^(n-1) terms) with a nonvanishing coefficient is
  !      positive.
  !
  ! Convention 1 is the one used in the construction of the Nielson--Koster
  ! tables; convention 2 and the resolution of the rare G2 tau degeneracies
  ! (repeated L within one U representation, f shell only) are deterministic
  ! choices documented here, and individual signs may differ from the
  ! printed tables by an overall term phase in those cases. Absolute values,
  ! orthonormality relations and all classification labels are identical.
  !
  ! Beyond the half-filled shell (n > 2l + 1) the recursion is simply
  ! continued, which yields a consistent set of CFPs for all n up to the
  ! full shell.
  !
  ! Conventions and usage
  ! ---------------------
  ! All angular momentum arguments are passed as TWICE their physical value,
  ! following the convention used throughout the library. The shell is
  ! identified by two_l = 0, 2, 4, 6 for s, p, d, f. The seniority and the
  ! electron number n are NOT angular momenta and are passed as plain
  ! integers.
  !
  ! The tables of a shell are built once on first use and cached in module
  ! state, so all subsequent calls are simple table lookups. Building the
  ! complete f-shell tables takes on the order of a second. Because of the
  ! cached state, the first call for a given shell is not thread safe.
  !
  ! The exported routines are:
  !
  !   cfp                    A single coefficient
  !                          (l^(n-1) alpha1; l SL {| l^n alpha).
  !   cfp_number_of_terms    The number of terms of l^n.
  !   cfp_terms              The labels of all terms of l^n.
  !   cfp_full_table         The full CFP matrix between l^n and l^(n-1).
  !   cfp_build_tables       Build (and cache) the tables of a shell.
  !   cfp_u_tensor_element   A reduced matrix element of the unit tensor
  !                          operator U^(k) between two terms of l^n.
  !   cfp_u_tensor_matrix    The full U^(k) reduced-matrix-element matrix.
  !   cfp_v11_tensor_element A reduced matrix element of the double tensor
  !                          operator V^(11) between two terms of l^n.
  !   cfp_v11_tensor_matrix  The full V^(11) reduced-matrix-element matrix.
  !
  ! Unit tensor operators
  ! ---------------------
  ! The one-electron unit tensor operators follow Racah's definitions, which
  ! are also the ones used in the Nielson--Koster tables:
  !
  !   U^(k)  = sum_i u^(k)(i)   with   (l || u^(k) || l) = 1 ,
  !   V^(11) = sum_i (s u^(1))(i), a double tensor of rank one both in spin
  !            and in orbital space, with (s || s^(1) || s) = sqrt(3/2).
  !
  ! The reduced matrix elements are evaluated from the CFP tables through
  ! the standard parent-expansion formulas (one 6j recoupling per space;
  ! see e.g. Racah III or Judd, Operator Techniques in Atomic Spectroscopy)
  ! and use the Wigner--Eckart convention of Edmonds,
  !
  !   <a J M| T^k_q |a' J' M'> = (-1)^(J-M) 3j(J k J'; -M q M') (a J||T||a' J') ,
  !
  ! applied in the spin and orbital spaces separately for V^(11). U^(k) is
  ! diagonal in the spin quantum number and V^(11) obeys |S - S'| <= 1;
  ! vanishing elements are returned for non-coupling term pairs.
  ! ===========================================================================

  use num_utils
  use math_utils
  use matrix_utils
  use angm_utils

  implicit none (type,external)

  ! The largest orbital angular momentum (f shell) and the corresponding
  ! largest electron number of a full shell.
  integer, parameter, private :: max_shell_l   = 3
  integer, parameter, private :: max_electrons = 4*max_shell_l + 2

  ! Generous bounds for the block bookkeeping inside build_level: the
  ! largest allowed number of terms sharing one (S, L), the largest allowed
  ! number of distinct (S, L) blocks of one configuration and the largest
  ! allowed 2L/2 value. These are checked at run time.
  integer, parameter, private :: max_block_terms = 40
  integer, parameter, private :: max_blocks      = 150
  integer, parameter, private :: max_orbital_l   = 16

  ! Cached tables. A shell is built on first use. The term labels are stored
  ! per (l, n) and the CFP matrices per (l, n) as
  ! coeff_store(daughter_term, parent_term, l, n).
  logical, dimension(0:max_shell_l), save, private :: shell_built = .false.

  integer, dimension(0:max_shell_l,0:max_electrons), save, private :: n_terms_store = 0

  integer, dimension(cfp_max_terms,0:max_shell_l,0:max_electrons), save, private :: &
       two_s_store = 0, two_ll_store = 0, seniority_store = 0, w_store = 0, g12_store = 0

  real(real64), dimension(cfp_max_terms,cfp_max_terms,0:max_shell_l,1:max_electrons), &
       save, private :: coeff_store

  private :: validate_configuration, validate_tensor_rank, ensure_shell, &
       build_shell, build_level, find_term, pair_overlap_factor, &
       phase_factor, binomial_coefficient, u_tensor_element_by_index, &
       v11_tensor_element_by_index

contains

  function cfp(two_l,n,two_s,two_ll,seniority,w,two_s_parent,two_ll_parent, &
       seniority_parent,w_parent) result(coefficient)
    ! Return the coefficient of fractional parentage
    !
    !   ( l^(n-1) (alpha1 S1 L1) l; S L {| l^n alpha S L )
    !
    ! for the shell defined by two_l (= 0, 2, 4, 6 for s, p, d, f) and the
    ! electron number n (1 <= n <= 4l+2). The daughter term of l^n is given
    ! by (two_s, two_ll, seniority, w) and the parent term of l^(n-1) by
    ! (two_s_parent, two_ll_parent, seniority_parent, w_parent), with all
    ! angular momenta as twice their physical values and the labels as
    ! returned by cfp_terms. For n = 1 the parent is the vacuum term of l^0
    ! with labels (0, 0, 0, 1). Zero is returned if the parent does not
    ! couple to the daughter; unknown term labels produce a fatal error.
    implicit none (type,external)

    integer, intent(in) :: two_l, n, two_s, two_ll, seniority, w
    integer, intent(in) :: two_s_parent, two_ll_parent, seniority_parent, w_parent
    real(real64)        :: coefficient

    integer :: l, daughter_index, parent_index

    call validate_configuration(two_l, n, .false., l)
    call ensure_shell(l)

    daughter_index = find_term(l, n,   two_s,        two_ll,        seniority,        w)
    parent_index   = find_term(l, n-1, two_s_parent, two_ll_parent, seniority_parent, w_parent)

    coefficient = coeff_store(daughter_index, parent_index, l, n)
  end function cfp


  function cfp_number_of_terms(two_l,n) result(number_of_terms)
    ! Return the number of terms of the configuration l^n for the shell
    ! defined by two_l (= 0, 2, 4, 6 for s, p, d, f). The electron number
    ! may be anything between 0 (the vacuum, one term) and the full shell
    ! 4l+2.
    implicit none (type,external)

    integer, intent(in) :: two_l, n
    integer             :: number_of_terms

    integer :: l

    call validate_configuration(two_l, n, .true., l)
    call ensure_shell(l)

    number_of_terms = n_terms_store(l, n)
  end function cfp_number_of_terms


  subroutine cfp_terms(two_l,n,number_of_terms,two_s_array,two_ll_array, &
       seniority_array,w_array,g2_casimir_x12_array)
    ! Return the labels of all terms of the configuration l^n in the
    ! canonical order (see the module documentation). The number_of_terms
    ! argument must equal the value returned by cfp_number_of_terms and
    ! defines the length of the output arrays:
    !
    !   two_s_array           2S of each term.
    !   two_ll_array          2L of each term.
    !   seniority_array       Seniority v of each term.
    !   w_array               Ordinal among the terms with the same (2S, 2L).
    !   g2_casimir_x12_array  Twelve times the G2 Casimir eigenvalue
    !                         (f shell; zero for s, p and d shells), which
    !                         identifies the G2 representation U = (u1 u2)
    !                         through 12g = u1^2+u2^2+u1*u2+5*u1+4*u2.
    implicit none (type,external)

    integer,                             intent(in)  :: two_l, n, number_of_terms
    integer, dimension(number_of_terms), intent(out) :: two_s_array, two_ll_array
    integer, dimension(number_of_terms), intent(out) :: seniority_array, w_array
    integer, dimension(number_of_terms), intent(out) :: g2_casimir_x12_array

    integer :: l

    call validate_configuration(two_l, n, .true., l)
    call ensure_shell(l)

    if (number_of_terms /= n_terms_store(l, n)) then
       write (*,*) 'ERROR in cfp_utils.'
       write (*,*) 'ERROR: The number_of_terms argument of cfp_terms does not match'
       write (*,*) '       the actual number of terms of the configuration. Use'
       write (*,*) '       cfp_number_of_terms to obtain the correct value.'
       write (*,*) 'Error termination.'
       error stop
    end if

    two_s_array          = two_s_store(1:number_of_terms, l, n)
    two_ll_array         = two_ll_store(1:number_of_terms, l, n)
    seniority_array      = seniority_store(1:number_of_terms, l, n)
    w_array              = w_store(1:number_of_terms, l, n)
    g2_casimir_x12_array = g12_store(1:number_of_terms, l, n)
  end subroutine cfp_terms


  subroutine cfp_full_table(two_l,n,n_daughter_terms,n_parent_terms,table)
    ! Return the full CFP matrix between the configurations l^n and
    ! l^(n-1) as table(daughter_term, parent_term), with both configurations
    ! ordered canonically as returned by cfp_terms. The dimensions
    ! n_daughter_terms and n_parent_terms must equal the term counts of l^n
    ! and l^(n-1) returned by cfp_number_of_terms. This bulk access avoids
    ! repeated table lookups when many coefficients are needed.
    implicit none (type,external)

    integer, intent(in) :: two_l, n, n_daughter_terms, n_parent_terms
    real(real64), dimension(n_daughter_terms,n_parent_terms), intent(out) :: table

    integer :: l

    call validate_configuration(two_l, n, .false., l)
    call ensure_shell(l)

    if (n_daughter_terms /= n_terms_store(l, n) .or. &
        n_parent_terms   /= n_terms_store(l, n-1)) then
       write (*,*) 'ERROR in cfp_utils.'
       write (*,*) 'ERROR: The dimensions passed to cfp_full_table do not match the'
       write (*,*) '       actual term counts. Use cfp_number_of_terms to obtain'
       write (*,*) '       the correct values.'
       write (*,*) 'Error termination.'
       error stop
    end if

    table = coeff_store(1:n_daughter_terms, 1:n_parent_terms, l, n)
  end subroutine cfp_full_table


  subroutine cfp_build_tables(two_l)
    ! Build and cache the complete CFP tables of the shell defined by two_l
    ! (= 0, 2, 4, 6 for s, p, d, f) for all electron numbers n = 1, ...,
    ! 4l+2. Calling this routine is optional: the tables are built
    ! automatically on first use by the other routines. It is provided so
    ! that the construction cost can be paid at a controlled point (the
    ! first call for a given shell is not thread safe).
    implicit none (type,external)

    integer, intent(in) :: two_l

    integer :: l

    call validate_configuration(two_l, 1, .false., l)
    call ensure_shell(l)
  end subroutine cfp_build_tables


  function cfp_u_tensor_element(two_l,n,two_s,two_ll,seniority,w, &
       two_s_2,two_ll_2,seniority_2,w_2,two_k) result(element)
    ! Return the reduced matrix element
    !
    !   ( l^n alpha S L || U^(k) || l^n alpha' S' L' )
    !
    ! of the unit tensor operator U^(k) = sum_i u^(k)(i), with
    ! (l||u^(k)||l) = 1, evaluated from the CFP tables through the parent
    ! expansion
    !
    !   (a||U^(k)||a') = n sqrt([L][L']) sum_p (a{|p)(a'{|p)
    !                    * (-1)^(Lp + l + L' + k) {l L Lp; L' l k} .
    !
    ! The shell is defined by two_l (= 0, 2, 4, 6 for s, p, d, f) and the
    ! electron number n. The bra term is given by (two_s, two_ll,
    ! seniority, w) and the ket term by (two_s_2, two_ll_2, seniority_2,
    ! w_2), with the labels as returned by cfp_terms. The rank is passed as
    ! two_k = 2k, following the convention that all angular momenta are
    ! twice their physical value. Zero is returned when the selection rules
    ! (S = S', triangularity of L, k, L', k <= 2l) are not satisfied. See
    ! the module documentation for the conventions.
    implicit none (type,external)

    integer, intent(in) :: two_l, n, two_s, two_ll, seniority, w
    integer, intent(in) :: two_s_2, two_ll_2, seniority_2, w_2, two_k
    real(real64)        :: element

    integer :: l, index_1, index_2

    call validate_configuration(two_l, n, .false., l)
    call validate_tensor_rank(two_k)
    call ensure_shell(l)

    index_1 = find_term(l, n, two_s,   two_ll,   seniority,   w)
    index_2 = find_term(l, n, two_s_2, two_ll_2, seniority_2, w_2)

    element = u_tensor_element_by_index(l, n, index_1, index_2, two_k)
  end function cfp_u_tensor_element


  subroutine cfp_u_tensor_matrix(two_l,n,two_k,number_of_terms,matrix)
    ! Return the full matrix of reduced matrix elements
    !
    !   matrix(i,j) = ( l^n term_i || U^(k) || l^n term_j )
    !
    ! of the unit tensor operator U^(k) over all terms of the configuration
    ! l^n in the canonical order returned by cfp_terms. The number_of_terms
    ! argument must equal the value returned by cfp_number_of_terms. The
    ! rank is passed as two_k = 2k. This bulk routine evaluates the whole
    ! matrix with memoized 6j symbols and is much faster than calling
    ! cfp_u_tensor_element in a loop. See cfp_u_tensor_element and the
    ! module documentation for the definitions and conventions.
    implicit none (type,external)

    integer, intent(in) :: two_l, n, two_k, number_of_terms
    real(real64), dimension(number_of_terms,number_of_terms), intent(out) :: matrix

    integer      :: l, i, j, p, np, e2
    real(real64) :: accumulator, coeff_1, coeff_2

    real(real64), dimension(0:max_orbital_l,0:max_orbital_l,0:max_orbital_l) :: sixj_memo
    real(real64) :: memo_empty

    call validate_configuration(two_l, n, .false., l)
    call validate_tensor_rank(two_k)
    call ensure_shell(l)

    if (number_of_terms /= n_terms_store(l, n)) then
       write (*,*) 'ERROR in cfp_utils.'
       write (*,*) 'ERROR: The number_of_terms argument of cfp_u_tensor_matrix does'
       write (*,*) '       not match the actual number of terms. Use'
       write (*,*) '       cfp_number_of_terms to obtain the correct value.'
       write (*,*) 'Error termination.'
       error stop
    end if

    memo_empty = huge(dp_one)
    sixj_memo  = memo_empty
    np = n_terms_store(l, n-1)
    matrix = dp_zero

    do j = 1, number_of_terms
       do i = 1, number_of_terms
          if (two_s_store(i,l,n) /= two_s_store(j,l,n)) cycle
          if (.not. triangular_condition(two_ll_store(i,l,n), two_k, &
               two_ll_store(j,l,n))) cycle
          accumulator = dp_zero
          do p = 1, np
             coeff_1 = coeff_store(i, p, l, n)
             if (abs(coeff_1) < really_small_number) cycle
             coeff_2 = coeff_store(j, p, l, n)
             if (abs(coeff_2) < really_small_number) cycle
             if (sixj_memo(two_ll_store(i,l,n)/2, two_ll_store(j,l,n)/2, &
                  two_ll_store(p,l,n-1)/2) == memo_empty) then
                sixj_memo(two_ll_store(i,l,n)/2, two_ll_store(j,l,n)/2, &
                     two_ll_store(p,l,n-1)/2) = &
                     wigner_6j(two_l, two_ll_store(i,l,n), two_ll_store(p,l,n-1), &
                               two_ll_store(j,l,n), two_l, two_k)
             end if
             e2 = two_ll_store(p,l,n-1) + two_l + two_ll_store(j,l,n) + two_k
             accumulator = accumulator + coeff_1 * coeff_2 * phase_factor(e2) &
                  * sixj_memo(two_ll_store(i,l,n)/2, two_ll_store(j,l,n)/2, &
                              two_ll_store(p,l,n-1)/2)
          end do
          matrix(i,j) = accumulator * real(n, real64) &
               * sqrt(real((two_ll_store(i,l,n) + 1) &
               *            (two_ll_store(j,l,n) + 1), real64))
       end do
    end do
  end subroutine cfp_u_tensor_matrix


  function cfp_v11_tensor_element(two_l,n,two_s,two_ll,seniority,w, &
       two_s_2,two_ll_2,seniority_2,w_2) result(element)
    ! Return the reduced matrix element
    !
    !   ( l^n alpha S L || V^(11) || l^n alpha' S' L' )
    !
    ! of the double tensor operator V^(11) = sum_i (s u^(1))(i), of rank
    ! one both in spin and in orbital space, with (s||s^(1)||s) =
    ! sqrt(3/2) and (l||u^(1)||l) = 1 (Racah's definition, as tabulated by
    ! Nielson and Koster). The element is evaluated from the CFP tables
    ! through the parent expansion
    !
    !   (a||V^(11)||a') = n sqrt(3/2) sqrt([S][S'][L][L'])
    !                     * sum_p (a{|p)(a'{|p)
    !                     * (-1)^(Sp + 1/2 + S' + 1) {1/2 S Sp; S' 1/2 1}
    !                     * (-1)^(Lp + l  + L' + 1)  {l   L Lp; L' l   1} ,
    !
    ! where the reduction is carried out in the spin and orbital spaces
    ! separately (see the module documentation for the conventions). The
    ! shell is defined by two_l (= 0, 2, 4, 6 for s, p, d, f) and the
    ! electron number n; the bra and ket terms are given by their labels as
    ! returned by cfp_terms. Zero is returned when the selection rules
    ! (triangularity of S, 1, S' and of L, 1, L') are not satisfied.
    implicit none (type,external)

    integer, intent(in) :: two_l, n, two_s, two_ll, seniority, w
    integer, intent(in) :: two_s_2, two_ll_2, seniority_2, w_2
    real(real64)        :: element

    integer :: l, index_1, index_2

    call validate_configuration(two_l, n, .false., l)
    call ensure_shell(l)

    index_1 = find_term(l, n, two_s,   two_ll,   seniority,   w)
    index_2 = find_term(l, n, two_s_2, two_ll_2, seniority_2, w_2)

    element = v11_tensor_element_by_index(l, n, index_1, index_2)
  end function cfp_v11_tensor_element


  subroutine cfp_v11_tensor_matrix(two_l,n,number_of_terms,matrix)
    ! Return the full matrix of reduced matrix elements
    !
    !   matrix(i,j) = ( l^n term_i || V^(11) || l^n term_j )
    !
    ! of the double tensor operator V^(11) over all terms of the
    ! configuration l^n in the canonical order returned by cfp_terms. The
    ! number_of_terms argument must equal the value returned by
    ! cfp_number_of_terms. See cfp_v11_tensor_element and the module
    ! documentation for the definitions and conventions.
    implicit none (type,external)

    integer, intent(in) :: two_l, n, number_of_terms
    real(real64), dimension(number_of_terms,number_of_terms), intent(out) :: matrix

    integer :: l, i, j

    call validate_configuration(two_l, n, .false., l)
    call ensure_shell(l)

    if (number_of_terms /= n_terms_store(l, n)) then
       write (*,*) 'ERROR in cfp_utils.'
       write (*,*) 'ERROR: The number_of_terms argument of cfp_v11_tensor_matrix does'
       write (*,*) '       not match the actual number of terms. Use'
       write (*,*) '       cfp_number_of_terms to obtain the correct value.'
       write (*,*) 'Error termination.'
       error stop
    end if

    do j = 1, number_of_terms
       do i = 1, number_of_terms
          matrix(i,j) = v11_tensor_element_by_index(l, n, i, j)
       end do
    end do
  end subroutine cfp_v11_tensor_matrix


  ! ==========================================================================
  ! Internal routines. These are not part of the exported interface.
  ! ==========================================================================

  function u_tensor_element_by_index(l,n,index_1,index_2,two_k) result(element)
    ! The U^(k) reduced matrix element between the terms with the given
    ! stored indices; see cfp_u_tensor_element for the formula. The tables
    ! of the shell must already have been built.
    implicit none (type,external)

    integer, intent(in) :: l, n, index_1, index_2, two_k
    real(real64)        :: element

    integer      :: p, np, e2
    real(real64) :: coeff_1, coeff_2, accumulator

    element = dp_zero
    if (two_s_store(index_1,l,n) /= two_s_store(index_2,l,n)) return
    if (.not. triangular_condition(two_ll_store(index_1,l,n), two_k, &
         two_ll_store(index_2,l,n))) return

    np = n_terms_store(l, n-1)
    accumulator = dp_zero
    do p = 1, np
       coeff_1 = coeff_store(index_1, p, l, n)
       if (abs(coeff_1) < really_small_number) cycle
       coeff_2 = coeff_store(index_2, p, l, n)
       if (abs(coeff_2) < really_small_number) cycle
       e2 = two_ll_store(p,l,n-1) + 2*l + two_ll_store(index_2,l,n) + two_k
       accumulator = accumulator + coeff_1 * coeff_2 * phase_factor(e2) &
            * wigner_6j(2*l, two_ll_store(index_1,l,n), two_ll_store(p,l,n-1), &
                        two_ll_store(index_2,l,n), 2*l, two_k)
    end do

    element = accumulator * real(n, real64) &
         * sqrt(real((two_ll_store(index_1,l,n) + 1) &
         *            (two_ll_store(index_2,l,n) + 1), real64))
  end function u_tensor_element_by_index


  function v11_tensor_element_by_index(l,n,index_1,index_2) result(element)
    ! The V^(11) reduced matrix element between the terms with the given
    ! stored indices; see cfp_v11_tensor_element for the formula. The
    ! tables of the shell must already have been built.
    implicit none (type,external)

    integer, intent(in) :: l, n, index_1, index_2
    real(real64)        :: element

    integer      :: p, np, e2_spin, e2_orbital
    real(real64) :: coeff_1, coeff_2, accumulator

    element = dp_zero
    if (.not. triangular_condition(two_s_store(index_1,l,n), 2, &
         two_s_store(index_2,l,n))) return
    if (.not. triangular_condition(two_ll_store(index_1,l,n), 2, &
         two_ll_store(index_2,l,n))) return

    np = n_terms_store(l, n-1)
    accumulator = dp_zero
    do p = 1, np
       coeff_1 = coeff_store(index_1, p, l, n)
       if (abs(coeff_1) < really_small_number) cycle
       coeff_2 = coeff_store(index_2, p, l, n)
       if (abs(coeff_2) < really_small_number) cycle
       e2_spin    = two_s_store(p,l,n-1) + 1 + two_s_store(index_2,l,n) + 2
       e2_orbital = two_ll_store(p,l,n-1) + 2*l + two_ll_store(index_2,l,n) + 2
       accumulator = accumulator + coeff_1 * coeff_2 &
            * phase_factor(e2_spin) * phase_factor(e2_orbital) &
            * wigner_6j(1, two_s_store(index_1,l,n), two_s_store(p,l,n-1), &
                        two_s_store(index_2,l,n), 1, 2) &
            * wigner_6j(2*l, two_ll_store(index_1,l,n), two_ll_store(p,l,n-1), &
                        two_ll_store(index_2,l,n), 2*l, 2)
    end do

    element = accumulator * real(n, real64) * sqrt(1.5d0) &
         * sqrt(real((two_s_store(index_1,l,n) + 1) &
         *            (two_s_store(index_2,l,n) + 1) &
         *            (two_ll_store(index_1,l,n) + 1) &
         *            (two_ll_store(index_2,l,n) + 1), real64))
  end function v11_tensor_element_by_index


  subroutine validate_tensor_rank(two_k)
    ! Check that a tensor rank argument is a valid doubled integer rank
    ! (non-negative and even; ranks are orbital-space quantities and are
    ! therefore integers, but they are passed as twice their value
    ! following the library convention). A fatal error is produced for
    ! invalid input.
    implicit none (type,external)

    integer, intent(in) :: two_k

    if (two_k < 0 .or. modulo(two_k, 2) /= 0) then
       write (*,*) 'ERROR in cfp_utils.'
       write (*,*) 'ERROR: The tensor rank two_k must be a non-negative even integer'
       write (*,*) '       (the rank k is an integer passed as twice its value).'
       write (*,*) 'Error termination.'
       error stop
    end if
  end subroutine validate_tensor_rank


  subroutine validate_configuration(two_l,n,allow_zero_electrons,l)
    ! Check that two_l defines one of the supported shells (s, p, d or f)
    ! and that the electron number n is inside the valid range. Returns the
    ! physical orbital angular momentum l = two_l / 2. A fatal error is
    ! produced for invalid input.
    implicit none (type,external)

    integer, intent(in)  :: two_l, n
    logical, intent(in)  :: allow_zero_electrons
    integer, intent(out) :: l

    integer :: n_min

    if (two_l /= 0 .and. two_l /= 2 .and. two_l /= 4 .and. two_l /= 6) then
       write (*,*) 'ERROR in cfp_utils.'
       write (*,*) 'ERROR: The shell angular momentum two_l must be 0, 2, 4 or 6'
       write (*,*) '       (an s, p, d or f shell, given as twice its value).'
       write (*,*) 'Error termination.'
       error stop
    end if

    l = two_l / 2

    n_min = 1
    if (allow_zero_electrons) n_min = 0

    if (n < n_min .or. n > 4*l + 2) then
       write (*,*) 'ERROR in cfp_utils.'
       write (*,*) 'ERROR: The electron number n is outside the valid range of the'
       write (*,*) '       shell (at most 4l+2 electrons).'
       write (*,*) 'Error termination.'
       error stop
    end if
  end subroutine validate_configuration


  subroutine ensure_shell(l)
    ! Build the tables of the shell with orbital angular momentum l if they
    ! have not been built yet.
    implicit none (type,external)

    integer, intent(in) :: l

    if (.not. shell_built(l)) call build_shell(l)
  end subroutine ensure_shell


  subroutine build_shell(l)
    ! Build the complete CFP tables of the shell with (physical) orbital
    ! angular momentum l for all electron numbers, starting from the trivial
    ! configurations l^0 and l^1 and recursing upwards with build_level.
    implicit none (type,external)

    integer, intent(in) :: l

    integer :: n

    coeff_store(:,:,l,:) = dp_zero

    ! The vacuum l^0: a single 1S "term" with seniority zero.
    n_terms_store(l, 0)      = 1
    two_s_store(1, l, 0)     = 0
    two_ll_store(1, l, 0)    = 0
    seniority_store(1, l, 0) = 0
    w_store(1, l, 0)         = 1
    g12_store(1, l, 0)       = 0

    ! The one-electron configuration l^1: a single 2S+1 = 2 term with L = l
    ! and seniority one. For the f shell the G2 representation is (10) with
    ! Casimir eigenvalue 1/2, i.e. 12g = 6.
    n_terms_store(l, 1)      = 1
    two_s_store(1, l, 1)     = 1
    two_ll_store(1, l, 1)    = 2*l
    seniority_store(1, l, 1) = 1
    w_store(1, l, 1)         = 1
    if (l == 3) then
       g12_store(1, l, 1) = 6
    else
       g12_store(1, l, 1) = 0
    end if
    coeff_store(1, 1, l, 1) = dp_one

    do n = 2, 4*l + 2
       call build_level(l, n)
    end do

    shell_built(l) = .true.
  end subroutine build_shell


  subroutine build_level(l,n)
    ! Construct the terms and the CFP table of the configuration l^n from
    ! the already constructed tables of l^(n-1) and l^(n-2). See the module
    ! documentation for a description of the algorithm. Several internal
    ! consistency checks (projector idempotency, dimension counting,
    ! integrality of the G2 Casimir eigenvalues) produce fatal errors if the
    ! construction is inconsistent.
    implicit none (type,external)

    integer, intent(in) :: l, n

    integer :: two_l, np, ng
    integer :: ip, ig, ib, jb, a, b, q, i, j, r, k
    integer :: nch, nphys, nold, nnew, group_first, group_last
    integer :: exponent2
    integer :: n_blocks, n_records, w_count
    integer(int64) :: dimension_sum

    ! Block bookkeeping. blk_channel maps the channel index of a block to
    ! the parent term index; blk_channel_of_parent is the inverse map (zero
    ! if the parent is not a channel of the block). blk_evec holds a
    ! provisional orthonormal basis of the physical eigenspace of each
    ! block; it is complete but its internal mixing is arbitrary until the
    ! classification step.
    integer, dimension(max_blocks)                :: blk_two_s, blk_two_ll
    integer, dimension(max_blocks)                :: blk_nch, blk_nphys
    integer, dimension(cfp_max_terms,max_blocks)  :: blk_channel, blk_channel_of_parent
    real(real64), allocatable, dimension(:,:,:)   :: blk_evec

    ! Current-block quantities used by the contained helper routines.
    integer :: cur_block, cur_two_s, cur_two_ll

    ! Lazily filled 6j memo tables for the exchange matrix of the current
    ! block. The spin table is indexed by (2Sq, 2S1, 2S1') and the orbital
    ! table by (Lq, L1, L1') (physical values; the orbital momenta of terms
    ! are always integers).
    real(real64), dimension(0:15,0:15,0:15) :: sixj_spin_memo
    real(real64), dimension(0:max_orbital_l,0:max_orbital_l,0:max_orbital_l) :: sixj_orb_memo
    real(real64) :: memo_empty

    ! Per-block work arrays.
    real(real64), allocatable, dimension(:,:) :: tmat, proj, old_mat, new_mat
    real(real64), allocatable, dimension(:,:) :: pnew, gram, gmat, tvals
    real(real64), allocatable, dimension(:,:) :: g_inverse_sqrt
    real(real64), allocatable, dimension(:)   :: evals, yvec, ovec, gevals, gvals
    integer,      allocatable, dimension(:)   :: old_v, old_g12, new_g12
    integer                                   :: n_intermediates

    ! Records of the classified terms of this level before sorting.
    integer,      dimension(cfp_max_terms)               :: rec_two_s, rec_two_ll
    integer,      dimension(cfp_max_terms)               :: rec_v, rec_g12, rec_birth
    real(real64), dimension(cfp_max_terms,cfp_max_terms) :: rec_coeff

    real(real64) :: accumulator, y_norm, o_norm, orbital_denominator, casimir_g
    real(real64) :: sixj_s, sixj_l, coeff_a, coeff_b

    two_l = 2*l
    np = n_terms_store(l, n-1)
    ng = n_terms_store(l, n-2)

    memo_empty = huge(dp_one)

    ! -----------------------------------------------------------------------
    ! Collect the candidate (S, L) blocks by coupling every parent term with
    ! one l electron.
    ! -----------------------------------------------------------------------
    n_blocks = 0
    do ip = 1, np
       do i = abs(two_s_store(ip,l,n-1) - 1), two_s_store(ip,l,n-1) + 1, 2
          do j = abs(two_ll_store(ip,l,n-1) - two_l), two_ll_store(ip,l,n-1) + two_l, 2
             ! Register the block (i, j) = (2S, 2L) if it is new.
             b = 0
             do a = 1, n_blocks
                if (blk_two_s(a) == i .and. blk_two_ll(a) == j) then
                   b = a
                   exit
                end if
             end do
             if (b == 0) then
                n_blocks = n_blocks + 1
                if (n_blocks > max_blocks) then
                   write (*,*) 'ERROR in cfp_utils.'
                   write (*,*) 'ERROR: Too many (S, L) blocks. Increase max_blocks.'
                   write (*,*) 'Error termination.'
                   error stop
                end if
                if (j/2 > max_orbital_l) then
                   write (*,*) 'ERROR in cfp_utils.'
                   write (*,*) 'ERROR: Orbital angular momentum too large. Increase'
                   write (*,*) '       max_orbital_l.'
                   write (*,*) 'Error termination.'
                   error stop
                end if
                blk_two_s(n_blocks)  = i
                blk_two_ll(n_blocks) = j
             end if
          end do
       end do
    end do

    allocate(blk_evec(cfp_max_terms, max_block_terms, n_blocks))
    blk_evec = dp_zero

    ! -----------------------------------------------------------------------
    ! First pass: for every block build the exchange matrix, form the
    ! antisymmetrizing projector and extract a provisional orthonormal basis
    ! of its eigenvalue-one eigenspace.
    ! -----------------------------------------------------------------------
    do ib = 1, n_blocks
       cur_block  = ib
       cur_two_s  = blk_two_s(ib)
       cur_two_ll = blk_two_ll(ib)

       ! Assemble the channels (parents coupling to this (S, L)).
       nch = 0
       blk_channel_of_parent(:, ib) = 0
       do ip = 1, np
          if (triangular_condition(two_s_store(ip,l,n-1), 1, cur_two_s) .and. &
              triangular_condition(two_ll_store(ip,l,n-1), two_l, cur_two_ll)) then
             nch = nch + 1
             blk_channel(nch, ib) = ip
             blk_channel_of_parent(ip, ib) = nch
          end if
       end do
       blk_nch(ib) = nch
       if (nch == 0) then
          blk_nphys(ib) = 0
          cycle
       end if

       ! The matrix T of the transposition of the last two electrons,
       !
       !   T(a,b) = sum_q cfp(a,q) cfp(b,q) (-1)^(1 + S1a + S1b + L1a + L1b)
       !            * sqrt([S1a][S1b][L1a][L1b])
       !            * {Sq 1/2 S1a; S 1/2 S1b} {Lq l L1a; L l L1b} ,
       !
       ! where q runs over the grandparent terms of l^(n-2) and [x] = 2x+1.
       allocate(tmat(nch,nch), proj(nch,nch), evals(nch))
       tmat = dp_zero
       sixj_spin_memo = memo_empty
       sixj_orb_memo  = memo_empty

       do a = 1, nch
          do b = a, nch
             accumulator = dp_zero
             do q = 1, ng
                coeff_a = coeff_store(blk_channel(a,ib), q, l, n-1)
                if (abs(coeff_a) < really_small_number) cycle
                coeff_b = coeff_store(blk_channel(b,ib), q, l, n-1)
                if (abs(coeff_b) < really_small_number) cycle

                sixj_s = spin_sixj(two_s_store(q,l,n-2), &
                     two_s_store(blk_channel(a,ib),l,n-1), &
                     two_s_store(blk_channel(b,ib),l,n-1))
                sixj_l = orbital_sixj(two_ll_store(q,l,n-2), &
                     two_ll_store(blk_channel(a,ib),l,n-1), &
                     two_ll_store(blk_channel(b,ib),l,n-1))

                accumulator = accumulator + coeff_a * coeff_b * sixj_s * sixj_l
             end do

             exponent2 = 2 + two_s_store(blk_channel(a,ib),l,n-1) &
                  + two_s_store(blk_channel(b,ib),l,n-1) &
                  + two_ll_store(blk_channel(a,ib),l,n-1) &
                  + two_ll_store(blk_channel(b,ib),l,n-1)
             accumulator = accumulator * phase_factor(exponent2) &
                  * sqrt(real((two_s_store(blk_channel(a,ib),l,n-1) + 1) &
                  *           (two_s_store(blk_channel(b,ib),l,n-1) + 1) &
                  *           (two_ll_store(blk_channel(a,ib),l,n-1) + 1) &
                  *           (two_ll_store(blk_channel(b,ib),l,n-1) + 1), real64))

             tmat(a,b) = accumulator
             tmat(b,a) = accumulator
          end do
       end do

       ! The antisymmetrizing projector and its spectral decomposition.
       proj = -real(n-1, real64) / real(n, real64) * tmat
       do a = 1, nch
          proj(a,a) = proj(a,a) + dp_one / real(n, real64)
       end do

       call diagonalize_real_symmetric_matrix(proj, evals, nch, .true.)

       nphys = 0
       do a = 1, nch
          if (abs(evals(a)) > cfp_projector_tolerance .and. &
              abs(evals(a) - dp_one) > cfp_projector_tolerance) then
             write (*,*) 'ERROR in cfp_utils.'
             write (*,*) 'ERROR: An eigenvalue of the antisymmetrizing projector is not'
             write (*,*) '       zero or one. This indicates an internal inconsistency.'
             write (*,*) 'Error termination.'
             error stop
          end if
          if (evals(a) > 0.5d0) nphys = nphys + 1
       end do

       if (nphys > max_block_terms) then
          write (*,*) 'ERROR in cfp_utils.'
          write (*,*) 'ERROR: Too many terms in one (S, L) block. Increase'
          write (*,*) '       max_block_terms.'
          write (*,*) 'Error termination.'
          error stop
       end if

       blk_nphys(ib) = nphys
       if (nphys > 0) then
          ! Eigenvalues are in ascending order: the last nphys columns span
          ! the physical space.
          blk_evec(1:nch, 1:nphys, ib) = proj(1:nch, nch-nphys+1:nch)
       end if

       deallocate(tmat, proj, evals)
    end do

    ! -----------------------------------------------------------------------
    ! Second pass: classify the physical states of every block into terms
    ! with v < n (pair-created images of l^(n-2) terms) and new terms with
    ! v = n, and collect the term records.
    ! -----------------------------------------------------------------------
    n_records = 0
    rec_coeff = dp_zero

    do ib = 1, n_blocks
       if (blk_nphys(ib) == 0) cycle

       cur_block  = ib
       cur_two_s  = blk_two_s(ib)
       cur_two_ll = blk_two_ll(ib)
       nch        = blk_nch(ib)
       nphys      = blk_nphys(ib)

       allocate(old_mat(nch, nphys), yvec(nch), ovec(nch))
       allocate(old_v(nphys), old_g12(nphys))

       ! ---- Terms with seniority v < n: antisymmetrized pair-created
       ! images of the l^(n-2) terms with the same (S, L). The image of a
       ! progenitor beta is the projection of the overlap vector
       !
       !   y(a) = cfp_(n-1)(a, beta) * R2(S1a, L1a) ,
       !
       ! where R2 is the recoupling of |[beta, l] S1 L1, l; S L> onto
       ! |beta, (l^2) 1S; S L>, onto the physical eigenspace. Images that
       ! vanish (seniorities forbidden beyond the half-filled shell) are
       ! dropped.
       nold = 0
       do ig = 1, ng
          if (two_s_store(ig,l,n-2)  /= cur_two_s)  cycle
          if (two_ll_store(ig,l,n-2) /= cur_two_ll) cycle

          y_norm = dp_zero
          do a = 1, nch
             ip = blk_channel(a, ib)
             coeff_a = coeff_store(ip, ig, l, n-1)
             if (abs(coeff_a) < really_small_number) then
                yvec(a) = dp_zero
             else
                yvec(a) = coeff_a * pair_overlap_factor(cur_two_s, cur_two_ll, &
                     two_s_store(ip,l,n-1), two_ll_store(ip,l,n-1), two_l)
             end if
             y_norm = y_norm + yvec(a)**2
          end do
          y_norm = sqrt(y_norm)
          if (y_norm < really_small_number) cycle

          ! Project onto the physical eigenspace: o = E (E^T y).
          ovec = dp_zero
          do r = 1, nphys
             accumulator = dp_zero
             do a = 1, nch
                accumulator = accumulator + blk_evec(a, r, ib) * yvec(a)
             end do
             ovec(1:nch) = ovec(1:nch) + accumulator * blk_evec(1:nch, r, ib)
          end do
          o_norm = sqrt(sum(ovec(1:nch)**2))

          if (o_norm <= cfp_drop_tolerance * y_norm) cycle

          nold = nold + 1
          old_mat(1:nch, nold) = ovec(1:nch)
          old_v(nold)   = seniority_store(ig, l, n-2)
          old_g12(nold) = g12_store(ig, l, n-2)
       end do

       ! Loewdin (symmetric) orthonormalization of the images. For images
       ! with distinct classification labels this reduces to a
       ! normalization; genuinely overlapping images (tau-degenerate pairs)
       ! are orthonormalized without introducing an order dependence.
       if (nold > 0) then
          allocate(gram(nold,nold), gevals(nold))
          gram = matmul(transpose(old_mat(1:nch,1:nold)), old_mat(1:nch,1:nold))
          call diagonalize_real_symmetric_matrix(gram, gevals, nold, .true.)
          if (gevals(1) < really_small_number * gevals(nold)) then
             write (*,*) 'ERROR in cfp_utils.'
             write (*,*) 'ERROR: Nearly linearly dependent pair-created states. This'
             write (*,*) '       indicates an internal inconsistency.'
             write (*,*) 'Error termination.'
             error stop
          end if
          ! old_mat <- old_mat * G^(-1/2) with G^(-1/2) = V g^(-1/2) V^T.
          allocate(g_inverse_sqrt(nold,nold))
          do i = 1, nold
             do j = 1, nold
                accumulator = dp_zero
                do k = 1, nold
                   accumulator = accumulator + gram(i,k) * gram(j,k) / sqrt(gevals(k))
                end do
                g_inverse_sqrt(i,j) = accumulator
             end do
          end do
          old_mat(1:nch,1:nold) = matmul(old_mat(1:nch,1:nold), g_inverse_sqrt)
          deallocate(gram, gevals, g_inverse_sqrt)
       end if

       nnew = nphys - nold
       if (nnew < 0) then
          write (*,*) 'ERROR in cfp_utils.'
          write (*,*) 'ERROR: More pair-created states than physical states in one'
          write (*,*) '       (S, L) block. This indicates an internal inconsistency.'
          write (*,*) 'Error termination.'
          error stop
       end if

       ! ---- New terms with seniority v = n: the orthogonal complement of
       ! the pair-created images inside the physical eigenspace.
       if (nnew > 0) then
          allocate(new_mat(nch, nnew), new_g12(nnew))

          allocate(pnew(nch,nch), evals(nch))
          pnew = matmul(blk_evec(1:nch,1:nphys,ib), &
               transpose(blk_evec(1:nch,1:nphys,ib)))
          if (nold > 0) then
             pnew = pnew - matmul(old_mat(1:nch,1:nold), &
                  transpose(old_mat(1:nch,1:nold)))
          end if
          call diagonalize_real_symmetric_matrix(pnew, evals, nch, .true.)
          do a = 1, nch
             if (abs(evals(a)) > cfp_projector_tolerance .and. &
                 abs(evals(a) - dp_one) > cfp_projector_tolerance) then
                write (*,*) 'ERROR in cfp_utils.'
                write (*,*) 'ERROR: The projector on the new (v = n) terms is not'
                write (*,*) '       idempotent. This indicates an internal'
                write (*,*) '       inconsistency.'
                write (*,*) 'Error termination.'
                error stop
             end if
          end do
          if (count(evals > 0.5d0) /= nnew) then
             write (*,*) 'ERROR in cfp_utils.'
             write (*,*) 'ERROR: The dimension of the space of the new (v = n) terms'
             write (*,*) '       is inconsistent.'
             write (*,*) 'Error termination.'
             error stop
          end if
          new_mat(1:nch, 1:nnew) = pnew(1:nch, nch-nnew+1:nch)
          deallocate(pnew, evals)

          if (nnew > 1 .and. l /= 3) then
             write (*,*) 'ERROR in cfp_utils.'
             write (*,*) 'ERROR: Unexpected degeneracy of new terms in an s, p or d'
             write (*,*) '       shell, where (S, L, v) is a complete classification.'
             write (*,*) 'Error termination.'
             error stop
          end if

          if (l == 3) then
             ! Classify (and, for a single new term, label) the new states
             ! with the G2 Casimir operator G = [3 (U1.U1) + 11 (U5.U5)]/4.
             ! The (U5.U5) matrix is evaluated with the resolution of the
             ! identity over the provisional bases of all blocks with the
             ! same spin, which is complete regardless of their internal
             ! classification.
             n_intermediates = 0
             do jb = 1, n_blocks
                if (blk_two_s(jb) == cur_two_s) &
                     n_intermediates = n_intermediates + blk_nphys(jb)
             end do
             allocate(tvals(nnew, n_intermediates))
             tvals = dp_zero
             i = 0
             do jb = 1, n_blocks
                if (blk_two_s(jb) /= cur_two_s) cycle
                do r = 1, blk_nphys(jb)
                   i = i + 1
                   do j = 1, nnew
                      tvals(j, i) = u5_reduced_element(new_mat(:,j), ib, &
                           blk_evec(:,r,jb), jb)
                   end do
                end do
             end do

             allocate(gmat(nnew,nnew), gvals(nnew))
             orbital_denominator = real(l*(l+1)*(2*l+1), real64)
             do i = 1, nnew
                do j = i, nnew
                   accumulator = dp_zero
                   do k = 1, n_intermediates
                      accumulator = accumulator + tvals(i,k) * tvals(j,k)
                   end do
                   accumulator = 11.0d0 * accumulator / real(cur_two_ll + 1, real64)
                   if (i == j) then
                      accumulator = accumulator + 3.0d0 &
                           * real((cur_two_ll/2) * (cur_two_ll/2 + 1), real64) &
                           / orbital_denominator
                   end if
                   gmat(i,j) = accumulator / 4.0d0
                   gmat(j,i) = gmat(i,j)
                end do
             end do
             deallocate(tvals)

             call diagonalize_real_symmetric_matrix(gmat, gvals, nnew, .true.)
             ! Rotate the new states into the Casimir eigenbasis (ascending
             ! eigenvalues).
             new_mat(1:nch,1:nnew) = matmul(new_mat(1:nch,1:nnew), gmat)

             do i = 1, nnew
                casimir_g = 12.0d0 * gvals(i)
                new_g12(i) = nint(casimir_g)
                if (abs(casimir_g - real(new_g12(i), real64)) > 1.0d-5) then
                   write (*,*) 'ERROR in cfp_utils.'
                   write (*,*) 'ERROR: A G2 Casimir eigenvalue is not twelve times an'
                   write (*,*) '       integer. This indicates an internal'
                   write (*,*) '       inconsistency.'
                   write (*,*) 'Error termination.'
                   error stop
                end if
             end do

             ! Canonicalize the bases of tau-degenerate groups (repeated L
             ! within one G2 representation) with a deterministic
             ! Gram--Schmidt procedure against the canonical channel order.
             group_first = 1
             do while (group_first <= nnew)
                group_last = group_first
                do while (group_last < nnew)
                   if (abs(gvals(group_last+1) - gvals(group_first)) &
                        < cfp_degeneracy_tolerance) then
                      group_last = group_last + 1
                   else
                      exit
                   end if
                end do
                if (group_last > group_first) then
                   call canonicalize_subspace(new_mat, nch, nnew, &
                        group_first, group_last)
                end if
                group_first = group_last + 1
             end do

             deallocate(gmat, gvals)
          else
             new_g12 = 0
          end if

          ! Fix the sign of each new term: the CFP to the first parent (in
          ! canonical order) with a nonvanishing coefficient is positive.
          do i = 1, nnew
             do a = 1, nch
                if (abs(new_mat(a,i)) > small_number) then
                   if (new_mat(a,i) < dp_zero) new_mat(1:nch,i) = -new_mat(1:nch,i)
                   exit
                end if
             end do
          end do
       end if

       ! ---- Collect the term records of this block: first the pair-created
       ! images (in the canonical order of their progenitors), then the new
       ! terms (in ascending Casimir order).
       do i = 1, nold
          n_records = n_records + 1
          if (n_records > cfp_max_terms) then
             write (*,*) 'ERROR in cfp_utils.'
             write (*,*) 'ERROR: Too many terms. Increase cfp_max_terms in num_utils.'
             write (*,*) 'Error termination.'
             error stop
          end if
          rec_two_s(n_records)  = cur_two_s
          rec_two_ll(n_records) = cur_two_ll
          rec_v(n_records)      = old_v(i)
          rec_g12(n_records)    = old_g12(i)
          rec_birth(n_records)  = n_records
          do a = 1, nch
             rec_coeff(n_records, blk_channel(a,ib)) = old_mat(a,i)
          end do
       end do
       do i = 1, nnew
          n_records = n_records + 1
          if (n_records > cfp_max_terms) then
             write (*,*) 'ERROR in cfp_utils.'
             write (*,*) 'ERROR: Too many terms. Increase cfp_max_terms in num_utils.'
             write (*,*) 'Error termination.'
             error stop
          end if
          rec_two_s(n_records)  = cur_two_s
          rec_two_ll(n_records) = cur_two_ll
          rec_v(n_records)      = n
          rec_g12(n_records)    = new_g12(i)
          rec_birth(n_records)  = n_records
          do a = 1, nch
             rec_coeff(n_records, blk_channel(a,ib)) = new_mat(a,i)
          end do
       end do

       deallocate(old_mat, yvec, ovec, old_v, old_g12)
       if (nnew > 0) deallocate(new_mat, new_g12)
    end do

    deallocate(blk_evec)

    ! -----------------------------------------------------------------------
    ! Sort the records into the canonical order (stable insertion sort),
    ! assign the w ordinals and store the level.
    ! -----------------------------------------------------------------------
    call sort_records()

    n_terms_store(l, n) = n_records
    do i = 1, n_records
       two_s_store(i, l, n)     = rec_two_s(i)
       two_ll_store(i, l, n)    = rec_two_ll(i)
       seniority_store(i, l, n) = rec_v(i)
       g12_store(i, l, n)       = rec_g12(i)
       w_count = 1
       do j = 1, i - 1
          if (rec_two_s(j) == rec_two_s(i) .and. rec_two_ll(j) == rec_two_ll(i)) &
               w_count = w_count + 1
       end do
       w_store(i, l, n) = w_count
       coeff_store(i, 1:np, l, n) = rec_coeff(i, 1:np)
    end do

    ! Dimension check: the multiplet dimensions of the terms must add up to
    ! the binomial dimension of the configuration.
    dimension_sum = 0
    do i = 1, n_records
       dimension_sum = dimension_sum &
            + int(rec_two_s(i) + 1, int64) * int(rec_two_ll(i) + 1, int64)
    end do
    if (dimension_sum /= binomial_coefficient(4*l + 2, n)) then
       write (*,*) 'ERROR in cfp_utils.'
       write (*,*) 'ERROR: The multiplet dimensions of the constructed terms do not'
       write (*,*) '       add up to the dimension of the configuration. This'
       write (*,*) '       indicates an internal inconsistency.'
       write (*,*) 'Error termination.'
       error stop
    end if

  contains

    function spin_sixj(two_sq,two_sa,two_sb) result(value)
      ! Memoized 6j symbol {Sq 1/2 S1a; S 1/2 S1b} of the exchange matrix of
      ! the current block.
      implicit none (type,external)

      integer, intent(in) :: two_sq, two_sa, two_sb
      real(real64)        :: value

      if (sixj_spin_memo(two_sq,two_sa,two_sb) == memo_empty) then
         sixj_spin_memo(two_sq,two_sa,two_sb) = &
              wigner_6j(two_sq, 1, two_sa, cur_two_s, 1, two_sb)
      end if
      value = sixj_spin_memo(two_sq,two_sa,two_sb)
    end function spin_sixj


    function orbital_sixj(two_lq,two_la,two_lb) result(value)
      ! Memoized 6j symbol {Lq l L1a; L l L1b} of the exchange matrix of the
      ! current block.
      implicit none (type,external)

      integer, intent(in) :: two_lq, two_la, two_lb
      real(real64)        :: value

      if (sixj_orb_memo(two_lq/2,two_la/2,two_lb/2) == memo_empty) then
         sixj_orb_memo(two_lq/2,two_la/2,two_lb/2) = &
              wigner_6j(two_lq, two_l, two_la, cur_two_ll, two_l, two_lb)
      end if
      value = sixj_orb_memo(two_lq/2,two_la/2,two_lb/2)
    end function orbital_sixj


    function u5_reduced_element(vector_a,block_a,vector_b,block_b) result(element)
      ! Reduced matrix element <a S La || U^(5) || b S Lb> of the unit
      ! tensor operator U^(5) = sum_i u^(5)(i) between two states of l^n
      ! given by their CFP components over the channels of their respective
      ! blocks (Edmonds' convention for reduced matrix elements),
      !
      !   <a||U5||b> = n sqrt([La][Lb]) sum_p cfp_a(p) cfp_b(p)
      !                * (-1)^(Lp + l + Lb + 5) {l La Lp; Lb l 5} .
      !
      ! The two blocks must have the same spin (enforced by the caller). The
      ! vectors are assumed-size arrays; only the leading blk_nch elements
      ! of the respective blocks are referenced.
      implicit none (type,external)

      integer,                    intent(in) :: block_a, block_b
      real(real64), dimension(*), intent(in) :: vector_a, vector_b
      real(real64)                           :: element

      integer :: ca, cb, parent, e2

      element = dp_zero
      if (.not. triangular_condition(blk_two_ll(block_a), 10, blk_two_ll(block_b))) return

      do ca = 1, blk_nch(block_a)
         if (abs(vector_a(ca)) < really_small_number) cycle
         parent = blk_channel(ca, block_a)
         cb = blk_channel_of_parent(parent, block_b)
         if (cb == 0) cycle
         if (abs(vector_b(cb)) < really_small_number) cycle
         e2 = two_ll_store(parent,l,n-1) + two_l + blk_two_ll(block_b) + 10
         element = element + vector_a(ca) * vector_b(cb) * phase_factor(e2) &
              * wigner_6j(two_l, blk_two_ll(block_a), two_ll_store(parent,l,n-1), &
                          blk_two_ll(block_b), two_l, 10)
      end do
      element = element * real(n, real64) &
           * sqrt(real((blk_two_ll(block_a) + 1) * (blk_two_ll(block_b) + 1), real64))
    end function u5_reduced_element


    subroutine canonicalize_subspace(matrix,n_channels,n_columns,first,last)
      ! Replace the columns first..last of matrix (which span a
      ! tau-degenerate subspace) by a deterministic orthonormal basis of the
      ! same subspace, constructed by Gram--Schmidt orthonormalization of
      ! the projections of the canonical channel unit vectors onto the
      ! subspace.
      implicit none (type,external)

      integer,                                        intent(in)    :: n_channels, n_columns
      integer,                                        intent(in)    :: first, last
      real(real64), dimension(n_channels,n_columns), intent(inout)  :: matrix

      real(real64), dimension(n_channels,last-first+1) :: subspace_projector_columns
      real(real64), dimension(n_channels)              :: candidate
      integer      :: dimension_needed, accepted, channel, column, repeat_index
      real(real64) :: norm, overlap

      dimension_needed = last - first + 1
      accepted = 0

      do channel = 1, n_channels
         ! The projection of the channel unit vector onto the subspace.
         candidate = dp_zero
         do column = first, last
            candidate = candidate + matrix(channel, column) * matrix(:, column)
         end do
         ! Orthogonalize against the already accepted vectors (twice, for
         ! numerical stability).
         do repeat_index = 1, 2
            do column = 1, accepted
               overlap = dot_product(subspace_projector_columns(:, column), candidate)
               candidate = candidate - overlap * subspace_projector_columns(:, column)
            end do
         end do
         norm = sqrt(sum(candidate**2))
         if (norm > small_number) then
            accepted = accepted + 1
            subspace_projector_columns(:, accepted) = candidate / norm
            if (accepted == dimension_needed) exit
         end if
      end do

      if (accepted /= dimension_needed) then
         write (*,*) 'ERROR in cfp_utils.'
         write (*,*) 'ERROR: Failed to canonicalize a tau-degenerate subspace. This'
         write (*,*) '       indicates an internal inconsistency.'
         write (*,*) 'Error termination.'
         error stop
      end if

      matrix(:, first:last) = subspace_projector_columns(:, 1:dimension_needed)
    end subroutine canonicalize_subspace


    subroutine sort_records()
      ! Stable insertion sort of the term records into the canonical order:
      ! decreasing 2S, then increasing 2L, then increasing seniority, then
      ! increasing G2 Casimir eigenvalue, then order of construction.
      implicit none (type,external)

      integer :: outer, inner
      integer :: key_two_s, key_two_ll, key_v, key_g12, key_birth
      logical :: precedes
      real(real64), dimension(cfp_max_terms) :: key_coeff

      do outer = 2, n_records
         key_two_s  = rec_two_s(outer)
         key_two_ll = rec_two_ll(outer)
         key_v      = rec_v(outer)
         key_g12    = rec_g12(outer)
         key_birth  = rec_birth(outer)
         key_coeff  = rec_coeff(outer, :)

         inner = outer - 1
         do while (inner >= 1)
            ! True if the key record must be placed before the record at
            ! position inner in the canonical order.
            if (key_two_s /= rec_two_s(inner)) then
               precedes = key_two_s > rec_two_s(inner)
            else if (key_two_ll /= rec_two_ll(inner)) then
               precedes = key_two_ll < rec_two_ll(inner)
            else if (key_v /= rec_v(inner)) then
               precedes = key_v < rec_v(inner)
            else if (key_g12 /= rec_g12(inner)) then
               precedes = key_g12 < rec_g12(inner)
            else
               precedes = key_birth < rec_birth(inner)
            end if
            if (.not. precedes) exit
            rec_two_s(inner+1)  = rec_two_s(inner)
            rec_two_ll(inner+1) = rec_two_ll(inner)
            rec_v(inner+1)      = rec_v(inner)
            rec_g12(inner+1)    = rec_g12(inner)
            rec_birth(inner+1)  = rec_birth(inner)
            rec_coeff(inner+1,:) = rec_coeff(inner,:)
            inner = inner - 1
         end do

         rec_two_s(inner+1)  = key_two_s
         rec_two_ll(inner+1) = key_two_ll
         rec_v(inner+1)      = key_v
         rec_g12(inner+1)    = key_g12
         rec_birth(inner+1)  = key_birth
         rec_coeff(inner+1,:) = key_coeff
      end do
    end subroutine sort_records

  end subroutine build_level


  function find_term(l,n,two_s,two_ll,seniority,w) result(term_index)
    ! Return the index of the term with the labels (two_s, two_ll,
    ! seniority, w) in the stored table of the configuration l^n. A fatal
    ! error with a diagnostic message is produced if no such term exists.
    implicit none (type,external)

    integer, intent(in) :: l, n, two_s, two_ll, seniority, w
    integer             :: term_index

    integer :: i

    do i = 1, n_terms_store(l, n)
       if (two_s_store(i,l,n) == two_s .and. two_ll_store(i,l,n) == two_ll .and. &
           seniority_store(i,l,n) == seniority .and. w_store(i,l,n) == w) then
          term_index = i
          return
       end if
    end do

    write (*,*) 'ERROR in cfp_utils.'
    write (*,*) 'ERROR: No term with the requested labels (2S, 2L, seniority, w) ='
    write (*,*) '      ', two_s, two_ll, seniority, w
    write (*,*) '       exists in the configuration with n =', n
    write (*,*) '       electrons. Use cfp_terms to list the valid term labels.'
    write (*,*) 'Error termination.'
    error stop
  end function find_term


  function pair_overlap_factor(two_s,two_ll,two_s1,two_l1,two_l) result(factor)
    ! The recoupling coefficient of the parent-coupled state
    ! |[beta, l] S1 L1, l; S L> onto the pair-coupled state
    ! |beta, (l^2 1S); S L> with beta = (S L), i.e.
    !
    !   factor = (-1)^(2S + 1) (-1)^(S + 1/2 + S1) sqrt([S1] / (2 [S]))
    !          * (-1)^(L + l + L1)                 sqrt([L1] / ([L][l])) ,
    !
    ! where [x] = 2x + 1. This is used in the construction of the
    ! seniority-v < n terms as antisymmetrized pair-created states.
    implicit none (type,external)

    integer, intent(in) :: two_s, two_ll, two_s1, two_l1, two_l
    real(real64)        :: factor

    integer :: exponent2

    exponent2 = 3*two_s + two_s1 + 3 + two_ll + two_l + two_l1
    factor = phase_factor(exponent2) &
         * sqrt(real((two_s1 + 1) * (two_l1 + 1), real64) &
         /      real(2 * (two_s + 1) * (two_ll + 1) * (two_l + 1), real64))
  end function pair_overlap_factor


  function phase_factor(exponent2) result(factor)
    ! Return (-1)**(exponent2/2) where exponent2 is TWICE the phase
    ! exponent. The doubled exponent must be even (i.e. the physical
    ! exponent must be an integer); an odd value indicates an internal
    ! inconsistency and produces a fatal error.
    implicit none (type,external)

    integer, intent(in) :: exponent2
    real(real64)        :: factor

    if (modulo(exponent2, 2) /= 0) then
       write (*,*) 'ERROR in cfp_utils.'
       write (*,*) 'ERROR: A phase exponent is not an integer. This indicates an'
       write (*,*) '       internal inconsistency.'
       write (*,*) 'Error termination.'
       error stop
    end if

    if (modulo(exponent2/2, 2) == 0) then
       factor = dp_one
    else
       factor = -dp_one
    end if
  end function phase_factor


  function binomial_coefficient(n_total,n_chosen) result(coefficient)
    ! The binomial coefficient n_total over n_chosen as an exact 64-bit
    ! integer, evaluated with the standard product formula which stays
    ! integer-valued at every step.
    implicit none (type,external)

    integer, intent(in) :: n_total, n_chosen
    integer(int64)      :: coefficient

    integer :: i

    coefficient = 1
    do i = 1, n_chosen
       coefficient = coefficient * int(n_total - n_chosen + i, int64) / int(i, int64)
    end do
  end function binomial_coefficient

end module cfp_utils
