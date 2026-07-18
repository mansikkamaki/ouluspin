! SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
! SPDX-License-Identifier: GPL-3.0-or-later

module math_utils
  
  ! A module containing various math functions used by angm_utils and possibly other
  ! modules. This module MUST NOT depend on any other module than num_utils.
  ! use, intrinsic :: iso_fortran_env
  
  use num_utils
  
  implicit none (type,external)
  
contains

  function real_close_to(a,b) result(test_result)
    ! Check if a and b are close enough to be considered equal.
    implicit none (type,external)

    real(real64), intent(in) :: a, b
    logical                  :: test_result

    if (abs(a-b) < small_number) then
       test_result = .true.
    else
       test_result = .false.
    end if
  end function real_close_to

  
  function complex_close_to(a,b) result(test_result)
    ! Check if a and b are close enough to be considered equal.
    implicit none (type,external)

    complex(real64), intent(in) :: a, b
    logical                     :: test_result

    if (abs(a-b) < small_number) then
       test_result = .true.
    else
       test_result = .false.
    end if
  end function complex_close_to

  
  function real_is_zero(a) result(test_result)
    ! Check if a is close enough to to zero to be considered zero.
    implicit none (type,external)

    real(real64), intent(in) :: a
    logical                  :: test_result

    if (abs(a) < small_number) then
       test_result = .true.
    else
       test_result = .false.
    end if
  end function real_is_zero

  
  function complex_is_zero(a) result(test_result)
    ! Check if a is close enough to to zero to be considered zero.
    implicit none (type,external)

    complex(real64), intent(in) :: a
    logical                     :: test_result

    if (abs(a) < small_number) then
       test_result = .true.
    else
       test_result = .false.
    end if
  end function complex_is_zero

  
  recursive function factorial(n) result(answer)
    ! A recursive function to evaluate the factorial of the input argument n.
    ! A negative argument produces a fatal error.
    implicit none (type,external)
    
    integer, intent(in) :: n
    integer(int64)      :: answer
    
    if (n < 0) then
       write (*,*) 'ERROR in math_utils.'
       write (*,*) 'ERROR: Negative argument in factorial.'
       write (*,*) 'Error termination.'
       error stop
    else if (n > 20) then
       write (*,*)  'ERROR in math_utils.'
       write (*,*)  'ERROR: Argument in factorial is larger than 20.'
       write (*,*)  '       Such a number cannot be stored as an integer.'
       write (*,*)  'Error termination.'
       error stop
    else if (n == 0) then
       answer = 1
    else
       answer = n * factorial(n-1)
    end if
  end function factorial
  
  
  function log_factorial(n) result(log_fact)
    ! Return log(n!) in quadruple precision. The values are tabulated with
    ! the recurrence log(n!) = log((n-1)!) + log(n) on the first call and
    ! the table is grown geometrically whenever a larger argument is
    ! encountered, so that repeated calls reduce to array lookups. The table
    ! persists between calls; because of it the first call is marginally
    ! more expensive and concurrent first calls from multiple threads are
    ! not safe. A negative argument produces a fatal error.
    !
    ! Note: this function is intended for Fortran-internal use. Its
    ! real(real128) result cannot be meaningfully accessed through the f2py
    ! interface.
    implicit none (type,external)

    integer, intent(in) :: n
    real(real128)       :: log_fact

    real(real128), allocatable, dimension(:), save :: table
    integer,                                  save :: table_size = -1

    real(real128), allocatable, dimension(:) :: old_table
    integer                                  :: i, new_size

    if (n < 0) then
       write (*,*) 'ERROR in math_utils.'
       write (*,*) 'ERROR: Negative argument in log_factorial.'
       write (*,*) 'Error termination.'
       error stop
    end if

    if (table_size < 0) then
       ! First call: build the initial table.
       table_size = max(64, 2*n)
       allocate(table(0:table_size))
       table(0) = 0.0_real128
       do i = 1, table_size
          table(i) = table(i-1) + log(real(i,real128))
       end do
    else if (n > table_size) then
       ! Grow the table geometrically.
       new_size = max(2*table_size, 2*n)
       call move_alloc(table, old_table)
       allocate(table(0:new_size))
       table(0:table_size) = old_table(0:table_size)
       do i = table_size + 1, new_size
          table(i) = table(i-1) + log(real(i,real128))
       end do
       table_size = new_size
       deallocate(old_table)
    end if

    log_fact = table(n)
  end function log_factorial


  function triangular_condition(j1,j2,J) result(holds)
    ! Check the validity of the triangular condition for j1, j2. and J3.
    ! In practice this means that J is one of the values
    !
    !  j1+j2, j1+j2-2, ... , |j1-j2|
    !
    ! An increment of two is used instead of one as the angular momentum values are
    ! multiples of two.
    implicit none (type,external)
    
    integer, intent(in) :: j1,j2,J
    logical             :: holds
    integer             :: j3 = 0
    
    do j3 = abs(j1-j2),j1+j2,2
       if (j3 == J) then
          holds = .true.
          return
       end if
    end do
    holds = .false.
  end function triangular_condition
  
  
  function generate_list_of_primes(number_of_primes) result(list_of_primes)
    ! Generates a list of the first n primes where n is given by the number_of_primes
    ! argument. Return the list as an array of integers.
    implicit none (type,external)
    
    integer, intent(in)                      :: number_of_primes
    integer, dimension(1:number_of_primes)   :: list_of_primes
    integer                                  :: i = 0, j = 0, found = 0
    logical                                  :: isprime = .true.
    
    list_of_primes = 0
    
    if (number_of_primes == 0) then
       write (*,*) 'ERROR in math utils.'
       write (*,*) 'ERROR: Cannot give a list of zero primes.'
       write (*,*) 'Error termination.'
       error stop
    else
       list_of_primes(1) = 2
       found = 1
    end if
    
    if (number_of_primes >= 2) then
       i = 3
       do
          do j = 1,found
             isprime = .true.
             if (modulo(i, list_of_primes(j)) == 0) then
                isprime = .false.
                exit
             end if
          end do
          
          if (isprime) then
             found = found + 1
             list_of_primes(found) = i
          end if
          
          i = i + 2
          if (found == number_of_primes) then
             exit
          end if
       end do
    end if
  end function generate_list_of_primes
  
  
  function largest_prime(n) result(max_index)
    ! Find and return the largest index (max_index) of the prime p so that
    ! n/p >= 1, where n is given as argument.
    implicit none (type,external)
    
    integer, intent(in)                   :: n
    integer                               :: max_index
    integer, dimension(1:relevant_primes) :: list_of_primes
    
    list_of_primes = generate_list_of_primes(relevant_primes)
    max_index = 0
    do
       max_index = max_index + 1
       if (floor(dble(n) &
            / dble(list_of_primes(max_index))) < 1) then
          exit
       else if (max_index == relevant_primes) then
          write (*,*) 'ERROR in math_utils.'
          write (*,*) 'ERROR: List of primes ran out when tryign to find a prime p'
          write (*,*) '       so that n/p >= 1. You probably need a longer list of'
          write (*,*) '       primes. Increase the value of the relevant_primes'
          write (*,*) '       parameter.'
          write (*,*) 'Error termination.'
          error stop
       end if
    end do
  end function largest_prime
  
  
  function evaluate_power_of_primes(power_of_primes, max_index) result(pp_value)
    ! Takes as input an array listing powers of primes and evaluates its value.
    implicit none (type,external)
    
    integer, intent(in)                         :: max_index
    integer, intent(in), dimension(1:max_index) :: power_of_primes
    real(real64)                                :: pp_value
    real(real64)                                :: product = 0.0
    integer                                     :: i = 0, k = 0, p = 0
    integer, dimension(1:max_index)             :: list_of_primes
    
    list_of_primes = generate_list_of_primes(max_index)
    
    product = 1.0
    do k = 1,max_index
       p = list_of_primes(k)
       i = power_of_primes(k)
       product = product * real(p,real64)**i
    end do

    pp_value = product
  end function evaluate_power_of_primes
  
  
  function power_of_primes_factorial(n, max_index) result(power_of_primes)
    ! Takes as input a factorial n! and as output give its decomposition
    ! as power of primes. The output array is a list of the powers of each
    ! prime starting from the lowest.
    implicit none (type,external)
    
    integer, intent(in)             :: n, max_index
    integer, dimension(1:max_index) :: power_of_primes
    integer, dimension(1:max_index) :: list_of_primes
    integer                         :: power = 0
    integer                         :: p = 0, i = 0, k = 0
    
    list_of_primes = generate_list_of_primes(max_index)
    
    ! Define the powers of each prime p according to Legendre's formula.
    ! See Wikipedia for more information.
    do k = 1,max_index
       p = list_of_primes(k)
       i = 0
       power = 0
       do
          i = i + 1
          if (p**i > n) then
             exit
          else
             power = power + floor(dble(n) / dble(p**i))
          end if
       end do
       power_of_primes(k) = power
    end do
  end function power_of_primes_factorial
  
  
  function levi_civita(i,j,k) result(l)
    ! Evaluate the value of the Levi--Civita symbol epsilon_ijk in
    ! three dimensions. The values of i, j and k can have values
    ! between 1 and 3.
    implicit none (type,external)
    
    integer, intent(in) :: i,j,k
    integer             :: l
    
    if ((i == j) .or. (i == k) .or. (j == k)) then
       l = 0
    else if ((i == 1) .and. (j == 2) .and. (k == 3)) then
       l = 1
    else if ((i == 2) .and. (j == 3) .and. (k == 1)) then
       l = 1
    else if ((i == 3) .and. (j == 1) .and. (k == 2)) then
       l = 1
    else if ((i == 3) .and. (j == 2) .and. (k == 1)) then
       l = -1
    else if ((i == 1) .and. (j == 3) .and. (k == 2)) then
       l = -1
    else if ((i == 2) .and. (j == 1) .and. (k == 3)) then
       l = -1
    else
       write (*,*) 'ERROR in math_utils.'
       write (*,*) 'ERROR: Unknown values in Levi--Civita symbol.'
       write (*,*) 'Error termination.'
       error stop
    end if
  end function levi_civita
  
end module math_utils
