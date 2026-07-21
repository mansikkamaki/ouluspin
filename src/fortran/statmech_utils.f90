! SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
! SPDX-License-Identifier: GPL-3.0-or-later

module statmech_utils
  
  use num_utils
  use math_utils
  use matrix_utils

  implicit none (type,external)

contains

  function boltzmann_factor(T,E,k_B) result(factor)
    ! Evaluate a Boltzmann factor:
    !
    !     exp(-E / k_B * T)
    !
    implicit none
    real(real64), intent(in) :: T, E, k_B
    real(real64)             :: factor

    factor = exp(-E / (T*k_B))
  end function boltzmann_factor


  function canonical_partition_function(T,E,k_B,n_basis) result(function_value)
    ! Evaluate the canonical partition function at temperature T:
    !
    !     sum_i exp(-E_i / k_B * T)
    !
    implicit none (type,external)

    real(real64), intent(in)                     :: T, k_B
    integer,      intent(in)                     :: n_basis
    real(real64), intent(in), dimension(n_basis) :: E
    real(real64)                                 :: function_value

    function_value = sum(exp(-E / (k_B*T)))
  end function canonical_partition_function


  subroutine equilibrium_expectation_value(A,eig,C,T_list,k_B,n_points,n_basis,value_list,normalize)
    ! Calculate the expectation value of a property at temperatures
    ! defined in the vector T_list. The routine takes as arguments the eigenvalue
    ! vector and eigenvector matrix of the Hamiltonian (eig, C) and a matrix
    ! representation of the operator A as well as the Boltzmann constant (k_B). The
    ! expectation value evaluated at each temperature are stored in the value_list
    ! vector.
    !
    ! The property operator is first  transformed to the Hamiltonian eigenstate basis.
    ! Within this basis, the expectation value of property A is evaluated as
    !
    !     <A> = 1 / Q(T) * sum_i <i|A|i>*exp(-E_i / k_B*T),
    !
    ! where i indexes the Hamiltonian eigenstates. The partition funciton is
    ! defined as
    !
    !     Q(T) = sum_i exp(-E_i / k_B*T)
    !
    ! and is evaluated using the canonical_partition_function function. If the
    ! argument normalize is set to .false., the expectation value sum will not
    ! be divided by the partition function. This is useful if alternative
    ! partition function is used by the calling procedure.
    !
    ! Only the diagonal elements <i|A|i> of the transformed operator enter
    ! the expectation value, so the full transformation C^H * A * C is not
    ! formed. The product X = A * C is evaluated with one call to zgemm and
    ! the diagonal is then picked up as
    !
    !     <i|A|i> = sum_p conjg(C(p,i)) * X(p,i),
    !
    ! which costs one matrix multiplication instead of the two of the full
    ! transformation and needs no second temporary matrix.
    implicit none (type,external)

    integer,                                     intent(in)  :: n_points, n_basis
    real(real64),                                intent(in)  :: k_B
    real(real64),    dimension(n_basis),         intent(in)  :: eig
    complex(real64), dimension(n_basis,n_basis), intent(in)  :: A,C
    real(real64),    dimension(n_points),        intent(in)  :: T_list
    logical,                                     intent(in)  :: normalize
    real(real64),    dimension(n_points),        intent(out) :: value_list

    complex(real64), dimension(n_basis,n_basis) :: X
    real(real64),    dimension(n_basis)         :: expectation_value_vector, propability_vector

    ! The local variables are deliberately left uninitialized in their
    ! declarations: an initializer would give them the SAVE attribute, i.e.
    ! one instance shared by every thread, and this routine is called from
    ! inside the parallel loops of powder_magnetization_utils.
    integer      :: T_index, i
    real(real64) :: Q, T

    ! BLAS parameters.
    complex(real64), parameter :: ALPHA = cmplx(1.0,0.0,kind=real64), BETA = cmplx(0.0,0.0,kind=real64)

    real(real64), external :: ddot
    external zgemm

    value_list = dp_zero

    call zgemm('N','N',n_basis,n_basis,n_basis,ALPHA,A,n_basis,C,n_basis,BETA,X,n_basis)

    do i = 1, n_basis
       expectation_value_vector(i) = real(sum(conjg(C(:,i))*X(:,i)),kind=real64)
    end do

    ! Evaluate the expectation value.
    do T_index = 1, n_points
       T = T_list(T_index)
       propability_vector = exp(-eig / (k_B*T))

       if (normalize .eqv. .true.) then
          Q = canonical_partition_function(T,eig,k_B,n_basis)
          value_list(T_index) = ddot(n_basis,expectation_value_vector,1,propability_vector,1) / Q
       else
          value_list(T_index) = ddot(n_basis,expectation_value_vector,1,propability_vector,1)
       end if
    end do
  end subroutine equilibrium_expectation_value
  
end module statmech_utils
