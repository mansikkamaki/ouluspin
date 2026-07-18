! SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
! SPDX-License-Identifier: GPL-3.0-or-later

module powder_magnetization_utils
  
  use num_utils
  use matrix_utils
  use statmech_utils

  implicit none (type,external)

contains

  subroutine diagonalize_hamiltonian(hamiltonian,magnetic_moment,eigenvalues,eigenvectors,B,n_basis, &
       compute_eigenvectors,translate_eigenvalues)
    ! Construct the full Hamiltonian from the field-free Hamiltonian and the Cartesian
    ! components of the magnetic moment operator in a specific field. Diagonalize the
    ! operator and return the eigenvalues and eigenvectors.
    implicit none (type,external)

    integer,         intent(in)                                :: n_basis
    real(real64),    intent(in),  dimension(3)                 :: B
    complex(real64), intent(in),  dimension(n_basis,n_basis)   :: hamiltonian
    complex(real64), intent(in),  dimension(3,n_basis,n_basis) :: magnetic_moment
    logical,         intent(in)                                :: compute_eigenvectors, translate_eigenvalues

    real(real64),    intent(out), dimension(n_basis)           :: eigenvalues
    complex(real64), intent(out), dimension(n_basis,n_basis)   :: eigenvectors

    integer :: i = 0

    real(real64) :: minimum

    eigenvectors = hamiltonian

    do i = 1,3
       eigenvectors = eigenvectors - B(i)*magnetic_moment(i,:,:)
    end do

    call diagonalize_complex_matrix(eigenvectors,eigenvalues,n_basis,compute_eigenvectors)

    if (translate_eigenvalues .eqv. .true.) then
       minimum = minval(eigenvalues)
       eigenvalues = eigenvalues - minimum
    end if
  end subroutine diagonalize_hamiltonian

  
  subroutine powder_magnetization(hamiltonian,magnetic_moment,T,B,M_scalar, &
       grid_vectors,grid_weights,k_B,n_basis,n_grid_points,n_T_points)
    ! Calculate the powder magnetization by spherical integration of the
    ! magnetization. The matrix representations of the field-free
    ! Hamiltonian and the Cartesian components of the magnetic moment
    ! operator are given as arguments, along with a vector containing
    ! the tempreature points, and the magnitude of the field. As a
    ! return value the subroutine provides a vector containing the scalar
    ! magnetization values evaluated at the different temperature points.
    !
    ! Note that no unit conversions or multiplications are carried out.
    implicit none (type,external)

    integer,         intent(in)                                :: n_basis, n_grid_points, n_T_points
    real(real64),    intent(in)                                :: B, k_B
    complex(real64), intent(in),  dimension(n_basis,n_basis)   :: hamiltonian
    complex(real64), intent(in),  dimension(3,n_basis,n_basis) :: magnetic_moment
    real(real64),    intent(in),  dimension(n_T_points)        :: T
    real(real64),    intent(in),  dimension(n_grid_points,3)   :: grid_vectors
    real(real64),    intent(in),  dimension(n_grid_points)     :: grid_weights
    real(real64),    intent(out), dimension(n_T_points)        :: M_scalar

    real(real64),    dimension(3)               :: B_vector
    real(real64),    dimension(n_basis)         :: eigenvalues
    real(real64),    dimension(n_T_points)      :: M_vector_component
    complex(real64), dimension(n_basis,n_basis) :: eigenvectors
    
    integer :: i = 0, j = 0

    M_scalar = dp_zero

    ! This loop should be parallelized.
    do i = 1, n_grid_points
       B_vector = B * grid_vectors(i,:)
       call diagonalize_hamiltonian(hamiltonian,magnetic_moment,eigenvalues,eigenvectors,B_vector,n_basis,.true.,.true.)

       M_vector_component = dp_zero
       do j = 1,3
          call equilibrium_expectation_value(magnetic_moment(j,:,:),eigenvalues,eigenvectors, &
               T,k_B,n_T_points,n_basis,M_vector_component,.true.)

          M_scalar = M_scalar + grid_weights(i) * grid_vectors(i,j) * M_vector_component
       end do
    end do

  end subroutine powder_magnetization


  subroutine free_rotation_magnetization(hamiltonian,magnetic_moment,T_list,B,M_scalar, &
       grid_vectors,grid_weights,k_B,n_basis,n_grid_points,n_T_points)
    ! Calculate the powder magnetization by spherical integration of the
    ! magnetization. The matrix representations of the field-free
    ! Hamiltonian and the Cartesian components of the magnetic moment
    ! operator are given as arguments, along with a vector containing
    ! the tempreature points, and the magnitude of the field. As a
    ! return value the subroutine provides a vector containing the scalar
    ! magentization values evaluated at the different temperature points.
    implicit none (type,external)

    integer,         intent(in)                                :: n_basis, n_grid_points, n_T_points
    real(real64),    intent(in)                                :: B, k_B
    complex(real64), intent(in),  dimension(n_basis,n_basis)   :: hamiltonian
    complex(real64), intent(in),  dimension(3,n_basis,n_basis) :: magnetic_moment
    real(real64),    intent(in),  dimension(n_T_points)        :: T_list
    real(real64),    intent(in),  dimension(n_grid_points,3)   :: grid_vectors
    real(real64),    intent(in),  dimension(n_grid_points)     :: grid_weights
    real(real64),    intent(out), dimension(n_T_points)        :: M_scalar

    real(real64),    dimension(3)                 :: B_vector
    real(real64),    dimension(n_basis)           :: eigenvalues, field_free_eigenvalues
    real(real64),    dimension(n_T_points)        :: M_vector_component, Q
    complex(real64), dimension(n_basis,n_basis)   :: eigenvectors
    
    integer :: i = 0, j = 0, p = 0

    real(real64) :: T = dp_zero

    M_scalar = dp_zero
    Q = dp_zero

    ! This loop should be parallelized.
    do i = 1, n_grid_points
       B_vector = B * grid_vectors(i,:)
       call diagonalize_hamiltonian(hamiltonian,magnetic_moment, &
            eigenvalues,eigenvectors,B_vector,n_basis,.true.,.false.)

       ! Evaluate magnetization without normalization
       M_vector_component = dp_zero
       do j = 1,3
          call equilibrium_expectation_value(magnetic_moment(j,:,:),eigenvalues,eigenvectors, &
               T_list,k_B,n_T_points,n_basis,M_vector_component,.false.)

          M_scalar = M_scalar + grid_weights(i) * grid_vectors(i,j) * M_vector_component
       end do

       ! Evaluate partition function
       do p = 1, n_T_points
          T = T_list(p)
          Q(p) = Q(p) + grid_weights(i)*sum(exp(-eigenvalues / (k_B * T)))
       end do
       
    end do

    do p = 1, n_T_points
       M_scalar(p) = M_scalar(p) / Q(p)
    end do

  end subroutine free_rotation_magnetization


  subroutine maximum_magnetization(hamiltonian,magnetic_moment,T_list,B,maximum_M,maximum_vectors, &
       grid_vectors,k_B,n_basis,n_grid_points,n_T_points)
    ! Calculate the magnetization along each grid vector at each
    ! temperature point defined in the argument T. For each temperature
    ! store the vector along with the magnetization obtaines a maximal
    ! value. Return both the vectors and the magnetization values.
    implicit none (type,external)

    integer,         intent(in)                                :: n_basis, n_grid_points, n_T_points
    real(real64),    intent(in)                                :: B, k_B
    complex(real64), intent(in),  dimension(n_basis,n_basis)   :: hamiltonian
    complex(real64), intent(in),  dimension(3,n_basis,n_basis) :: magnetic_moment
    real(real64),    intent(in),  dimension(n_T_points)        :: T_list
    real(real64),    intent(in),  dimension(n_grid_points,3)   :: grid_vectors
    real(real64),    intent(out), dimension(n_T_points)        :: maximum_M
    real(real64),    intent(out), dimension(n_T_points,3)      :: maximum_vectors

    real(real64),    dimension(3)               :: B_vector
    real(real64),    dimension(n_basis)         :: eigenvalues
    real(real64),    dimension(n_T_points)      :: M_vector_component, M_scalar
    complex(real64), dimension(n_basis,n_basis) :: eigenvectors
    
    integer :: i = 0, j = 0

    maximum_vectors = dp_zero
    maximum_M = dp_zero

    do i = 1, n_grid_points
       B_vector = B * grid_vectors(i,:)
       call diagonalize_hamiltonian(hamiltonian,magnetic_moment,eigenvalues,eigenvectors,B_vector,n_basis,.true.,.true.)

       M_vector_component = dp_zero
       M_scalar = dp_zero
       do j = 1,3
          call equilibrium_expectation_value(magnetic_moment(j,:,:),eigenvalues,eigenvectors, &
               T_list,k_B,n_T_points,n_basis,M_vector_component,.true.)

          M_scalar = M_scalar + grid_vectors(i,j) * M_vector_component
       end do

       do j = 1, n_T_points
          if (M_scalar(j) > maximum_M(j)) then
             maximum_M(j)         = M_scalar(j)
             maximum_vectors(j,:) = grid_vectors(i,:)
          end if
       end do
    end do
  end subroutine maximum_magnetization


end module powder_magnetization_utils
