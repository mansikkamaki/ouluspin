! SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
! SPDX-License-Identifier: GPL-3.0-or-later

module powder_magnetization_utils

  use num_utils
  use matrix_utils
  use statmech_utils

  implicit none (type,external)

  ! The Cartesian components of the magnetic moment operator are indexed
  ! last, i.e. the operators are passed as magnetic_moment(n_basis,n_basis,3),
  ! so that each component is contiguous in memory. The components are read
  ! one at a time and handed over to BLAS, which needs them contiguous; with
  ! the component indexed first every such reference would be a strided copy.

contains

  subroutine project_magnetic_moment(magnetic_moment,direction,projected_moment,n_basis)
    ! Project the magnetic moment operator on the direction given as an
    ! argument, i.e. construct the operator
    !
    !     mu(n) = sum_j n_j * mu_j.
    !
    ! This is the only combination of the Cartesian components that the
    ! magnetization along a field direction needs: it is both the operator
    ! entering the Zeeman term of the Hamiltonian and the one whose
    ! equilibrium expectation value is the magnetization along that
    ! direction. Forming it once per grid point costs O(n_basis**2) and
    ! saves the two O(n_basis**3) basis transformations that evaluating the
    ! three Cartesian components separately would need.
    implicit none (type,external)

    integer,         intent(in)                                :: n_basis
    complex(real64), intent(in),  dimension(n_basis,n_basis,3) :: magnetic_moment
    real(real64),    intent(in),  dimension(3)                 :: direction
    complex(real64), intent(out), dimension(n_basis,n_basis)   :: projected_moment

    integer :: j

    projected_moment = direction(1)*magnetic_moment(:,:,1)

    do j = 2,3
       projected_moment = projected_moment + direction(j)*magnetic_moment(:,:,j)
    end do
  end subroutine project_magnetic_moment


  subroutine diagonalize_projected_hamiltonian(hamiltonian,projected_moment,eigenvalues,eigenvectors, &
       B,n_basis,compute_eigenvectors,translate_eigenvalues)
    ! Construct the Hamiltonian of a field of magnitude B applied along the
    ! direction the magnetic moment operator has already been projected on,
    !
    !     H(B) = H - B * mu(n),
    !
    ! and diagonalize it. See project_magnetic_moment for the projected
    ! operator.
    implicit none (type,external)

    integer,         intent(in)                              :: n_basis
    real(real64),    intent(in)                              :: B
    complex(real64), intent(in),  dimension(n_basis,n_basis) :: hamiltonian, projected_moment
    logical,         intent(in)                              :: compute_eigenvectors, translate_eigenvalues

    real(real64),    intent(out), dimension(n_basis)         :: eigenvalues
    complex(real64), intent(out), dimension(n_basis,n_basis) :: eigenvectors

    eigenvectors = hamiltonian - B*projected_moment

    call diagonalize_complex_matrix(eigenvectors,eigenvalues,n_basis,compute_eigenvectors)

    if (translate_eigenvalues .eqv. .true.) then
       eigenvalues = eigenvalues - minval(eigenvalues)
    end if
  end subroutine diagonalize_projected_hamiltonian


  subroutine diagonalize_hamiltonian(hamiltonian,magnetic_moment,eigenvalues,eigenvectors,B,n_basis, &
       compute_eigenvectors,translate_eigenvalues)
    ! Construct the full Hamiltonian from the field-free Hamiltonian and the Cartesian
    ! components of the magnetic moment operator in a specific field. Diagonalize the
    ! operator and return the eigenvalues and eigenvectors.
    !
    ! The field is given as the Cartesian vector B, i.e. its magnitude is the
    ! norm of that vector. The routine is a convenience wrapper that projects
    ! the magnetic moment on the field direction and calls
    ! diagonalize_projected_hamiltonian; the integration routines of this
    ! module call the latter directly, since they reuse the projected
    ! operator.
    implicit none (type,external)

    integer,         intent(in)                                :: n_basis
    real(real64),    intent(in),  dimension(3)                 :: B
    complex(real64), intent(in),  dimension(n_basis,n_basis)   :: hamiltonian
    complex(real64), intent(in),  dimension(n_basis,n_basis,3) :: magnetic_moment
    logical,         intent(in)                                :: compute_eigenvectors, translate_eigenvalues

    real(real64),    intent(out), dimension(n_basis)           :: eigenvalues
    complex(real64), intent(out), dimension(n_basis,n_basis)   :: eigenvectors

    complex(real64), dimension(n_basis,n_basis) :: projected_moment

    ! The projection is taken along B itself and the magnitude of the field
    ! is set to one, which reproduces H - sum_j B_j * mu_j for any B.
    call project_magnetic_moment(magnetic_moment,B,projected_moment,n_basis)

    call diagonalize_projected_hamiltonian(hamiltonian,projected_moment,eigenvalues,eigenvectors, &
         dp_one,n_basis,compute_eigenvectors,translate_eigenvalues)
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
    ! At each grid point the magnetization along the field direction is
    ! the equilibrium expectation value of the projected magnetic moment
    ! operator mu(n) (see project_magnetic_moment), so one expectation
    ! value is evaluated per grid point rather than one per Cartesian
    ! component.
    !
    ! Note that no unit conversions or multiplications are carried out.
    implicit none (type,external)

    integer,         intent(in)                                :: n_basis, n_grid_points, n_T_points
    real(real64),    intent(in)                                :: B, k_B
    complex(real64), intent(in),  dimension(n_basis,n_basis)   :: hamiltonian
    complex(real64), intent(in),  dimension(n_basis,n_basis,3) :: magnetic_moment
    real(real64),    intent(in),  dimension(n_T_points)        :: T
    real(real64),    intent(in),  dimension(n_grid_points,3)   :: grid_vectors
    real(real64),    intent(in),  dimension(n_grid_points)     :: grid_weights
    real(real64),    intent(out), dimension(n_T_points)        :: M_scalar

    real(real64),    dimension(n_basis)         :: eigenvalues
    real(real64),    dimension(n_T_points)      :: M_direction
    complex(real64), dimension(n_basis,n_basis) :: eigenvectors, projected_moment

    integer :: i

    M_scalar = dp_zero

    !$omp parallel do default(shared) schedule(static) &
    !$omp private(i,eigenvalues,eigenvectors,projected_moment,M_direction) &
    !$omp reduction(+:M_scalar)
    do i = 1, n_grid_points
       call project_magnetic_moment(magnetic_moment,grid_vectors(i,:),projected_moment,n_basis)

       call diagonalize_projected_hamiltonian(hamiltonian,projected_moment,eigenvalues,eigenvectors, &
            B,n_basis,.true.,.true.)

       call equilibrium_expectation_value(projected_moment,eigenvalues,eigenvectors, &
            T,k_B,n_T_points,n_basis,M_direction,.true.)

       M_scalar = M_scalar + grid_weights(i) * M_direction
    end do
    !$omp end parallel do

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
    !
    ! The molecules are allowed to rotate freely, so the orientations are
    ! weighted by the Boltzmann statistics of the whole sample: the
    ! expectation values are accumulated without normalization and divided
    ! by the partition function summed over the grid at the end.
    implicit none (type,external)

    integer,         intent(in)                                :: n_basis, n_grid_points, n_T_points
    real(real64),    intent(in)                                :: B, k_B
    complex(real64), intent(in),  dimension(n_basis,n_basis)   :: hamiltonian
    complex(real64), intent(in),  dimension(n_basis,n_basis,3) :: magnetic_moment
    real(real64),    intent(in),  dimension(n_T_points)        :: T_list
    real(real64),    intent(in),  dimension(n_grid_points,3)   :: grid_vectors
    real(real64),    intent(in),  dimension(n_grid_points)     :: grid_weights
    real(real64),    intent(out), dimension(n_T_points)        :: M_scalar

    real(real64),    dimension(n_basis)         :: eigenvalues
    real(real64),    dimension(n_T_points)      :: M_direction, Q
    complex(real64), dimension(n_basis,n_basis) :: eigenvectors, projected_moment

    integer :: i, p

    M_scalar = dp_zero
    Q = dp_zero

    !$omp parallel do default(shared) schedule(static) &
    !$omp private(i,p,eigenvalues,eigenvectors,projected_moment,M_direction) &
    !$omp reduction(+:M_scalar,Q)
    do i = 1, n_grid_points
       call project_magnetic_moment(magnetic_moment,grid_vectors(i,:),projected_moment,n_basis)

       call diagonalize_projected_hamiltonian(hamiltonian,projected_moment,eigenvalues,eigenvectors, &
            B,n_basis,.true.,.false.)

       ! Evaluate magnetization without normalization
       call equilibrium_expectation_value(projected_moment,eigenvalues,eigenvectors, &
            T_list,k_B,n_T_points,n_basis,M_direction,.false.)

       M_scalar = M_scalar + grid_weights(i) * M_direction

       ! Evaluate partition function
       do p = 1, n_T_points
          Q(p) = Q(p) + grid_weights(i)*sum(exp(-eigenvalues / (k_B * T_list(p))))
       end do

    end do
    !$omp end parallel do

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
    complex(real64), intent(in),  dimension(n_basis,n_basis,3) :: magnetic_moment
    real(real64),    intent(in),  dimension(n_T_points)        :: T_list
    real(real64),    intent(in),  dimension(n_grid_points,3)   :: grid_vectors
    real(real64),    intent(out), dimension(n_T_points)        :: maximum_M
    real(real64),    intent(out), dimension(n_T_points,3)      :: maximum_vectors

    real(real64),    dimension(n_basis)         :: eigenvalues
    real(real64),    dimension(n_T_points)      :: M_scalar
    complex(real64), dimension(n_basis,n_basis) :: eigenvectors, projected_moment

    integer :: i, j

    maximum_vectors = dp_zero
    maximum_M = dp_zero

    ! The loop is left serial: the grid points are compared against a
    ! running maximum, and the vector stored with it, so the iterations are
    ! not independent of each other.
    do i = 1, n_grid_points
       call project_magnetic_moment(magnetic_moment,grid_vectors(i,:),projected_moment,n_basis)

       call diagonalize_projected_hamiltonian(hamiltonian,projected_moment,eigenvalues,eigenvectors, &
            B,n_basis,.true.,.true.)

       call equilibrium_expectation_value(projected_moment,eigenvalues,eigenvectors, &
            T_list,k_B,n_T_points,n_basis,M_scalar,.true.)

       do j = 1, n_T_points
          if (M_scalar(j) > maximum_M(j)) then
             maximum_M(j)         = M_scalar(j)
             maximum_vectors(j,:) = grid_vectors(i,:)
          end if
       end do
    end do
  end subroutine maximum_magnetization


end module powder_magnetization_utils
