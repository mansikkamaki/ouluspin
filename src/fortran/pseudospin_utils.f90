! SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
! SPDX-License-Identifier: GPL-3.0-or-later

module pseudospin_utils

  use num_utils
  use math_utils
  use matrix_utils

  implicit none (type,external)

contains
  
  subroutine pseudospin_transformation(H_full,mu_full,H_eff,n_full_basis,n_eff_basis)
    ! Construct an effective Hamiltonian representation of a pseudospin Hamiltonian.
    ! Two matrices are given as input arguments: H_full and mu_full. The operator mu
    ! will be projected onto the n_eff_basis lowest eigenstates of H_full. This
    ! operator will then be diagonalized and the respective eigenvector matrix
    ! will be used to transform the n_eff_basis lowest eigenstates of_full H into the
    ! effective Hamiltonian H_eff.
    implicit none (type,external)

    integer,                                               intent(in)  :: n_full_basis, n_eff_basis
    complex(real64), dimension(n_full_basis,n_full_basis), intent(in)  :: H_full, mu_full
    complex(real64), dimension(n_eff_basis,n_eff_basis),   intent(out) :: H_eff

    real(real64),    dimension(n_full_basis)              :: H_eig
    complex(real64), dimension(n_full_basis,n_full_basis) :: mu_trans, C_H
    complex(real64), dimension(n_eff_basis,n_eff_basis)   :: mu_eff, H_diag
    real(real64),    dimension(n_eff_basis)               :: mu_eig

    integer :: i = 0

    C_H = H_full

    call diagonalize_complex_matrix(C_H,H_eig,n_full_basis,.true.)
    call basis_transformation(C_H,mu_full,mu_trans,n_full_basis,0)

    mu_eff = mu_trans(1:n_eff_basis,1:n_eff_basis)
    H_diag = cmplx(0.0,kind=real64)

    do i = 1,n_eff_basis
       H_diag(i,i) = H_eig(i)
    end do
    
    call diagonalize_complex_matrix(mu_eff,mu_eig,n_eff_basis,.true.)
    call basis_transformation(mu_eff,H_diag,H_eff,n_eff_basis,0)
    
  end subroutine pseudospin_transformation

end module pseudospin_utils
