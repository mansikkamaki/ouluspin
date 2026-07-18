! SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
! SPDX-License-Identifier: GPL-3.0-or-later

module iwahara_utils

  use num_utils
  use math_utils
  use matrix_utils
  use cg_utils
  
  implicit none (type,external)

  contains
    function iwahara_operator(J1,M1,J2,M2,k,q) result(matrix_element)
      ! Evaluate a matrix element 
      !
      !     <J1,M1|O_kq/O_k0|J2,M2>
      !
      ! of the Iwahara definition of Stevens operators following equation (A28) in
      !
      !     N. Iwahara, L. F. Chibotaru. Phys. Rev. B. 2015, 91, 174438.
      !
      ! This function should be used only when some separate evaluation of an
      ! Iwahara operator is requested.
      implicit none (type,external)

      integer, intent(in) :: J1,M1,J2,M2,k,q
      real(real64)        :: matrix_element

      if (J1 == J2 .and. k == 0 .and. q == 0) then
         ! Let's keep things simple for the trivial case of an identity operator.
         matrix_element = dp_one
      else if (J1 == J2) then
         matrix_element = cg(J2,M2,k,q,J1,M1) / special_cg(J1,k)
      else
         matrix_element = 0.0
      end if

    end function iwahara_operator


    subroutine construct_iwahara_operator_matrix_for_one_rank(J_array,M_array,k_vector,q_vector, &
         n_basis,n_sites,operator_matrix,only_triangle)
      ! Evaluate the matrix representation of an Iwahara operator for one set of
      ! parameters k1,q1,k2,q2, ... 
      implicit none (type,external)
      
      integer,                                     intent(in)  :: n_basis, n_sites
      integer,         dimension(n_basis,n_sites), intent(in)  :: J_array, M_array
      integer,         dimension(n_sites),         intent(in)  :: k_vector, q_vector
      complex(real64), dimension(n_basis,n_basis), intent(out) :: operator_matrix
      logical,                                     intent(in)  :: only_triangle

      integer, dimension(n_sites) :: conservation_test_vector

      integer      :: bra_basis_index = 0, ket_basis_index = 0, limit_index = 0
      integer      :: site_index = 0
      integer      :: J1 = 0, J2 = 0, M1 = 0, M2 = 0, k = 0, q = 0
      real(real64) :: cg_product = 0.0

      operator_matrix = cmplx(0.0,kind=real64)
      
      do ket_basis_index = 1, n_basis
         if (only_triangle) then
            limit_index = ket_basis_index
         else
            limit_index = n_basis
         end if
         
         do bra_basis_index = 1, limit_index
            ! We use vector addtion to check the conservation of angular momentum
            ! projection in each CG coefficient in each term.
            conservation_test_vector = M_array(ket_basis_index,:) + q_vector(:) - M_array(bra_basis_index,:)
            if (all(conservation_test_vector == 0)) then
               cg_product = 1.0
               do site_index = 1, n_sites
                  J1 = J_array(bra_basis_index,site_index)
                  J2 = J_array(ket_basis_index,site_index)
                  M1 = M_array(bra_basis_index,site_index)
                  M2 = M_array(ket_basis_index,site_index)
                  k  = k_vector(site_index)
                  q  = q_vector(site_index)
                  cg_product = cg_product * iwahara_operator(J1,M1,J2,M2,k,q)
                  ! If we have an exactly zero coefficient, no point in calculating
                  ! the rest of the product. Note that the comparison must be exact:
                  ! the Iwahara operators are not normalized and the factors
                  ! 1/special_cg of the remaining sites can be of the order of 1e3
                  ! or more for high ranks, so truncating a small but non-zero
                  ! partial product would produce significant errors in the matrix
                  ! elements.
                  if (cg_product == dp_zero) then
                     exit
                  end if
               end do
               
               operator_matrix(bra_basis_index,ket_basis_index) = cmplx(cg_product,dp_zero,kind=real64)
            end if
         end do
      end do
    end subroutine construct_iwahara_operator_matrix_for_one_rank


    subroutine construct_general_iwahara_operator_matrix(J_array,M_array,k_array,q_array,parameters, &
         n_basis,n_sites,n_ranks,operator_matrix,only_triangle)
      ! Construct the matrix representation of a Iwahara--Chibotaru spherical
      ! tensor operator acting on a pseudospin basis of the type
      !
      !        |S0,M0> (x) |S1,M1> (x) ...
      !
      ! where S0 and M0 are the pseudospin and its projection for site 0, S1 and M1
      ! are the pseudospin and its projection for site 1 etc. The symbol (x) stands
      ! for tensor product.
      !
      ! Only the upper triangle of the matrix will be constructed as this is the part
      ! used by LAPACK ZHEEV() routine in the diagonalize_complex_matrix wrapper
      ! routine.
      implicit none (type,external)
      
      integer,                                     intent(in)  :: n_basis, n_sites, n_ranks
      integer,         dimension(n_basis,n_sites), intent(in)  :: J_array, M_array
      integer,         dimension(n_ranks,n_sites), intent(in)  :: k_array, q_array
      complex(real64), dimension(n_ranks),         intent(in)  :: parameters
      complex(real64), dimension(n_basis,n_basis), intent(out) :: operator_matrix
      logical,                                     intent(in)  :: only_triangle

      complex(real64), dimension(n_basis,n_basis) :: tmp_matrix
      integer                                     :: rank_index = 0
      integer                                     :: i = 0, j = 0

      operator_matrix = cmplx(0.0,kind=real64)

      ! Construct matrix. Only the upper triangle of the matrix will be constructed
      ! explicitly as the matrix is taken as hermitian.
      do rank_index = 1, n_ranks
         ! If the parameter is exactly zero, do not calculate the matrix element.
         ! Note that the comparison must be exact: the Iwahara operators are not
         ! normalized and their matrix elements grow as 1/special_cg with the
         ! rank, reaching the order of 1e3--1e4 for the highest ranks of large
         ! pseudospins. Skipping parameters based on a finite threshold would
         ! therefore produce significant errors in the operator matrix.
         if (parameters(rank_index) /= cmplx(0.0,kind=real64)) then
            call construct_iwahara_operator_matrix_for_one_rank(J_array,M_array, &
                 k_array(rank_index,:),q_array(rank_index,:), &
                 n_basis,n_sites,tmp_matrix,.true.)
            
            operator_matrix = operator_matrix + tmp_matrix*parameters(rank_index)
         end if
      end do

      ! If the full matrix is requested, evaluate rest of the elements.
      if (only_triangle .eqv. .false.) then
         do i = 1,n_basis
            do j = 1,i - 1
               !operator_matrix(j,i) = conjg(operator_matrix(i,j))
               operator_matrix(i,j) = conjg(operator_matrix(j,i))
            end do
         end do
      end if
      
    end subroutine construct_general_iwahara_operator_matrix


    subroutine diagonalize_general_iwahara_operator_matrix(J_array,M_array,k_array,q_array,parameters, &
         n_basis,n_sites,n_ranks,eigenvalues, &
         eigenvectors)
      ! Construct and diagonalize the matrix representation of a Iwahara--Chibotaru spherical
      ! tensor operator acting on a pseudospin basis of the type
      !
      !        |S0,M0> (x) |S1,M1> (x) ...
      !
      ! where S0 and M0 are the pseudospin and its projection for site 0, S1 and M1
      ! are the pseudospin and its projection for site 1 etc. The symbol (x) stands
      ! for tensor product.
      !
      ! The operator is assumed Hermitian. This will not be checked, and the subroutine will
      ! produce spurious resuts if the matrix is not Hermitian. Only the upper triangle of
      ! the matrix will be constructed as this is the part used by LAPACK ZHEEV() routine
      ! in the diagonalize_complex_matrix wrapper routine.
      implicit none (type,external)

      integer,                                     intent(in)  :: n_basis, n_sites, n_ranks
      integer,         dimension(n_basis,n_sites), intent(in)  :: J_array, M_array
      integer,         dimension(n_ranks,n_sites), intent(in)  :: k_array, q_array
      complex(real64), dimension(n_ranks),         intent(in)  :: parameters
      real(real64),    dimension(n_basis),         intent(out) :: eigenvalues
      complex(real64), dimension(n_basis,n_basis), intent(out) :: eigenvectors

      complex(real64), dimension(n_basis,n_basis) :: operator_matrix

      call construct_general_iwahara_operator_matrix(J_array,M_array,k_array,q_array,parameters, &
           n_basis,n_sites,n_ranks,operator_matrix,.true.)

      call diagonalize_complex_matrix(operator_matrix,eigenvalues,n_basis,.true.)

      eigenvectors = operator_matrix
      
    end subroutine diagonalize_general_iwahara_operator_matrix


    subroutine construct_iwahara_chibotaru_tensor_from_matrix(matrix,J_array,M_array, &
         k_array,q_array,parameters, &
         n_basis,n_sites,n_ranks)
      ! Takes as input a complex matrix representation of the operator in a basis defined
      ! by the J_array and M_array arrays. This is used to construct a set of
      ! parameters and respective ranks defining an Iwahara--Chibotaru spherical tensor
      ! operator. The operator parameters are constructed using equation (39) in 
      !
      !     N. Iwahara, L. F. Chibotaru. Phys. Rev. B. 2015, 91, 174438.
      !
      ! Note that in the reference projection is explicitly given for a two-site system
      ! but the generalization to one or more than two sites is trivial.
      !
      ! If the argument enforce_time_reversal_symmetry is set to .true., all parameters
      ! corresponding to rank tuples where the sum of k values is not even, will be
      ! set to zero as a consequence of time-reversal symmetry.
      implicit none (type,external)
      
      integer,                                      intent(in)  :: n_basis, n_sites, n_ranks
      complex(real64), dimension(n_basis,n_basis),  intent(in)  :: matrix
      integer,         dimension(n_basis,n_sites),  intent(in)  :: J_array, M_array
      integer,         dimension(n_ranks,n_sites),  intent(in)  :: k_array, q_array
      complex(real64), dimension(n_ranks),          intent(out) :: parameters

      complex(real64), dimension(n_basis,n_basis) :: tmp_product_matrix
      complex(real64), dimension(n_basis,n_basis) :: Q_matrix

      integer         :: i = 0, k = 0, q = 0, J = 0
      integer         :: rank_index = 0, site_index = 0
      integer         :: phase_exponent
      real(real64)    :: cg_product = dp_zero
      complex(real64) :: trace = dp_zero

      ! BLAS parameters.
      complex(real64), parameter :: ALPHA = cmplx(1.0,0.0,kind=real64), BETA = cmplx(0.0,0.0,kind=real64)

      external zgemm

      parameters = cmplx(1.0,0.0,kind=real64)

      do rank_index = 1,n_ranks
         ! Evaluate the phase and the square root / CG product.
         cg_product     = 1.0
         phase_exponent = 0
         do site_index = 1,n_sites
            k = k_array(rank_index,site_index)
            q = q_array(rank_index,site_index)
            ! NOTE: We are assuming here that all values of J for a given site are the same!
            J = J_array(1,site_index)
            
            phase_exponent = phase_exponent + q
            ! We use k+1 instead of 2k+1 as k is already multiplied by two.
            cg_product = cg_product * (real(k,kind=real64)+1.0) / (real(J,kind=real64)+1.0) * (special_cg(J,k))**2
            
         end do

         ! Check that everything is ok with the phase.
         if (mod(phase_exponent,2) /= 0) then
            write (*,*) 'ERROR in iwahara_utils.'
            write (*,*) 'ERROR: Phase of projection is non-integer.'
            write (*,*) '       This is strange and should not be possible.'
            write (*,*) 'Error termination.'
            error stop
         end if
         
         ! Note the negative sign on the q vector.
         call construct_iwahara_operator_matrix_for_one_rank(J_array,M_array, &
              k_array(rank_index,:),-q_array(rank_index,:), &
              n_basis,n_sites,Q_matrix,.false.)
         
         call zgemm('N','N',n_basis,n_basis,n_basis,ALPHA,Q_matrix,n_basis,matrix,n_basis,BETA,tmp_product_matrix,n_basis)
         trace = cmplx(dp_zero,kind=real64)
         do i = 1,n_basis
            trace = trace + tmp_product_matrix(i,i)
         end do

         parameters(rank_index) = (-1)**(phase_exponent/2) * cg_product * trace
      end do
    end subroutine construct_iwahara_chibotaru_tensor_from_matrix


end module iwahara_utils
