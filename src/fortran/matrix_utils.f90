! SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
! SPDX-License-Identifier: GPL-3.0-or-later

module matrix_utils
  
  use num_utils
  use math_utils
  
  implicit none (type,external)

contains
  subroutine commutator(A,B,C,n_basis)
    ! Calculate the commutator
    !
    !     C = [A,B] = AB - BA.
    !
    implicit none (type,external)

    integer,                                     intent(in)  :: n_basis
    complex(real64), dimension(n_basis,n_basis), intent(in)  :: A, B
    complex(real64), dimension(n_basis,n_basis), intent(out) :: C

    ! This is a temporary matrix for the calculation.
    complex(real64), dimension(n_basis,n_basis) :: X
    
    ! BLAS parameters.
    complex(real64), parameter :: ALPHA = cmplx(1.0,0.0,kind=real64), BETA = cmplx(0.0,0.0,kind=real64)

    external zgemm

    call zgemm('N','N',n_basis,n_basis,n_basis,ALPHA,A,n_basis,B,n_basis,BETA,C,n_basis)
    call zgemm('N','N',n_basis,n_basis,n_basis,ALPHA,B,n_basis,A,n_basis,BETA,X,n_basis)

    C = C - X
  end subroutine commutator


  function check_hermicity(A,n_basis) result(test_value)
    ! Test whether the matrix A is Hermitian.
    implicit none (type,external)

    integer,                                     intent(in) :: n_basis
    complex(real64), dimension(n_basis,n_basis), intent(in) :: A

    logical :: test_value
    integer :: i = 0, j = 0

    test_value = .true.
    do i = 1, n_basis
       do j = 1, n_basis
          if ((complex_close_to(A(i,j),conjg(A(j,i)))) .eqv. .false.) then
             test_value = .false.
             return
          end if
       end do
    end do
  end function check_hermicity


  subroutine external_diagonalize_complex_matrix(matrix,eigenvalues,eigenvectors,n_basis)
    ! This is a version of diagonalize_complex_matrix that can be called from
    ! Python code. The difference in this version is that the operator and
    ! eigenvector matrices are separate matrices so that there is no need
    ! to have an inout intent. This means that more memory is used as one
    ! additional matrix needs to be allocated.
    implicit none (type,external)

    integer,                                     intent(in)  :: n_basis
    complex(real64), dimension(n_basis,n_basis), intent(in)  :: matrix
    complex(real64), dimension(n_basis,n_basis), intent(out) :: eigenvectors
    real(real64),    dimension(n_basis),         intent(out) :: eigenvalues

    ! Work scalars and arrays for LAPACK
    integer                                :: info, lwork
    real(real64), dimension(3*n_basis - 2) :: rwork
    ! This is the maximum size for the ZHEEV work array. The value should really be
    ! fixed outside the program to fit the available memory. This needs to be fixed
    ! in the future. ZHEEV will raise a runtime error if the value is too small. At
    ! the moment it is fixed to one megabyte of memory in a 64 bit system.
    integer, parameter                     :: lwmax = 65536
    complex(real64), dimension(lwmax)      :: work

    external zheev

    ! This if statement is here to avoid problems if we are dealing with a 1 x 1 matrix.
    if (n_basis == 1) then
       eigenvalues(1)    = real(matrix(1,1),kind=real64)
       eigenvectors(1,1) = cmplx(1.0,0.0,kind=real64)
    else
       eigenvectors = matrix
       
       ! Query optimal workspace for the diagonalization.
       lwork = -1
       call zheev('V','U',n_basis,eigenvectors,n_basis,eigenvalues,work,lwork,rwork,info)
       lwork = min(lwmax, int(work(1)))
       
       ! Diagonalize.
       call zheev('V','U',n_basis,eigenvectors,n_basis,eigenvalues,work,lwork,rwork,info)

    end if
    
  end subroutine external_diagonalize_complex_matrix

  
  subroutine diagonalize_complex_matrix(matrix,eigenvalues,n_basis,compute_eigenvectors)
    ! This subroutines takes as input a complex matrix and returns its eigenvectors
    ! in the same matrix and its eigenvalues as a vector. This subroutine is mostly
    ! a wrapper for the LAPACK zheev() routine.
    implicit none (type,external)

    integer,                                     intent(in)    :: n_basis
    logical,                                     intent(in)    :: compute_eigenvectors
    complex(real64), dimension(n_basis,n_basis), intent(inout) :: matrix
    real(real64),    dimension(n_basis),         intent(out)   :: eigenvalues

    ! Work scalars and arrays for LAPACK
    integer                                :: info, lwork
    real(real64), dimension(3*n_basis - 2) :: rwork
    ! This is the maximum size for the ZHEEV work array. The value should really be
    ! fixed outside the program to fit the available memory. This needs to be fixed
    ! in the future. ZHEEV will raise a runtime error if the value is too small. At
    ! the moment it is fixed to one megabyte of memory in a 64 bit system.
    integer, parameter                     :: lwmax = 65536
    complex(real64), dimension(lwmax)      :: work

    external zheev

    ! This if statement is here to avoid problems if we are dealing with a 1 x 1 matrix.
    if (n_basis == 1) then
       eigenvalues(1) = real(matrix(1,1),kind=real64)
       matrix(1,1)    = cmplx(1.0,0.0,kind=real64)
    else if (compute_eigenvectors .eqv. .false.) then
       ! Query optimal workspace for the diagonalization.
       lwork = -1
       call zheev('N','U',n_basis,matrix,n_basis,eigenvalues,work,lwork,rwork,info)
       lwork = min(lwmax, int(work(1)))
       
       ! Diagonalize.
       call zheev('N','U',n_basis,matrix,n_basis,eigenvalues,work,lwork,rwork,info)
    else
       ! Query optimal workspace for the diagonalization.
       lwork = -1
       call zheev('V','U',n_basis,matrix,n_basis,eigenvalues,work,lwork,rwork,info)
       lwork = min(lwmax, int(work(1)))
       
       ! Diagonalize.
       call zheev('V','U',n_basis,matrix,n_basis,eigenvalues,work,lwork,rwork,info)
    end if
  end subroutine diagonalize_complex_matrix


  subroutine diagonalize_real_symmetric_matrix(matrix,eigenvalues,n_basis,compute_eigenvectors)
    ! This subroutine takes as input a real symmetric matrix and returns its
    ! eigenvectors in the same matrix (as columns) and its eigenvalues, in
    ! ascending order, as a vector. This subroutine is mostly a wrapper for
    ! the LAPACK dsyev() routine and is the real symmetric counterpart of
    ! diagonalize_complex_matrix.
    implicit none (type,external)

    integer,                                  intent(in)    :: n_basis
    logical,                                  intent(in)    :: compute_eigenvectors
    real(real64), dimension(n_basis,n_basis), intent(inout) :: matrix
    real(real64), dimension(n_basis),         intent(out)   :: eigenvalues

    ! Work scalars and arrays for LAPACK
    integer                                :: info, lwork
    ! This is the maximum size for the DSYEV work array. The value should really be
    ! fixed outside the program to fit the available memory. This needs to be fixed
    ! in the future. DSYEV will raise a runtime error if the value is too small. At
    ! the moment it is fixed to half a megabyte of memory in a 64 bit system.
    integer, parameter                     :: lwmax = 65536
    real(real64), dimension(lwmax)         :: work

    external dsyev

    ! This if statement is here to avoid problems if we are dealing with a 1 x 1 matrix.
    if (n_basis == 1) then
       eigenvalues(1) = matrix(1,1)
       matrix(1,1)    = dp_one
    else if (compute_eigenvectors .eqv. .false.) then
       ! Query optimal workspace for the diagonalization.
       lwork = -1
       call dsyev('N','U',n_basis,matrix,n_basis,eigenvalues,work,lwork,info)
       lwork = min(lwmax, int(work(1)))

       ! Diagonalize.
       call dsyev('N','U',n_basis,matrix,n_basis,eigenvalues,work,lwork,info)
    else
       ! Query optimal workspace for the diagonalization.
       lwork = -1
       call dsyev('V','U',n_basis,matrix,n_basis,eigenvalues,work,lwork,info)
       lwork = min(lwmax, int(work(1)))

       ! Diagonalize.
       call dsyev('V','U',n_basis,matrix,n_basis,eigenvalues,work,lwork,info)
    end if
  end subroutine diagonalize_real_symmetric_matrix


  subroutine basis_transformation(C,A,B,n_basis,direction)
    ! Transform the complex matrix A given as an argument using the unitary matrix
    ! C also given as an argument as
    !
    !     B = C^H * A * C  if  direction = 0
    !
    ! or 
    !
    !     B = C * A * C^H  if  direction = 1.
    !
    ! ^H indicates Hermitian conjugation and * indicates a matrix product.
    ! This subroutine is mostly a wrapper for BLAS zgemm() routine.
    implicit none (type,external)

    integer,                                     intent(in)  :: n_basis, direction
    complex(real64), dimension(n_basis,n_basis), intent(in)  :: C, A
    complex(real64), dimension(n_basis,n_basis), intent(out) :: B

    ! This is a temporary matrix for calculations.
    complex(real64), dimension(n_basis,n_basis) :: X 

    ! BLAS parameters.
    complex(real64), parameter :: ALPHA = cmplx(1.0,0.0,kind=real64), BETA = cmplx(0.0,0.0,kind=real64)

    external zgemm

    if (direction == 0) then
       call zgemm('N','N',n_basis,n_basis,n_basis,ALPHA,A,n_basis,C,n_basis,BETA,X,n_basis)
       call zgemm('C','N',n_basis,n_basis,n_basis,ALPHA,C,n_basis,X,n_basis,BETA,B,n_basis)
    else if (direction == 1) then
       call zgemm('N','C',n_basis,n_basis,n_basis,ALPHA,A,n_basis,C,n_basis,BETA,X,n_basis)
       call zgemm('N','N',n_basis,n_basis,n_basis,ALPHA,C,n_basis,X,n_basis,BETA,B,n_basis)
    else
       write (*,*) 'ERROR in matrix_utils.'
       write (*,*) 'ERROR: Unknown transformation direction.'
       write (*,*) 'Error termination.'
       error stop
    end if
    
  end subroutine basis_transformation


  subroutine spectral_representation(eig,C,n_basis,matrix)
    ! Construct a matrix representation of an operator from the eigenvalue vector (eig) and
    ! eigenvector matrix (C) using the spectral decomposition.
    implicit none (type,external)

    integer,                                     intent(in)  :: n_basis
    real(real64),    dimension(n_basis),         intent(in)  :: eig
    complex(real64), dimension(n_basis,n_basis), intent(in)  :: C
    complex(real64), dimension(n_basis,n_basis), intent(out) :: matrix

    complex(real64), dimension(n_basis,n_basis) :: diagonal_matrix

    integer :: i = 0, j = 0

    diagonal_matrix = cmplx(0.0,0.0,kind=real64)

    do i = 1,n_basis
       diagonal_matrix(i,i) = cmplx(eig(i),0.0,kind=real64)
    end do

    call basis_transformation(C,diagonal_matrix,matrix,n_basis,1)
          
  end subroutine spectral_representation
  

  subroutine rotate_vector_operator_matrix(V_init,V_final,R,n_basis)
    ! Rotate the matrix representations of the three Cartesian components
    ! of a vector operator using rotation matrix R:
    !
    !     V_final = R . V_init,
    !
    ! where . indicates a dot product.
    implicit none (type,external)

    integer,                                       intent(in)  :: n_basis
    complex(real64), dimension(3,n_basis,n_basis), intent(in)  :: V_init
    complex(real64), dimension(3,n_basis,n_basis), intent(out) :: V_final
    real(real64),    dimension(3,3),               intent(in)  :: R

    integer :: i = 0, j = 0

    V_final = cmplx(0.0,kind=real64)

    do i = 1,3
       do j = 1,3
          V_final(i,:,:) = V_final(i,:,:) + R(i,j) * V_init(j,:,:)
       end do
    end do
  end subroutine rotate_vector_operator_matrix


  subroutine squared_vector_operator(vector_operator,scalar_operator,n_basis)
    ! Takes as an argument the matrix representations of the three
    ! Cartesian components of a vector operator and returns the matrix
    ! representation of a scalar operator defined as
    !
    !     s = v^T . v = v_x^2 + v_y^2 + v_z^2,
    !
    ! where ^T indicates transposition and . a dot product.
    implicit none (type,external)

    integer,                                       intent(in)  :: n_basis
    complex(real64), dimension(3,n_basis,n_basis), intent(in)  :: vector_operator
    complex(real64), dimension(n_basis,n_basis),   intent(out) :: scalar_operator
    
    complex(real64), dimension(n_basis,n_basis) :: X

    integer :: i = 0
    
    ! BLAS parameters.
    complex(real64), parameter :: ALPHA = cmplx(1.0,0.0,kind=real64), BETA = cmplx(0.0,0.0,kind=real64)

    external zgemm

    X = cmplx(dp_zero,kind=real64)
    scalar_operator = cmplx(dp_zero,kind=real64)
    
    do i = 1,3
       call zgemm('N','N',n_basis,n_basis,n_basis,ALPHA,vector_operator(i,:,:),n_basis,vector_operator(i,:,:),n_basis,BETA,X,n_basis)

       scalar_operator = scalar_operator + X
    end do
    
  end subroutine squared_vector_operator


  subroutine hadamard_product(A,B,C,n_basis)
    ! Calculate the Hadamard product of A and B and store it in C.
    implicit none (type,external)

    integer,                                     intent(in)  :: n_basis
    complex(real64), dimension(n_basis,n_basis), intent(in)  :: A,B
    complex(real64), dimension(n_basis,n_basis), intent(out) :: C

    C = A*B
  end subroutine hadamard_product


  subroutine phase_matrix(input_matrix,output_matrix,n_basis)
    ! Takes as input a complex matrix and returns a matrix consisting of the
    ! complex phases of each matrix element.
    implicit none (type,external)

    integer, intent(in) :: n_basis

    complex(real64), dimension(n_basis,n_basis), intent(in)  :: input_matrix
    real(real64),    dimension(n_basis,n_basis), intent(out) :: output_matrix

    integer :: i = 0, j = 0

    output_matrix = dp_zero

    do i = 1,n_basis
       do j = 1,n_basis
          output_matrix(i,j) = atan2(aimag(input_matrix(i,j)),real(input_matrix(i,j),kind=real64))
       end do
    end do
  end subroutine phase_matrix


  subroutine complex_matrix_product(A,B,C,conj_A,conj_B,n_basis)
    ! Evaluate the matrix product
    !
    !     C = A . B
    !
    ! This is mostly a wrapper for zgemm that can be called from
    ! the Python code.
    !
    ! If conj_A = 'N', A is used as given, if conj_A = 'C', the
    ! Hermitian conjugate of A is taken. Similarly for B.
    !
    implicit none (type,external)
    integer,                                     intent(in)  :: n_basis
    complex(real64), dimension(n_basis,n_basis), intent(in)  :: A, B
    character(len=1),                            intent(in)  :: conj_A, conj_B
    complex(real64), dimension(n_basis,n_basis), intent(out) :: C

    ! BLAS parameters.
    complex(real64), parameter :: ALPHA = cmplx(1.0,0.0,kind=real64), BETA = cmplx(0.0,0.0,kind=real64)

    external zgemm

    if (((conj_A /= 'N') .and. (conj_A /= 'C')) .or. ((conj_B /= 'N') .and. (conj_B /= 'C'))) then
       write (*,*) 'ERROR in matrix_utils.'
       write (*,*) 'ERROR: Unknown matrix conjugation.'
       write (*,*) 'Error termination.'
       error stop
    end if

    call zgemm(conj_A,conj_B,n_basis,n_basis,n_basis,ALPHA,A,n_basis,B,n_basis,BETA,C,n_basis)
  end subroutine complex_matrix_product

end module matrix_utils
