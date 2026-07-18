! SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
! SPDX-License-Identifier: GPL-3.0-or-later

module grid_utils
  ! This module contains an interface to the Lebedev-Laikov grid
  ! routines written in Fortran 77. The routines should eventually
  ! be translated to the Fortran 2018 standard.
  !
  ! The code in lebedev_laikov.f has been obtained from
  !
  !   http://www.ccl.net/cca/software/SOURCES/FORTRAN/Lebedev-Laikov-Grids/
  !
  ! on 17 January 2022. The code has not been modified and is used in
  ! its provided form.
  !
  ! The original accompanying README file is:
  !
  !   Lebedev grids of orders n=6m+5 where m=0,1,...,21 in 16 digit precision
  ! =======================================================================
  ! 
  ! The file Lebedev-Laikov.F implements a set of subroutines providing 
  ! Lebedev-Laikov grids of order n=2m+1, where m=1,2,...,15, and additionally
  ! grids of order n=6m+5, where m=5,6,...,21. The parameters ensure 
  ! that angular integration of polynomials x**k * y**l * z**m, where k+l+m <= 131 
  ! can be performed with a relative accuracy of 2e-14 [1]. Note that the weights
  ! are normalised to add up to 1.0.
  ! 
  ! For each order n a separate subroutine is provided named 
  ! LD. The parameters X, Y, Z are arrays for the 
  ! cartesian components of each point, and the parameter W is an array for the
  ! weights. The subroutines increase the integer parameter N by number of grid
  ! points generated. All these routines use the subroutine gen_oh which takes care 
  ! of the octahedral symmetry of the grids.
  !
  ! Christoph van Wuellen (Ruhr-Universitaet, Bochum, Germany) generated the 
  ! routines in Lebedev-Laikov.F by translating the original C-routines kindly 
  ! provided by Dmitri Laikov (Moscow State University, Moscow, Russia). We 
  ! are in debt to Dmitri Laikov for giving us permission to make these routines
  ! publically available.
  !
  ! Huub van Dam
  ! Daresbury Laboratory, Daresbury, United Kingdom
  ! April, 2000
  !
  ! References
  ! ==========
  !
  ! [1] V.I. Lebedev, and D.N. Laikov
  !     "A quadrature formula for the sphere of the 131st
  !      algebraic order of accuracy"
  !     Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
  !

  ! use, intrinsic :: iso_fortran_env

  use num_utils
  
  implicit none (type,external)

contains
  
  subroutine generate_lebedev_laikov_grid(n_grid_points,vectors,weights)
    ! This is an interface for the Fortran 77 LD routines in
    ! lebedev_laikov.f. This routine is intended to be called
    ! from the Python code.
    implicit none (type,external)

    integer, intent(in) :: n_grid_points
    
    real(real64), dimension(n_grid_points,3), intent(out) :: vectors
    real(real64), dimension(n_grid_points),   intent(out) :: weights

    integer :: test_n_grid_points = 0

    external LD0006
    external LD0014
    external LD0026
    external LD0038
    external LD0050
    external LD0074
    external LD0086
    external LD0110
    external LD0146
    external LD0170
    external LD0194
    external LD0230
    external LD0266
    external LD0302
    external LD0350
    external LD0434
    external LD0590
    external LD0770
    external LD0974
    external LD1202
    external LD1454
    external LD1730
    external LD2030
    external LD2354
    external LD2702
    external LD3074
    external LD3470
    external LD3890
    external LD4334
    external LD4802
    external LD5294
    external LD5810


    if (n_grid_points == 6) then
       call LD0006(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 14) then
       call LD0014(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 26) then
       call LD0026(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 38) then
       call LD0038(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 50) then
       call LD0050(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 74) then
       call LD0074(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 86) then
       call LD0086(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 110) then
       call LD0110(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 146) then
       call LD0146(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 170) then
       call LD0170(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 194) then
       call LD0194(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 230) then
       call LD0230(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 266) then
       call LD0266(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 302) then
       call LD0302(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 350) then
       call LD0350(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 434) then
       call LD0434(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 590) then
       call LD0590(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 770) then
       call LD0770(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 974) then
       call LD0974(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 1202) then
       call LD1202(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 1454) then
       call LD1454(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 1730) then
       call LD1730(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 2030) then
       call LD2030(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 2354) then
       call LD2354(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 2702) then
       call LD2702(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 3074) then
       call LD3074(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 3470) then
       call LD3470(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 3890) then
       call LD3890(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 4334) then
       call LD4334(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 4802) then
       call LD4802(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 5294) then
       call LD5294(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else if (n_grid_points == 5810) then
       call LD5810(vectors(:,1),vectors(:,2),vectors(:,3),weights,test_n_grid_points)
    else
       write (*,*) 'ERROR in grid_utils.'
       write (*,*) 'ERROR: Unsupported number of grid points requested.'
       write (*,*) 'Error termination.'
       error stop
    end if

    if (test_n_grid_points /= n_grid_points) then
       write (*,*) 'ERROR in grid_utils.'
       write (*,*) 'ERROR: Number of requested grid points does not match the number of generated.'
       write (*,*) '       grid points. This is strange and should not be possible.'
       write (*,*) 'Error termination.'
       error stop
    end if
  end subroutine generate_lebedev_laikov_grid
  
end module grid_utils
