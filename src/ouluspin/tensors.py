# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

import sys

from math import sqrt, cos, sin, acos, asin, atan2, pi, factorial
from copy import deepcopy

import numpy as np
import numpy.linalg as la

from ouluspin._fortran import fortran_utils as fu
from ouluspin import result_table


class ChibotaruUngurSphericalTensor:
    """A class to store ITO expansion coefficients of a single-site tensor following the
    notation defined in:

        L. F. Chibotaru and L. Ungur. J. Chem. Phys., 2012, 137, 064112.

    The SINGLE_ANISO module follows this notation in its outputs. Some
    information is also given in other operator definitions, but this is the most general form.

    The main purpose of this class is to convert the Chibotaru--Ungur (CU) ITO expansion
    coefficients into the Iwahara--Chibotaru (IC) definition defined in

        N. Iwahara and L. F. Chibotaru. Phys. Rev. B, 2015, 91, 174438

    and

        N. Iwahara, L. Ungur and L. F. Chibotaru. Phys. Rev. B, 2018, 98, 054436.

    The IC definition will be used in all calculations in this program.

    NOTE!!! All angular momentum quantum numbers are given as multiplets of two in order to treat
            half-integer angular momenta as integer numbers.

    Arguments
    ---------
    real_rank_list : list of tuple of int
        A list consisting of tuples of the form (m,n) givng the rank and compoent of each
        parameter in the listing of the ITO decomposition parameters. For each rank m
        there exists m+1 componets n ranging from 0 to m.
    real_parameter_list : list of float
        A list of the real parameters (the coefficient of the opertor O_n^m in eq. (37) of
        Chibotaru--Ungur 2012) in the ITO decomposition in the same order as in rank_list.
    imag_parameter_list : list of float
        A list of the imaginary parameters (the coefficients of the operator \Omega_n^m in eq.
        (37) of Chibotaru--Ungur 2012) in the ITO decomposition in the same order as in
        rank_list.
    pseudospin : int
        Value of the pseudospin used in construction of the equivalent operators. This is
        needed so that the CU operators can be converted to the IC definition.

    Optional arguments
    ------------------

    Attributes
    ----------
    real_rank_list : list of tuple of int
        A list consisting of tuples of the form (m,n) givng the rank and compoent of each
        parameter in the listing of the CU ITO decomposition parameters. For each rank m
        there exists m+1 componets n ranging from 0 to m.
    real_parameter_list : list of float
        A list of the real parameters (the coefficient of the opertor O_n^m in eq. (37) of
        Chibotaru--Ungur 2012) in the ITO decomposition in the same order as in rank_list.
    imag_parameter_list : list of float
        A list of the imaginary parameters (the coefficients of the operator \Omega_n^m in eq.
        (37) of Chibotaru--Ungur 2012) in the ITO decomposition in the same order as in
        rank_list.
    pseudospin : int
        Value of the pseudospin used in construction of the equivalent operators.
    frame : str or None
        A free-form label of the coordinate frame in which the tensor is
        expressed (e.g. 'principal magnetic axis frame' or 'input axis
        frame'), or None when the frame is not known. The label is
        reported by the printing methods, propagated through additions
        (only when both operands carry the same label) and scalar
        multiplications, and passed on to the
        Iwahara--Chibotaru tensor returned by
        iwahara_chibotaru_spherical_tensor.

    Private methods
    ---------------
    __stevens_operator(k) : float
        Returns the value of the matrix elment <SS|O_{kq}|SS>, where O_{kq} is a Stevens
        operator in its original definition and S is the pseudospin.
    __convert_parameters() : list of tuple of int, list of complex
        Convert the CU parameters into IC parameters and return them.

    PUBLIC METHODS
    --------------
    iwahara_chibotaru_spherical_tensor() : IwaharaChibotaruSphericalTensor
        Return a one-site Iwahara--Chibotaru spherical tensor representation of this
        Chibotaru--Ungur spherical tensor.
    parameter_table(title=None) : ResultTable
        Return a table of the Chibotaru--Ungur expansion parameters.

    Class methods
    -------------
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if all
        tests passed.
    """

    def __stevens_operator(self, k):
        """Returns the value of the matrix elment <SS|O_{kq}|SS>, where O_{kq} is a Stevens
        operator in its original definition.

        The operators are defined in

            S. A. Al'tshuler and B. M. Kozyrev. Electron Paramagnetic Resonance in Compounds
            of Transition Elements. Halsted Press, 1974, NY, USA.

        and

            C. Rudowicz and C. Y. Chung. J. Phys.; Condens. Matter, 2004, 16, 5825.

        The former contains defitions for ranks 1 to 6 and the latter contains the definitions
        for ranks 3, 5, 7, 8, 10 and 12.

        The highest-weight matrix element is evaluated here for a general
        rank k with the closed form

            <S,S|O_k^0|S,S> = (2S)! / ( (2S-k)! * 2**s2(k) ) ,

        where s2(k) is the number of ones in the binary representation of
        the rank k. The equality of this expression with the explicit
        polynomial forms tabulated in the references above (ranks 1-6, 8,
        10 and 12) is asserted in run_tests. The evaluation is carried out
        in exact integer arithmetic. Ranks larger than 2S, for which the
        Stevens operators vanish identically, produce a fatal error as they
        cannot appear in a valid ITO decomposition.
        """
        # NOTE!!! The rank argument k is given as a multiple of two,
        # following the library convention, and self.pseudospin is 2S.
        rank = k//2

        if rank < 1 or rank > self.pseudospin:
            print("ERROR in ChibotaruUngurSphericalTensor.")
            print("ERROR: Unsupported rank in __stevens_operator(): " + str(rank) + ".")
            print("ERROR: The rank must be between 1 and 2S = " + str(self.pseudospin) + ".")
            print("Error termination.")
            sys.exit(1)

        return float(factorial(self.pseudospin) // factorial(self.pseudospin - rank)) \
               / float(2**bin(rank).count('1'))


    def __convert_parameters(self):
        """Convert the CU parameters into IC parameters and return them."""
        complex_parameter_list = []
        complex_rank_list      = []
        
        for i in range(0,len(self.real_rank_list)):
            m = self.real_rank_list[i][0]
            n = self.real_rank_list[i][1]

            A  = self.real_parameter_list[i]
            C  = self.imag_parameter_list[i]
            SS = self.__stevens_operator(m)
            
            if n == 0:
                B = SS * complex(A,0.0)
                complex_parameter_list.append(B)
                complex_rank_list.append((m,0))
            else:
                B_plus  = 0.5 * SS * (-1)**(n//2) * complex(A,-C)
                B_minus = 0.5 * SS * complex(A,C)

                complex_parameter_list.append(B_plus)
                complex_parameter_list.append(B_minus)

                complex_rank_list.append((m,n))
                complex_rank_list.append((m,-n))

        complex_parameter_list = [x for _,x in sorted(zip(complex_rank_list,
                                                          complex_parameter_list))]
        complex_rank_list      = sorted(complex_rank_list)

        return complex_rank_list, complex_parameter_list


    def iwahara_chibotaru_spherical_tensor(self):
        """Return a one-site Iwahara--Chibotaru spherical tensor representation of this
        Chibotaru--Ungur spherical tensor.
        """
        complex_rank_list, complex_parameter_list = self.__convert_parameters()

        converted_tensor = IwaharaChibotaruSphericalTensor(complex_rank_list,complex_parameter_list)
        converted_tensor.frame = self.frame
        return converted_tensor
        
        
    def parameter_table(self, title=None):
        """Return a listing of the Chibotaru--Ungur expansion parameters as
        an instance of ResultTable. Each row contains the rank m and the
        component n of one term of the expansion followed by the real
        parameter A and the imaginary parameter C of that term.

        Optional arguments
        ------------------
        title : str or None
            The title of the table. Default is None, in which case no
            title is printed.
        """
        if self.frame is None:
            notes = ["Coordinate frame: unspecified"]
        else:
            notes = ["Coordinate frame: " + self.frame]

        rows = []
        for i in range(0,len(self.real_rank_list)):
            rows.append([self.real_rank_list[i][0] // 2,
                         self.real_rank_list[i][1] // 2,
                         self.real_parameter_list[i],
                         self.imag_parameter_list[i]])

        return result_table.ResultTable(rows,
                                        column_headers=["m","n","A","C"],
                                        title=title,
                                        notes=notes,
                                        formats=[None,None,'.12f','.12f'],
                                        table_type='spherical_tensor')


    def __repr__(self):
        """Return a human-readable string of the parameters."""
        return str(self.parameter_table())


    def __init__(self, real_rank_list, real_parameter_list, imag_parameter_list, pseudospin):
        """Upon class initiation convert the CU parameters into IC parameters and store them
        as an attribute.
        """
        # See IwaharaChibotaruSphericalTensor for the meaning of the frame
        # label.
        self.frame = None

        self.real_parameter_list = [x for _,x in sorted(zip(real_rank_list,real_parameter_list))]
        self.imag_parameter_list = [x for _,x in sorted(zip(real_rank_list,imag_parameter_list))]
        self.real_rank_list      = sorted(real_rank_list)
        self.pseudospin          = pseudospin

        if not len(self.real_parameter_list) == len(self.imag_parameter_list):
            print("ERROR in ChibotaruUngurSphericalTensor.")
            print("Error: Inconsistent number of real and imaginary parameters.")
            print("Error termination.")
            sys.exit(1)
        if not len(self.real_parameter_list) == len(self.real_rank_list):
            print("ERROR in ChibotaruUngurSphericalTensor.")
            print("Error: Inconsistent number of ranks and parameters.")
            print("Error termination.")
            sys.exit(1)


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class and run a set of tests on the class constructor
        and the class methods. Return True if all tests passed and False
        otherwise.

        Optional arguments
        ------------------
        print_output : boolean
            Whether to print the outcome of each test. Default is True.
        """
        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('ChibotaruUngurSphericalTensor',
                                                       test_name,condition,print_output))

        pseudospin = 3   # S = 3/2

        # A rank-1 axial decomposition: the Stevens operator O_1^0 = S_z with
        # coefficient g converts to the IC parameter X_10 = S*g, which must
        # match the Cartesian z-operator constructor.
        cu  = cls([(2,0)],[2.0],[0.0],pseudospin)
        ic  = cu.iwahara_chibotaru_spherical_tensor()
        ref = IwaharaChibotaruSphericalTensor\
              .from_one_site_cartesian_operator(2.0,'z',pseudospin)
        check('rank-1 axial conversion matches Cartesian z operator', ic == ref)

        # A rank-2 axial decomposition: <SS|O_2^0|SS> = 3S^2 - S(S+1) = 3
        # for S = 3/2, so E(2,0) = 1.5 gives the IC parameter X_20 = 4.5.
        cu = cls([(4,0)],[1.5],[0.0],pseudospin)
        ic = cu.iwahara_chibotaru_spherical_tensor()
        check('rank-2 axial conversion',
              ([4,0] in ic.rank_list) and
              abs(ic.parameter_list[ic.rank_list.index([4,0])] - 4.5) < 1.0e-12)

        # The converted parameters of a Hermitian operator must obey
        # X_k,-q = (-1)^q * conjugate(X_k,q).
        cu = cls([(4,2)],[0.7],[0.3],pseudospin)
        ic = cu.iwahara_chibotaru_spherical_tensor()
        B_plus  = ic.parameter_list[ic.rank_list.index([4,2])]
        B_minus = ic.parameter_list[ic.rank_list.index([4,-2])]
        check('Hermiticity of converted parameters',
              abs(B_minus + B_plus.conjugate()) < 1.0e-12)

        # The closed form of the highest-weight Stevens matrix elements,
        # <S,S|O_k^0|S,S> = (2S)!/((2S-k)! * 2**s2(k)), used by
        # __stevens_operator must reproduce the explicit polynomial forms
        # tabulated in Al'tshuler--Kozyrev (ranks 1-6) and Rudowicz--Chung
        # (ranks 8, 10 and 12) for all pseudospins.
        def legacy_stevens(rank, S):
            X = S*(S + 1.0)
            if rank == 1:
                return S
            elif rank == 2:
                return 3*S**2 - X
            elif rank == 3:
                return 5*S**3 - (3*X - 1)*S
            elif rank == 4:
                return 35.0*S**4 - S**2 * (30.0*X - 25.0) + (3.0*X**2 - 6.0*X)
            elif rank == 5:
                return 63*S**5 - (70*X - 105)*S**3 + (15*X**2 - 50*X + 12)*S
            elif rank == 6:
                return 231.0*S**6 - S**4 * (315.0*X - 735.0) \
                       + S**2 * (105.0*X**2 - 525.0*X + 294.0) \
                       - (5.0*X**3 - 40.0*X**2 + 60.0*X)
            elif rank == 8:
                return 0.5*(12870.0*S**8 - 12012*S**6 * (-9.0 + 2*X)
                            + 2310.0*S**4 * (81.0 - 56.0*X + 6.0*X**2)
                            + 70.0*X * (-144.0 + 108.0*X - 20.0*X**2 + X**3)
                            - 12.0*S**2 * (-4566.0 + 9898.0*X - 3045.0*X**2 + 210.0*X**3))
            elif rank == 10:
                return 46189.0*S**10 \
                       - 36465.0*S**8 *(-22.0 + 3.0*X) \
                       + 3003.0*S**6 * (1199.0 - 450.0*X + 30.0*X**2) \
                       - 715*S**4 * (-6248.0 + 5481.0*X - 966.0*X**2 + 42.0*X**3) \
                       - 63*X * (2880.0 - 2304.0*X + 508.0*X**2 - 40.0*X**3 + X**4) \
                       + 33*S**2 *(32208.0 - 78900.0*X + 29680.0*X**2 - 3290*X**3 + 105.0*X**4)
            elif rank == 12:
                return 676039.0*S**12 \
                       - 323323.0*S**10 * (-65.0 + 6.0*X) \
                       + 138567.0*S**8 * (1391.0 - 330.0*X + 15.0*X**2) \
                       - 17017.0*S**6 * (-35945.0 + 17622.0*X - 2010.0*X**2 + 60*X**3) \
                       + 1001.0*S**4 * (606164.0 - 618090.0*X + 139245.0*X**2 - 10200.0*X**3 + 225.0*X**4) \
                       + 231.0*X * (-86400.0 + 72000.0*X - 17544.0*X**2 + 1708.0*X**3 - 70.0*X**4 + X**5) \
                       - 39.0*S**2 * (-3176160.0 + 8488392*X - 3601048.0*X**2 + 501116*X**3 - 26565.0*X**4 + 462.0*X**5)

        stevens_ok = True
        for two_s in range(1,17):
            probe = cls([(2,0)],[1.0],[0.0],two_s)
            for rank in [1,2,3,4,5,6,8,10,12]:
                if rank > two_s:
                    continue
                closed = probe._ChibotaruUngurSphericalTensor__stevens_operator(2*rank)
                legacy = legacy_stevens(rank,two_s/2.0)
                if abs(closed - legacy) > 1.0e-6*max(1.0,abs(legacy)):
                    stevens_ok = False
        check('closed-form Stevens elements match tabulated polynomials',
              stevens_ok)

        # A rank not previously supported by the explicit polynomials
        # (needed e.g. for the rank-14 crystal-field parameters of a
        # J = 15/2 multiplet): check the closed form against a directly
        # evaluated value, <15/2,15/2|O_14^0|15/2,15/2> = 15!/(1! * 2**3).
        probe = cls([(2,0)],[1.0],[0.0],15)
        check('rank-14 Stevens element',
              abs(probe._ChibotaruUngurSphericalTensor__stevens_operator(28)
                  - factorial(15)/8.0) < 1.0e-3)

        check('string representation renders', len(str(cu)) > 0)

        return debug_output.test_summary('ChibotaruUngurSphericalTensor',
                                         result_list,print_output)


class IwaharaChibotaruSphericalTensor:
    """A class to store an ITO expansion using the Iwahara--Chibotaru definition of ITOs
    given in 

        N. Iwahara and L. F. Chibotaru. Phys. Rev. B, 2015, 91, 174438

    and

        N. Iwahara, L. Ungur and L. F. Chibotaru. Phys. Rev. B, 2018, 98, 054436.

    Note that the definition is based on the two above-mentioned papers. Iwahara et al.
    have since updated the definition in

        N. Iwahara, Z. Huang, I. Neefjes, L. F. Chibotaru. Phys. Rev. B. 2022, 105,
        144401.

    The definition in the 2022 paper differs from the earlier papers and will not be used
    here.

    This will be the form of ITOs used by the rest of this program. The tensor parameters
    X_(k0,q0,k1,q1,...) are defined by a set of ranks k and projections or components q.
    The numbers correspond to different spin sites of the system.

    NOTE!!! All angular momentum quantum numbers are given as multiplets of two in order to
            treat half-integer angular momenta as integer numbers.

    Two instances can be added witch corresponds to simply adding the values of each
    expansion parameter of equal k,q rank. The instance can be multiplied by a scalar,
    which corresponds to multiplying all expansion parameteres by that number.

    The class can store tensors corresponding to a single or multiple spin sites. In the
    case of a single site, the rank and component list simply corresponds to values
    k1 and q1, whereas in the case of multiple sites the values k1 and q1 correspond
    to site 1, values k2 and q2 to site 2 etc.

    Arguments
    ---------
    rank_list : list of list of int
        A list of lists of the type [k0,q0,k1,q1, ... ] where the k and q
        values correspond to the operators acting on spin sites 0, 1, etc.
        The innermost list can also be a tuple and it will be automatically
        converted to a list.
    parameter_list : list of complex
        A list of the complex-valued ITO decomposition parameters in the same order as in
        rank_list.

    Optional arguments
    ------------------

    Attributes
    ----------
    rank_list : list of list of int
        A list of lists of the type [k0,q0,k1,q1, ... ] where the k and q
        values correspond to the operators acting on spin sites 0, 1, etc. 
    parameter_list : list of complex
        A list of the complex-valued ITO decomposition parameters in the same order as in
        rank_list.
    n_sites : int
        The number of spin sites.
    n_ranks : int
        The number of ranks, i.e., the length of rank_list.
    frame : str or None
        A free-form label of the coordinate frame in which the tensor is
        expressed (e.g. 'principal magnetic axis frame' or 'input axis
        frame'), or None when the frame is not known. The label is
        reported by the printing methods, propagated through additions
        (only when both operands carry the same label) and scalar
        multiplications, and reset by rotations, after which the caller
        must relabel the frame if it is known.

    Private methods
    ---------------
    __reorder_parameter_list()
        Order the parameter_list and rank_list so that lowest ranks are listed first in
        ascending order.

    Public methods
    --------------
    cartesian_vector(pseudospin) : array of float64
        Return a Cartesian vector (rank one tensor) representation of the rank k=1 parameters.
    one_site_cartesian_tensor(pseudospin) : CartesianTensor
        Return a Cartesian rank-two tensor representation of the rank k=0 and k=2 parameters.
        It is assumed that this tensors corresponds to a single site.
    two_site_cartesian_tensor(pseudospin_A,pseudospin_B) : CartesianTensor
        Return a Cartesian rank-two tensor representation of a two-site tensor.
    rotate(rotation)
        Rotate the tensor using the Rotation instance given as an argument.
    ITO_table(symbol="X",title=None,order_of_magnitude=0,rank_threshold=0.0,half_table=False) : ResultTable
        Return a listing of all ITO expansions parameters.
    inflate_dimension(site_list)
        Inflate the tensor IN PLACE so that it corresponds to a tensor acting on a
        system with more spin sites than included in this tensor. The tensor
        operator will still only act on the same spin sites as before, but it will
        include identity operators acting on other sites. The method modifies the
        instance and returns nothing, so a tensor that is still needed in its
        original form must be copied before it is inflated.
    reorder_spin_sites(reorder_list) : IwaharaChibotaruTensor
        Reorder the spin sites according to indices in the list given as an argument,
        and return the reordered tensor. The list contains numbers 0,1,2,3,... up to the
        total number of spin sites in the system. The numbers refer to the order of the spin
        sites in the old tensor and their order determines the order of the spin sites in the
        new tensor.
    purge_ranks(threshold=1.0e-6)
        Remove terms (both ranks and parameters) with an absolute parameter value less
        than threshold.
    couple_two_site_tensor() : IwaharaChibotaruSphericalTensor
        If this instance is a two-site tensor, construct a one-site tensor by coupling
        the tensor product.
    fortran_rank_arrays() : array of int32, array of int32
        Construct and return the k and q value NumPy arrays used by the Fortran routines
        in construction of the operator matrices.
    time_reversal_conjugate() : IwaharaChibotaruSphericalTensor
        Returns the time-reversal conjugate of this tensor.
    hermitian_conjugate() : IwaharaChibotaruSphericalTensor
        Returns the Hermitian conjugate of this tensor.
    separate_tensors() : list of IwaharaChibotaruSphericalTensor, IwaharaChibotaruSphericalTensor
        Separate the tensor into a list of one-site tensors, one for each spin site,
        containing the terms acting on just that site, and a single multi-site tensor
        containing the terms acting on multiple sites together with the constant term
        (all ranks and components zero).

    Class methods
    -------------
    from_one_site_cartesian_operator(parameter,component,pseudospin)
        Construct an ITO corresponding to an operator of the type X*S_a, where X is a complex
        scalar parameter and a is a Cartesian index. For example: the magnetic moment operator
        where X is the isotropic g value multiplied by the Bohr magneton.
    from_two_site_isotropic_operator(parameter,pseudospin_A,pseudospin_B)
        Construct an ITO corresponding to an operator of the type X*S_A*S_B, where X is a
        complex scalar parameter and S_A*S_B is the vector peroduct of two vector 
        pseudospin operators S_A and S_B acting on the two sites. For example: a Heisenberg
        Hamiltonian, where X is the (negative) exchange coupling constant.
    from_two_site_ising_operator(parameter,pseudospin_A,pseudospin_B)
        Construct an ITO corresponding to an operator of the type X*S_A,z*S_B,z, where X is a
        complex scalar parameter and S_A,z and S_B,z are pseudospin operators acting on the
        projection of the pseudospin at sites A and B. For example: an Ising Hamiltonian, where
        X is the (negative) exchange coupling constant.
    from_one_site_cartesian_tensor(matrix,pseudospin)
        Construct an ITO corresponding to an operator of the type S*X*S, where X is a
        complex 3x3 matrix, and S is a pseudospin operator. For example: the zero-field
        splitting operator, where X is ZFS D tensor.
    from_two_site_cartesian_tensor(matrix,pseudospin_A,pseudospin_B)
        Construct an ITO corresponding to an operator of the type S_A*X*S_B, where X is a
        complex 3x3 matrix, and S_A and S_B are pseudospin operators acting on sites A and B,
        respectively. For example, the asymmetric or symmetric anisotropic contributions to
        exchange coupling where X is the exchange J tensor.
    from_matrix_representation(matrix,basis)
        Construct the operator from a matrix representation of the operator in a given basis.
        Both the matrix and the basis are given as arguments.
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if all
        tests passed.
    """
    
    def __reorder_parameter_list(self):
        """Order the parameter_list and rank_list so that lowest ranks are listed first in
        ascending order.

        The sorting is carried out on the ranks only. Sorting zipped
        (rank, parameter) pairs would compare the complex parameters whenever
        two ranks are equal, and complex numbers have no ordering.
        """
        order = sorted(range(0,len(self.rank_list)), key=lambda i: self.rank_list[i])
        self.parameter_list = [self.parameter_list[i] for i in order]
        self.rank_list      = [self.rank_list[i] for i in order]

    
    def cartesian_vector(self,pseudospin):
        """Return a Cartesian vector (rank one tensor) representation of the rank k=1
        parameters.
        """
        if not self.n_sites == 1:
            print("ERROR in IwaharaChibotaruSphericalTensor.")
            print("Error: Cannot construct a Cartesian vector from multiple-site tensor.")
            print("Error termination.")
            sys.exit(1)
        # Note that the rank lists are stored as lists of lists, so the
        # lookups must be done with lists, not tuples.
        if [2,2] in self.rank_list:
            X1 = self.parameter_list[self.rank_list.index([2,2])]
        else:
            X1 = complex(0.0,0.0)

        if [2,0] in self.rank_list:
            X0 = self.parameter_list[self.rank_list.index([2,0])]
        else:
            X0 = complex(0.0,0.0)

        # Use real-valued pseudospin as opposed to an integer multiple of two.
        pseudospin = float(pseudospin) / 2.0

        vector = np.zeros(3, dtype=np.float64)
        vector[0] = -sqrt(2.0)/pseudospin * X1.real
        vector[1] =  sqrt(2.0)/pseudospin * X1.imag
        vector[2] = X0.real/pseudospin

        return vector

    
    def one_site_cartesian_tensor(self,pseudospin):
        """Return a Cartesian rank-two tensor representation of the rank k=0 and k=2 parameters.
        It is assumed that this tensors corresponds to a single site.
        """
        if not self.n_sites == 1:
            print("ERROR in IwaharaChibotaruSphericalTensor.")
            print("Error: Cannot construct a one-site Cartesian tensor from multiple-site tensor.")
            print("Error termination.")
            sys.exit(1)

        pseudospin = float(pseudospin) / 2.0

        # Note that the rank lists are stored as lists of lists, so the
        # lookups must be done with lists, not tuples.
        # The rank-zero parameter is the coefficient of the identity operator;
        # the isotropic part of the Cartesian tensor is obtained from it by
        # dividing out the factor S(S+1) coming from S . (A_I * 1) . S.
        if [0,0] in self.rank_list:
            Z = self.parameter_list[self.rank_list.index([0,0])].real \
                / (pseudospin*(pseudospin + 1.0))
        else:
            Z = 0.0

        if [4,4] in self.rank_list:
            X2 = self.parameter_list[self.rank_list.index([4,4])]
        else:
            X2 = complex(0.0,0.0)

        if [4,2] in self.rank_list:
            X1 = self.parameter_list[self.rank_list.index([4,2])]
        else:
            X1 = complex(0.0,0.0)

        if [4,0] in self.rank_list:
            X0 = self.parameter_list[self.rank_list.index([4,0])]
        else:
            X0 = complex(0.0,0.0)

        matrix = np.zeros((3,3), dtype=np.float64)

        a = sqrt(6.0)
        b = 3.0*pseudospin**2 - pseudospin*(pseudospin + 1.0)

        matrix[0][0] = ( a*X2.real - X0.real) / b + Z
        matrix[1][1] = (-a*X2.real - X0.real) / b + Z
        matrix[2][2] = 2.0 * X0.real / b + Z
        matrix[0][1] = -a * X2.imag / b
        matrix[0][2] = -a * X1.real / b
        matrix[1][2] =  a * X1.imag / b
        
        matrix[1][0] = matrix[0][1]
        matrix[2][0] = matrix[0][2]
        matrix[2][1] = matrix[1][2]

        return CartesianTensor(matrix)

    
    def two_site_cartesian_tensor(self,pseudospin_A,pseudospin_B):
        """Return a Cartesian rank-two tensor representation of a two-site tensor.
        It is assumed that this tensor corresponds to two sites.
        """
        if not self.n_sites == 2:
            print("ERROR in IwaharaChibotaruSphericalTensor.")
            print("Error: Can only construct a two-site Cartesian tensor from a two-site tensor..")
            print("Error termination.")
            sys.exit(1)

        # Note that the rank lists are stored as lists of lists, so the
        # lookups must be done with lists, not tuples.
        if [2,2,2,2] in self.rank_list:
            Xpp = self.parameter_list[self.rank_list.index([2,2,2,2])]
        else:
            Xpp = complex(0.0,0.0)

        if [2,0,2,0] in self.rank_list:
            X00 = self.parameter_list[self.rank_list.index([2,0,2,0])]
        else:
            X00 = complex(0.0,0.0)

        if [2,2,2,-2] in self.rank_list:
            Xpm = self.parameter_list[self.rank_list.index([2,2,2,-2])]
        else:
            Xpm = complex(0.0,0.0)

        if [2,2,2,0] in self.rank_list:
            Xp0 = self.parameter_list[self.rank_list.index([2,2,2,0])]
        else:
            Xp0 = complex(0.0,0.0)

        if [2,0,2,2] in self.rank_list:
            X0p = self.parameter_list[self.rank_list.index([2,0,2,2])]
        else:
            X0p = complex(0.0,0.0)

            
        SS = float(pseudospin_A / 2.0) * float(pseudospin_B / 2.0)

        # These are the inverses of the conversion factors used in
        # from_two_site_cartesian_tensor.
        a = 1.0 / SS
        b = sqrt(2.0) / SS

        matrix = np.zeros((3,3), dtype=np.float64)

        matrix[0][0] =  a * ( Xpp.real - Xpm.real)
        matrix[1][1] = -a * ( Xpp.real + Xpm.real)
        matrix[2][2] =  a * X00.real
        matrix[0][1] = -a * ( Xpp.imag + Xpm.imag)
        matrix[1][0] =  a * (-Xpp.imag + Xpm.imag)
        matrix[0][2] = -b * Xp0.real
        matrix[2][0] = -b * X0p.real
        matrix[1][2] =  b * Xp0.imag
        matrix[2][1] =  b * X0p.imag

        return CartesianTensor(matrix)

    
    def rotate(self,rotation):
        """Rotate the tensor using the Rotation instance given as an argument.
        
        The rotation is given by R * X * R^H, where X is the ITO, R is the
        rotation operator and ^H indicates Hermitean conjugation.

        The new ITO parameters X'_k1q1,k2q2,... are given by

            X'_k1q1,k2q2,... = sum_q1'q2'... X_k1q1',k2q2',... * D_q1q1'^k1(R) * D_q2q2'^k2(R) ...,

        where X_kq' are the original ITO parameters and D_qq'^k(R) is an
        element of the Wigner matrix representing the rotation R.

        Note that the rotation mixes all components q of a given rank k.
        Therefore the rotated tensor is constructed on a complete set of
        rank tuples: for each combination of k values present in this
        tensor all q components are included, even if the original tensor
        only contained a subset of the q components.
        """

        # Collect the unique combinations of k values present in the tensor.
        k_combination_list = []
        for rank_tuple in self.rank_list:
            k_combination = []
            for l in range(0,self.n_sites):
                k_combination.append(rank_tuple[2*l])
            if k_combination not in k_combination_list:
                k_combination_list.append(k_combination)

        # For each k combination construct all rank tuples [k1,q1,k2,q2,...]
        # with the q values running over the complete ranges -k,...,k.
        new_rank_list = []
        for k_combination in k_combination_list:
            partial_tuple_list = [[]]
            for l in range(0,self.n_sites):
                k = k_combination[l]
                expanded_tuple_list = []
                for partial_tuple in partial_tuple_list:
                    for q in range(-k,k+2,2):
                        expanded_tuple_list.append(partial_tuple + [k,q])
                partial_tuple_list = expanded_tuple_list
            new_rank_list.extend(partial_tuple_list)

        new_parameter_list = []

        for i in range(0,len(new_rank_list)):
            new_rank_tuple = new_rank_list[i]

            tmp_sum = complex(0.0,0.0)
            for j in range(0,len(self.rank_list)):
                old_rank_tuple = self.rank_list[j]

                tmp_product = self.parameter_list[j]

                for l in range(0,self.n_sites):
                    if new_rank_tuple[2*l] == old_rank_tuple[2*l]:
                        k1 = new_rank_tuple[2*l]
                        q1 = new_rank_tuple[2*l+1]
                        q2 = old_rank_tuple[2*l+1]
                        tmp_product = tmp_product * rotation.wigner_D(k1,q1,q2)
                    else:
                        tmp_product = complex(0.0,0.0)
                        break

                tmp_sum += tmp_product

            new_parameter_list.append(tmp_sum)

        self.rank_list      = new_rank_list
        self.parameter_list = new_parameter_list
        self.n_ranks        = len(self.rank_list)

        self.__reorder_parameter_list()

        # The rotation changes the coordinate frame; the caller must relabel
        # the frame if it is known.
        self.frame = None

    
    def ITO_table(self, symbol="X", title=None, order_of_magnitude=0, rank_threshold=0.0, half_table=False):
        """Return a listing of all ITO expansions parameters as an instance
        of ResultTable. Printing the returned instance, or converting it
        into a str, gives the human-readable table.

        Each row of the table contains the ranks k and the components q of
        the operators of one term of the expansion, one pair per spin site,
        followed by the real part, the imaginary part and the magnitude of
        the corresponding expansion parameter.

        Optional arguments:
        -------------------
        symbol : str
            A symbol used to label the tensor elements in the output. Default is 'X'.
        title : str
            A title of the tensor table used in the output. Default is None, in which
            case no title is printed.
        order_of_magnitude : int
            Before printing, the values of the parameters will be multiplied by
            10^order_of_magnitude. Default is 0.
        rank_threshold : float
            If the magnitudes of all parameters of a given rank combination (i.e.,
            specific combination of k1,k2,k3,...) are below this threshold value,
            the parameters are not printed for this rank combination. The default value
            is 0.0 in which case all parameters are printed.
        half_table : boolean
            Since the parameters with all values of q inverted are related to
            each other by a simple phase difference, only print the values that are
            unique. These are chosen so that q1 is always non-negative in the printed
            output.

        The optional argument symbol gives the str used to label the tensor
        elements.
        """
        factor = 10.0**(order_of_magnitude)

        index_str          = ""
        for i in range(0,self.n_sites):
            index_str          += "k" + str(i+1) + "q" + str(i+1)
            if not i == self.n_sites-1:
                index_str          += ","

        column_headers = []
        for i in range(0,self.n_sites):
            column_headers.append('k' + str(i+1))
            column_headers.append('q' + str(i+1))
        column_headers.extend(["Re(" + symbol + "_" + index_str + ")",
                               "Im(" + symbol + "_" + index_str + ")",
                               "|" + symbol + "_" + index_str + "|"])

        # The ranks and the components are printed as integers and the
        # parameters as floating-point numbers.
        formats = 2*self.n_sites*[None] + 3*['.6f']

        notes = []
        if self.frame is None:
            notes.append("Coordinate frame: unspecified")
        else:
            notes.append("Coordinate frame: " + self.frame)
        if not order_of_magnitude == 0:
            notes.append("The parameters are multiplied by 10^{0}."
                         .format(order_of_magnitude))

        rows = []

        # Find the largest parameter magnitude of each rank combination
        # (k1,k2,...) for the rank_threshold filter. The threshold is compared
        # against the unscaled parameter magnitudes, i.e., before multiplication
        # by 10^order_of_magnitude (consistent with the threshold of purge_ranks).
        max_magnitude = {}
        for i in range(0,len(self.rank_list)):
            k_combination = tuple(self.rank_list[i][0::2])
            magnitude     = abs(self.parameter_list[i])
            if (not k_combination in max_magnitude) or (magnitude > max_magnitude[k_combination]):
                max_magnitude[k_combination] = magnitude

        for i in range(0,len(self.rank_list)):
            rank_tuple = self.rank_list[i]

            # Skip rank combinations where the magnitudes of all parameters are
            # below the threshold.
            if max_magnitude[tuple(rank_tuple[0::2])] < rank_threshold:
                continue

            # In the half table only one of the two parameters related by an
            # inversion of all q values is printed. The printed representative
            # is chosen so that the first non-zero q value is positive, which
            # in particular means that q1 is always non-negative in the output.
            if half_table:
                first_nonzero_q = 0
                for q in rank_tuple[1::2]:
                    if not q == 0:
                        first_nonzero_q = q
                        break
                if first_nonzero_q < 0:
                    continue

            row = []
            for j in range(0,self.n_sites):
                k = rank_tuple[2*j]
                q = rank_tuple[2*j+1]
                row.append(k//2)
                row.append(q//2)
            row.extend([factor*self.parameter_list[i].real,
                        factor*self.parameter_list[i].imag,
                        abs(factor*self.parameter_list[i])])

            rows.append(row)

        return result_table.ResultTable(rows,
                                        column_headers=column_headers,
                                        title=title,
                                        notes=notes,
                                        formats=formats,
                                        table_type='spherical_tensor')


    def inflate_dimension(self,site_list):
        """Inflate this Iwahara--Chibotaru spherical tensor IN PLACE so that it
        corresponds to a tensor acting on a system with more spin sites than included
        in this tensor. The tensor operator will still only act on the same spin sites
        as before, but it will include identity operators acting on other sites.

        The method modifies the instance and returns nothing. A tensor that is still
        needed in its original form, e.g. one belonging to the caller, must therefore
        be copied (deepcopy) before it is inflated; inflating it a second time is an
        error, since its number of sites has already grown.

        The site list argument is a list of ones and zeros. The ones stand for the positions
        of the old sites in the new site list whereas zeros correspond to the sites to be
        added. The total number of ones must equal the current number of sites.
        """
        for site_index in site_list:
            if not ((site_index == 1) or (site_index == 0)):
                print("ERROR in IwaharaChibotaruSphericalTensor.")
                print("ERROR: Unkown site index: " + str(site_index) + ".")
                print("Error termination.")
                sys.exit(1)

        if not sum(site_list) == self.n_sites:
            print("ERROR in IwaharaChibotaruSphericalTensor.")
            print("ERROR: Inconsistent number of old sites in site_list.")
            print("Error termination.")
            sys.exit(1)

        new_rank_list = []
        for i in range(0,len(self.rank_list)):
            new_rank_tuple_list = []
            old_index = 0
            for site_index in site_list:
                if site_index == 0:
                    new_rank_tuple_list.append(0)
                    new_rank_tuple_list.append(0)
                elif site_index == 1:
                    new_rank_tuple_list.append(self.rank_list[i][old_index])
                    new_rank_tuple_list.append(self.rank_list[i][old_index+1])
                    old_index += 2

            # The rank list must remain a list of lists (not tuples) so that
            # rank lookups and sorting stay consistent with the constructor.
            new_rank_list.append(new_rank_tuple_list)

        self.n_sites   = len(site_list)
        self.rank_list = deepcopy(new_rank_list)

        
    def reorder_spin_sites(self,reorder_list):
        """Reorder the spin sites according to indices in the list given as an argument,
        and return the reordered tensor. The list contains numbers 0,1,2,3,... up to the
        total number of spin sites in the system. The numbers refer to the order of the spin
        sites in the old tensor and their order determines the order of the spin sites in the
        new tensor.
        """
        if not len(reorder_list) == self.n_sites:
            print("ERROR in IwaharaChibotaruSphericalTensor.")
            print("ERROR: Reorder list must have the same number of elements as the number of spin sites.")
            print("Error termination.")
            sys.exit(1)

        if not list(range(0,self.n_sites)) == sorted(reorder_list):
            print("ERROR in IwaharaChibotaruSphericalTensor.")
            print("ERROR: Reorder list must contain numbers 0,1,2,... up to the number of spin sites.")
            print("Error termination.")
            sys.exit(1)
        
        new_rank_list = []

        for i in range(0,self.n_ranks):
            rank_tuple = []
            for j in range(0,self.n_sites):
                rank_tuple.append(0)
                rank_tuple.append(0)

            for j in range(0,self.n_sites):
                k = reorder_list[j]
                rank_tuple[2*j]   = deepcopy(self.rank_list[i][2*k])
                rank_tuple[2*j+1] = deepcopy(self.rank_list[i][2*k+1])

            new_rank_list.append(deepcopy(rank_tuple))

        return IwaharaChibotaruSphericalTensor(new_rank_list, deepcopy(self.parameter_list))

    
    def purge_ranks(self,threshold=1.0e-6):
        """Remove terms (both ranks and parameters) with an absolute parameter value less
        than threshold.
        """
        old_parameter_list = deepcopy(self.parameter_list)
        old_rank_list      = deepcopy(self.rank_list)

        new_parameter_list = []
        new_rank_list      = []

        for i in range(0,self.n_ranks):
            if abs(old_parameter_list[i]) >= threshold:
                new_parameter_list.append(old_parameter_list[i])
                new_rank_list.append(old_rank_list[i])

        self.parameter_list = new_parameter_list
        self.rank_list      = new_rank_list
                
        self.n_ranks = len(self.rank_list)


    def couple_two_site_tensor(self):
        """If this instance is a two-site tensor, construct a one-site tensor by
        coupling the tensor product.
        """
        if not self.n_sites == 2:
            print("ERROR in IwaharaChibotaruSphericalTensor.")
            print("Error: Can only construct a coupled tensor from a two-site tensor.")
            print("Error termination.")
            sys.exit(1)

        coupled_rank_list = []
        coupled_parameter_list = []
        
        k1_max = 0
        k2_max = 0
        for rank_tuple in self.rank_list:
            if rank_tuple[0] > k1_max:
                k1_max = rank_tuple[0]
            if rank_tuple[2] > k2_max:
                k2_max = rank_tuple[2]

        # The coupled ranks must start from zero: even though the maximum
        # ranks k1_max and k2_max limit the largest coupled rank to
        # k1_max + k2_max, rank pairs with k1 = k2 can couple to K = 0
        # regardless of the values of k1_max and k2_max. Ranks that cannot
        # be produced by any pair simply obtain zero-valued parameters from
        # the vanishing Clebsch--Gordan coefficients.
        for K in range(0,k1_max+k2_max+2,2):
            for Q in range(-K,K+2,2):
                coupled_rank_list.append((K,Q))

        for i in range(0,len(coupled_rank_list)):
            K = coupled_rank_list[i][0]
            Q = coupled_rank_list[i][1]
            
            tmp_parameter = 0.0
            
            for j in range(0,self.n_ranks):
                k1 = self.rank_list[j][0]
                q1 = self.rank_list[j][1]
                k2 = self.rank_list[j][2]
                q2 = self.rank_list[j][3]
                
                tmp_parameter += self.parameter_list[j] * fu.cg_utils.cg(k1,q1,k2,q2,K,Q)

            coupled_parameter_list.append(tmp_parameter)

        return IwaharaChibotaruSphericalTensor(coupled_rank_list,coupled_parameter_list)
                

    def fortran_rank_arrays(self):
        """Construct and return the k and q value NumPy arrays with dimensions
        n_ranks x n_sites used by the Fortran routines in construction of the operator
        matrices.
        """
        k_array = np.zeros((self.n_ranks,self.n_sites), dtype=np.int32)
        q_array = np.zeros((self.n_ranks,self.n_sites), dtype=np.int32)

        for rank_index in range(0,self.n_ranks):
            for site_index in range(0,self.n_sites):
                k_array[rank_index][site_index] = self.rank_list[rank_index][2*site_index]
                q_array[rank_index][site_index] = self.rank_list[rank_index][2*site_index+1]

        return k_array, q_array


    def time_reversal_conjugate(self):
        """Returns the time-reversal conjugate of this tensor."""
        new_rank_list      = []
        new_parameter_list = []
        
        for i in range(0,self.n_ranks):
            current_rank_set = self.rank_list[i]
            inverse_rank_set = []

            phase = 0
            
            for j in range(0,self.n_sites):
                inverse_rank_set.append(current_rank_set[2*j])
                inverse_rank_set.append(-current_rank_set[2*j+1])

                phase += current_rank_set[2*j] - current_rank_set[2*j+1]

            if phase % 2 == 0:
                phase = phase // 2
            else:
                print("ERROR in IwaharaChibotaruSphericalTensor.")
                print("Error: Non-integer phase in time_reversal_conjugate.")
                print("Error termination.")
                sys.exit(1)

            if inverse_rank_set in self.rank_list:
                l = self.rank_list.index(inverse_rank_set)
                current_B = self.parameter_list[l]
                new_B     = (-1)**phase * current_B.conjugate()

                new_rank_list.append(current_rank_set)
                new_parameter_list.append(new_B)

        return IwaharaChibotaruSphericalTensor(new_rank_list,new_parameter_list)


    def hermitian_conjugate(self):
        """Returns the Hermitian conjugate of this tensor."""
        new_rank_list      = []
        new_parameter_list = []
        
        for i in range(0,self.n_ranks):
            current_rank_set = self.rank_list[i]
            inverse_rank_set = []

            phase = 0
            
            for j in range(0,self.n_sites):
                inverse_rank_set.append(current_rank_set[2*j])
                inverse_rank_set.append(-current_rank_set[2*j+1])

                phase += -current_rank_set[2*j+1]

            if phase % 2 == 0:
                phase = phase // 2
            else:
                print("ERROR in IwaharaChibotaruSphericalTensor.")
                print("Error: Non-integer phase in time_inversion_conjugate.")
                print("Error termination.")
                sys.exit(1)

            if inverse_rank_set in self.rank_list:
                l = self.rank_list.index(inverse_rank_set)
                current_B = self.parameter_list[l]
                new_B     = (-1)**phase * current_B.conjugate()

                new_rank_list.append(current_rank_set)
                new_parameter_list.append(new_B)

        return IwaharaChibotaruSphericalTensor(new_rank_list,new_parameter_list)
    
        
    def separate_tensors(self):
        """Separate the tensor into one-site tensors and a multi-site tensor.

        Each term whose rank set has a non-zero rank on exactly one site is
        collected into the one-site tensor of that site. All other terms --
        those with non-zero ranks on two or more sites, as well as the
        constant term with all ranks and components zero -- are collected
        into a single multi-site tensor with the same number of sites as
        this tensor. If a returned tensor receives no terms, it is given a
        single zero-valued term of all-zero ranks so that all returned
        tensors are valid instances.

        Returns
        -------
        tensor_list : list of IwaharaChibotaruSphericalTensor
            A list of the one-site tensors, one for each spin site.
        multi_site_tensor : IwaharaChibotaruSphericalTensor
            A tensor acting on all spin sites of this tensor, containing the
            constant term and the terms acting on multiple sites.
        """
        one_site_rank_lists       = [[] for n in range(0,self.n_sites)]
        one_site_parameter_lists  = [[] for n in range(0,self.n_sites)]
        multi_site_rank_list      = []
        multi_site_parameter_list = []

        for i in range(0,self.n_ranks):
            active_sites = [n for n in range(0,self.n_sites)
                            if self.rank_list[i][2*n] != 0]

            if len(active_sites) == 1:
                n = active_sites[0]
                one_site_rank_lists[n].append([self.rank_list[i][2*n],
                                               self.rank_list[i][2*n+1]])
                one_site_parameter_lists[n].append(self.parameter_list[i])
            else:
                multi_site_rank_list.append(self.rank_list[i])
                multi_site_parameter_list.append(self.parameter_list[i])

        tensor_list = []

        for n in range(0,self.n_sites):
            if len(one_site_rank_lists[n]) == 0:
                one_site_rank_lists[n].append([0,0])
                one_site_parameter_lists[n].append(complex(0.0,0.0))

            tensor_list.append(IwaharaChibotaruSphericalTensor(one_site_rank_lists[n],
                                                               one_site_parameter_lists[n]))

        if len(multi_site_rank_list) == 0:
            multi_site_rank_list.append([0]*(2*self.n_sites))
            multi_site_parameter_list.append(complex(0.0,0.0))

        multi_site_tensor = IwaharaChibotaruSphericalTensor(multi_site_rank_list,
                                                            multi_site_parameter_list)

        return tensor_list, multi_site_tensor


    def __add__(self, other):
        """Add the parameters from another Chibotaru--Iwahara ITO decomposition into the
        parameters of this decomposition for each rank k,q.
        """
        if other is None:
            return deepcopy(self)

        else:
            if not self.n_sites == other.n_sites:
                print("ERROR in IwaharaChibotaruSphericalTensor.")
                print("ERROR: Cannot add two tensors with different number of sites.")
                print("Error termination.")
                sys.exit(1)
            
            copy_of_self = deepcopy(self)

            # The frame label survives the addition only when both tensors
            # carry the same label.
            if getattr(other,'frame',None) != self.frame:
                copy_of_self.frame = None

            for i in range(0,len(other.rank_list)):
                current_rank_set = other.rank_list[i]
                B = other.parameter_list[i]
                
                if current_rank_set in copy_of_self.rank_list:
                    j = copy_of_self.rank_list.index(current_rank_set)
                    copy_of_self.parameter_list[j] += B
                else:
                    copy_of_self.parameter_list.append(B)
                    copy_of_self.rank_list.append(current_rank_set)
                copy_of_self.__reorder_parameter_list()
                copy_of_self.n_ranks = len(copy_of_self.rank_list)
            return copy_of_self

    
    def __radd__(self, other):
        """Right-hand addition; see __add__. Supports sum() through the zero start value."""
        if other == 0:
            return deepcopy(self)
        else:
            return self.__add__(other)

        
    def __mul__(self, number):
        """Multiply all ITO decomposition parameters by the number given as an argument.
        """
        try:
            complex(number)
        except (TypeError, ValueError):
            print("ERROR in IwaharaChibotaruSphericalTensor.")
            print("ERROR: Can only multiply the instance with a number.")
            print("Error termination.")
            sys.exit(1)

        copy_of_self = deepcopy(self)

        for i in range(0,len(copy_of_self.rank_list)):
            copy_of_self.parameter_list[i] *=  number

        return copy_of_self


    def __rmul__(self, other):
        """Right-hand multiplication by a number; see __mul__."""
        return self.__mul__(other)


    def __eq__(self, other):
        """Check the equality of two tensors by comparing their nonzero parameters rank by rank."""
        if not isinstance(other, IwaharaChibotaruSphericalTensor):
            return False

        # Before carrying out the comparison, we create copies of both tensors,
        # and purge ranks with zero parameters, so that we can compare two tensors
        # with different number of ranks as long as the differing ranks correspond
        # to zero-valued parameters.

        copy_of_self  = deepcopy(self)
        copy_of_other = deepcopy(other)

        copy_of_self.purge_ranks()
        copy_of_other.purge_ranks()

        if not copy_of_self.n_ranks == copy_of_other.n_ranks:
            return False
        
        for i in range(0,copy_of_self.n_ranks):
            B_self    = copy_of_self.parameter_list[i]
            rank_self = copy_of_self.rank_list[i]

            if rank_self in copy_of_other.rank_list:
                j = copy_of_other.rank_list.index(rank_self)
                B_other = copy_of_other.parameter_list[j]

                if not np.allclose(B_self.real,B_other.real):
                    return False
                if not np.allclose(B_self.imag,B_other.imag):
                    return False
            else:
                return False

        return True


    def __repr__(self):
        """Return the ITO parameter table of the tensor."""
        return str(self.ITO_table())
    

    def __init__(self, rank_list, parameter_list):
        """Upon class initiation, store the parameters given as an argument"""
        # A free-form label of the coordinate frame in which the tensor is
        # expressed (e.g. 'principal magnetic axis frame' or 'input axis
        # frame'), or None when the frame is not known. The label is
        # reported by the printing methods, propagated through the
        # arithmetic operations when unambiguous and reset by rotations.
        self.frame = None

        # To make things simple, tuples in the rank_list list of lists
        # will be converted to lists.
        self.rank_list = []
        for rank_tuple in rank_list:
            self.rank_list.append(list(rank_tuple))

        self.parameter_list = parameter_list

        self.n_ranks = len(self.rank_list)
        self.n_sites = len(self.rank_list[0])//2
        
        for rank_tuple in self.rank_list:
            if not len(rank_tuple) == 2*self.n_sites:
                print("ERROR in IwaharaChibotaruSphericalTensor.")
                print("ERROR: Inconsistent number of rank parameters.")
                print("Error termination.")
                sys.exit(1)

        self.__reorder_parameter_list()

        
    @classmethod
    def from_one_site_cartesian_operator(cls,parameter,component,pseudospin):
        """Construct an ITO corresponding to an operator of the type X*S_a, where X is a complex
        scalar parameter and a is a Cartesian index. For example: the magnetic moment operator
        where X is the isotropic g value multiplied by the Bohr magneton.
        """
        X = parameter
        S = float(pseudospin)/2.0

        if (component == 'x') or (component == 0):
            # Note the signs: S_x = (T_1,-1 - T_1,+1)/sqrt(2) in the
            # spherical-component convention T_1,+/-1 = -/+ (S_x +/- i*S_y)/sqrt(2)
            # used throughout the library.
            Xp = -S/sqrt(2) * complex(X,0.0)
            Xm =  S/sqrt(2) * complex(X,0.0)

            rank_list      = [(2,-2),(2,2)]
            parameter_list = [Xm,Xp]
            
        elif (component == 'y') or (component == 1):
            Xp =  S/sqrt(2) * complex(0.0,X)
            Xm = -S/sqrt(2) * complex(0.0,-X)
            
            rank_list      = [(2,-2),(2,2)]
            parameter_list = [Xm,Xp]
            
        elif (component == 'z') or (component == 2):
            X0 = complex(S*X,0.0)
            rank_list      = [(2,0)]
            parameter_list = [X0]
            
        else:
            print("ERROR in IwaharaChibotaruSphericalTensor.")
            print("ERROR: Unkown Cartesian component: " + str(component) + ".")
            print("Error termination.")
            sys.exit(1)

        return cls(rank_list,parameter_list)


    @classmethod
    def from_two_site_isotropic_operator(cls,parameter,pseudospin_A,pseudospin_B):
        """Construct an ITO corresponding to an operator of the type X*S_A*S_B, where X is a
        complex scalar parameter and S_A*S_B is the vector peroduct of two vector 
        pseudospin operators S_A and S_B acting on the two sites. For example: a Heisenberg
        Hamiltonian, where X is the (negative) exchange coupling constant.
        """
        S_A = float(pseudospin_A)/2.0
        S_B = float(pseudospin_B)/2.0
        X   = parameter

        X_pm_real = -S_A*S_B / 2.0 * 2*X
        X_00_real = S_A*S_B * X

        rank_list = [(2, 2,2,-2),
                     (2, 0,2, 0),
                     (2,-2,2, 2)]

        parameter_list = [complex(X_pm_real,0.0),
                          complex(X_00_real,0.0),
                          complex(X_pm_real,0.0)]

        return cls(rank_list,parameter_list)


    @classmethod
    def from_two_site_ising_operator(cls,parameter,pseudospin_A,pseudospin_B):
        """Construct an ITO corresponding to an operator of the type X*S_A,z*S_B,z, where X is a
        complex scalar parameter and S_A,z and S_B,z are pseudospin operators acting on the
        projection of the pseudospin at sites A and B. For example: an Ising Hamiltonina, where
        X is the (negative) exchange coupling constant.
        """
        S_A = float(pseudospin_A)/2.0
        S_B = float(pseudospin_B)/2.0
        X   = S_A * S_B * parameter

        rank_list      = [(2,0,2,0)]
        parameter_list = [complex(X,0.0)]

        return cls(rank_list,parameter_list)


    @classmethod
    def from_one_site_cartesian_tensor(cls,matrix,pseudospin):
        """Construct an ITO corresponding to an operator of the type S*X*S, where X is a
        complex 3x3 matrix, and S is a pseudospin operator. For example: the zero-field
        splitting operator, where X is ZFS D tensor.
        """
        S = float(pseudospin) / 2.0

        a = 3*S**2 - S*(S+1.0)
        b = 1.0/sqrt(6.0)

        A_I = np.trace(matrix) / 3.0
        A_S = 0.5*(matrix + matrix.T) - A_I*np.identity(3)
        A_A = 0.5*(matrix - matrix.T)

        if not np.allclose(A_A,0.0):
            print("ERROR in IwaharaChibotaruSphericalTensor.")
            print("ERROR: Cannot construct a one-site rank-two ITO from a Cartesian tensor with a non-zero")
            print("       asymmetric part.")
            print("Error termination.")
            sys.exit(1)

        # The rank-two parameters are constructed from the traceless
        # symmetric part of the tensor. The isotropic part enters only the
        # rank-zero parameter below: S . (A_I * 1) . S = A_I*S(S+1) * identity.
        X_xx = A_S[0][0]
        X_yy = A_S[1][1]
        X_zz = A_S[2][2]
        X_xy = A_S[0][1]
        X_xz = A_S[0][2]
        X_yz = A_S[1][2]

        ReX_20 = 0.5*a   * X_zz
        ImX_20 = 0.0
        ReX_22 = 0.5*a*b * (X_xx - X_yy)
        ImX_22 =    -a*b * X_xy
        ReX_21 =    -a*b * X_xz
        ImX_21 =     a*b * X_yz

        rank_list = [(0,0),(4,-4),(4,-2),(4,0),(4,2),(4,4)]

        # The negative-q parameters follow from Hermiticity:
        # X_k,-q = (-1)^q * conjugate(X_k,q).
        parameter_list = [complex(A_I*S*(S + 1.0),0.0),
                          complex( ReX_22,-ImX_22),
                          complex(-ReX_21, ImX_21),
                          complex( ReX_20, ImX_20),
                          complex( ReX_21, ImX_21),
                          complex( ReX_22, ImX_22)]

        return cls(rank_list,parameter_list)


    @classmethod
    def from_two_site_cartesian_tensor(cls,matrix,pseudospin_A,pseudospin_B):
        """Construct an ITO corresponding to an operator of the type S_A*X*S_B, where X is a
        complex 3x3 matrix, and S_A and S_B are pseudospin operators acting on sites A and B,
        respectively. For example, the asymmetric or symmetric anisotropic contributions to
        exchange coupling where X is the exchange J tensor.
        """
        S_A = float(pseudospin_A) / 2.0
        S_B = float(pseudospin_B) / 2.0

        # The purely transverse (q_A, q_B = +-1) terms carry two factors of
        # 1/sqrt(2) from the spherical components of the two spin operators,
        # whereas the mixed transverse-longitudinal terms carry only one.
        a = 0.5*S_A*S_B
        c = S_A*S_B / sqrt(2.0)

        rank_list = [(+2,-2,+2,-2),
                     (+2,-2,+2, 0),
                     (+2,-2,+2,+2),
                     (+2, 0,+2,-2),
                     (+2, 0,+2, 0),
                     (+2, 0,+2,+2),
                     (+2,+2,+2,-2),
                     (+2,+2,+2, 0),
                     (+2,+2,+2,+2)]
        
        ReXpp =  a *(matrix[0][0] - matrix[1][1])
        ImXpp = -a *(matrix[0][1] + matrix[1][0])
        ReXpm = -a *(matrix[0][0] + matrix[1][1])
        ImXpm = -a *(matrix[0][1] - matrix[1][0])

        ReX00 = 2.0 * a * matrix[2][2]
        ImX00 = 0.0

        ReXp0 = -c*matrix[0][2]
        ImXp0 =  c*matrix[1][2]
        ReX0p = -c*matrix[2][0]
        ImX0p =  c*matrix[2][1]

        parameter_list = [complex( ReXpp,-ImXpp),
                          complex(-ReXp0, ImXp0),
                          complex( ReXpm,-ImXpm),
                          complex(-ReX0p, ImX0p),
                          complex( ReX00, ImX00),
                          complex( ReX0p, ImX0p),
                          complex( ReXpm, ImXpm),
                          complex( ReXp0, ImXp0),
                          complex( ReXpp, ImXpp)]

        return cls(rank_list,parameter_list)


    @classmethod
    def from_matrix_representation(cls,matrix,basis):
        """Construct the operator from a matrix representation of the operator in a given basis.
        Both the matrix and the basis are given as arguments.
        """
        n_basis = basis.n_basis
        n_sites = basis.n_sites

        S_array, M_array = basis.construct_fortran_pseudospin_arrays()

        # Construct the rank lists that will be used by the class constructor.
        
        k_max_list = []
        for i in range(0,n_sites):
            S = basis.basis_state_list[0][i][0]

            # We will ignore ranks larger than k = 16 to avoid numerical problems
            # in the evaluation of CG coefficients. The CG coefficients have been
            # verified against exact rational arithmetic to be accurate to machine
            # precision up to k = 16 for pseudospins up to S = 8, which covers the
            # complete decomposition for all lanthanide and actinide ground
            # multiplets.
            if 2*S > 32:
                k_max_list.append(32)
            else:
                k_max_list.append(2*S)

        rank_list = []

        # First construct the rank list for just the first site.
        for k in range(0,k_max_list[0]+2,2):
            for q in range(-k,k+2,2):
                rank_list.append([k,q])

        # Recursively append each site to the rank list.
        for i in range(1,n_sites):
            k_max = k_max_list[i]
            new_rank_list  = []
            old_rank_list  = deepcopy(rank_list)
            rank_list = []

            for k in range(0,k_max+2,2):
                for q in range(-k,k+2,2):
                    new_rank_list.append([k,q])

            for old_rank_parttuple in old_rank_list:
                for new_rank_parttuple in new_rank_list:
                    new_rank_tuple = deepcopy(old_rank_parttuple) + new_rank_parttuple
                    rank_list.append(new_rank_tuple)

        n_ranks = len(rank_list)

        # Construct the rank arrays used by the Fortran code.
        k_array = np.zeros((n_ranks,n_sites), dtype=np.int32)
        q_array = np.zeros((n_ranks,n_sites), dtype=np.int32)

        for rank_index in range(0,n_ranks):
            for site_index in range(0,n_sites):
                k_array[rank_index][site_index] = rank_list[rank_index][2*site_index]
                q_array[rank_index][site_index] = rank_list[rank_index][2*site_index+1]

        parameter_list = fu.iwahara_utils\
                           .construct_iwahara_chibotaru_tensor_from_matrix(matrix,S_array,M_array,
                                                                           k_array,q_array)

        return cls(rank_list,parameter_list)


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class and run a set of tests on the class constructor
        and the class methods. Return True if all tests passed and False
        otherwise.

        The constructors are tested against operator matrices built from
        explicit spin matrices, so that the tests verify the physics and not
        only the internal consistency of the class.

        Optional arguments
        ------------------
        print_output : boolean
            Whether to print the outcome of each test. Default is True.
        """
        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('IwaharaChibotaruSphericalTensor',
                                                       test_name,condition,print_output))

        pseudospin = 3   # S = 3/2
        S_ops = debug_output.spin_matrices(pseudospin)

        # One-site Cartesian operators against explicit spin matrices.
        all_components_ok = True
        for i, component in enumerate(('x','y','z')):
            t = cls.from_one_site_cartesian_operator(1.3,component,pseudospin)
            M = debug_output.operator_matrix_from_tensor(t,[pseudospin])
            all_components_ok = all_components_ok and np.allclose(M,1.3*S_ops[i])
        check('one-site Cartesian operators match spin matrices', all_components_ok)

        t = cls.from_one_site_cartesian_operator(1.3,'x',pseudospin)
        check('cartesian_vector round trip',
              np.allclose(t.cartesian_vector(pseudospin),[1.3,0.0,0.0]))

        # One-site rank-2 tensor: operator matrix of S.D.S and round trip.
        D = np.array([[ 1.2, 0.3,-0.4],
                      [ 0.3,-0.7, 0.2],
                      [-0.4, 0.2, 0.9]])
        t = cls.from_one_site_cartesian_tensor(D,pseudospin)
        M = debug_output.operator_matrix_from_tensor(t,[pseudospin])
        M_ref = sum(D[a][b]*np.dot(S_ops[a],S_ops[b])
                    for a in range(0,3) for b in range(0,3))
        check('one-site rank-2 tensor matches S.D.S matrix', np.allclose(M,M_ref))
        check('one-site Cartesian tensor round trip',
              np.allclose(t.one_site_cartesian_tensor(pseudospin).tensor,D))

        # Two-site rank-2 tensor with unequal pseudospins, including an
        # antisymmetric contribution.
        pseudospin_A, pseudospin_B = 2, 1
        SA = debug_output.spin_matrices(pseudospin_A)
        SB = debug_output.spin_matrices(pseudospin_B)
        J = np.array([[ 0.5, 0.1,-0.3],
                      [-0.1, 0.5, 0.2],
                      [ 0.3,-0.2, 0.8]])
        t = cls.from_two_site_cartesian_tensor(J,pseudospin_A,pseudospin_B)
        M = debug_output.operator_matrix_from_tensor(t,[pseudospin_A,pseudospin_B])
        M_ref = sum(J[a][b]*np.kron(SA[a],SB[b])
                    for a in range(0,3) for b in range(0,3))
        check('two-site rank-2 tensor matches S_A.J.S_B matrix', np.allclose(M,M_ref))
        check('two-site Cartesian tensor round trip',
              np.allclose(t.two_site_cartesian_tensor(pseudospin_A,pseudospin_B).tensor,J))

        # The isotropic operator must equal the Cartesian tensor of X*identity.
        t_isotropic = cls.from_two_site_isotropic_operator(-2.0,pseudospin_A,pseudospin_B)
        t_identity  = cls.from_two_site_cartesian_tensor(-2.0*np.identity(3),
                                                         pseudospin_A,pseudospin_B)
        check('isotropic operator equals isotropic Cartesian tensor',
              t_isotropic == t_identity)

        t_ising = cls.from_two_site_ising_operator(1.0,pseudospin_A,pseudospin_B)
        M = debug_output.operator_matrix_from_tensor(t_ising,[pseudospin_A,pseudospin_B])
        check('Ising operator matches S_Az*S_Bz matrix',
              np.allclose(M,np.kron(SA[2],SB[2])))

        # Tensor algebra and conjugations.
        check('addition and scalar multiplication',
              (t_isotropic + t_isotropic) == 2.0*t_isotropic)
        check('Hermitian conjugate of a Hermitian tensor',
              t_isotropic.hermitian_conjugate() == t_isotropic)
        check('time-reversal conjugate of a time-even tensor',
              t_isotropic.time_reversal_conjugate() == t_isotropic)

        # Rotation: rotating the ITO must equal converting the rotated
        # Cartesian tensor.
        alpha, beta, gamma = 0.4, 0.9, -1.2
        R_z = lambda angle: np.array([[cos(angle),-sin(angle),0.0],
                                      [sin(angle), cos(angle),0.0],
                                      [0.0,0.0,1.0]])
        R_y = lambda angle: np.array([[cos(angle),0.0,sin(angle)],
                                      [0.0,1.0,0.0],
                                      [-sin(angle),0.0,cos(angle)]])
        R = np.dot(R_z(gamma),np.dot(R_y(beta),R_z(alpha)))
        rotation = Rotation(R)

        t_rotated = cls.from_one_site_cartesian_tensor(D,pseudospin)
        t_rotated.rotate(rotation)
        t_reference = cls.from_one_site_cartesian_tensor(np.dot(R,np.dot(D,R.T)),
                                                         pseudospin)
        check('rotation matches Cartesian rotation', t_rotated == t_reference)

        # inflate_dimension, reorder_spin_sites and separate_tensors.
        t_A = cls.from_one_site_cartesian_operator(1.0,'z',pseudospin)
        t_A.inflate_dimension([0,1])
        check('inflate_dimension',
              (t_A.n_sites == 2) and (t_A.rank_list == [[0,0,2,0]]))
        t_B = t_A.reorder_spin_sites([1,0])
        check('reorder_spin_sites', t_B.rank_list == [[2,0,0,0]])
        separated, multi_site = (t_A + t_B).separate_tensors()
        t_ref = cls.from_one_site_cartesian_operator(1.0,'z',pseudospin)
        check('separate_tensors one-site parts',
              (len(separated) == 2) and (separated[0] == t_ref)
              and (separated[1] == t_ref))
        check('separate_tensors empty multi-site part',
              (multi_site.n_sites == 2)
              and (multi_site.rank_list == [[0,0,0,0]])
              and (multi_site.parameter_list == [complex(0.0,0.0)]))

        # The constant (all-zero-rank) term and genuine inter-site terms must
        # both end up in the multi-site tensor, and only there.
        t_C = cls([[0,0,0,0],[2,0,2,0]],[complex(3.0,0.0),complex(0.5,0.0)])
        separated, multi_site = (t_A + t_B + t_C).separate_tensors()
        check('separate_tensors multi-site part',
              (multi_site == t_C) and (separated[0] == t_ref)
              and (separated[1] == t_ref))

        # Coupling a Heisenberg tensor must produce a pure scalar (K = 0).
        coupled = t_isotropic.couple_two_site_tensor()
        coupled.purge_ranks()
        check('coupled Heisenberg tensor is a scalar', coupled.rank_list == [[0,0]])

        # Utility methods.
        k_array, q_array = t_isotropic.fortran_rank_arrays()
        check('fortran_rank_arrays shape',
              k_array.shape == (t_isotropic.n_ranks,2))
        check('ITO table renders', len(str(t_isotropic.ITO_table())) > 0)
        check('ITO table is a ResultTable',
              isinstance(t_isotropic.ITO_table(),result_table.ResultTable))

        # The filtering options of ITO_table, checked on the parameter rows
        # stored in the returned table.
        def table_rows(table):
            return table.rows

        # A one-site tensor with ranks (k,q) = (0,0), (1,-1), (1,0), (1,1)
        # and (2,0), where all rank k = 2 parameters are negligibly small.
        t_table = cls([[0,0],[2,-2],[2,0],[2,2],[4,0]],
                      [complex(1.0,0.0),complex(0.5,-0.5),complex(0.2,0.0),
                       complex(-0.5,0.5),complex(1.0e-8,0.0)])
        check('ITO table prints all rows by default',
              len(table_rows(t_table.ITO_table())) == 5)
        half_rows = table_rows(t_table.ITO_table(half_table=True))
        check('ITO table half_table drops negative q1',
              (len(half_rows) == 4)
              and all(int(row[1]) >= 0 for row in half_rows))
        check('ITO table rank_threshold drops small rank combinations',
              len(table_rows(t_table.ITO_table(rank_threshold=1.0e-6))) == 4)
        check('ITO table combined filters',
              len(table_rows(t_table.ITO_table(rank_threshold=1.0e-6,
                                               half_table=True))) == 3)
        # In a multi-site tensor the half table representative of terms with
        # q1 = 0 is decided by the first non-zero q value.
        t_table = cls([[0,0,0,0],[0,0,2,2],[0,0,2,-2]],
                      [complex(1.0,0.0),complex(0.5,0.5),complex(-0.5,0.5)])
        check('ITO table half_table with q1 = 0',
              len(table_rows(t_table.ITO_table(half_table=True))) == 2)

        # Coordinate frame labels: the default frame is unspecified, an
        # attached label is printed, addition keeps the label only when
        # both operands carry the same one, and a rotation resets it.
        tmp_tensor = cls([(2,0)],[1.0])
        check('default frame is unspecified',
              tmp_tensor.frame is None and 'unspecified' in str(tmp_tensor))
        tmp_tensor.frame = 'principal magnetic axis frame'
        check('frame label is printed',
              'principal magnetic axis frame' in str(tmp_tensor))
        other_tensor = cls([(2,0)],[1.0])
        other_tensor.frame = 'principal magnetic axis frame'
        check('frame label survives addition of equal frames',
              (tmp_tensor + other_tensor).frame == 'principal magnetic axis frame')
        other_tensor.frame = 'input axis frame'
        check('frame label dropped on addition of different frames',
              (tmp_tensor + other_tensor).frame is None)
        check('frame label survives scalar multiplication',
              (2.0*tmp_tensor).frame == 'principal magnetic axis frame')
        tmp_tensor.rotate(Rotation(np.identity(3)))
        check('frame label reset by rotation', tmp_tensor.frame is None)

        return debug_output.test_summary('IwaharaChibotaruSphericalTensor',
                                         result_list,print_output)



class MixedCartesianIwaharaChibotaruSphericalTensor:
    """A class to store an ITO expansion of a Cartesian vector operator using the
    Iwahara--Chibotaru definition of ITOs given in

        N. Iwahara and L. F. Chibotaru. Phys. Rev. B, 2015, 91, 174438

    and

        N. Iwahara, L. Ungur and L. F. Chibotaru. Phys. Rev. B, 2018, 98, 054436.

    The tensor parameters g^a_(k1,q1,k2,q2,...) carry one Cartesian index a (x, y or z)
    in addition to the ranks k and components q of the spherical indices of the different
    spin sites. The main use of the mixed Cartesian--spherical tensors is to treat vector
    operators such as the magnetic moment operator. The Zeeman Hamiltonian can be written
    as

        H_Zeeman = sum_{a,k1,q1,k2,q2,...} B_a * g^a_{k1,q1,k2,q2,...}
                                             * O_{k1,q1}(J_1) * O_{k2,q2}(J_2) * ...

    where B_a is a Cartesian component of the magnetic field, and the mixed tensor g can
    be thought of as a generalization of the usual g tensor.

    The class is a higher-level container: the three Cartesian components are stored as
    three instances of IwaharaChibotaruSphericalTensor, and the details of the spherical
    rank structure are stored in the respective component instances. This class is to
    IwaharaChibotaruSphericalTensor what GeneralVectorOperatorMatrix is to
    GeneralOperatorMatrix.

    Note that the mathematical structure differs from the scalar spherical tensors: under
    a rotation the Cartesian index transforms with the Cartesian rotation matrix while
    the spherical indices transform with the Wigner D matrices, and under time reversal
    the tensor is odd in the rank structure of each component (a Cartesian vector operator
    such as the magnetic moment is time-odd).

    Two instances can be added, which corresponds to adding the three Cartesian component
    tensors. An instance can be multiplied by a scalar, which corresponds to multiplying
    all component tensors by that number.

    NOTE!!! All angular momentum quantum numbers are given as multiples of two in order to
            treat half-integer angular momenta as integer numbers.

    Arguments
    ---------
    component_list : list of IwaharaChibotaruSphericalTensor
        A list of exactly three tensors giving the Cartesian components of the vector
        operator in the order x, y, z. All three tensors must correspond to the same
        number of spin sites.

    Optional arguments
    ------------------

    Attributes
    ----------
    component_list : list of IwaharaChibotaruSphericalTensor
        The three Cartesian components of the tensor in the order x, y, z.
    n_sites : int
        The number of spin sites.
    frame : str or None
        A free-form label of the coordinate frame in which the tensor is
        expressed (e.g. 'principal magnetic axis frame' or 'input axis
        frame'), or None when the frame is not known. The label is
        reported by the printing methods, propagated through additions
        (only when both operands carry the same label) and scalar
        multiplications, and reset by rotations, after which the caller
        must relabel the frame if it is known.

    Private methods
    ---------------

    Public methods
    --------------
    component(component) : IwaharaChibotaruSphericalTensor
        Return the Cartesian component given as an argument (either 'x', 'y', 'z' or
        0, 1, 2). The instance can equivalently be indexed directly with the [] operator.
    one_site_cartesian_tensor(pseudospin) : CartesianTensor
        Return a Cartesian rank-two tensor representation (the generalized g tensor)
        constructed from the rank k=1 parameters of the three components. It is assumed
        that this tensor corresponds to a single site.
    contract_with_vector(vector) : IwaharaChibotaruSphericalTensor
        Contract the Cartesian index of the tensor with the Cartesian vector given as
        an argument and return the resulting scalar spherical tensor. For example:
        contraction of the Zeeman g tensor with the magnetic-field vector gives the
        spherical tensor of the Zeeman Hamiltonian.
    rotate(rotation)
        Rotate the tensor using the Rotation instance given as an argument. The Cartesian
        index is rotated with the Cartesian rotation matrix and the spherical indices with
        the Wigner D matrices.
    ITO_table(symbol="g",title=None,order_of_magnitude=0,rank_threshold=0.0,half_table=False) : ResultTable
        Return a listing of all ITO expansion parameters. The first index column is the
        Cartesian index, followed by the k1, q1, k2, q2, ... indices.
    inflate_dimension(site_list)
        Inflate all three components to a system with more spin sites (see
        IwaharaChibotaruSphericalTensor.inflate_dimension).
    reorder_spin_sites(reorder_list) : MixedCartesianIwaharaChibotaruSphericalTensor
        Reorder the spin sites of all three components and return the reordered tensor
        (see IwaharaChibotaruSphericalTensor.reorder_spin_sites).
    purge_ranks(threshold=1.0e-6)
        Remove terms with an absolute parameter value less than threshold from all three
        components.
    time_reversal_conjugate() : MixedCartesianIwaharaChibotaruSphericalTensor
        Returns the time-reversal conjugate of this tensor, constructed component-wise.
        The Cartesian index is unaffected by time reversal.
    hermitian_conjugate() : MixedCartesianIwaharaChibotaruSphericalTensor
        Returns the Hermitian conjugate of this tensor, constructed component-wise.
    separate_tensors() : list of MixedCartesianIwaharaChibotaruSphericalTensor, MixedCartesianIwaharaChibotaruSphericalTensor
        Separate the tensor into a list of one-site mixed tensors, one for each spin
        site, and a single multi-site mixed tensor (see
        IwaharaChibotaruSphericalTensor.separate_tensors).

    Class methods
    -------------
    from_general_vector_operator_matrix(vector_operator_matrix,basis)
        Construct the mixed tensor from an instance of GeneralVectorOperatorMatrix by
        constructing an IwaharaChibotaruSphericalTensor from the matrix representation of
        each of the three Cartesian components.
    from_one_site_cartesian_tensor(matrix,pseudospin)
        Construct a one-site mixed tensor from a Cartesian 3x3 matrix g, corresponding
        to the vector operator with components mu_a = sum_b g_ab * S_b. For example: the
        Zeeman interaction where g is the g tensor.
    from_one_site_isotropic_operator(parameter,pseudospin)
        Construct a one-site mixed tensor corresponding to the vector operator with
        components mu_a = X*S_a, where X is a scalar parameter. For example: the magnetic
        moment operator of an isotropic spin where X is the isotropic g value.
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if all
        tests passed.
    """

    def component(self,component):
        """Return the Cartesian component given as an argument (either 'x', 'y', 'z'
        or 0, 1, 2).
        """
        component_index_dictionary = {'x' : 0, 'y' : 1, 'z' : 2,
                                      0 : 0, 1 : 1, 2 : 2}

        if not component in component_index_dictionary:
            print("ERROR in MixedCartesianIwaharaChibotaruSphericalTensor.")
            print("ERROR: Unkown Cartesian component: " + str(component) + ".")
            print("Error termination.")
            sys.exit(1)

        return self.component_list[component_index_dictionary[component]]


    def one_site_cartesian_tensor(self,pseudospin):
        """Return a Cartesian rank-two tensor representation (the generalized g tensor)
        constructed from the rank k=1 parameters of the three components. It is assumed
        that this tensor corresponds to a single site.

        The rows of the returned tensor correspond to the Cartesian index of the mixed
        tensor: component a of the vector operator is mu_a = sum_b g_ab * S_b. This is
        the inverse of from_one_site_cartesian_tensor.
        """
        if not self.n_sites == 1:
            print("ERROR in MixedCartesianIwaharaChibotaruSphericalTensor.")
            print("Error: Cannot construct a one-site Cartesian tensor from multiple-site tensor.")
            print("Error termination.")
            sys.exit(1)

        matrix = np.zeros((3,3), dtype=np.float64)
        for alpha in range(0,3):
            matrix[alpha] = self.component_list[alpha].cartesian_vector(pseudospin)

        return CartesianTensor(matrix)


    def contract_with_vector(self,vector):
        """Contract the Cartesian index of the tensor with the Cartesian vector given
        as an argument and return the resulting scalar spherical tensor

            X_k1q1,k2q2,... = sum_a v_a * g^a_k1q1,k2q2,...

        For example: contraction of the Zeeman g tensor with the magnetic-field vector
        gives the spherical tensor of the Zeeman Hamiltonian.
        """
        if not len(vector) == 3:
            print("ERROR in MixedCartesianIwaharaChibotaruSphericalTensor.")
            print("ERROR: The vector must have exactly three Cartesian components.")
            print("Error termination.")
            sys.exit(1)

        contracted_tensor = None
        for alpha in range(0,3):
            contracted_tensor = vector[alpha]*self.component_list[alpha] \
                                + contracted_tensor

        return contracted_tensor


    def rotate(self,rotation):
        """Rotate the tensor using the Rotation instance given as an argument.

        The mixed tensor transforms differently from the scalar spherical tensors: the
        Cartesian index is rotated with the Cartesian rotation matrix R while the
        spherical indices of each component are rotated with the Wigner D matrices,

            g'^a_k1q1,k2q2,... = sum_b R_ab sum_q1'q2'... g^b_k1q1',k2q2',...
                                                          * D_q1q1'^k1(R) * D_q2q2'^k2(R) ...

        For a one-site rank-1 tensor this reduces to the usual transformation of the
        Cartesian g tensor: g' = R * g * R^T.
        """
        old_component_list = deepcopy(self.component_list)
        for old_component in old_component_list:
            old_component.rotate(rotation)

        R = rotation.rotation_matrix

        new_component_list = []
        for alpha in range(0,3):
            new_component = None
            for beta in range(0,3):
                new_component = R[alpha][beta]*old_component_list[beta] + new_component
            new_component_list.append(new_component)

        self.component_list = new_component_list

        # The rotation changes the coordinate frame; the caller must relabel
        # the frame if it is known.
        self.frame = None


    def ITO_table(self, symbol="g", title=None, order_of_magnitude=0, rank_threshold=0.0, half_table=False):
        """Return a listing of all ITO expansion parameters as an instance of
        ResultTable. Printing the returned instance, or converting it into a
        str, gives the human-readable table. The first index column is the
        Cartesian index a, followed by the k1, q1, k2, q2, ... indices and by
        the real part, the imaginary part and the magnitude of the expansion
        parameter.

        Optional arguments:
        -------------------
        symbol : str
            A symbol used to label the tensor elements in the output. Default is 'g'.
        title : str
            A title of the tensor table used in the output. Default is None, in which
            case no title is printed.
        order_of_magnitude : int
            Before printing, the values of the parameters will be multiplied by
            10^order_of_magnitude. Default is 0.
        rank_threshold : float
            If the magnitudes of all parameters of a given Cartesian component and rank
            combination (i.e., specific combination of a,k1,k2,k3,...) are below this
            threshold value, the parameters are not printed for this combination. The
            default value is 0.0 in which case all parameters are printed.
        half_table : boolean
            Since the parameters with all values of q inverted are related to
            each other by a simple phase difference, only print the values that are
            unique. These are chosen so that q1 is always non-negative in the printed
            output.
        """
        factor = 10.0**(order_of_magnitude)

        component_label_list = ['x','y','z']

        index_str = "a,"
        for i in range(0,self.n_sites):
            index_str += "k" + str(i+1) + "q" + str(i+1)
            if not i == self.n_sites-1:
                index_str += ","

        column_headers = ["a"]
        for i in range(0,self.n_sites):
            column_headers.append('k' + str(i+1))
            column_headers.append('q' + str(i+1))
        column_headers.extend(["Re(" + symbol + "_" + index_str + ")",
                               "Im(" + symbol + "_" + index_str + ")",
                               "|" + symbol + "_" + index_str + "|"])

        # The Cartesian label is a string and the ranks and the components
        # are integers; only the parameters are floating-point numbers.
        formats = (1 + 2*self.n_sites)*[None] + 3*['.6f']

        notes = []
        if self.frame is None:
            notes.append("Coordinate frame: unspecified")
        else:
            notes.append("Coordinate frame: " + self.frame)
        if not order_of_magnitude == 0:
            notes.append("The parameters are multiplied by 10^{0}."
                         .format(order_of_magnitude))

        rows = []

        for alpha in range(0,3):
            tensor = self.component_list[alpha]

            # Find the largest parameter magnitude of each rank combination
            # (k1,k2,...) of this Cartesian component for the rank_threshold
            # filter. The threshold is compared against the unscaled parameter
            # magnitudes, i.e., before multiplication by 10^order_of_magnitude
            # (consistent with the threshold of purge_ranks).
            max_magnitude = {}
            for i in range(0,len(tensor.rank_list)):
                k_combination = tuple(tensor.rank_list[i][0::2])
                magnitude     = abs(tensor.parameter_list[i])
                if (not k_combination in max_magnitude) or (magnitude > max_magnitude[k_combination]):
                    max_magnitude[k_combination] = magnitude

            for i in range(0,len(tensor.rank_list)):
                rank_tuple = tensor.rank_list[i]

                # Skip rank combinations where the magnitudes of all parameters
                # are below the threshold.
                if max_magnitude[tuple(rank_tuple[0::2])] < rank_threshold:
                    continue

                # In the half table only one of the two parameters related by an
                # inversion of all q values is printed. The printed representative
                # is chosen so that the first non-zero q value is positive, which
                # in particular means that q1 is always non-negative in the output.
                if half_table:
                    first_nonzero_q = 0
                    for q in rank_tuple[1::2]:
                        if not q == 0:
                            first_nonzero_q = q
                            break
                    if first_nonzero_q < 0:
                        continue

                row = [component_label_list[alpha]]
                for j in range(0,self.n_sites):
                    k = rank_tuple[2*j]
                    q = rank_tuple[2*j+1]
                    row.append(k//2)
                    row.append(q//2)
                row.extend([factor*tensor.parameter_list[i].real,
                            factor*tensor.parameter_list[i].imag,
                            abs(factor*tensor.parameter_list[i])])

                rows.append(row)

        return result_table.ResultTable(rows,
                                        column_headers=column_headers,
                                        title=title,
                                        notes=notes,
                                        formats=formats,
                                        table_type='spherical_tensor')


    def inflate_dimension(self,site_list):
        """Inflate all three components IN PLACE to a system with more spin sites. The
        tensor operator will still only act on the same spin sites as before, but it
        will include identity operators acting on other sites. The method modifies the
        instance and returns nothing, so a tensor that is still needed in its original
        form must be copied before it is inflated.

        The site list argument is a list of ones and zeros. The ones stand for the
        positions of the old sites in the new site list whereas zeros correspond to the
        sites to be added. The total number of ones must equal the current number of
        sites.
        """
        for component in self.component_list:
            component.inflate_dimension(site_list)

        self.n_sites = len(site_list)


    def reorder_spin_sites(self,reorder_list):
        """Reorder the spin sites of all three components according to indices in the
        list given as an argument, and return the reordered tensor. The list contains
        numbers 0,1,2,3,... up to the total number of spin sites in the system. The
        numbers refer to the order of the spin sites in the old tensor and their order
        determines the order of the spin sites in the new tensor.
        """
        new_component_list = []
        for component in self.component_list:
            new_component_list.append(component.reorder_spin_sites(reorder_list))

        return MixedCartesianIwaharaChibotaruSphericalTensor(new_component_list)


    def purge_ranks(self,threshold=1.0e-6):
        """Remove terms (both ranks and parameters) with an absolute parameter value less
        than threshold from all three components.
        """
        for component in self.component_list:
            component.purge_ranks(threshold=threshold)


    def time_reversal_conjugate(self):
        """Returns the time-reversal conjugate of this tensor.

        The Cartesian index is unaffected by time reversal, so the conjugate is
        constructed component-wise. Note that a physical Cartesian vector operator
        such as the magnetic moment is time-odd: its time-reversal conjugate equals
        the tensor multiplied by -1.
        """
        new_component_list = []
        for component in self.component_list:
            new_component_list.append(component.time_reversal_conjugate())

        return MixedCartesianIwaharaChibotaruSphericalTensor(new_component_list)


    def hermitian_conjugate(self):
        """Returns the Hermitian conjugate of this tensor. The Cartesian index is
        unaffected by Hermitian conjugation, so the conjugate is constructed
        component-wise.
        """
        new_component_list = []
        for component in self.component_list:
            new_component_list.append(component.hermitian_conjugate())

        return MixedCartesianIwaharaChibotaruSphericalTensor(new_component_list)


    def separate_tensors(self):
        """Separate the tensor into one-site mixed tensors and a multi-site mixed tensor.

        Each Cartesian component is separated with
        IwaharaChibotaruSphericalTensor.separate_tensors, and the separated components
        are collected back into mixed tensors.

        Returns
        -------
        tensor_list : list of MixedCartesianIwaharaChibotaruSphericalTensor
            A list of the one-site mixed tensors, one for each spin site.
        multi_site_tensor : MixedCartesianIwaharaChibotaruSphericalTensor
            A mixed tensor acting on all spin sites of this tensor, containing the
            constant terms and the terms acting on multiple sites.
        """
        separated_component_lists  = []
        multi_site_component_list  = []

        for component in self.component_list:
            one_site_tensor_list, multi_site_tensor = component.separate_tensors()
            separated_component_lists.append(one_site_tensor_list)
            multi_site_component_list.append(multi_site_tensor)

        tensor_list = []
        for n in range(0,self.n_sites):
            tensor_list.append(MixedCartesianIwaharaChibotaruSphericalTensor(
                [separated_component_lists[alpha][n] for alpha in range(0,3)]))

        multi_site_tensor = MixedCartesianIwaharaChibotaruSphericalTensor(multi_site_component_list)

        return tensor_list, multi_site_tensor


    def __getitem__(self,component):
        """Return the Cartesian component tensor with the given index (0 = x, 1 = y, 2 = z)."""
        return self.component(component)


    def __add__(self, other):
        """Add another mixed tensor into this tensor component-wise."""
        if other is None:
            return deepcopy(self)

        else:
            if not isinstance(other, MixedCartesianIwaharaChibotaruSphericalTensor):
                print("ERROR in MixedCartesianIwaharaChibotaruSphericalTensor.")
                print("ERROR: Can only add two mixed Cartesian--spherical tensors.")
                print("Error termination.")
                sys.exit(1)

            if not self.n_sites == other.n_sites:
                print("ERROR in MixedCartesianIwaharaChibotaruSphericalTensor.")
                print("ERROR: Cannot add two tensors with different number of sites.")
                print("Error termination.")
                sys.exit(1)

            new_component_list = []
            for alpha in range(0,3):
                new_component_list.append(self.component_list[alpha]
                                          + other.component_list[alpha])

            new_tensor = MixedCartesianIwaharaChibotaruSphericalTensor(new_component_list)
            # The frame label survives the addition only when both tensors
            # carry the same label.
            if getattr(other,'frame',None) == self.frame:
                new_tensor.frame = self.frame
            return new_tensor


    def __radd__(self, other):
        """Right-hand addition; see __add__. Supports sum() through the zero start value."""
        if other == 0:
            return deepcopy(self)
        else:
            return self.__add__(other)


    def __mul__(self, number):
        """Multiply all ITO decomposition parameters of all three components by the
        number given as an argument.
        """
        try:
            complex(number)
        except (TypeError, ValueError):
            print("ERROR in MixedCartesianIwaharaChibotaruSphericalTensor.")
            print("ERROR: Can only multiply the instance with a number.")
            print("Error termination.")
            sys.exit(1)

        new_component_list = []
        for component in self.component_list:
            new_component_list.append(number*component)

        new_tensor = MixedCartesianIwaharaChibotaruSphericalTensor(new_component_list)
        new_tensor.frame = self.frame
        return new_tensor


    def __rmul__(self, other):
        """Right-hand multiplication by a number; see __mul__."""
        return self.__mul__(other)


    def __eq__(self, other):
        """Check the equality of two mixed tensors by comparing their Cartesian components."""
        if not isinstance(other, MixedCartesianIwaharaChibotaruSphericalTensor):
            return False

        for alpha in range(0,3):
            if not self.component_list[alpha] == other.component_list[alpha]:
                return False

        return True


    def __repr__(self):
        """Return the ITO parameter table of the tensor."""
        return str(self.ITO_table())


    def __init__(self, component_list):
        """Upon class initiation, store copies of the three Cartesian component tensors
        given as an argument and check their consistency.
        """
        # See IwaharaChibotaruSphericalTensor for the meaning of the frame
        # label.
        self.frame = None

        if not len(component_list) == 3:
            print("ERROR in MixedCartesianIwaharaChibotaruSphericalTensor.")
            print("ERROR: Exactly three Cartesian components must be given.")
            print("Error termination.")
            sys.exit(1)

        for component in component_list:
            if not isinstance(component, IwaharaChibotaruSphericalTensor):
                print("ERROR in MixedCartesianIwaharaChibotaruSphericalTensor.")
                print("ERROR: The components must be instances of IwaharaChibotaruSphericalTensor.")
                print("Error termination.")
                sys.exit(1)

        self.component_list = deepcopy(list(component_list))

        self.n_sites = self.component_list[0].n_sites

        for component in self.component_list:
            if not component.n_sites == self.n_sites:
                print("ERROR in MixedCartesianIwaharaChibotaruSphericalTensor.")
                print("ERROR: Inconsistent number of spin sites in the components.")
                print("Error termination.")
                sys.exit(1)


    @classmethod
    def from_general_vector_operator_matrix(cls,vector_operator_matrix,basis):
        """Construct the mixed tensor from an instance of GeneralVectorOperatorMatrix.
        An IwaharaChibotaruSphericalTensor is constructed from the matrix representation
        of each of the three Cartesian components (see
        IwaharaChibotaruSphericalTensor.from_matrix_representation), and these are used
        as the components of the mixed tensor. The basis is given as an instance of
        PseudoSpinBasis.
        """
        if not vector_operator_matrix.n_operators == 3:
            print("ERROR in MixedCartesianIwaharaChibotaruSphericalTensor.")
            print("ERROR: The vector operator must have exactly three components.")
            print("Error termination.")
            sys.exit(1)

        component_list = []
        for alpha in range(0,3):
            matrix = vector_operator_matrix.operator_list[alpha].matrix
            component_list.append(IwaharaChibotaruSphericalTensor
                                  .from_matrix_representation(matrix,basis))

        return cls(component_list)


    @classmethod
    def from_one_site_cartesian_tensor(cls,matrix,pseudospin):
        """Construct a one-site mixed tensor from a Cartesian 3x3 matrix g, corresponding
        to the vector operator with components mu_a = sum_b g_ab * S_b. For example: the
        Zeeman interaction where g is the g tensor. Unlike in the scalar rank-two case,
        the matrix may contain an antisymmetric part.
        """
        component_list = []
        for alpha in range(0,3):
            component = None
            for beta in range(0,3):
                component = IwaharaChibotaruSphericalTensor\
                            .from_one_site_cartesian_operator(matrix[alpha][beta],
                                                              beta,pseudospin) \
                            + component
            component_list.append(component)

        return cls(component_list)


    @classmethod
    def from_one_site_isotropic_operator(cls,parameter,pseudospin):
        """Construct a one-site mixed tensor corresponding to the vector operator with
        components mu_a = X*S_a, where X is a scalar parameter. For example: the magnetic
        moment operator of an isotropic spin, where X is the isotropic g value.
        """
        component_list = []
        for component in ('x','y','z'):
            component_list.append(IwaharaChibotaruSphericalTensor
                                  .from_one_site_cartesian_operator(parameter,component,
                                                                    pseudospin))

        return cls(component_list)


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class and run a set of tests on the class constructor
        and the class methods. Return True if all tests passed and False
        otherwise.

        The constructors are tested against operator matrices built from
        explicit spin matrices, so that the tests verify the physics and not
        only the internal consistency of the class.

        Optional arguments
        ------------------
        print_output : boolean
            Whether to print the outcome of each test. Default is True.
        """
        from ouluspin import pseudospin_operators
        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('MixedCartesianIwaharaChibotaruSphericalTensor',
                                                       test_name,condition,print_output))

        pseudospin = 3   # S = 3/2
        S_ops = debug_output.spin_matrices(pseudospin)

        # The isotropic operator mu_a = X*S_a against explicit spin matrices.
        X = 1.3
        t_isotropic = cls.from_one_site_isotropic_operator(X,pseudospin)
        all_components_ok = (t_isotropic.n_sites == 1)
        for alpha in range(0,3):
            M = debug_output.operator_matrix_from_tensor(t_isotropic[alpha],[pseudospin])
            all_components_ok = all_components_ok and np.allclose(M,X*S_ops[alpha])
        check('isotropic operator matches spin matrices', all_components_ok)

        # A general Cartesian g matrix, including an antisymmetric part:
        # component a of the operator is mu_a = sum_b g_ab*S_b.
        g = np.array([[ 2.1, 0.3,-0.4],
                      [ 0.1, 1.7, 0.2],
                      [-0.4, 0.5, 2.4]])
        t_g = cls.from_one_site_cartesian_tensor(g,pseudospin)
        all_components_ok = True
        for alpha in range(0,3):
            M     = debug_output.operator_matrix_from_tensor(t_g[alpha],[pseudospin])
            M_ref = sum(g[alpha][beta]*S_ops[beta] for beta in range(0,3))
            all_components_ok = all_components_ok and np.allclose(M,M_ref)
        check('Cartesian tensor operator matches g.S matrices', all_components_ok)
        check('one-site Cartesian tensor round trip',
              np.allclose(t_g.one_site_cartesian_tensor(pseudospin).tensor,g))

        # Component access by label and by index.
        check('component access',
              (t_g.component('y') == t_g[1]) and (t_g.component(2) == t_g['z']))

        # Construction from a GeneralVectorOperatorMatrix must reproduce the
        # tensor constructed directly from the Cartesian g matrix.
        moment_matrix_list = []
        for alpha in range(0,3):
            moment_matrix_list.append(sum(g[alpha][beta]*S_ops[beta]
                                          for beta in range(0,3)))
        vector_operator = pseudospin_operators\
                          .GeneralVectorOperatorMatrix(moment_matrix_list)
        basis = pseudospin_operators.PseudoSpinBasis([pseudospin])
        t_matrix = cls.from_general_vector_operator_matrix(vector_operator,basis)
        check('construction from GeneralVectorOperatorMatrix', t_matrix == t_g)

        # Contraction with a Cartesian vector: the contracted scalar tensor
        # must correspond to the operator sum_ab v_a*g_ab*S_b.
        v = np.array([0.3,-1.2,0.8])
        contracted = t_g.contract_with_vector(v)
        M     = debug_output.operator_matrix_from_tensor(contracted,[pseudospin])
        M_ref = sum(v[alpha]*g[alpha][beta]*S_ops[beta]
                    for alpha in range(0,3) for beta in range(0,3))
        check('contraction with a Cartesian vector',
              isinstance(contracted,IwaharaChibotaruSphericalTensor)
              and np.allclose(M,M_ref))

        # Tensor algebra.
        check('addition and scalar multiplication',
              (t_g + t_g) == 2.0*t_g)
        check('addition with None', (t_g + None) == t_g)

        # Conjugations: the components of a Hermitian vector operator are
        # Hermitian, and a Cartesian vector operator built from real g is
        # time-odd.
        check('Hermitian conjugate of a Hermitian tensor',
              t_g.hermitian_conjugate() == t_g)
        check('time-reversal conjugate of a time-odd tensor',
              t_g.time_reversal_conjugate() == -1.0*t_g)

        # Rotation: the Cartesian index rotates with R and the spherical
        # indices with the Wigner D matrices, so that for a one-site rank-1
        # tensor the rotation must equal converting the rotated Cartesian
        # tensor R*g*R^T.
        alpha_angle, beta_angle, gamma_angle = 0.4, 0.9, -1.2
        R_z = lambda angle: np.array([[cos(angle),-sin(angle),0.0],
                                      [sin(angle), cos(angle),0.0],
                                      [0.0,0.0,1.0]])
        R_y = lambda angle: np.array([[cos(angle),0.0,sin(angle)],
                                      [0.0,1.0,0.0],
                                      [-sin(angle),0.0,cos(angle)]])
        R = np.dot(R_z(gamma_angle),np.dot(R_y(beta_angle),R_z(alpha_angle)))
        rotation = Rotation(R)

        t_rotated = cls.from_one_site_cartesian_tensor(g,pseudospin)
        t_rotated.rotate(rotation)
        t_reference = cls.from_one_site_cartesian_tensor(np.dot(R,np.dot(g,R.T)),
                                                         pseudospin)
        check('rotation matches Cartesian rotation', t_rotated == t_reference)

        # inflate_dimension, reorder_spin_sites and separate_tensors.
        t_A = cls.from_one_site_isotropic_operator(1.0,pseudospin)
        t_A.inflate_dimension([0,1])
        check('inflate_dimension',
              (t_A.n_sites == 2) and (t_A[2].rank_list == [[0,0,2,0]]))
        t_B = t_A.reorder_spin_sites([1,0])
        check('reorder_spin_sites', t_B[2].rank_list == [[2,0,0,0]])

        t_ref = cls.from_one_site_isotropic_operator(1.0,pseudospin)
        separated, multi_site = (t_A + t_B).separate_tensors()
        check('separate_tensors one-site parts',
              (len(separated) == 2) and (separated[0] == t_ref)
              and (separated[1] == t_ref))
        check('separate_tensors empty multi-site part',
              (multi_site.n_sites == 2)
              and all(multi_site[alpha].rank_list == [[0,0,0,0]]
                      and multi_site[alpha].parameter_list == [complex(0.0,0.0)]
                      for alpha in range(0,3)))

        # purge_ranks removes the negligibly small terms of all components.
        t_purge = 1.0e-9*t_g
        t_purge.purge_ranks()
        check('purge_ranks empties all components',
              all(t_purge[alpha].n_ranks == 0 for alpha in range(0,3)))

        # The equality operator must tolerate differing zero-valued ranks.
        t_padded = t_g + 0.0*t_isotropic
        check('equality with zero-padded ranks', t_padded == t_g)

        # The filtering options of ITO_table, checked on the parameter rows
        # stored in the returned table.
        def table_rows(table):
            return table.rows

        # Each component of t_table has ranks (k,q) = (1,-1), (1,0) and
        # (1,1) (zero-valued parameters included); the parameters of the
        # y component are negligibly small.
        t_table = cls.from_one_site_cartesian_tensor(np.diag([1.0,1.0e-8,2.0]),
                                                     pseudospin)
        rows = table_rows(t_table.ITO_table())
        check('ITO table prints all rows by default',
              (len(rows) == 9) and (rows[0][0] == 'x') and (rows[-1][0] == 'z'))
        half_rows = table_rows(t_table.ITO_table(half_table=True))
        check('ITO table half_table drops negative q1',
              (len(half_rows) == 6)
              and all(int(row[2]) >= 0 for row in half_rows))
        check('ITO table rank_threshold drops small components',
              all(row[0] in ('x','z') for row in
                  table_rows(t_table.ITO_table(rank_threshold=1.0e-6))))
        check('string representation renders', len(str(t_table)) > 0)

        return debug_output.test_summary('MixedCartesianIwaharaChibotaruSphericalTensor',
                                         result_list,print_output)



class CartesianTensor:
    """A class to store a Cartesian rank-two tensor. THe main purpose of this class is to
    store the tensor and convert it into either a single rank-two tensor given in terms
    of rank-two Iwahara--Chibotaru parameters or into three Cartesian vectors given in terms
    of rank-one Iwahara--Chibotaru parameters.

    The Iwahara--Chibotaru definition of an ITO decomposition is given in

        N. Iwahara and L. F. Chibotaru. Phys. Rev. B, 2015, 91, 174438

    and

        N. Iwahara, L. Ungur and L. F. Chibotaru. Phys. Rev. B, 2018, 98, 054436.

    This will be the form of ITOs used by the rest of this program.

    NOTE!!! All angular momentum quantum numbers are given as multipless of two in order to treat
            half-integer angular momenta as integer numbers.

    Arguments
    ---------
    tensor : array of float64
        A matrix representation of the tensor.

    Optional arguments
    ------------------

    Attributes
    ----------
    tensor : array of float64
        A matrix representation of the tensor.
    eigenvectors : array of float64
        A matrix containing the eigenvectors on its columns.
    eigenvalues : list of float
        The principal values of the tensor.
    symmetric : boolean
        Whether the tensor is symmetric or not.
    frame : str or None
        A free-form label of the coordinate frame in which the tensor is
        expressed (e.g. 'principal magnetic axis frame' or 'input axis
        frame'), or None when the frame is not known. The label is
        reported by the printing methods, propagated through additions
        (only when both operands carry the same label) and scalar
        multiplications, and reset by rotations, after which the caller
        must relabel the frame if it is known.

    Private methods
    ---------------

    Public methods
    --------------
    rank_one_ito(axis,pseudospin,transpose=False,part='full') : IwaharaChibotaruSphericalTensor
        Return an ITO decomposition of rank one from a vector (defined by the argument axis,
        which can have values x, y or z) of the Cartesian tensor.
    rank_two_ito(pseudospin) : IwaharaChibotaruSphericalTensor
        Return an ITO decomposition of rank two based on the tensor.
    trace() : float
        Return the trace of the tensor.
    isotropic_part() : float
        Return the isotropic part of the tensor defined as one third of the trace.
    symmetric_part() : array of float64
        Return the traceless symmetric part of the tensor as a matrix.
    antisymmetric_part() : array of float64
        Return the asymmetric part of the tensor as a matrix.
    rotate(rotation_matrix)
        Rotate the tensor using the rotation matrix given as an argument.
    tensor_table(order_of_magnitude=0) : ResultTable
        Return a table of the matrix representation of the tensor, including the full
        tensor, its symmetric traceless and anti-symmetric contributions and the
        isotropic part.
    principal_axis_table() : ResultTable or None
        Return a table of the principal values and principal axes of a symmetric
        tensor, or None when the tensor is not symmetric.

    Class methods
    -------------
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if all
        tests passed.
    """
    def rank_one_ito(self,axis,pseudospin,
                     transpose=False,
                     part='full'):
        """Return an ITO decomposition of rank one from a vector (defined by the argument axis,
        which can have values x, y or z) of the Cartesian tensor. 
        """
        if transpose:
            tmp_tensor = self.tensor.transpose()
        else:
            tmp_tensor = self.tensor

        tmp_pseudospin = float(pseudospin)/2.0
        
        rank_list = [(2,-2),(2,0),(2,2)]
        parameter_list = []
        
        parameter_list.append(tmp_pseudospin/sqrt(2.0) * complex( tmp_tensor[0][axis], tmp_tensor[1][axis]))
        parameter_list.append(tmp_pseudospin*tmp_tensor[2][axis])
        parameter_list.append(tmp_pseudospin/sqrt(2.0) * complex(-tmp_tensor[0][axis], tmp_tensor[1][axis]))

        return IwaharaChibotaruSphericalTensor(rank_list,parameter_list)

    
    def rank_two_ito(self, pseudospin):
        """Return an ITO decomposition up to rank two based on the tensor.

        The inverse conversions are given in equations (19)--(24) of the supporting
        information of

            A. Mansikkamäki, A. A. Popov, Q. Deng, N. Iwahara and L. F. Chibotaru.
            J. Chem. Phys., 2017, 147, 124305.

        The tensor is assumed as symmetric, in which case it can be represented
        as a sum of rank-zero and rank-two ITOs.

        The pseudospin is given as a multiple of two like everywhere else in
        the library. The conversion is identical to the one implemented in
        IwaharaChibotaruSphericalTensor.from_one_site_cartesian_tensor and is
        the inverse of IwaharaChibotaruSphericalTensor.one_site_cartesian_tensor.
        """
        if not np.allclose(self.tensor,self.tensor.T):
            print("ERROR in CartesianTensor.")
            print("Error: Rank-two ito decomposition of a asymmetric tensor requested.")
            print("Error termination.")
            sys.exit(1)

        tmp_pseudospin = float(pseudospin) / 2.0

        a = 3.0*tmp_pseudospin**2 - tmp_pseudospin*(tmp_pseudospin + 1.0)
        b = 1.0/sqrt(6.0)

        symmetric_part = self.symmetric_part()
        isotropic_part = self.isotropic_part()

        X20_real = 0.5*a   * symmetric_part[2][2]
        X22_real = 0.5*a*b * (symmetric_part[0][0] - symmetric_part[1][1])
        X22_imag =    -a*b * symmetric_part[0][1]
        X21_real =    -a*b * symmetric_part[0][2]
        X21_imag =     a*b * symmetric_part[1][2]

        # The negative-q parameters follow from Hermiticity:
        # X_k,-q = (-1)^q * conjugate(X_k,q). The rank-zero parameter is the
        # coefficient of the identity operator: S . (A_I * 1) . S = A_I*S(S+1).
        rank_list = [(0,0),(4,-4),(4,-2),(4,0),(4,2),(4,4)]
        parameter_list = [complex(isotropic_part*tmp_pseudospin*(tmp_pseudospin + 1.0),0.0),
                          complex( X22_real,-X22_imag),
                          complex(-X21_real, X21_imag),
                          complex( X20_real,0.0),
                          complex( X21_real, X21_imag),
                          complex( X22_real, X22_imag)]

        return IwaharaChibotaruSphericalTensor(rank_list,parameter_list)


    def trace(self):
        """Return the trace of the tensor."""
        return np.trace(self.tensor)

    
    def symmetric_part(self):
        """Return the traceless symmetric part of the tensor as a matrix.
        Note that the isotropic part is removed from the returned matrix.
        """
        return 0.5*(self.tensor + np.transpose(self.tensor)) - np.identity(3)*self.isotropic_part()
    

    def antisymmetric_part(self):
        """Return the asymmetric part of the tensor as a matrix."""
        return 0.5*(self.tensor - np.transpose(self.tensor))
    

    def isotropic_part(self):
        """Return the isotropic part of the tensor defined as one third of the trace."""
        return np.trace(self.tensor) / 3.0


    def rotate(self,rotation):
        """Rotate the tensor using the rotation  given as an argument."""
        old_trace = np.trace(self.tensor)
        self.tensor = np.dot(rotation.rotation_matrix, np.dot(self.tensor,np.transpose(rotation.rotation_matrix)))
        new_trace = np.trace(self.tensor)

        if abs(old_trace - new_trace) > 1.0e-6:
            print("ERROR in CartesianTensor.")
            print("Error: Tensor trace changed under rotation.")
            print("  Old trace: {0:20.8f}".format(old_trace))
            print("  New trace: {0:20.8f}".format(new_trace))
            print("Error termination.")
            sys.exit(1)

        if self.symmetric:
            self.eigenvalues, self.eigenvectors = la.eigh(self.tensor)
            self.eigenvalues = list(self.eigenvalues)

        # The rotation changes the coordinate frame; the caller must relabel
        # the frame if it is known.
        self.frame = None

        
    def tensor_table(self, order_of_magnitude=0):
        """Return a table of the matrix representation of the tensor as an
        instance of ResultTable. Printing the returned instance, or
        converting it into a str, gives the human-readable table.

        The table lists the full tensor and its symmetric traceless and
        anti-symmetric contributions as separate sections, one Cartesian
        row per line, and reports the isotropic part below the table. The
        principal values and axes of a symmetric tensor are tabulated
        separately by the principal_axis_table method.

        Optional arguments
        ------------------
        order_of_magnitude : int
            Before printing, the values of the tensor elements will be
            multiplied by 10^order_of_magnitude. Default is 0.
        """
        factor = 10**order_of_magnitude

        isotropic    = self.isotropic_part()
        full_matrix  = self.tensor
        symm_matrix  = self.symmetric_part()
        asymm_matrix = self.antisymmetric_part()

        component_label_list = ['x','y','z']

        rows        = []
        row_headers = []

        for section_label, matrix in (("Full tensor",full_matrix),
                                      ("Symmetric traceless contribution",symm_matrix),
                                      ("Anti-symmetric contribution",asymm_matrix)):
            rows.append(section_label)
            for i in range(0,3):
                rows.append([factor*matrix[i][j] for j in range(0,3)])
                row_headers.append(component_label_list[i])

        if self.frame is None:
            notes = ["Coordinate frame: unspecified"]
        else:
            notes = ["Coordinate frame: " + self.frame]
        if not order_of_magnitude == 0:
            notes.append("The tensor elements are multiplied by 10^{0}."
                         .format(order_of_magnitude))

        summary = ["Isotropic part: {0:12.6f}".format(factor*isotropic)]

        return result_table.ResultTable(rows,
                                        column_headers=component_label_list,
                                        row_headers=row_headers,
                                        title="MATRIX REPRESENTATION OF THE TENSOR",
                                        notes=notes,
                                        summary=summary,
                                        table_type='cartesian_tensor')


    def principal_axis_table(self):
        """Return a table of the principal values and the principal axes of
        a symmetric tensor as an instance of ResultTable, or None when the
        tensor is not symmetric and the principal values are therefore not
        available. Each line contains one principal value followed by the
        Cartesian components of the corresponding principal axis.
        """
        if not self.symmetric:
            return None

        component_label_list = ['x','y','z']

        rows = []
        for i in range(0,3):
            rows.append([self.eigenvalues[i]]
                        + [self.eigenvectors[j][i] for j in range(0,3)])

        if self.frame is None:
            notes = ["Coordinate frame: unspecified"]
        else:
            notes = ["Coordinate frame: " + self.frame]
        notes.append("The principal axes are written on the rows following the")
        notes.append("principal value they belong to.")

        return result_table.ResultTable(rows,
                                        column_headers=["Principal value",
                                                        "x","y","z"],
                                        row_headers=component_label_list,
                                        title="PRINCIPAL VALUES AND AXES OF THE TENSOR",
                                        notes=notes,
                                        table_type='cartesian_tensor')


    def __repr__(self):
        """Return the matrix representation of the tensor followed, for a
        symmetric tensor, by its principal values and axes.
        """
        tmp_str = str(self.tensor_table())

        principal_axis_table = self.principal_axis_table()
        if principal_axis_table is not None:
            tmp_str += str(principal_axis_table)

        return tmp_str

    
    def __add__(self,other):
        """Return the element-wise sum of two Cartesian tensors. The frame label is kept when both operands carry the same one."""
        new_tensor = CartesianTensor(deepcopy(self.tensor) + other.tensor)
        # The frame label survives the addition only when both tensors carry
        # the same label.
        if getattr(other,'frame',None) == self.frame:
            new_tensor.frame = self.frame
        return new_tensor
                

    def __init__(self, tensor):
        """Upon class initiation store the tensor. If the tensor is symmetric, also
        evaluate its eigenvalues and vectors.
        """
        # See IwaharaChibotaruSphericalTensor for the meaning of the frame
        # label.
        self.frame = None

        self.tensor = tensor

        if np.allclose(self.antisymmetric_part(),0.0,atol=1.0e-6):
            self.symmetric = True

            self.eigenvalues, self.eigenvectors = la.eigh(self.tensor)
            self.eigenvalues = list(self.eigenvalues)
        else:
            self.symmetric = False


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class and run a set of tests on the class constructor
        and the class methods. Return True if all tests passed and False
        otherwise.

        Optional arguments
        ------------------
        print_output : boolean
            Whether to print the outcome of each test. Default is True.
        """
        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('CartesianTensor',test_name,
                                                       condition,print_output))

        matrix = np.array([[2.0,0.5,0.0],
                           [0.5,1.0,0.0],
                           [0.0,0.0,3.0]])
        ct = cls(matrix.copy())

        check('symmetric tensor recognized', ct.symmetric)
        check('trace', abs(ct.trace() - 6.0) < 1.0e-12)
        check('isotropic part', abs(ct.isotropic_part() - 2.0) < 1.0e-12)
        check('symmetric part is traceless',
              abs(np.trace(ct.symmetric_part())) < 1.0e-12)
        check('antisymmetric part of a symmetric tensor vanishes',
              np.allclose(ct.antisymmetric_part(),0.0))
        check('decomposition reconstructs the tensor',
              np.allclose(ct.symmetric_part() + ct.antisymmetric_part()
                          + ct.isotropic_part()*np.identity(3),matrix))

        ct_diagonal = cls(np.diag([1.0,2.0,3.0]))
        check('eigenvalues of a diagonal tensor',
              np.allclose(sorted(ct_diagonal.eigenvalues),[1.0,2.0,3.0]))

        asymmetric = cls(matrix + np.array([[0.0,0.2,0.0],
                                            [-0.2,0.0,0.0],
                                            [0.0,0.0,0.0]]))
        check('asymmetric tensor recognized', not asymmetric.symmetric)

        # rank_two_ito must be the inverse of one_site_cartesian_tensor.
        pseudospin = 3
        ito = ct.rank_two_ito(pseudospin)
        check('rank_two_ito round trip',
              np.allclose(ito.one_site_cartesian_tensor(pseudospin).tensor,matrix))

        # rank_one_ito of column a must equal the sum of the Cartesian
        # one-site operators weighted by the column elements.
        ito = ct.rank_one_ito(2,pseudospin)
        ito_reference = sum(IwaharaChibotaruSphericalTensor
                            .from_one_site_cartesian_operator(matrix[a][2],component,
                                                              pseudospin)
                            for a, component in enumerate(('x','y','z')))
        check('rank_one_ito matches Cartesian operator sum', ito == ito_reference)

        # Rotation preserves the trace and the eigenvalues.
        alpha, beta, gamma = 0.3, 1.1, -0.7
        R = np.dot(np.array([[cos(gamma),-sin(gamma),0.0],
                             [sin(gamma), cos(gamma),0.0],
                             [0.0,0.0,1.0]]),
                   np.dot(np.array([[cos(beta),0.0,sin(beta)],
                                    [0.0,1.0,0.0],
                                    [-sin(beta),0.0,cos(beta)]]),
                          np.array([[cos(alpha),-sin(alpha),0.0],
                                    [sin(alpha), cos(alpha),0.0],
                                    [0.0,0.0,1.0]])))
        old_eigenvalues = sorted(ct.eigenvalues)
        ct.rotate(Rotation(R))
        check('rotation preserves eigenvalues',
              np.allclose(sorted(ct.eigenvalues),old_eigenvalues))

        check('addition', np.allclose((ct + ct).tensor,2.0*ct.tensor))
        check('tensor table renders', len(str(ct.tensor_table())) > 0)
        check('tensor table is a ResultTable',
              isinstance(ct.tensor_table(),result_table.ResultTable))
        check('the tensor table has one row per Cartesian row of each part',
              len([row for row in ct.tensor_table().rows
                   if isinstance(row,list)]) == 9)
        check('the principal axis table of a symmetric tensor renders',
              len(str(ct.principal_axis_table())) > 0)
        check('a non-symmetric tensor has no principal axis table',
              CartesianTensor(np.array([[0.0,1.0,0.0],
                                        [-1.0,0.0,0.0],
                                        [0.0,0.0,0.0]])).principal_axis_table() is None)

        # Coordinate frame label in the printed table.
        tmp_tensor = cls(np.identity(3))
        check('default frame is unspecified in the table',
              'unspecified' in str(tmp_tensor))
        tmp_tensor.frame = 'input axis frame'
        check('frame label is printed in the table',
              'input axis frame' in str(tmp_tensor))

        return debug_output.test_summary('CartesianTensor',result_list,print_output)



class Rotation:
    """A class to store a rotation matrix and the related Euler angles.

    The Euler angles correspond to a rotation using the Z--Y--Z convention,
    where the rotation operators are applied using angles alpha, beta and
    gamma in that order: R = R_z(gamma)*R_y(beta)*R_z(alpha).

    Arguments
    ---------
    rotation_matrix : array of float64
        The rotation matrix.

    Optional arguments
    ------------------

    Attributes
    ----------
    rotation_matrix : array of float64
        The rotation matrix.
    alpha : float
        The alpha Euler angle.
    beta : float
        The beta Euler angle.
    gamma : float
        The gamma Euler angle.

    Private methods
    ---------------
    __evaluate_euler_angles()
        Calculate a set of Euler angles based on the rotation
        matrix and store it as an attribute.
    __construct_matrix_from_angles(alpha,beta,gamma):
        Construct the rotation matrix based on the Euler angles
        given as arguments and return it.

    Public methods
    --------------
    wigner_D(J,M1,M2) : complex
        Return a Wigner D matrix corresponding to the rotation.
    inverse(construction="inversion") : Rotation
        Return a copy of self with inverse rotation.

    Class methods
    -------------
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if all
        tests passed.
    """
    
    def __evaluate_euler_angles(self):
        """Calculate a set of Euler angles based on the rotation
        matrix and store it as an attribute.
        """
        R11 = self.rotation_matrix[0][0]
        R13 = self.rotation_matrix[0][2]
        R21 = self.rotation_matrix[1][0]
        R23 = self.rotation_matrix[1][2]
        R31 = self.rotation_matrix[2][0]
        R32 = self.rotation_matrix[2][1]
        R33 = self.rotation_matrix[2][2]

        if abs(R33 - 1.0) < 1.0e-6:
            # For beta = 0 only the sum alpha + gamma is determined, and it
            # is assigned entirely to alpha. Note that atan2 must be used
            # instead of asin(R21), so that rotations about z by angles
            # outside the interval [-pi/2, pi/2] (e.g. by pi) are also
            # extracted correctly.
            self.beta = 0.0
            self.alpha = atan2(R21,R11)
            self.gamma = 0.0
        else:
            self.beta = acos(R33)
            self.alpha = atan2(R23,R13)
            self.gamma = atan2(R32,-R31)

    
    def __construct_matrix_from_angles(self,alpha,beta,gamma):
        """Construct the rotation matrix based on the Euler angles
        given as arguments and return it.

        The form of the matrix is given in equation (54), in
        Section 1.4.5 of

            D. A. Varshalovich, A. N. Moskalev, V. K. Khersonskii.
            Quantum Theory of Angular Momentum.
            1988, World Scientific, Singapore.
        """
        constructed_matrix = np.zeros((3,3), dtype=np.float64)

        constructed_matrix[0][0] = cos(alpha)*cos(beta)*cos(gamma) - sin(alpha)*sin(gamma)
        constructed_matrix[1][0] = sin(alpha)*cos(beta)*cos(gamma) + cos(alpha)*sin(gamma)
        constructed_matrix[2][0] = -sin(beta)*cos(gamma)
        constructed_matrix[0][1] = -cos(alpha)*cos(beta)*sin(gamma) - sin(alpha)*cos(gamma)
        constructed_matrix[1][1] = -sin(alpha)*cos(beta)*sin(gamma) + cos(alpha)*cos(gamma)
        constructed_matrix[2][1] = sin(beta)*sin(gamma)
        constructed_matrix[0][2] = cos(alpha)*sin(beta)
        constructed_matrix[1][2] = sin(alpha)*sin(beta)
        constructed_matrix[2][2] = cos(beta)

        return constructed_matrix

    
    def wigner_D(self,J,M1,M2):
        """Return a Wigner D matrix corresponding to the rotation."""
        return fu.angm_utils.wigner_big_d(J,M1,M2,self.alpha,self.beta,self.gamma)


    def inverse(self, construction="inversion"):
        """Return a copy of self with inverse rotation.

        If the optional argument is "inversion", the matrix is
        constructed by direct matrix inversion, and if it is "euler",
        then the inverse is constructed by reversing the rotation based
        on the Euler angles. Both methods should of course produce the
        same matrix and this is mostly intended for debugging
        purposes.

        """
        if construction == 'inversion':
            inverse_rotation_matrix = la.inv(self.rotation_matrix)
        elif construction == 'euler':
            reverse_alpha = -self.gamma
            reverse_beta = -self.beta
            reverse_gamma = -self.alpha

            inverse_rotation_matrix = self.__construct_matrix_from_angles(reverse_alpha,reverse_beta,reverse_gamma)
        else:
            print("ERROR in Rotation.")
            print("ERROR: Unrecognized argument for inverse: " + str(construction))
            print("Error termination.")
            sys.exit(1)
            
        return Rotation(inverse_rotation_matrix)
    

    def __repr__(self):
        """Return a human-readable string of the rotation matrix and the Euler angles."""
        tmp_str = "    Rotation matrix:\n\n"
        tmp_str += "    --------------------------------------\n"
        for i in range(0,3):
            tmp_str += "   "
            for j in range(0,3):
                tmp_str += " {0:12.8f}".format(self.rotation_matrix[i][j])
            tmp_str += "\n"
        tmp_str += "    --------------------------------------\n"
        tmp_str += "               Euler alpha angle {0:9.4f}\n".format(360.0*self.alpha / (2*pi))
        tmp_str += "               Euler beta  angle {0:9.4f}\n".format(360.0*self.beta / (2*pi))
        tmp_str += "               Euler gamma angle {0:9.4f}\n".format(360.0*self.gamma / (2*pi))
        tmp_str += "    --------------------------------------\n\n"
        
        return tmp_str

    
    def __init__(self, rotation_matrix):
        """Upon class iniations, store the rotation matrix and
        evaluate the Euler angles.
        """
        self.rotation_matrix = rotation_matrix

        # Check that the rotation matrix belongs to SO(3).
        if not abs(la.det(self.rotation_matrix) - 1.0) < 1.0e-6:
            print("ERROR in Rotation.")
            print("ERROR: Rotation matrix is not unimodular.")
            print("Error termination.")
            sys.exit(1)

        if not np.allclose(self.rotation_matrix.imag, 0.0):
            print("ERROR in Rotation.")
            print("ERROR: Rotation matrix is not real.")
            print("Error termination.")
            sys.exit(1)

        if not np.allclose(np.dot(np.transpose(self.rotation_matrix),self.rotation_matrix), np.identity(3)):
            print("ERROR in Rotation.")
            print("ERROR: Rotation matrix is not orthogonal.")
            print("Error termination.")
            sys.exit(1)


        self.__evaluate_euler_angles()

        # Note that the thresholds for the comparison have to be reduced here. If R33 is very close
        # to one, the R31 and R32 elements, for example, can still have values of the order 10^-5.
        matrix_constructed_from_angles = self.__construct_matrix_from_angles(self.alpha,self.beta,self.gamma)
        
        if not np.allclose(self.rotation_matrix,matrix_constructed_from_angles, atol=1.0e-4):
            print("ERROR in Rotation.")
            print("ERROR: Matrix constructed from Euler angles does not match the original matrix.")
            print()
            print("     {0:>38}     {1:>38}".format("Original rotation matrix:","Matrix constructed from Euler Angles:"))
            for i in range(0,3):
                print("    ", end="")
                for j in range(0,3):
                    print(" {0:12.6f}".format(self.rotation_matrix[i][j]), end="")
                print("    ", end="")
                for j in range(0,3):
                    print(" {0:12.6f}".format(matrix_constructed_from_angles[i][j]), end="")
                print()
            print()
            print("Error termination.")
            sys.exit(1)


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class and run a set of tests on the class constructor
        and the class methods. Return True if all tests passed and False
        otherwise.

        Optional arguments
        ------------------
        print_output : boolean
            Whether to print the outcome of each test. Default is True.
        """
        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('Rotation',test_name,
                                                       condition,print_output))

        alpha, beta, gamma = 0.4, 0.9, -1.2
        R_z = lambda angle: np.array([[cos(angle),-sin(angle),0.0],
                                      [sin(angle), cos(angle),0.0],
                                      [0.0,0.0,1.0]])
        R_y = lambda angle: np.array([[cos(angle),0.0,sin(angle)],
                                      [0.0,1.0,0.0],
                                      [-sin(angle),0.0,cos(angle)]])
        # This is the Z-Y-Z composition corresponding to eq. (54) in Section
        # 1.4.5 of Varshalovich et al., which is the matrix implemented in
        # __construct_matrix_from_angles.
        R = np.dot(R_z(alpha),np.dot(R_y(beta),R_z(gamma)))

        # The constructor itself validates that the matrix is a proper
        # rotation and that the extracted Euler angles reproduce the matrix.
        rotation = cls(R)
        check('Euler angle extraction',
              (abs(rotation.alpha - alpha) < 1.0e-6) and
              (abs(rotation.beta  - beta)  < 1.0e-6) and
              (abs(rotation.gamma - gamma) < 1.0e-6))

        inverse_a = rotation.inverse(construction='inversion')
        inverse_b = rotation.inverse(construction='euler')
        check('inverse by inversion and by Euler angles agree',
              np.allclose(inverse_a.rotation_matrix,inverse_b.rotation_matrix,
                          atol=1.0e-8))
        check('inverse rotation',
              np.allclose(np.dot(R,inverse_a.rotation_matrix),np.identity(3)))

        # Wigner D matrix: unitarity for a rank-2 (k = 4 in doubled units)
        # representation, and the identity rotation gives the unit matrix.
        k = 4
        unitary = True
        for q1 in range(-k,k+2,2):
            row_sum = 0.0
            for q2 in range(-k,k+2,2):
                row_sum += abs(rotation.wigner_D(k,q1,q2))**2
            unitary = unitary and (abs(row_sum - 1.0) < 1.0e-10)
        check('Wigner D matrix is unitary', unitary)

        identity_rotation = cls(np.identity(3))
        identity_ok = True
        for q1 in range(-k,k+2,2):
            for q2 in range(-k,k+2,2):
                D = identity_rotation.wigner_D(k,q1,q2)
                if q1 == q2:
                    identity_ok = identity_ok and (abs(D - 1.0) < 1.0e-10)
                else:
                    identity_ok = identity_ok and (abs(D) < 1.0e-10)
        check('Wigner D matrix of the identity rotation', identity_ok)

        # A rotation about z by pi exercises the beta = 0 branch of the
        # Euler angle extraction, which must use atan2 instead of asin so
        # that angles outside [-pi/2, pi/2] are extracted correctly.
        rotation_z_pi = cls(np.diag([-1.0,-1.0,1.0]))
        check('Euler angles of a rotation about z by pi',
              (abs(abs(rotation_z_pi.alpha) - pi) < 1.0e-6) and
              (abs(rotation_z_pi.beta) < 1.0e-6))

        check('string representation renders', len(str(rotation)) > 0)

        return debug_output.test_summary('Rotation',result_list,print_output)

