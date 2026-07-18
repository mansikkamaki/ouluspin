# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import time

from copy import deepcopy
from cmath import phase as cphase
from math import sqrt, pi

import numpy as np
import numpy.linalg as la

from ouluspin import pseudospin_operators
from ouluspin import properties
from ouluspin import tensors
from ouluspin._fortran import fortran_utils as fu

from ouluspin import _debug as output


class ElectronExchangeSystem:
    """A class to set up the different exchange and Zeeman interactions in a general
    pseudospin system consisting of multiple spin sites. Each described by a Zeeman
    and zero-field splitting (ZFS) or crystal-field (CF) tensors given in the general
    Iwahara--Chibotaru form. The main purpose of this class is to use it for the
    evaluation of magnetization and magnetic susceptibility.

    Coordinate frame convention: all tensors given to this class are
    assumed to be expressed in one common coordinate frame, conventionally
    the principal magnetic axis frame of the ground Kramers/quasi doublet
    (for a model system without an ab initio reference the frame is
    arbitrary but must be shared by all tensors). The properties evaluated
    by this class (powder magnetization and susceptibility) are invariant
    under the choice of this frame.
    
    No assumptions are made on the field strenghts, and both the magnetization and 
    the susceptibility are calculated using powder-integration and explicitly treat
    the Zeeman interaction. The susceptibility is evaluated as linear susceptibility
    based on the calculated magnetization. No differential susceptibility is
    calculated.

    NOTE!!! The magnetic moment tensor given as input implies the g tensor, not the
            actual magnetic moment. Multiplication by negative Bohr magneton is
            carried out as part of the procedure.

    Arguments
    ---------
    pseudospin_list : list of int
        A list of pseudospin for each spin site, each given as a multiple of two.
    hamiltonian_tensor_tuple_list : list of tuple of int and IwaharaChibotaruSphericalTensor
        A list of tuples defining the tensors describing the system at zero
        external field. Each list element is a tuple. The first element of the
        tuple is an instance of IwaharaChibotaruSpherical tensor, and the following
        elements list the spin sites the tensor corresponds to. For a two-site
        tensor there must be two integer numbers etc.
    magnetic_moment_tensor_tuple_list : list of tuple of int and MixedCartesianIwaharaChibotaruSphericalTensor
        A list of tuples defining the tensors describing the magnetic moment operators
        of each spins site. Each list element is a tuple. The first element of the
        tuple is an instance of MixedCartesianIwaharaChibotaruSphericalTensor
        containing the three Cartesian components of the magnetic moment operator,
        and the following elements list the spin sites the tensor corresponds to.
        For a two-site tensor there must be two integer numbers etc. It is possible
        to define magnetic moment operators that act on multiple sites whatever the
        physical implication of such operators is.
    units : EnergyUnitSystem
        The nergy unit system.

    Optional arguments
    ------------------
    print_output : boolean
        Whether to print output. Default is False.
    energy_print_threshold : float or None
        Threshold for printing eigenvalues and eigenvectors of the Hamiltonian.
        If the value is set to None, which is the default, all values and
        vectors will be printed.

    Attributes
    ----------
    pseudospin_list : list of int
        A list of pseudospin for each spin site, each given as a multiple of two.
    hamiltonian_tensor_tuple_list : list of tuple of int and IwaharaChibotaruSphericalTensor
        A list of tuples defining the tensors describing the system at zero
        external field. Each list element is a tuple. The first element of the
        tuple is an instance of IwaharaChibotaruSpherical tensor, and the following
        elements list the spin sites the tensor corresponds to. For a two-site
        tensor there must be two integer numbers etc.
    magnetic_moment_tensor_tuple_list : list of tuple of int and MixedCartesianIwaharaChibotaruSphericalTensor
        A list of tuples defining the tensors describing the magnetic moment operators
        of each spins site. Each list element is a tuple. The first element of the
        tuple is an instance of MixedCartesianIwaharaChibotaruSphericalTensor
        containing the three Cartesian components of the magnetic moment operator,
        and the following elements list the spin sites the tensor corresponds to.
        For a two-site tensor there must be two integer numbers etc. It is possible
        to define magnetic moment operators that act on multiple sites whatever the
        physical implication of such operators is.
    units : EnergyUnitSystem
        The nergy unit system.
    print_output : boolean
        Whether to print output.
    hamiltonian : PseudoSpinOperator
        The field-free part of the Hamiltonian. I.e., the exchange and ZFS terms.
    magnetic_moment : PseudoSpinVectorOperator
        A vector operator consisting of the three Cartesian components of the
        magnetic moment.
    n_sites : int
        The number of spin sites in the system.
    basis : PseudoSpinBasis
        The basis used in construction of the operators.
    energy_print_threshold : float or None
        The energy threshold above which eigenvalues are omitted from the
        printed output; see the corresponding optional argument.

    Private methods
    ---------------
    __construct_hamiltonian(B)
        Construct the Hamiltonian consisting of Zeeman, ZFS and exchange terms
        using the field vector given as an argument, and store it as an attribute.
        The Hamiltonian is diagonalized upon construction.
    __construct_magnetic_moment_operators()
        Construct the operators of the three Cartesian components of the magnetic
        moment and store them as attributes. Construction of the operator matrices
        is requested upon their construction but they are not diagonalized.

    Public methods
    --------------

    Class methods
    -------------
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if all
        tests passed.
    """

    def __construct_hamiltonian(self):
        """Construct the field-independent terms of the Hamiltonian and store them
        as an attribute. The Hamiltonian matrix is stored in the instance but
        not diagonalized.
        """
        tensor_list = []

        tikk = time.time()
        if self.print_output:
            print("    Constructing the Hamiltonian ...")
            print("      ITOs ...")

        for tensor_tuple in self.hamiltonian_tensor_tuple_list:

            inflation_list = []
            for i in range(0,self.n_sites):
                if i in tensor_tuple:
                    inflation_list.append(1)
                else:
                    inflation_list.append(0)

            tensor = tensor_tuple[0]
            tensor.inflate_dimension(inflation_list)
            tensor_list.append(tensor)
                        
        # Operator matrix
        if self.print_output:
            print("      Operator matrix ...")
                
        self.hamiltonian = pseudospin_operators\
            .PseudoSpinOperator(self.basis, tensor_list, self.units,
                                eigenvalue_print_limit=self.energy_print_threshold,
                                diagonalize_operator_matrix=True,
                                store_operator_matrix=True,
                                translate_eigenvalues=True)

        tokk = time.time()
        if self.print_output:
            print("    Done.")
            print("    Time spent {0:12.3f}".format(tokk-tikk))
            print()

    
    def __construct_magnetic_moment_operators(self):
        """Construct the operators of the three Cartesian components of the magnetic
        moment and store them as attributes. Construction of the operator matrices
        is requested upon their construction but they are not diagonalized.
        """
        tikk = time.time()
        
        if self.print_output:
            print("    Constructing magnetic moment operators ...")
            

        if self.print_output:
            print("      ITOs ...")

        mixed_tensor_list = []

        for tensor_tuple in self.magnetic_moment_tensor_tuple_list:
            inflation_list = []
            for i in range(0,self.n_sites):
                if i in tensor_tuple:
                    inflation_list.append(1)
                else:
                    inflation_list.append(0)

            # The multiplication returns a new instance, so the inflation does
            # not modify the mixed tensor stored in the tuple list.
            mixed_tensor = -self.units.mu_B * tensor_tuple[0]
            mixed_tensor.inflate_dimension(inflation_list)

            mixed_tensor_list.append(mixed_tensor)

        if self.print_output:
            print("      Operator matrix ...")

        self.magnetic_moment = pseudospin_operators.PseudoSpinVectorOperator(self.basis, mixed_tensor_list, self.units,
                                                                             diagonalize_operator_matrix=False,
                                                                             store_operator_matrix=True,
                                                                             eigenvalue_print_limit=self.energy_print_threshold,
                                                                             translate_eigenvalues=False)

        tokk = time.time()
        if self.print_output:
            print("    Done.")
            print("    Time spent {0:12.3f}".format(tokk-tikk))
            print()

    
    def __repr__(self):
        # Note that this class does not itself store susceptibility or
        # magnetization instances; they may be attached by external code.
        """Return a human-readable summary of the system."""
        tmp_str = ""
        if getattr(self,'susceptibility',None) is not None:
            tmp_str += str(self.susceptibility)
        if getattr(self,'magnetization',None) is not None:
            tmp_str += str(self.magnetization)

        return tmp_str
    

    def __init__(self, pseudospin_list, hamiltonian_tensor_tuple_list, magnetic_moment_tensor_tuple_list, units,
                 print_output=False,
                 energy_print_threshold=None):

        """Upon class initiation, check the consistency of the tensor and site lists, sum the tensors site by site and construct the pseudospin Hamiltonian and magnetic moment operators."""
        self.pseudospin_list                   = pseudospin_list
        self.hamiltonian_tensor_tuple_list     = hamiltonian_tensor_tuple_list
        self.magnetic_moment_tensor_tuple_list = magnetic_moment_tensor_tuple_list
        self.units                             = units

        self.print_output           = print_output
        self.energy_print_threshold = energy_print_threshold

        self.n_sites = len(self.pseudospin_list)

        tikk = time.time()
        if self.print_output:
            print(self.units)
            print("    Constructing basis ...")
        self.basis = pseudospin_operators.PseudoSpinBasis(self.pseudospin_list)

        tokk = time.time()
        if self.print_output:
            print("    Done.")
            print("    Time spent {0:12.3f}".format(tokk-tikk))
            print()
            print(self.basis)
            
        self.__construct_magnetic_moment_operators()
        self.__construct_hamiltonian()


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
        from ouluspin import units
        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('ElectronExchangeSystem',
                                                       test_name,condition,print_output))

        tmp_units = units.EnergyUnitSystem('wavenumber')

        # A Heisenberg dimer of two S = 1/2 spins with J = 10 cm^-1 and
        # isotropic g = 2 on both sites.
        J = 10.0
        heisenberg = tensors.IwaharaChibotaruSphericalTensor\
                            .from_two_site_isotropic_operator(J,1,1)

        moment_tensor_tuple_list = []
        for site in (0,1):
            moment_tensor = tensors.MixedCartesianIwaharaChibotaruSphericalTensor\
                                   .from_one_site_isotropic_operator(2.0,1)
            moment_tensor_tuple_list.append((moment_tensor,site))

        system = cls([1,1],[(heisenberg,0,1)],moment_tensor_tuple_list,tmp_units)

        check('number of sites and basis dimension',
              (system.n_sites == 2) and (system.basis.n_basis == 4))
        check('Heisenberg singlet-triplet gap',
              np.allclose(system.hamiltonian.eigenvalues,[0.0,J,J,J]))
        check('Hamiltonian matrix is Hermitian',
              np.allclose(system.hamiltonian.matrix,
                          system.hamiltonian.matrix.conj().T))

        moment_matrix_list = system.magnetic_moment.matrix_list()
        check('three Cartesian magnetic moment components',
              len(moment_matrix_list) == 3)
        check('magnetic moment matrices are Hermitian and traceless',
              all(np.allclose(matrix,matrix.conj().T)
                  and abs(np.trace(matrix)) < 1.0e-10
                  for matrix in moment_matrix_list))

        # The moment includes the factor -mu_B*g: the largest eigenvalue of
        # mu_z of two parallel spins is 2*(g/2)*mu_B = 2*mu_B.
        mu_z_eigenvalues = np.linalg.eigvalsh(moment_matrix_list[2])
        check('magnitude of the total magnetic moment',
              abs(max(abs(mu_z_eigenvalues)) - 2.0*tmp_units.mu_B) < 1.0e-10)

        return debug_output.test_summary('ElectronExchangeSystem',
                                         result_list,print_output)



class AbInitioElectronExchangeSystem:
    """A class to construct a pseudospin system from a the results of a
    quantum chemical calculation. The class takes as arguments matrix
    representations of the Hamiltonian, the Cartesian components of
    the magnetic moment operator and the Cartesian components of the
    spin operator that have been calculated using some
    quantum-chemical methods. The class also takes as argument an
    instance of PseudoSpinBasis that defines the pseudospin system.
    The ZFS and Zeeman g-tensors are then constructed from the results.

    The pseudospin basis states are chosen as the eigenstates of the
    mu_z component of the magnetic moment operator within the space of
    the lowest Hamiltonian eigenstates. The arbitrary phases left over
    from the numerical diagonalization are fixed using the mu_x
    component of the magnetic moment operator, so that the basis
    behaves as a proper |S,M> basis under time reversal.

    By default the pseudospin is quantized along the principal
    magnetic axis of the ground Kramers/Ising/pseudo doublet. An
    alternative rotation matrix can be passed as an optional argument.

    Arguments
    ---------
    hamiltonian : GeneralOperatorMatrix
        A matrix representation of the Hamiltonian. The instance must
        contain the eigenvalues and eigenvectors as an attribute.
    magnetic_moment : GeneralVectorOperatorMatrix
        Matrix representations of the vector components of the
        magnetic moment vector operator.
    basis : PseudoSpinBasis
        The basis defining the pseudospin system.
    units : EnergyUnitSystem
        The energy unit system.

    Optional arguments
    ------------------
    reorder_list : list of int
        The pseudospin states are chosen as the eigenstates of the
        magnetic moment operator. By default it is assumed that the
        largest magnetic moment is the lowest in energy and this
        increases equally. If this order needs to be changed, the
        new order can be given as a list with the new pseudospin
        state indices.
    R : array of real64
        A 3 by 3 rotation matrix defining the transformatin from the
        input coordinate system to a system where the quantization
        axis coincides with the z axis. The default is None, in which
        case the transformation matrix will be calculated from the
        ground Kramers/Ising/pseudo doublet of the system.
    print_output : boolean
        Whether to print output. The default is False.

    Attributes
    ----------
    hamiltonian : GeneralOperatorMatrix
        A matrix representation of the Hamiltonian. The instance must
        contain the eigenvalue and eigenvectors as an attribute.
    magnetic_moment : GeneralVectorOperatorMatrix
        Matrix representations of the vector components of the
        magnetic moment vector operator.
    basis : PseudoSpinBasis
        The basis defining the pseudospin system.
    units : EnergyUnitSystem
        The energy unit system.
    kramers_system : boolean
        Whether this system is a Kramers (True) or a non-Kramers (False)
        system.
    reorder_list : list of int
        The pseudospin states are chosen as the eigenstates of the
        magnetic moment operator. By default it is assumed that the
        largest magnetic moment is the lowest in energy and this
        increases equally. If this order needs to be changed, the
        new order can be given as a list with the new pseudospin
        state indices.
    R : array of real64
        A 3 by 3 rotation matrix defining the transformation from the
        input coordinate system to a system where the quantization
        axis coincides with the z axis. NOTE: this is the rotation that
        was APPLIED to the operators during the construction of the
        instance. For systems whose operators were rotated before the
        construction (e.g. those built by from_average_aniso_data, which
        passes the identity here) it does not describe the relation to the
        input frame; the input_frame_rotation attribute below is the
        authoritative record of that relation and should be used for all
        frame bookkeeping.
    input_frame_rotation : Rotation
        The full rotation from the input coordinate frame (the frame of the
        operator matrices given to the constructor, e.g. the frame of a
        SINGLE_ANISO datafile) to the frame of the pseudospin operators,
        stored as a Rotation instance. For a system constructed with
        from_average_aniso_data this is the composition of the rotation
        into the principal frame of the first datafile and the rotation
        into the principal frame of the averaged ground doublet. It can be
        passed to PseudoSpinDoublet so that the magnetic axes of the
        doublets are reported with respect to the input frame.
    tensor_frame : str
        The coordinate frame label attached to the tensors returned by
        hamiltonian_tensor and magnetic_moment_tensor:
        'principal magnetic axis frame' when the frame rotation was
        determined from the ground doublet (R was not given), and
        'user-defined axis frame' when an explicit R was passed to the
        constructor.
    print_output : boolean
        Whether to print output.
    n_basis : int
        The number of pseudospin basis states.
    n_full_basis : int
        The dimension of the operator matrices given as class
        arguments.
    pseudospin_hamiltonian_matrix : array of complex128
        Matrix representation of the pseudospin Hamiltonian in the
        pseudospin eigenbasis.
    pseudospin_magnetic_moment_matrix : list of array of complex128
        A list of the matrix representations of the three Cartesian
        components of the magnetic moment operators in the 
        pseudospin eigenbasis.

    Public methods
    --------------
    hamiltonian_tensor() : IwaharaChibotaruSphericalTensor
        Construct and return an irreducible tensor representation of
        the Hamiltonian.
    magnetic_moment_tensor() : MixedCartesianIwaharaChibotaruSphericalTensor
        Construct and return a mixed Cartesian--spherical tensor
        representation of the magnetic moment operator containing the
        three Cartesian components.
    hamiltonian_operator() : PseudoSpinOperator
        Construct and return the pseudospin Hamiltonian of the system as
        a PseudoSpinOperator instance.
    magnetic_moment_operator() : PseudoSpinVectorOperator
        Construct and return the pseudospin magnetic moment operator of
        the system as a PseudoSpinVectorOperator instance.
    static_transition_magnetic_moments(n_states=None) : StaticTransitionMagneticMoments
        Construct and return the StaticTransitionMagneticMoments instance
        of the system.
    pseudospin_doublet(states) : PseudoSpinDoublet
        Construct and return the PseudoSpinDoublet instance of the doublet
        spanned by the two pseudospin eigenstates given in the states
        tuple, with the g-tensor reported in the input axis frame.
    pseudospin_doublet_table(doublets) : str
        Construct and return a tabulation string of one or several
        pseudospin doublets, with the g-tensor axes given in the input
        axis frame.

    Class methods
    -------------
    from_aniso_data(filename,pseudospin,units,...)
        Construct the system from a single SINGLE_ANISO datafile (a .aniso
        type file containing the operator matrices, not the SINGLE_ANISO
        output file). The pseudospin structure can be given as a single
        pseudospin, a list of pseudospins or a complete PseudoSpinBasis.
    from_average_aniso_data(filename_1,filename_2,pseudospin,lande_g_factor,units,...)
        Construct the system by averaging the crystal fields of two
        SINGLE_ANISO datafiles. The crystal field of the second system is
        rotated into the principal magnetic frame of the first before the
        averaging, the magnetic moment is constructed from the Lande
        g-factor, and the averaged system is finally rotated into the
        principal magnetic frame of its own ground doublet.
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if all
        tests passed.

    Private methods
    ---------------
    __calculate_rotation() : void
        Calculate g-tensor of the ground Kramers/Ising/pseudo doublet
        and store the transformation into its principal axis system
        as an attribute.
    __calculate_reorder_matrix() : array of float64
        Calculate and return an orthogonal matrix based on the reorder_list
        attribute that can be used to reorder the states.
    __construct_basis(pseudospin) : PseudoSpinBasis
        Static helper turning a pseudospin, a list of pseudospins or a
        ready PseudoSpinBasis into a PseudoSpinBasis instance.
    __setup_numerical_matrices() : array of complex128, list of array of complex128
        Construct numerical matrices of the Hamiltonian and the componentes of the
        Cartesian components of the magnetic moment operator that will be used in the
        construction of the pseudospin operators. Return the matrices.
    __construct_pseudospin_operator_matrices()
        Construct matrix representations of the pseudospin Hamiltonian
        and magnetic moment operators in the basis given as a class
        argument and store them as attributes.
   __correct_phases()
        Correct the arbitrary relative phases of the pseudospin basis states
        left over from the numerical diagonalization of mu_z, so that the
        elements of mu_x between basis states differing in the projection of
        exactly one spin site by one step are real and positive and the basis
        behaves as a proper product |S,M> basis under time reversal.
    __check_time_reversal() : boolean, str
        Check whether the pseudospin operators have the correct
        properties under the operation of time reversal. Retun a boolean
        value of whether all of the operators (Hamiltonian and magnetic
        moment) pass the test and a string containing a human-readable
        summary of the results.
    """
    
    def __calculate_rotation(self):
        """Calculate g-tensor of the ground Kramers/Ising/pseudo doublet
        and store the transformation into its principal axis system
        as an attribute. The parity of the basis dimension determines
        whether the ground doublet is a Kramers doublet or a non-Kramers
        (quasi-)doublet, and this is passed on to the doublet explicitly.
        """
        ground_kd = properties.PseudoSpinDoublet\
                              .from_general_operator_matrix((0,1),self.hamiltonian,self.magnetic_moment,
                                                            self.units,
                                                            kramers=self.kramers_system,
                                                            print_output=self.print_output)

        if self.print_output:
            print("    Ground doublet Zeeman g-tensor:\n")
            print(ground_kd)

        # The rotation that is applied to the magnetic moment operator is
        # the inverse of the rotation that is applied to the g tensor. The
        # signs of the (directionless) principal axes returned by the
        # diagonalization are arbitrary; one axis is flipped when necessary
        # so that the rotation is always proper (determinant +1).
        eigenvectors = np.array(ground_kd.g_tensor.eigenvectors, dtype=np.float64)
        if la.det(eigenvectors) < 0.0:
            eigenvectors[:,0] = -eigenvectors[:,0]

        self.R = la.inv(eigenvectors)


    def __calculate_reorder_matrix(self):
        """Calculate and return an orthogonal permutation matrix based on the
        reorder_list attribute that can be used to reorder the states.
        """
        P = np.zeros((self.n_basis,self.n_basis), dtype=np.float64)

        for i in range(0,self.n_basis):
            j = self.reorder_list[i]
            P[i][j] = 1.0
            P[j][i] = 1.0

        if not np.allclose(np.dot(P,P.T),np.identity(self.n_basis, dtype=np.float64)):
            print("ERROR in AbInitioElectronicPseudoSpinSystem.")
            print("Error: Reorder matrix is not orthogonal.")
            print("Error termination.")
            sys.exit(1)

        return P


    def __setup_numerical_matrices(self):
        """Construct numerical matrices of the Hamiltonian and the componentes of the
        Cartesian components of the magnetic moment operator that will be used in the
        construction of the pseudospin operators. Return the matrices.

        The operators are first projected onto a space with the same dimension as the
        pseudospin operators, then transformed to the correct axis frame.
        """
        # The magnetic moment operator matrices are already written in
        # the basis of the Hamiltonian eigenstates. The Hamiltonian
        # needs to be diagonalized first. Then project the operators
        # onto the space spanned by the n_basis lowest eigenstates.
        initial_hamiltonian = np.zeros((self.n_basis,self.n_basis), dtype=np.complex128)
        for i in range(0,self.n_basis):
            initial_hamiltonian[i][i] = complex(self.hamiltonian.eigenvalues[i],0.0)
        
        initial_magnetic_moment = []
        for i in range(0,3):
            tmp_magnetic_moment = deepcopy(self.magnetic_moment.operator_list[i].matrix[:self.n_basis,:self.n_basis])
            initial_magnetic_moment.append(tmp_magnetic_moment)

        # Transform the magnetic moment and spin to the correct quantization
        # axis frame.
        initial_magnetic_moment = fu.matrix_utils.\
            rotate_vector_operator_matrix(initial_magnetic_moment,self.R)

        return initial_hamiltonian, initial_magnetic_moment


    def __construct_pseudospin_operator_matrices(self):
        """Construct matrix representations of the pseudospin Hamiltonian
        and magnetic moment operators in the basis given as a class
        argument and store them as attributes.

        The theory is oulined in the papers:

            L. F. Chibotaru and L. Ungur. J. Chem. Phys. 2012, 137, 064112.
            L. Ungur and L. F. Chibotaru. Chem. Eur. J. 2017, 23, 3708--3718.

        """
        initial_hamiltonian, initial_magnetic_moment = self.__setup_numerical_matrices()

        # Find the transformation into the pseudospin basis.
        _, C = fu.matrix_utils.external_diagonalize_complex_matrix(initial_magnetic_moment[2])

        P = self.__calculate_reorder_matrix()
        C = np.dot(C,P)

        # Transform
        self.pseudospin_magnetic_moment_matrix = []
        for i in range(0,4):
            if i == 3:
                self.pseudospin_hamiltonian_matrix = fu.matrix_utils.basis_transformation(C,initial_hamiltonian,0)
            else:
                self.pseudospin_magnetic_moment_matrix.append(fu.matrix_utils.basis_transformation(C,initial_magnetic_moment[i],0))


    def __correct_phases(self):
        """Correct the arbitrary relative phases of the pseudospin basis states.

        The pseudospin basis states are obtained by numerically diagonalizing
        the mu_z operator, which leaves each state with an arbitrary complex
        phase. The phases are fixed so that the basis behaves as a proper
        product |S,M> basis under time reversal, i.e. so that the corrected
        operators A satisfy

            U conj(A) U^dagger = s_A A,

        where U is the unitary part of the time-reversal operator of the
        basis and s_A is the time-reversal parity of the operator (+1 for
        the Hamiltonian, -1 for the magnetic moment). Without this
        correction the ITO decomposition of the Hamiltonian contains
        spurious time-reversal-odd contributions and the weights of the
        time-reversal-even contributions are distributed incorrectly.

        The basis states are connected by a spanning tree following the
        lexicographic product-basis ordering of PseudoSpinBasis: the parent
        of each state is the state in which the projection of the last site
        with a non-minimal projection is lowered by one step. The elements
        of mu_x along the tree edges are dominated by the physical ladder
        contributions of the magnetic moment of a single site and are
        therefore non-zero. The phases are then fixed in two steps:

          1. The phases of the states in the first half of the basis, up to
             and including the state that crosses the middle of the basis,
             are fixed by a Condon--Shortley-like convention requiring the
             elements of mu_x along the tree edges to be real and positive.
             For a single spin site this is the chain <i|mu_x|i+1> over
             adjacent basis states. This fixes the free orientation of the
             x and y axes and the remaining time-reversal gauge.
          2. Time reversal connects each basis state i to the conjugate
             state with all projections reversed, and constrains only the
             phase sums psi_i = phi_i + phi_conj(i) of the conjugate pairs.
             Specifically, requiring the standard behavior above for
             A = mu_x on an element (k,l) gives

                 psi_k - psi_l = arg(mu_x_kl) + arg(mu_x_conj(k)conj(l))
                                 + arg(-u_conj(k) u_conj(l)),

             where u_i is the diagonal sign of U over the conjugate pairs.
             These differences are accumulated along the tree edges of the
             first half of the basis, anchored to the pair whose phases are
             both fixed by step 1, and the phases of the second half of the
             basis are obtained as phi_conj(i) = psi_i - phi_i.

        The phases of the second half cannot be fixed by the real-positive
        convention of step 1: when the spin sites are coupled, the basis
        states are not products of single-site states and the elements of
        mu_x between them carry intrinsic complex phases, so forcing all of
        them real and positive is inconsistent with time reversal. Step 1
        applied to the full basis is correct only for a single spin site,
        where mu_x is proportional to the ladder operators of a single
        |S,M> multiplet by the Wigner--Eckart theorem; for a single site
        both schemes coincide.
        """
        mu_x = self.pseudospin_magnetic_moment_matrix[0]

        n_basis = self.n_basis
        n_sites = self.basis.n_sites

        # The stride of a site is the difference between the indices of two
        # basis states that differ only in the projection of that site by
        # one step in the lexicographic ordering of PseudoSpinBasis.
        stride_list = []
        for a in range(0,n_sites):
            stride = 1
            for b in range(a+1,n_sites):
                stride *= self.basis.n_states_per_site[b]
            stride_list.append(stride)

        def tree_parent(i):
            local_index_list = self.basis.local_state_index_list[i]
            for a in range(n_sites-1,-1,-1):
                if local_index_list[a] > 0:
                    return i - stride_list[a]

        # The conjugate of basis state i, with all projections reversed, is
        # the state n_basis-1-i in the lexicographic ordering. The signs of
        # the unitary part of the time-reversal operator over the conjugate
        # pairs are collected in u.
        U = self.basis.unitary_part_of_time_reversal_operator()
        u = np.zeros(n_basis, dtype=np.float64)
        for i in range(0,n_basis):
            u[i] = U[n_basis-1-i][i]

        phi = np.zeros(n_basis, dtype=np.float64)

        # Step 1: Condon--Shortley-like convention for the first half of the
        # basis and the state crossing the middle of the basis.
        for i in range(1,n_basis//2 + 1):
            j = tree_parent(i)
            phi[i] = phi[j] - cphase(mu_x[j][i])

        # Step 2: accumulate the time-reversal constraints on the pair phase
        # sums psi along the tree edges of the first half of the basis. The
        # accumulated differences delta are anchored to the pair
        # (anchor, n_basis-1-anchor), which is the only pair whose phases
        # are both fixed by step 1. (For an odd basis dimension the anchor
        # is the self-conjugate middle state.)
        anchor = (n_basis-1)//2

        delta = np.zeros(anchor+1, dtype=np.float64)
        for r in range(1,anchor+1):
            p  = tree_parent(r)
            rc = n_basis-1-r
            pc = n_basis-1-p

            delta[r] = delta[p] + cphase(-u[rc]*u[pc]*mu_x[r][p]*mu_x[rc][pc])

        psi_anchor = phi[anchor] + phi[n_basis-1-anchor]

        for i in range(n_basis//2 + 1,n_basis):
            r = n_basis-1-i
            phi[i] = psi_anchor + delta[r] - delta[anchor] - phi[r]

        D = np.diag(np.exp(1j*phi))

        self.pseudospin_hamiltonian_matrix = \
            np.dot(D.conj().T,np.dot(self.pseudospin_hamiltonian_matrix,D))
        for i in range(0,3):
            self.pseudospin_magnetic_moment_matrix[i] = \
                np.dot(D.conj().T,np.dot(self.pseudospin_magnetic_moment_matrix[i],D))


    def hamiltonian_tensor(self):
        """Construct and return an irreducible tensor representation of
        the Hamiltonian. The tensor carries the coordinate frame label of
        the system (see the tensor_frame attribute).
        """
        hamiltonian_tensor = tensors.IwaharaChibotaruSphericalTensor\
                      .from_matrix_representation(self.pseudospin_hamiltonian_matrix,self.basis)
        hamiltonian_tensor.frame = self.tensor_frame

        return hamiltonian_tensor

    
    def magnetic_moment_tensor(self):
        """Construct and return a mixed Cartesian--spherical tensor representation
        of the magnetic moment operator containing the three Cartesian components.
        """
        vector_operator = pseudospin_operators\
                          .GeneralVectorOperatorMatrix(self.pseudospin_magnetic_moment_matrix)

        moment_tensor = tensors.MixedCartesianIwaharaChibotaruSphericalTensor\
                      .from_general_vector_operator_matrix(vector_operator,self.basis)
        moment_tensor.frame = self.tensor_frame
        for component in moment_tensor.component_list:
            component.frame = self.tensor_frame

        return moment_tensor


    def hamiltonian_operator(self):
        """Construct and return the pseudospin Hamiltonian of the system as
        a PseudoSpinOperator instance, with the operator matrix stored and
        the eigenvalues translated so that the lowest one is zero. The
        operator is expressed in the coordinate frame of the tensors of
        this system (see the tensor_frame attribute).
        """
        return pseudospin_operators.PseudoSpinOperator(self.basis,
                                                       [self.hamiltonian_tensor()],
                                                       self.units,
                                                       translate_eigenvalues=True,
                                                       store_operator_matrix=True)


    def magnetic_moment_operator(self):
        """Construct and return the pseudospin magnetic moment operator of
        the system as a PseudoSpinVectorOperator instance, with the
        operator matrices stored. The operator is expressed in the
        coordinate frame of the tensors of this system (see the
        tensor_frame attribute).
        """
        return pseudospin_operators.PseudoSpinVectorOperator(self.basis,
                                                             [self.magnetic_moment_tensor()],
                                                             self.units,
                                                             diagonalize_operator_matrix=False,
                                                             store_operator_matrix=True,
                                                             translate_eigenvalues=False)


    def static_transition_magnetic_moments(self, n_states=None):
        """Construct and return the StaticTransitionMagneticMoments
        instance of the system, evaluated from the pseudospin Hamiltonian
        and magnetic moment operators of this system.

        Optional arguments
        ------------------
        n_states : int
            The number of lowest eigenstates included. The default is
            None, in which case all pseudospin states are included.
        """
        if n_states is None:
            n_states = self.n_basis

        return properties.StaticTransitionMagneticMoments(self.magnetic_moment_operator(),
                                                          self.hamiltonian_operator(),
                                                          n_states,
                                                          self.units)


    def pseudospin_doublet(self, states):
        """Construct and return the PseudoSpinDoublet instance of the
        doublet spanned by the two pseudospin eigenstates whose indices are
        given in the states tuple.

        The coordinate frames are tracked: the doublet is constructed from
        the pseudospin operators of this system (expressed in the frame of
        the tensor_frame attribute) and receives the input_frame_rotation
        of this system, so that its g-tensor is reported in the input axis
        frame following the convention of the library. The Kramers
        classification of the system (the kramers_system attribute) is
        passed on to the doublet.

        Arguments
        ---------
        states : tuple of int
            The indices of the two pseudospin eigenstates spanning the
            doublet.
        """
        return properties.PseudoSpinDoublet\
                         .from_pseudospin_operator(states,
                                                   self.hamiltonian_operator(),
                                                   self.magnetic_moment_operator(),
                                                   self.units,
                                                   kramers=self.kramers_system,
                                                   rotation=self.input_frame_rotation)


    def pseudospin_doublet_table(self, doublets):
        """Construct and return a human-readable tabulation string of one
        or several pseudospin doublets of the system.

        Each tabulated doublet is constructed with the pseudospin_doublet
        method: the g-tensors and their principal magnetic axes are given
        in the INPUT axis frame (the frame of the ab initio data), which
        is also stated explicitly in the output. The energy of each
        doublet is the eigenvalue of the lower of its two states.

        Arguments
        ---------
        doublets : tuple of int or list of tuple of int
            Either a single tuple with the indices of the two pseudospin
            eigenstates spanning a doublet, or a list of such tuples, in
            which case all the listed doublets are tabulated.
        """
        if isinstance(doublets,tuple):
            doublet_list = [doublets]
        elif isinstance(doublets,list):
            doublet_list = doublets
        else:
            print("ERROR in AbInitioElectronExchangeSystem.")
            print("Error: The doublets argument must be a tuple of two state indices")
            print("       or a list of such tuples.")
            print("Error termination.")
            sys.exit(1)

        hamiltonian_operator     = self.hamiltonian_operator()
        magnetic_moment_operator = self.magnetic_moment_operator()

        tmp_str  = "    PSEUDOSPIN DOUBLETS\n\n"
        tmp_str += "    The g-tensors and their principal magnetic axes are given in the\n"
        tmp_str += "    input axis frame.\n\n"

        for states in doublet_list:
            if not len(states) == 2:
                print("ERROR in AbInitioElectronExchangeSystem.")
                print("Error: Each doublet must be given as a tuple of two state indices.")
                print("Error termination.")
                sys.exit(1)

            doublet = properties.PseudoSpinDoublet\
                                .from_pseudospin_operator(tuple(states),
                                                          hamiltonian_operator,
                                                          magnetic_moment_operator,
                                                          self.units,
                                                          kramers=self.kramers_system,
                                                          rotation=self.input_frame_rotation)

            tmp_str += "    DOUBLET ({0},{1}),   E = {2:12.4f} {3}\n\n"\
                       .format(states[0],states[1],
                               hamiltonian_operator.eigenvalues[states[0]],
                               self.units.energy_unit_str)
            tmp_str += str(doublet)
            tmp_str += "\n"

        return tmp_str

    
    def __check_time_reversal(self):
        """Check whether the pseudospin operators have the correct
        properties under the operation of time reversal. Retun a boolean
        value of whether all of the operators (Hamiltonian and magnetic
        moment) pass the test and a string containing a human-readable
        summary of the results.

        The comparisons use a tolerance relative to the largest matrix
        element of each operator, because the ab initio states themselves
        break time-reversal symmetry at a small but finite numerical level,
        which propagates into the pseudospin operators.
        """
        result_list = []
        U = self.basis.unitary_part_of_time_reversal_operator()

        for alpha in range(0,4):
            if alpha == 3:
                matrix = self.pseudospin_hamiltonian_matrix
            else:
                matrix = self.pseudospin_magnetic_moment_matrix[alpha]

            conjugate_matrix = np.conj(matrix)
            transformed_matrix = fu.matrix_utils.basis_transformation(U,matrix,0)

            tolerance = 1.0e-6*max(1.0,np.max(np.abs(matrix)))

            if np.allclose(transformed_matrix,conjugate_matrix,atol=tolerance):
                result_list.append(1)
            elif np.allclose(transformed_matrix,-1.0*conjugate_matrix,atol=tolerance):
                result_list.append(-1)
            else:
                result_list.append(0)

        tmp_str = "    Effect of time-reversal transformation on the pseudospin operators:\n\n"
        label_list = ['mu_x','mu_y','mu_z','H']
        correct_result_list = [-1,-1,-1,1]
        for i in range(0,4):
            tmp_str += "        Operator {0:>4}    result: {1:2}     correct result: {2:2}\n"\
                       .format(label_list[i],result_list[i],correct_result_list[i])
        tmp_str += "\n"

        if result_list == correct_result_list:
            return True, tmp_str
        else:
            return False, tmp_str

    
    def __init__(self,hamiltonian,magnetic_moment,basis,units,
                 reorder_list=None,
                 R=None,
                 print_output=False):
        """Construct the tensors and store them as attributes."""
        self.hamiltonian     = hamiltonian
        self.magnetic_moment = magnetic_moment
        self.basis           = basis
        self.units           = units
        self.print_output    = print_output

        self.n_basis = self.basis.n_basis
        self.n_full_basis = self.hamiltonian.n_basis

        if self.n_basis % 2 == 0:
            self.kramers_system = True
        else:
            self.kramers_system = False

        if not self.magnetic_moment.n_basis == self.n_full_basis:
            print("ERROR in AbInitioElectronicPseudoSpinSystem.")
            print("Error: Inconsistent dimensions in input matrices.")
            print("Error termination.")
            sys.exit(1)

        if R is None:
            self.__calculate_rotation()
            self.tensor_frame = 'principal magnetic axis frame'
        else:
            # The magnetic moment is an axial vector and is rotated with
            # the polar-vector rule, which is valid only for proper
            # rotations; an improper R would silently mirror the physics.
            R = np.array(R, dtype=np.float64)
            if not np.allclose(np.dot(R,R.T),np.identity(3),atol=1.0e-6) \
               or la.det(R) < 0.0:
                print("ERROR in AbInitioElectronExchangeSystem.")
                print("Error: The rotation matrix R must be a proper rotation")
                print("       (orthogonal with determinant +1). The magnetic moment")
                print("       is an axial vector, and rotating it with an improper")
                print("       matrix would silently mirror the physics.")
                print("Error termination.")
                sys.exit(1)
            self.R = R
            self.tensor_frame = 'user-defined axis frame'

        # The rotation from the input coordinate frame (the frame of the
        # operator matrices given to the constructor) to the frame of the
        # pseudospin operators, stored as a Rotation instance. Should the
        # frame transformation be improper (which can only happen when an
        # improper R was passed in explicitly), the overall sign is flipped
        # for this record, which leaves the directionless magnetic axes
        # unaffected. The from_average_aniso_data class method overrides
        # this record with the full rotation from the input frame of the
        # first datafile.
        rotation_record = np.array(self.R, dtype=np.float64)
        if la.det(rotation_record) < 0.0:
            rotation_record = -rotation_record
        self.input_frame_rotation = tensors.Rotation(rotation_record)

        if reorder_list is None:
            self.reorder_list = list(range(self.n_basis-1,-1,-1))
        else:
            self.reorder_list = reorder_list

        self.__construct_pseudospin_operator_matrices()
        self.__correct_phases()
        result, result_str = self.__check_time_reversal()

        if not result:
            print("WARNING in AbInitioElectronExchangeSystem.")
            print("Warning: Pseudospin operators have incorrect behavior under time reversal.")
            print(result_str)


    @staticmethod
    def __construct_basis(pseudospin):
        """Turn the pseudospin argument of the from_*_aniso_data class
        methods into a PseudoSpinBasis instance. The argument can be a
        ready PseudoSpinBasis (returned as is), a single integer pseudospin
        (a single-site system) or a list of integer pseudospins (a
        multi-site system). All pseudospins are given as multiples of two,
        as everywhere in the library. Invalid input produces a fatal error.
        """
        if isinstance(pseudospin,pseudospin_operators.PseudoSpinBasis):
            return pseudospin
        elif isinstance(pseudospin,(int,np.integer)):
            return pseudospin_operators.PseudoSpinBasis([int(pseudospin)])
        elif isinstance(pseudospin,(list,tuple)):
            return pseudospin_operators.PseudoSpinBasis(list(pseudospin))
        else:
            print("ERROR in AbInitioElectronExchangeSystem.")
            print("Error: The pseudospin argument must be a PseudoSpinBasis, a single")
            print("       integer pseudospin or a list of integer pseudospins (all")
            print("       pseudospins given as multiples of two).")
            print("Error termination.")
            sys.exit(1)


    @classmethod
    def from_aniso_data(cls, filename, pseudospin, units,
                        include_bohr_magneton=True,
                        reorder_list=None,
                        R=None,
                        print_output=False):
        """Construct the system from a single SINGLE_ANISO datafile (a
        .aniso type file containing the operator matrices, not the
        SINGLE_ANISO output file).

        The datafile is read with OrcaAnisoFile: the Hamiltonian (the
        spin-orbit energies) and the matrices of the magnetic moment in the
        basis of the spin-orbit eigenstates are extracted and passed to the
        class constructor together with the pseudospin basis.

        Arguments
        ---------
        filename : str
            Name of the SINGLE_ANISO datafile.
        pseudospin : PseudoSpinBasis or int or list of int
            The pseudospin structure of the system: either a ready
            PseudoSpinBasis, a single pseudospin (a single-site system) or
            a list of pseudospins (a multi-site system), the pseudospins
            given as multiples of two. When pseudospins are given, the
            complete basis is constructed from them.
        units : EnergyUnitSystem
            The unit system.

        Optional arguments
        ------------------
        include_bohr_magneton : boolean
            Whether the magnetic moment operators read from the datafile
            are multiplied by the Bohr magneton of the unit system (so that
            they are in units of energy per tesla, which is the convention
            used by the property classes of the library). Default is True.
        reorder_list : list of int
            Passed on to the class constructor; see the class documentation.
        R : array of real64
            Passed on to the class constructor; see the class documentation.
        print_output : boolean
            Whether to print output. The default is False.
        """
        from ouluspin.qc import orca

        basis = cls.__construct_basis(pseudospin)

        calculation = orca.OrcaAnisoFile(filename,units)

        hamiltonian = pseudospin_operators\
                      .GeneralOperatorMatrix(calculation.hamiltonian(),
                                             diagonalize_operator_matrix=True,
                                             translate_eigenvalues=True)
        magnetic_moment = pseudospin_operators\
                          .GeneralVectorOperatorMatrix(calculation.magnetic_moment(
                                                           include_bohr_magneton=include_bohr_magneton),
                                                       diagonalize_operator_matrix=True,
                                                       translate_eigenvalues=False)

        return cls(hamiltonian,magnetic_moment,basis,units,
                   reorder_list=reorder_list,
                   R=R,
                   print_output=print_output)


    @classmethod
    def from_average_aniso_data(cls, filename_1, filename_2, pseudospin, lande_g_factor, units,
                                reorder_list=None,
                                print_output=False):
        """Construct the system by averaging the crystal fields of two
        SINGLE_ANISO datafiles (.aniso type files containing the operator
        matrices, not the SINGLE_ANISO output files), for example those of
        a calculation with one electron added to and one electron removed
        from a given configuration.

        Each datafile is first projected onto its own pseudospin system
        (with the quantization axis along the main magnetic axis of its
        ground doublet). The Hamiltonian tensor of the second system is
        then rotated into the principal magnetic frame of the first system,
        so that both crystal fields are expressed in the same coordinate
        frame, and the average Hamiltonian is formed as the mean of the two
        tensors. The magnetic moment operators of the averaged system are
        constructed from the Lande g-factor as mu_a = -g_J * mu_B * J_a
        (isotropic, and therefore independent of the coordinate frame).

        Finally, the averaged Hamiltonian is brought into its own principal
        magnetic frame: the g-tensor of the ground Kramers/quasi doublet of
        the averaged system is evaluated with PseudoSpinDoublet, the
        averaged Hamiltonian tensor is rotated into the principal axis
        frame of this g-tensor, and the operator matrices passed to the
        class constructor are built from the rotated tensor. The returned
        system is therefore quantized along the main magnetic axis of its
        own ground doublet, exactly as a system constructed directly from a
        single datafile.

        Arguments
        ---------
        filename_1 : str
            Name of the first SINGLE_ANISO datafile. The principal magnetic
            frame of this calculation is used as the common frame in which
            the two crystal fields are averaged.
        filename_2 : str
            Name of the second SINGLE_ANISO datafile.
        pseudospin : PseudoSpinBasis or int or list of int
            The pseudospin structure of the system; see from_aniso_data.
            Only single-site systems are supported by this class method.
        lande_g_factor : float
            The Lande g-factor g_J used in the construction of the magnetic
            moment operators of the averaged system.
        units : EnergyUnitSystem
            The unit system.

        Optional arguments
        ------------------
        reorder_list : list of int
            Passed on to the class constructor; see the class documentation.
        print_output : boolean
            Whether to print output. The default is False.
        """
        basis = cls.__construct_basis(pseudospin)

        if not basis.n_sites == 1:
            print("ERROR in AbInitioElectronExchangeSystem.")
            print("Error: from_average_aniso_data only supports single-site systems,")
            print("       as a single Lande g-factor defines the magnetic moment of")
            print("       one spin site only.")
            print("Error termination.")
            sys.exit(1)

        # Project each datafile onto its own pseudospin system (each in the
        # principal magnetic frame of its own ground doublet).
        system_1 = cls.from_aniso_data(filename_1,basis,units,print_output=print_output)
        system_2 = cls.from_aniso_data(filename_2,basis,units,print_output=print_output)

        # Rotate the Hamiltonian tensor of the second system into the
        # principal magnetic frame of the first system. The stored frame
        # rotations relate the tensor components of frame i to the input
        # frame by v_i = R_i * v, so the components transform from frame 2
        # to frame 1 with the relative rotation R_1 * R_2^T. The rotation
        # is made proper (determinant +1) by an overall sign change when
        # necessary, which is allowed because the Hamiltonian contains only
        # parity-even tensor components, on which the inversion acts
        # trivially.
        relative_rotation = np.dot(system_1.R,system_2.R.T)
        if la.det(relative_rotation) < 0.0:
            relative_rotation = -relative_rotation

        hamiltonian_tensor_1 = system_1.hamiltonian_tensor()
        hamiltonian_tensor_2 = system_2.hamiltonian_tensor()
        hamiltonian_tensor_2.rotate(tensors.Rotation(relative_rotation))

        if print_output:
            rotation_cosine = 0.5*(np.trace(relative_rotation) - 1.0)
            rotation_angle  = np.degrees(np.arccos(min(1.0,max(-1.0,rotation_cosine))))
            print("    The crystal field of the second system was rotated by"
                  " {0:8.3f} degrees".format(rotation_angle))
            print("    into the principal magnetic frame of the first system.")
            print()

        # The average Hamiltonian in the common frame.
        average_tensor = 0.5*(hamiltonian_tensor_1 + hamiltonian_tensor_2)

        # The magnetic moment from the Lande g-factor.
        moment_tensor = tensors.MixedCartesianIwaharaChibotaruSphericalTensor\
                               .from_one_site_isotropic_operator(-lande_g_factor*units.mu_B,
                                                                 basis.pseudospin_list[0])
        average_magnetic_moment = pseudospin_operators\
                                  .PseudoSpinVectorOperator(basis,[moment_tensor],units,
                                                            diagonalize_operator_matrix=False,
                                                            store_operator_matrix=True,
                                                            translate_eigenvalues=False)

        # Determine the principal magnetic frame of the averaged system from
        # the g-tensor of its ground Kramers/quasi doublet and rotate the
        # averaged Hamiltonian tensor into that frame.
        average_hamiltonian = pseudospin_operators\
                              .PseudoSpinOperator(basis,[average_tensor],units,
                                                  translate_eigenvalues=True,
                                                  store_operator_matrix=True)

        ground_doublet = properties.PseudoSpinDoublet\
                                   .from_pseudospin_operator((0,1),
                                                             average_hamiltonian,
                                                             average_magnetic_moment,
                                                             units,
                                                             kramers=(basis.n_basis % 2 == 0))

        principal_rotation = la.inv(ground_doublet.g_tensor.eigenvectors)
        if la.det(principal_rotation) < 0.0:
            principal_rotation = -principal_rotation

        average_tensor.rotate(tensors.Rotation(principal_rotation))

        # Build the operator matrices from the rotated tensor in the same
        # form in which they appear in a SINGLE_ANISO datafile: the
        # Hamiltonian diagonal in its eigenbasis and the magnetic moment
        # matrices written in that eigenbasis. As the tensors are already
        # expressed in the principal magnetic frame, the class constructor
        # is called with the identity as the frame rotation.
        rotated_hamiltonian = pseudospin_operators\
                              .PseudoSpinOperator(basis,[average_tensor],units,
                                                  translate_eigenvalues=True,
                                                  store_operator_matrix=True)

        hamiltonian_matrix = np.diag(np.array(rotated_hamiltonian.eigenvalues,
                                              dtype=np.complex128))
        moment_matrix_list = []
        for alpha in range(0,3):
            moment_matrix_list.append(fu.matrix_utils.basis_transformation(
                rotated_hamiltonian.eigenvectors,
                average_magnetic_moment.operator_list[alpha].matrix,0))

        hamiltonian = pseudospin_operators\
                      .GeneralOperatorMatrix(hamiltonian_matrix,
                                             diagonalize_operator_matrix=True,
                                             translate_eigenvalues=True)
        magnetic_moment = pseudospin_operators\
                          .GeneralVectorOperatorMatrix(moment_matrix_list,
                                                       diagonalize_operator_matrix=True,
                                                       translate_eigenvalues=False)

        system = cls(hamiltonian,magnetic_moment,basis,units,
                     reorder_list=reorder_list,
                     R=np.identity(3),
                     print_output=print_output)

        # The operators were rotated into the principal magnetic frame of
        # the averaged ground doublet before the construction, so the
        # tensors of the returned system are in the principal magnetic
        # axis frame even though the constructor was called with the
        # identity rotation.
        system.tensor_frame = 'principal magnetic axis frame'

        # The full rotation from the input coordinate frame (the frame of
        # the operator matrices of the first datafile) to the frame of the
        # averaged system: first into the principal magnetic frame of the
        # first system, then into the principal magnetic frame of the
        # ground doublet of the averaged system.
        total_rotation = np.dot(principal_rotation,system_1.R)
        if la.det(total_rotation) < 0.0:
            total_rotation = -total_rotation
        system.input_frame_rotation = tensors.Rotation(total_rotation)

        return system


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class and run a set of tests on the class constructor
        and the class methods. Return True if all tests passed and False
        otherwise.

        The tests construct a synthetic "ab initio" system from explicit
        operator matrices of an S = 3/2 spin with a zero-field splitting
        D*S_z^2 + E*(S_x^2 - S_y^2) and an isotropic magnetic moment
        mu = -g*mu_B*S, and check that the extracted pseudospin operators
        and tensors reproduce the input. The phases of the eigenstates of
        the synthetic Hamiltonian are scrambled with random phase factors
        to emulate the arbitrary phases produced by numerical
        diagonalization in a real ab initio calculation; the class must
        correct these phases so that the extracted Hamiltonian is even
        under time reversal and its ITO decomposition contains only
        even-rank terms.

        The same checks are repeated for a two-site system in which the
        S = 3/2 ion is exchange-coupled to an isotropic S = 1/2 radical,
        which tests the time-reversal-based phase correction of a coupled
        product basis, and for a non-Kramers S = 1 ion, whose odd basis
        dimension exercises the phase correction of a self-conjugate
        middle basis state.

        Optional arguments
        ------------------
        print_output : boolean
            Whether to print the outcome of each test. Default is True.
        """
        from ouluspin import units
        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('AbInitioElectronExchangeSystem',
                                                       test_name,condition,print_output))

        tmp_units  = units.EnergyUnitSystem('wavenumber')
        pseudospin = 3   # S = 3/2
        D, E, g    = 5.0, 1.0, 2.0

        spin_matrix_list = debug_output.spin_matrices(pseudospin)

        # The Hamiltonian D*S_z^2 + E*(S_x^2 - S_y^2) gives two doublets
        # separated by 2*sqrt(D^2 + 3*E^2).
        hamiltonian_matrix = D*np.dot(spin_matrix_list[2],spin_matrix_list[2]) \
                             + E*(np.dot(spin_matrix_list[0],spin_matrix_list[0])
                                  - np.dot(spin_matrix_list[1],spin_matrix_list[1]))
        hamiltonian = pseudospin_operators\
                      .GeneralOperatorMatrix(hamiltonian_matrix,
                                             diagonalize_operator_matrix=True,
                                             translate_eigenvalues=True)

        # The magnetic moment operators are transformed to the eigenbasis of
        # the Hamiltonian, as would be the case for real ab initio data. The
        # phases of the eigenvectors are scrambled with random phase factors
        # to emulate the arbitrary phases of a numerical diagonalization.
        rng = np.random.default_rng(1)
        C = np.dot(hamiltonian.eigenvectors,
                   np.diag(np.exp(2.0j*np.pi*rng.random(pseudospin+1))))
        moment_matrix_list = []
        for spin_matrix in spin_matrix_list:
            moment_matrix = -g*tmp_units.mu_B*spin_matrix
            moment_matrix_list.append(np.dot(C.conj().T,np.dot(moment_matrix,C)))
        magnetic_moment = pseudospin_operators\
                          .GeneralVectorOperatorMatrix(moment_matrix_list)

        basis = pseudospin_operators.PseudoSpinBasis([pseudospin])

        # The quantization axis is passed explicitly so that the test result
        # does not depend on the ordering of degenerate principal axes.
        system = cls(hamiltonian,magnetic_moment,basis,tmp_units,
                     R=np.identity(3))

        gap = 2.0*np.sqrt(D**2 + 3.0*E**2)
        check('Kramers system recognized', system.kramers_system)
        check('explicit R gives the user-defined frame label',
              system.tensor_frame == 'user-defined axis frame'
              and system.hamiltonian_tensor().frame == 'user-defined axis frame')
        check('pseudospin Hamiltonian eigenvalues',
              np.allclose(np.linalg.eigvalsh(system.pseudospin_hamiltonian_matrix),
                          [0.0,0.0,gap,gap]))

        # The corrected pseudospin basis must behave as a proper |S,M> basis
        # under time reversal: the Hamiltonian is even and the ITO
        # decomposition contains only even-rank terms.
        U = basis.unitary_part_of_time_reversal_operator()
        H_ps = system.pseudospin_hamiltonian_matrix
        check('pseudospin Hamiltonian is even under time reversal',
              np.allclose(np.dot(U,np.dot(H_ps.conj(),U.T)),H_ps))

        tmp_tensor = system.hamiltonian_tensor()
        odd_rank_maximum = max(abs(parameter)
                               for rank,parameter in zip(tmp_tensor.rank_list,
                                                         tmp_tensor.parameter_list)
                               if (rank[0]//2) % 2 == 1)
        check('no odd ranks in the Hamiltonian tensor',
              odd_rank_maximum < 1.0e-8)

        # In the corrected pseudospin basis the Hamiltonian matrix must
        # coincide with the input Hamiltonian up to the constant shift of
        # the eigenvalues. (The phase convention corresponds at most to a
        # rotation about z by pi, under which both the D and E terms are
        # invariant.)
        shift = np.trace(H_ps).real/(pseudospin + 1) \
                - np.trace(hamiltonian_matrix).real/(pseudospin + 1)
        check('pseudospin Hamiltonian matrix reproduces the input Hamiltonian',
              np.allclose(H_ps - shift*np.identity(pseudospin + 1),
                          hamiltonian_matrix))

        # The pseudospin Hamiltonian must correspond to the axial ZFS: the
        # (2,0) ITO parameter of the traceless part is 0.5*(3S^2-S(S+1))*s_zz
        # with s_zz = 2*D/3, i.e. D for S = 3/2.
        hamiltonian_tensor = system.hamiltonian_tensor()
        hamiltonian_tensor.purge_ranks()
        check('axial ZFS parameter recovered',
              ([4,0] in hamiltonian_tensor.rank_list) and
              abs(hamiltonian_tensor.parameter_list[hamiltonian_tensor
                                                    .rank_list.index([4,0])] - D) < 1.0e-8)

        # The ITO decomposition must reproduce the pseudospin operator matrix.
        reconstructed = debug_output\
                        .operator_matrix_from_tensor(system.hamiltonian_tensor(),
                                                     [pseudospin])
        check('Hamiltonian tensor reproduces the operator matrix',
              np.allclose(reconstructed,system.pseudospin_hamiltonian_matrix))

        # The magnetic moment tensor is returned as a mixed Cartesian--spherical
        # tensor. The z component: mu_z = X*O_10 with X = -g*mu_B*S.
        moment_tensor = system.magnetic_moment_tensor()
        check('magnetic moment tensor is a mixed Cartesian--spherical tensor',
              isinstance(moment_tensor,
                         tensors.MixedCartesianIwaharaChibotaruSphericalTensor))
        moment_tensor_z = moment_tensor.component('z')
        moment_tensor_z.purge_ranks()
        S = pseudospin/2.0
        check('magnetic moment tensor of the z component',
              ([2,0] in moment_tensor_z.rank_list) and
              abs(moment_tensor_z.parameter_list[moment_tensor_z
                                                 .rank_list.index([2,0])]
                  - (-g*tmp_units.mu_B*S)) < 1.0e-8)

        # The full mixed tensor of the isotropic moment mu = -g*mu_B*S must
        # equal the tensor of the isotropic operator with X = -g*mu_B. The
        # phase convention of the pseudospin basis is fixed only up to a
        # rotation of the pseudospin about z by pi, which flips the sign of
        # the spherical parts of the x and y components while the laboratory
        # Cartesian index is unaffected; the corresponding reference with
        # the Cartesian matrix -g*mu_B*diag(-1,-1,1) is also accepted.
        reference_moment = tensors.MixedCartesianIwaharaChibotaruSphericalTensor\
                                  .from_one_site_isotropic_operator(-g*tmp_units.mu_B,
                                                                    pseudospin)
        rotated_reference = tensors.MixedCartesianIwaharaChibotaruSphericalTensor\
                                   .from_one_site_cartesian_tensor(-g*tmp_units.mu_B
                                                                   *np.diag([-1.0,-1.0,1.0]),
                                                                   pseudospin)
        check('mixed magnetic moment tensor equals the isotropic reference',
              (moment_tensor == reference_moment)
              or (moment_tensor == rotated_reference))

        # ------------------------------------------------------------------
        # Two-site test: the same S = 3/2 ion exchange-coupled to an
        # isotropic S = 1/2 radical. The pseudospin basis is now a coupled
        # product basis, and the phases of the pseudospin basis states can
        # no longer be fixed by requiring all elements of mu_x between
        # adjacent basis states to be real and positive; this tests the
        # time-reversal-based phase correction of the second half of the
        # basis. The g factor of the radical differs from the g factor of
        # the ion so that the spectrum of mu_z is non-degenerate.
        # ------------------------------------------------------------------
        pseudospin_list = [pseudospin,1]
        J_exchange      = 2.0
        g_radical       = 1.0

        radical_matrix_list = debug_output.spin_matrices(1)

        ion_identity     = np.identity(pseudospin+1, dtype=np.complex128)
        radical_identity = np.identity(2, dtype=np.complex128)

        two_site_hamiltonian_matrix = np.kron(hamiltonian_matrix,radical_identity)
        for a in range(0,3):
            two_site_hamiltonian_matrix += J_exchange*np.kron(spin_matrix_list[a],
                                                              radical_matrix_list[a])
        two_site_hamiltonian = pseudospin_operators\
                               .GeneralOperatorMatrix(two_site_hamiltonian_matrix,
                                                      diagonalize_operator_matrix=True,
                                                      translate_eigenvalues=True)

        C = np.dot(two_site_hamiltonian.eigenvectors,
                   np.diag(np.exp(2.0j*np.pi*rng.random(2*(pseudospin+1)))))
        moment_matrix_list = []
        for a in range(0,3):
            moment_matrix = -tmp_units.mu_B*(g*np.kron(spin_matrix_list[a],radical_identity)
                                             + g_radical*np.kron(ion_identity,radical_matrix_list[a]))
            moment_matrix_list.append(np.dot(C.conj().T,np.dot(moment_matrix,C)))
        two_site_moment = pseudospin_operators\
                          .GeneralVectorOperatorMatrix(moment_matrix_list)

        two_site_basis = pseudospin_operators.PseudoSpinBasis(pseudospin_list)

        two_site_system = cls(two_site_hamiltonian,two_site_moment,two_site_basis,
                              tmp_units,R=np.identity(3))

        U = two_site_basis.unitary_part_of_time_reversal_operator()
        H_ps = two_site_system.pseudospin_hamiltonian_matrix
        check('two-site pseudospin Hamiltonian is even under time reversal',
              np.allclose(np.dot(U,np.dot(H_ps.conj(),U.T)),H_ps))

        # The time-reversal parity of a product of ITOs is (-1)^(k1+k2), so
        # the decomposition of the Hamiltonian may only contain terms with
        # an even sum of the ranks.
        two_site_tensor = two_site_system.hamiltonian_tensor()
        odd_rank_maximum = max(abs(parameter)
                               for rank,parameter in zip(two_site_tensor.rank_list,
                                                         two_site_tensor.parameter_list)
                               if ((rank[0]+rank[2])//2) % 2 == 1)
        check('no time-reversal-odd ranks in the two-site Hamiltonian tensor',
              odd_rank_maximum < 1.0e-8)

        # In the corrected pseudospin basis the Hamiltonian matrix must
        # coincide with the input Hamiltonian up to the constant shift of
        # the eigenvalues. (The phase convention corresponds at most to
        # rotations of both sites about z by pi, under which the D and E
        # terms and the isotropic exchange are invariant.)
        n_two_site = two_site_basis.n_basis
        shift = (np.trace(H_ps).real
                 - np.trace(two_site_hamiltonian_matrix).real)/n_two_site
        check('two-site pseudospin Hamiltonian matrix reproduces the input Hamiltonian',
              np.allclose(H_ps - shift*np.identity(n_two_site),
                          two_site_hamiltonian_matrix))

        # The exchange part of the extracted tensor must match the ITO
        # parameters of the isotropic exchange operator.
        reference_tensor = tensors.IwaharaChibotaruSphericalTensor\
                                  .from_two_site_isotropic_operator(J_exchange,
                                                                    pseudospin_list[0],
                                                                    pseudospin_list[1])
        exchange_ok = True
        for rank,parameter in zip(reference_tensor.rank_list,
                                  reference_tensor.parameter_list):
            i = two_site_tensor.rank_list.index(rank)
            if abs(two_site_tensor.parameter_list[i] - parameter) > 1.0e-8:
                exchange_ok = False
        check('isotropic exchange parameters recovered',exchange_ok)

        # The ITO decomposition must reproduce the pseudospin operator matrix.
        reconstructed = debug_output.operator_matrix_from_tensor(two_site_tensor,
                                                                 pseudospin_list)
        check('two-site Hamiltonian tensor reproduces the operator matrix',
              np.allclose(reconstructed,H_ps))

        # ------------------------------------------------------------------
        # Non-Kramers test: an S = 1 ion with the ZFS D*S_z^2 +
        # E*(S_x^2 - S_y^2) and an isotropic magnetic moment. The basis
        # dimension is odd and the middle basis state is its own
        # time-reversal conjugate, which exercises the phase correction
        # of a self-conjugate anchor state.
        # ------------------------------------------------------------------
        nk_pseudospin       = 2   # S = 1
        nk_spin_matrix_list = debug_output.spin_matrices(nk_pseudospin)

        nk_hamiltonian_matrix = D*np.dot(nk_spin_matrix_list[2],nk_spin_matrix_list[2]) \
                                + E*(np.dot(nk_spin_matrix_list[0],nk_spin_matrix_list[0])
                                     - np.dot(nk_spin_matrix_list[1],nk_spin_matrix_list[1]))
        nk_hamiltonian = pseudospin_operators\
                         .GeneralOperatorMatrix(nk_hamiltonian_matrix,
                                                diagonalize_operator_matrix=True,
                                                translate_eigenvalues=True)

        C = np.dot(nk_hamiltonian.eigenvectors,
                   np.diag(np.exp(2.0j*np.pi*rng.random(nk_pseudospin+1))))
        moment_matrix_list = []
        for nk_spin_matrix in nk_spin_matrix_list:
            moment_matrix = -g*tmp_units.mu_B*nk_spin_matrix
            moment_matrix_list.append(np.dot(C.conj().T,np.dot(moment_matrix,C)))
        nk_moment = pseudospin_operators\
                    .GeneralVectorOperatorMatrix(moment_matrix_list)

        nk_basis = pseudospin_operators.PseudoSpinBasis([nk_pseudospin])

        nk_system = cls(nk_hamiltonian,nk_moment,nk_basis,tmp_units,
                        R=np.identity(3))

        check('non-Kramers system recognized', not nk_system.kramers_system)

        U = nk_basis.unitary_part_of_time_reversal_operator()
        H_ps = nk_system.pseudospin_hamiltonian_matrix
        check('non-Kramers pseudospin Hamiltonian is even under time reversal',
              np.allclose(np.dot(U,np.dot(H_ps.conj(),U.T)),H_ps))

        shift = (np.trace(H_ps).real
                 - np.trace(nk_hamiltonian_matrix).real)/(nk_pseudospin+1)
        check('non-Kramers pseudospin Hamiltonian matrix reproduces the input Hamiltonian',
              np.allclose(H_ps - shift*np.identity(nk_pseudospin+1),
                          nk_hamiltonian_matrix))

        # ------------------------------------------------------------------
        # The from_aniso_data and from_average_aniso_data class methods,
        # tested with synthetic SINGLE_ANISO datafiles of S = 3/2 systems
        # with H = D*S_z^2 + E*(S_x^2 - S_y^2), D < 0 (ground doublet close
        # to |+-3/2>) and mu = -g*mu_B*S. Datafile A is an axial reference
        # (E = 0), datafile C has twice its axial splitting, and datafiles
        # F (rhombic) and G (the same rhombic system with its coordinate
        # frame rotated by 90 degrees) test the frame alignment of the
        # averaging.
        # ------------------------------------------------------------------
        import os
        import tempfile

        hartree = tmp_units.convert_energy_unit(1.0,'hartree')
        Sx, Sy, Sz = debug_output.spin_matrices(3)

        def datafile_content(hamiltonian_matrix,moment_list):
            """Synthetic datafile of the system: the eigenvalues of the
            Hamiltonian and the moment and spin matrices written in the
            ascending-energy eigenbasis (moments in units of mu_B)."""
            eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
            content  = "$format\n2020\n"
            content += "$nss\n4\n"
            content += "$eso\n4\n"
            content += " ".join("{0:22.14e}".format(value/hartree)
                                for value in eigenvalues) + "\n"
            labels = ('x','y','z')
            for matrix_set, tag in ((moment_list,'magn'),
                                    ((Sx,Sy,Sz),'spin')):
                for a in range(0,3):
                    matrix = np.dot(eigenvectors.conj().T,
                                    np.dot(matrix_set[a],eigenvectors))
                    for part, values in (('r',matrix.real),('i',matrix.imag)):
                        content += "$" + tag + "_" + labels[a] + part + "\n4\n"
                        for i in range(0,4):
                            content += " ".join("{0:22.14e}".format(value)
                                                for value in values[i]) + "\n"
            return content

        # The datafiles store the magnetic moment in units of the Bohr
        # magneton (mu = -g*S for a pure spin with g = 2). The system of
        # datafile G is rotated by 90 degrees about y in the laboratory
        # frame, which rotates the moment components as (x,y,z) -> (z,y,-x)
        # while the Hamiltonian eigenvalues are unchanged. Its zero-field
        # splitting parameters differ from those of F so that the g-tensor
        # of the averaged ground doublet has no degenerate principal values
        # and its principal axes are well defined.
        D_f, E_f = -10.0, 1.0
        D_g, E_g = -14.0, 0.5
        hamiltonian_a = D_f*np.dot(Sz,Sz)
        hamiltonian_f = D_f*np.dot(Sz,Sz) + E_f*(np.dot(Sx,Sx) - np.dot(Sy,Sy))
        hamiltonian_g = D_g*np.dot(Sz,Sz) + E_g*(np.dot(Sx,Sx) - np.dot(Sy,Sy))
        moment_plain   = [-g*Sx,-g*Sy,-g*Sz]
        moment_rotated = [-g*Sz,-g*Sy,+g*Sx]

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename_a = os.path.join(tmp_dir,'a.anisofile')
            filename_c = os.path.join(tmp_dir,'c.anisofile')
            filename_f = os.path.join(tmp_dir,'f.anisofile')
            filename_g = os.path.join(tmp_dir,'g.anisofile')
            with open(filename_a,'w') as f:
                f.write(datafile_content(hamiltonian_a,moment_plain))
            with open(filename_c,'w') as f:
                f.write(datafile_content(2.0*hamiltonian_a,moment_plain))
            with open(filename_f,'w') as f:
                f.write(datafile_content(hamiltonian_f,moment_plain))
            with open(filename_g,'w') as f:
                f.write(datafile_content(hamiltonian_g,moment_rotated))

            # from_aniso_data with the three forms of the pseudospin argument.
            system_int   = cls.from_aniso_data(filename_a,3,tmp_units)
            system_list  = cls.from_aniso_data(filename_a,[3],tmp_units)
            system_basis = cls.from_aniso_data(filename_a,
                                               pseudospin_operators.PseudoSpinBasis([3]),
                                               tmp_units)
            check('from_aniso_data accepts an integer pseudospin',
                  np.allclose(np.linalg.eigvalsh(system_int.pseudospin_hamiltonian_matrix),
                              [0.0,0.0,2.0*abs(D_f),2.0*abs(D_f)]))
            check('from_aniso_data accepts a pseudospin list',
                  np.allclose(system_list.pseudospin_hamiltonian_matrix,
                              system_int.pseudospin_hamiltonian_matrix))
            check('from_aniso_data accepts a basis',
                  np.allclose(system_basis.pseudospin_hamiltonian_matrix,
                              system_int.pseudospin_hamiltonian_matrix))
            check('from_aniso_data tensors carry the principal frame label',
                  system_int.tensor_frame == 'principal magnetic axis frame'
                  and system_int.hamiltonian_tensor().frame
                      == 'principal magnetic axis frame')

            # The convenience constructors of the pseudospin operators and
            # the property classes, tested on the axial system A (ground
            # doublet |+-3/2>, gap 2|D_f|, mu = -g*mu_B*S).
            hamiltonian_operator = system_int.hamiltonian_operator()
            check('hamiltonian_operator eigenvalues',
                  np.allclose(np.sort(hamiltonian_operator.eigenvalues),
                              [0.0,0.0,2.0*abs(D_f),2.0*abs(D_f)]))

            magnetic_moment_operator = system_int.magnetic_moment_operator()
            check('magnetic_moment_operator matrices',
                  all(np.allclose(magnetic_moment_operator.operator_list[alpha].matrix,
                                  system_int.pseudospin_magnetic_moment_matrix[alpha],
                                  atol=1.0e-8)
                      for alpha in range(0,3)))

            # The ground states are |+-3/2> with <mu_z> = -+ g*mu_B*3/2;
            # StaticTransitionMagneticMoments reports the expectation
            # values in units of the Bohr magneton.
            transition_moments = system_int.static_transition_magnetic_moments()
            check('static_transition_magnetic_moments expectation values',
                  np.allclose(np.sort(np.abs(transition_moments.expectation_values)),
                              [1.0,1.0,3.0,3.0]))

            # The doublet method must attach the frame bookkeeping: the
            # main magnetic axis of the ground doublet in the INPUT frame
            # of datafile A is the z axis, and the doublet is a Kramers
            # doublet with gz = 2*g*|M| = 6.
            doublet = system_int.pseudospin_doublet((0,1))
            input_g = np.array(doublet.input_frame_g_tensor.eigenvalues)
            input_axes = np.array(doublet.input_frame_g_tensor.eigenvectors)
            largest = int(np.argmax(input_g))
            check('pseudospin_doublet g values',
                  np.allclose(np.sort(input_g),[0.0,0.0,6.0],atol=1.0e-8))
            check('pseudospin_doublet input-frame main axis',
                  abs(abs(input_axes[2][largest]) - 1.0) < 1.0e-8)
            check('pseudospin_doublet Kramers classification',
                  doublet.kramers is True)

            # The tabulation accepts a single tuple as well as a list of
            # tuples, states the input frame explicitly and contains the
            # doublet energies.
            table_single = system_int.pseudospin_doublet_table((0,1))
            table_list   = system_int.pseudospin_doublet_table([(0,1),(2,3)])
            check('pseudospin_doublet_table states the input frame',
                  'input axis frame' in table_single)
            check('pseudospin_doublet_table accepts a tuple and a list',
                  table_single.count('DOUBLET (') == 1
                  and table_list.count('DOUBLET (') == 2
                  and '{0:12.4f}'.format(2.0*abs(D_f)) in table_list)

            # The self-average must reproduce the single-file system.
            average_aa = cls.from_average_aniso_data(filename_a,filename_a,3,g,tmp_units)
            check('self-average reproduces the single-file spectrum',
                  np.allclose(np.linalg.eigvalsh(average_aa.pseudospin_hamiltonian_matrix),
                              np.linalg.eigvalsh(system_int.pseudospin_hamiltonian_matrix)))

            # The ground doublet of the averaged system: gz = 2*g*|M| = 6
            # for the |+-3/2> doublet with the Lande moment construction.
            doublet = properties.PseudoSpinDoublet\
                                .from_general_operator_matrix((0,1),
                                                              average_aa.hamiltonian,
                                                              average_aa.magnetic_moment,
                                                              tmp_units,
                                                              kramers=True)
            check('Lande moments give the correct ground-doublet g values',
                  np.allclose(np.sort(doublet.g_tensor.eigenvalues),[0.0,0.0,6.0]))

            # Averaging systems with different axial splittings: the gaps
            # average.
            average_ac = cls.from_average_aniso_data(filename_a,filename_c,3,g,tmp_units)
            check('average of different axial splittings',
                  np.allclose(np.linalg.eigvalsh(average_ac.pseudospin_hamiltonian_matrix),
                              [0.0,0.0,3.0*abs(D_f),3.0*abs(D_f)]))

            # Frame handling of misaligned systems: datafile G contains the
            # rhombic system of datafile F rotated by 90 degrees about y in
            # the common laboratory frame. Because the crystal field of G
            # is rotated into the frame of F through the stored frame
            # rotations, the average must reproduce the direct
            # laboratory-frame average of the two Hamiltonians,
            # H_avg = (H_F + H_G)/2, up to an overall rotation (the final
            # rotation into the principal frame of the averaged ground
            # doublet does not change the spectrum or the g values).
            average_fg = cls.from_average_aniso_data(filename_f,filename_g,3,g,tmp_units)

            hamiltonian_g_lab = D_g*np.dot(Sx,Sx) + E_g*(np.dot(Sz,Sz) - np.dot(Sy,Sy))
            hamiltonian_average = 0.5*(hamiltonian_f + hamiltonian_g_lab)
            reference_eigenvalues = np.linalg.eigvalsh(hamiltonian_average)
            reference_eigenvalues = reference_eigenvalues - reference_eigenvalues[0]

            check('averaged system tensors carry the principal frame label',
                  average_fg.tensor_frame == 'principal magnetic axis frame')
            check('average of misaligned systems: laboratory-frame spectrum',
                  np.allclose(np.linalg.eigvalsh(average_fg.pseudospin_hamiltonian_matrix),
                              reference_eigenvalues))

            reference_hamiltonian = pseudospin_operators\
                                    .GeneralOperatorMatrix(hamiltonian_average.astype(np.complex128),
                                                           diagonalize_operator_matrix=True,
                                                           translate_eigenvalues=True)
            reference_moment = pseudospin_operators\
                               .GeneralVectorOperatorMatrix([-g*tmp_units.mu_B*Sx,
                                                             -g*tmp_units.mu_B*Sy,
                                                             -g*tmp_units.mu_B*Sz])
            doublet_reference = properties.PseudoSpinDoublet\
                                          .from_general_operator_matrix((0,1),
                                                                        reference_hamiltonian,
                                                                        reference_moment,
                                                                        tmp_units,
                                                                        kramers=True)
            doublet_fg = properties.PseudoSpinDoublet\
                                   .from_general_operator_matrix((0,1),
                                                                 average_fg.hamiltonian,
                                                                 average_fg.magnetic_moment,
                                                                 tmp_units,
                                                                 kramers=True,
                                                                 rotation=average_fg.input_frame_rotation)
            check('average of misaligned systems: ground-doublet g values',
                  np.allclose(np.sort(doublet_fg.g_tensor.eigenvalues),
                              np.sort(doublet_reference.g_tensor.eigenvalues),
                              atol=1.0e-6))

            # The input_frame_rotation attribute must be a proper rotation,
            # and attaching it to a doublet must express the magnetic axes
            # in the input (laboratory) frame: the main axis of the ground
            # doublet of the averaged system must coincide with that of the
            # directly evaluated laboratory-frame reference.
            check('input_frame_rotation is a proper rotation',
                  abs(np.linalg.det(average_fg.input_frame_rotation.rotation_matrix)
                      - 1.0) < 1.0e-10)

            reference_g    = np.array(doublet_reference.g_tensor.eigenvalues)
            reference_axis = np.array(doublet_reference.g_tensor.eigenvectors)\
                             [:,int(np.argmax(reference_g))]
            averaged_g     = np.array(doublet_fg.input_frame_g_tensor.eigenvalues)
            averaged_axis  = np.array(doublet_fg.input_frame_g_tensor.eigenvectors)\
                             [:,int(np.argmax(averaged_g))]
            check('doublet magnetic axes reported in the input frame',
                  abs(abs(np.dot(reference_axis,averaged_axis)) - 1.0) < 1.0e-6)

        return debug_output.test_summary('AbInitioElectronExchangeSystem',
                                         result_list,print_output)
            




        
