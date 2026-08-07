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
from ouluspin import result_table
from ouluspin._fortran import fortran_utils as fu

from ouluspin import _debug as output


class PseudoSpinSystem:
    """The common base of the pseudospin systems of the library.

    The class collects the analysis that depends only on the pseudospin
    Hamiltonian and the pseudospin magnetic moment operator of a system,
    and not on where those operators came from. Both
    ElectronExchangeSystem, which builds them from tensors given by the
    user, and AbInitioElectronExchangeSystem, which extracts them from the
    results of a quantum-chemical calculation, inherit from it, so that the
    two are analysed through one and the same interface.

    The class is not instantiated on its own and takes no constructor
    arguments of its own; it is a base class only. A subclass must provide
    the following, which the methods here use:

    basis : PseudoSpinBasis
        The basis of the pseudospin system.
    units : EnergyUnitSystem
        The energy unit system.
    kramers_system : boolean
        Whether the system is a Kramers (True) or a non-Kramers (False)
        system; see the kramers_system_from_basis class method.
    input_frame_rotation : Rotation
        The rotation from the input coordinate frame to the frame the
        pseudospin operators are expressed in, so that the magnetic axes
        can be reported with respect to the input frame. The identity when
        the operators are already expressed in the input frame.
    hamiltonian_operator() : PseudoSpinOperator
        The pseudospin Hamiltonian of the system, carrying its matrix
        representation and its eigenvalues.
    magnetic_moment_operator() : PseudoSpinVectorOperator
        The pseudospin magnetic moment operator of the system, carrying the
        matrix representations of its three Cartesian components.

    A subclass is free to store these operators or to construct them on
    request; the methods here call the two methods and do not assume
    either. Since the class cannot be instantiated by itself it carries no
    run_tests class method of its own, and its methods are covered by the
    run_tests of both subclasses.

    Public methods
    --------------
    kramers_system_from_basis(basis) : boolean
        Static method returning whether the system spanned by the given
        basis is a Kramers system.
    static_transition_magnetic_moments(n_states=None) : StaticTransitionMagneticMoments
        Construct and return the StaticTransitionMagneticMoments instance
        of the system.
    static_magnetic_properties(grid,...) : StaticMagneticProperties
        Construct and return the StaticMagneticProperties instance of the
        system, i.e. the powder magnetization and susceptibility.
    quantization_axis() : array of float64
        Return the quantization axis of the system, i.e. the z axis of the
        frame the pseudospin operators and the spherical tensors are
        written in, as a unit vector in the input coordinate frame.
    ground_doublet_magnetic_axis() : array of float64
        Return the principal magnetic axis of the ground
        Kramers/Ising/pseudo doublet as a unit vector in the input
        coordinate frame.
    pseudospin_doublet(states) : PseudoSpinDoublet
        Construct and return the PseudoSpinDoublet instance of the doublet
        spanned by the two pseudospin eigenstates given in the states
        tuple, with the g-tensor reported in the input axis frame.
    pseudospin_doublet_index_list(pseudospin) : list of tuple of int
        Return the states of the pseudospin multiplet grouped into
        doublets, as the list of index tuples the doublet methods take.
    pseudospin_doublet_list(doublets) : list
        Construct and return the PseudoSpinDoublet instances of one or
        several pseudospin doublets.
    pseudospin_doublet_summary_table(doublets) : ResultTable
        Construct and return a compound table of one or several pseudospin
        doublets with one line per doublet.
    pseudospin_doublet_table(doublets) : str
        Construct and return a tabulation string of one or several
        pseudospin doublets, each tabulated in full.

    Private methods
    ---------------
    __error(message)
        Report an error and stop.
    __warning(message)
        Report a warning and continue.
    """

    @staticmethod
    def kramers_system_from_basis(basis):
        """Return whether the pseudospin system spanned by the given basis
        is a Kramers system, i.e. one whose states are degenerate in pairs
        by Kramers' theorem in the absence of a magnetic field.

        A system is a Kramers system when it holds an odd number of
        electrons, i.e. when its TOTAL spin is half-integer. The total spin
        of a system of several sites is half-integer exactly when the sum
        of the pseudospins of the sites is half-integer, i.e. when the sum
        of the pseudospins in the doubled form used throughout the library
        is odd.

        Note that the parity of the DIMENSION of the basis does not decide
        this for a system of more than one site: the dimension is the
        product of the dimensions of the sites, so it is even as soon as a
        single site carries a half-integer pseudospin. Two S = 1/2 sites,
        for instance, span a basis of four states but hold two electrons,
        so the system is not a Kramers system at all: its states are a
        singlet and a triplet, none of them degenerate by Kramers'
        theorem. For a system of one site the two criteria agree.

        Arguments
        ---------
        basis : PseudoSpinBasis
            The basis of the pseudospin system.
        """
        return sum(basis.pseudospin_list) % 2 == 1


    def __error(self, message):
        """Report an error and stop. The message is printed one line at a
        time, so that every line of a message of several lines is marked as
        an error, and the header names the class of the instance, so that an
        error raised in a base class points at the class that was built.
        """
        print("ERROR in " + type(self).__name__ + ".")
        for line in message.split("\n"):
            print("ERROR: " + line)
        print("Error termination.")
        sys.exit(1)


    def __warning(self, message):
        """Report a warning and continue. The message is printed the way the
        one of __error is, and the calculation goes on.
        """
        print("WARNING in " + type(self).__name__ + ".")
        for line in message.split("\n"):
            print("Warning: " + line)
        print()


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
            n_states = self.basis.n_basis

        return properties.StaticTransitionMagneticMoments(self.magnetic_moment_operator(),
                                                          self.hamiltonian_operator(),
                                                          n_states,
                                                          self.units)


    def static_magnetic_properties(self, grid,
                                   susceptibility=None,
                                   magnetization=None,
                                   sample_orientation='powder',
                                   n=1.0,
                                   print_output=None):
        """Construct and return the StaticMagneticProperties instance of
        the system, i.e. the powder magnetization and magnetic
        susceptibility evaluated from the pseudospin Hamiltonian and the
        pseudospin magnetic moment operator of this system.

        The properties are NOT calculated here; the returned instance
        calculates them when its calculate_magnetization and
        calculate_susceptibility methods are called, so that a system can
        be handed the specifications of several calculations in turn.

        Both properties are obtained by integrating over the orientations
        of the molecule, so both are invariant under the choice of the
        coordinate frame the operators of the system are expressed in: the
        frame does not enter the result, and a system whose quantization
        axis was chosen one way gives the same magnetization and
        susceptibility as the same system quantized another way.

        Arguments
        ---------
        grid : ZCWGrid or LebedevLaikovGrid or SimpleGrid
            The grid used for the powder integration. See
            StaticMagneticProperties for which grid suits which property
            and for the recommended grid qualities.

        Optional arguments
        ------------------
        susceptibility : StaticMagneticSusceptibility or None
            The specification of the susceptibility calculation. Default is
            None, in which case the susceptibility is not calculated.
        magnetization : IsothermalStaticMagnetization or None
            The specification of the magnetization calculation. Default is
            None, in which case the magnetization is not calculated.
        sample_orientation : str
            How the orientations of the individual molecules are taken into
            account, i.e. 'powder' (default), 'free' or 'maximal'. See
            StaticMagneticProperties.
        n : float
            The number of magnetic subsystems per mole of the sample.
            Default is 1.0.
        print_output : boolean or None
            Whether the calculation prints output. Default is None, in
            which case the setting of this system is used.
        """
        if print_output is None:
            print_output = getattr(self,'print_output',False)

        return properties.StaticMagneticProperties(
            self.hamiltonian_operator(),
            self.magnetic_moment_operator(),
            grid,
            self.units,
            susceptibility=susceptibility,
            magnetization=magnetization,
            print_output=print_output,
            sample_orientation=sample_orientation,
            n=n)


    def __oriented_unit_vector(self, vector, name):
        """Return the given vector normalized to unit length and given the
        sign convention of the axes of the library.

        An axis has no direction, so the sign of a vector representing one
        is arbitrary. It is fixed here by making the FIRST component that
        is not numerically zero positive, so that the same system always
        gives the same vector.

        The sign is deliberately not decided by the component of the
        largest magnitude: the largest magnitude is often shared by two
        components (an axis such as (1,1,-1)/sqrt(3) is nothing unusual for
        a molecule of a high symmetry), the tie is then broken by the
        floating-point noise of the last digits, and the same axis can come
        out with either sign. Which component comes first is settled before
        any arithmetic, so only a component that is numerically zero needs
        a tolerance, and the components are scanned in order until one is
        clearly nonzero.

        The name of the axis is used in the error message raised for a
        vector that cannot be normalized.
        """
        axis = np.array(vector, dtype=np.float64)
        norm = la.norm(axis)

        if norm <= 0.0:
            self.__error("The " + name + " of the system came out as a vector of\n"
                         "zero length and cannot be normalized.")

        axis = axis/norm

        for i in range(0,3):
            if abs(axis[i]) > 1.0e-8:
                if axis[i] < 0.0:
                    axis = -axis
                break

        return axis


    def quantization_axis(self):
        """Return the quantization axis of the system as a unit vector
        expressed in the input coordinate frame, i.e. in the frame the
        operators of the system were given or calculated in.

        The quantization axis is the axis the pseudospin is quantized
        along: the z axis of the frame the pseudospin operators and the
        spherical tensors of the system are written in. It is the axis of
        the projection M that labels the pseudospin basis states, and the
        component q of every Iwahara--Chibotaru operator O_{k,q} is counted
        with respect to it.

        How the axis was chosen is recorded in the tensor_frame attribute
        of the system. When it was chosen as the principal magnetic axis of
        the ground Kramers/Ising/pseudo doublet, which is the usual choice,
        this method and ground_doublet_magnetic_axis return the same
        vector; when it was chosen otherwise, the two differ.

        The axis is read off the input_frame_rotation attribute, which
        relates the input frame to the frame of the pseudospin operators by
        v_current = R * v_input. The axis that appears as the z axis in the
        current frame is therefore R^T applied to the unit vector along z,
        i.e. the third row of R.

        The axis has no direction, so the sign of the vector is fixed by
        making the first component that is not numerically zero positive.
        """
        rotation_matrix = np.array(self.input_frame_rotation.rotation_matrix,
                                   dtype=np.float64)

        return self.__oriented_unit_vector(rotation_matrix[2,:],
                                           'quantization axis')


    def ground_doublet_magnetic_axis(self):
        """Return the principal magnetic axis of the ground
        Kramers/Ising/pseudo doublet of the system as a unit vector
        expressed in the input coordinate frame, i.e. in the frame the
        operators of the system were given or calculated in.

        The principal (main) magnetic axis of a doublet is the principal
        axis of its g-tensor belonging to the largest principal g value,
        which is the convention of the field for the strongly axial
        doublets of a single-molecule magnet. The axis is evaluated here
        from the g-tensor of the doublet spanned by the two lowest
        pseudospin eigenstates, expressed in the input frame, i.e. from
        pseudospin_doublet((0,1)).input_frame_g_tensor.

        This is the axis the pseudospin is quantized along whenever the
        quantization axis was chosen as the principal magnetic axis of the
        ground doublet; see quantization_axis, which returns the axis that
        was actually used.

        The definition of the main axis is only meaningful for a doublet
        that is axial enough for the largest principal g value to stand
        alone. For a planar or nearly isotropic doublet, whose two largest
        principal g values are (nearly) degenerate, the axis is fixed by
        numerical noise in the diagonalization and carries no physical
        meaning; this is a property of the definition and not of the
        evaluation.

        The axis has no direction, so the sign of the vector is fixed by
        making the first component that is not numerically zero positive.
        """
        g_tensor = self.pseudospin_doublet((0,1)).input_frame_g_tensor

        g_values     = np.array(g_tensor.eigenvalues, dtype=np.float64)
        eigenvectors = np.array(g_tensor.eigenvectors, dtype=np.float64)

        main_axis = eigenvectors[:,int(np.argmax(np.abs(g_values)))]

        return self.__oriented_unit_vector(main_axis,
                                           'ground doublet magnetic axis')


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


    def pseudospin_doublet_index_list(self, pseudospin):
        """Return the states of the pseudospin multiplet grouped into
        doublets, as the list of index tuples the doublet methods take.

        The pseudospin is given in the doubled form used throughout the
        library, i.e. 15 for S = 15/2, and the multiplet holds pseudospin
        + 1 states. How they group depends on the parity of the multiplet:

        A Kramers system, i.e. one of a half-integer pseudospin (an odd
        value of the argument), holds an even number of states, which are
        exactly degenerate in pairs by Kramers' theorem. The states are
        grouped into the doublets (0,1), (2,3), ... and every state
        belongs to one.

        A non-Kramers system, i.e. one of an integer pseudospin (an even
        value of the argument), holds an odd number of states, so one of
        them is left over as a singlet. The doublets of such a system are
        quasi-doublets, i.e. pairs of states split by the tunneling gap,
        so the states are paired to leave the smallest total splitting
        within the pairs: the singlet is placed where it costs the least.
        It is returned as a tuple of a single index, which the
        pseudospin_doublet_list method turns into the energy of the state.

        The singlet cannot be the ground state, since the principal
        magnetic axes of the system are those of the ground doublet and a
        singlet carries none; that case is an error.

        Arguments
        ---------
        pseudospin : int
            The pseudospin of the multiplet in the doubled form, i.e. 15
            for the J = 15/2 multiplet of a Dy(III) ion. The multiplet must
            fit into the basis of the system.
        """
        n_states = pseudospin + 1

        if n_states > self.basis.n_basis:
            self.__error("The pseudospin " + str(pseudospin) + " needs "
                         + str(n_states) + " states,\n"
                         "but the basis of the system holds only "
                         + str(self.basis.n_basis) + ".")

        # A half-integer pseudospin, i.e. a Kramers system: the states are
        # degenerate in pairs and every state belongs to a doublet.
        if pseudospin % 2 == 1:
            return [(2*i,2*i+1) for i in range(0,n_states//2)]

        # An integer pseudospin, i.e. a non-Kramers system. The states are
        # ordered by energy, so the pairs are formed of neighbouring states
        # and the singlet splits the multiplet in two: the states below it
        # pair among themselves and so do the states above it. The singlet
        # therefore stands at an even position, and the one leaving the
        # smallest total splitting within the pairs is chosen.
        energy_list = self.hamiltonian_operator().eigenvalues[:n_states]

        def pairing_cost(singlet_index):
            """The total splitting within the pairs when the state of the
            given index is left as the singlet.
            """
            cost = 0.0

            for i in range(0,singlet_index,2):
                cost += abs(energy_list[i+1] - energy_list[i])

            for i in range(singlet_index+1,n_states-1,2):
                cost += abs(energy_list[i+1] - energy_list[i])

            return cost

        singlet_index = 0
        smallest_cost = None

        for candidate in range(0,n_states,2):
            cost = pairing_cost(candidate)

            if smallest_cost is None or cost < smallest_cost:
                smallest_cost = cost
                singlet_index = candidate

        if singlet_index == 0:
            self.__error("The ground state of the non-Kramers multiplet is a singlet,\n"
                         "i.e. it is not part of a quasi-doublet. The principal\n"
                         "magnetic axes of the system are those of the ground\n"
                         "doublet, so they cannot be determined for such a system.")

        index_list = [(i,i+1) for i in range(0,singlet_index,2)]
        index_list.append((singlet_index,))
        index_list.extend([(i,i+1) for i in range(singlet_index+1,n_states-1,2)])

        return index_list


    def pseudospin_doublet_list(self, doublets):
        """Construct and return a list of PseudoSpinDoublet instances of the
        listed pseudospin doublets of the system. The doublets are
        constructed with the pseudospin_doublet method, i.e. with the full
        frame bookkeeping of the system, but the Hamiltonian and the
        magnetic moment operators are constructed only once for the whole
        list.

        Doublets that have already been constructed are passed through
        unchanged, so that a list obtained from this method can be given
        to the tabulation methods without constructing the doublets a
        second time.

        Arguments
        ---------
        doublets : tuple of int, PseudoSpinDoublet or list
            Either a single tuple with the indices of the two pseudospin
            eigenstates spanning a doublet, a single already constructed
            PseudoSpinDoublet, or a list of either, in which case all the
            listed doublets are constructed.
        """
        if isinstance(doublets,(tuple,properties.PseudoSpinDoublet)):
            doublet_list = [doublets]
        elif isinstance(doublets,list):
            doublet_list = doublets
        else:
            self.__error("The doublets argument must be a tuple of two state indices,\n"
                         "a PseudoSpinDoublet, or a list of either.")

        # The operators are needed only when a doublet still has to be
        # constructed.
        hamiltonian_operator     = None
        magnetic_moment_operator = None

        instance_list = []

        for states in doublet_list:
            if isinstance(states,properties.PseudoSpinDoublet):
                instance_list.append(states)
                continue

            # The energy of a singlet state, i.e. the result of an earlier
            # call of this method, is passed through as it is, so that a
            # list obtained from this method can be handed to the
            # tabulation methods as it stands.
            if isinstance(states,(int,float,np.integer,np.floating)):
                instance_list.append(float(states))
                continue

            if hamiltonian_operator is None:
                hamiltonian_operator     = self.hamiltonian_operator()
                magnetic_moment_operator = self.magnetic_moment_operator()

            # A singlet state of a non-Kramers system is given as a tuple
            # of one index. It spans no doublet, so there is nothing to
            # construct; its energy is passed on, which is what the
            # tabulation methods take for a state that is not part of a
            # doublet.
            if len(states) == 1:
                instance_list.append(
                    float(hamiltonian_operator.eigenvalues[states[0]]))
                continue

            if not len(states) == 2:
                self.__error("Each doublet must be given as a tuple of two state indices,\n"
                             "or a singlet as a tuple of one index.")

            instance_list.append(
                properties.PseudoSpinDoublet
                          .from_pseudospin_operator(tuple(states),
                                                    hamiltonian_operator,
                                                    magnetic_moment_operator,
                                                    self.units,
                                                    kramers=self.kramers_system,
                                                    rotation=self.input_frame_rotation))

        return instance_list


    def pseudospin_doublet_summary_table(self, doublets):
        """Construct and return a compound table of the listed pseudospin
        doublets of the system as an instance of ResultTable, with one line
        per doublet. The table is built by the
        pseudospin_doublet_compound_table class method of ResultTable; see
        it for the structure of the table, which differs between Kramers
        and non-Kramers systems.

        Arguments
        ---------
        doublets : tuple of int, PseudoSpinDoublet or list
            The doublets to tabulate, in any of the forms accepted by the
            pseudospin_doublet_list method. Passing a list of already
            constructed doublets avoids constructing them twice when both
            tabulation methods are used.
        """
        return result_table.ResultTable\
                           .pseudospin_doublet_compound_table(
                               self.pseudospin_doublet_list(doublets),
                               self.units)


    def pseudospin_doublet_table(self, doublets):
        """Construct and return a human-readable tabulation string of one
        or several pseudospin doublets of the system, with the properties
        of each doublet tabulated separately and in full.

        Each tabulated doublet is constructed with the pseudospin_doublet
        method: the g-tensors and their principal magnetic axes are given
        in the INPUT axis frame, i.e. the frame the operators of the system
        were given or calculated in, which is also stated explicitly in the
        output. The energy of each
        doublet is the eigenvalue of the lower of its two states. A
        compact one-line-per-doublet summary of the same doublets is given
        by the pseudospin_doublet_summary_table method.

        Arguments
        ---------
        doublets : tuple of int, PseudoSpinDoublet or list
            The doublets to tabulate, in any of the forms accepted by the
            pseudospin_doublet_list method. Passing a list of already
            constructed doublets avoids constructing them twice when both
            tabulation methods are used.
        """
        tmp_str  = "    PSEUDOSPIN DOUBLETS\n\n"
        tmp_str += "    The g-tensors and their principal magnetic axes are given in the\n"
        tmp_str += "    input axis frame.\n\n"

        for doublet in self.pseudospin_doublet_list(doublets):
            # A singlet state of a non-Kramers system spans no doublet and
            # has none of the properties tabulated below, so only its
            # energy is stated.
            if not isinstance(doublet,properties.PseudoSpinDoublet):
                tmp_str += "    SINGLET,   E = {0:12.4f} {1}\n\n"\
                           .format(float(doublet),self.units.energy_unit_str)
                continue

            if doublet.state_energies is None:
                energy_str = "not available"
            else:
                energy_str = "{0:12.4f} {1}".format(min(doublet.state_energies),
                                                    self.units.energy_unit_str)

            tmp_str += "    DOUBLET ({0},{1}),   E = {2}\n\n"\
                       .format(doublet.states[0],doublet.states[1],energy_str)
            tmp_str += str(doublet)
            tmp_str += "\n"

        return tmp_str


class ElectronExchangeSystem(PseudoSpinSystem):
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
    n_basis : int
        The dimension of the basis, i.e. the number of states of the system.
    kramers_system : boolean
        Whether the system is a Kramers (True) or a non-Kramers (False)
        system, i.e. whether the sum of the pseudospins of the sites is
        half-integer; see PseudoSpinSystem.kramers_system_from_basis.
    hamiltonian_tensor_list : list of IwaharaChibotaruSphericalTensor
        The tensors of the Hamiltonian, each a copy of the tensor given in
        the corresponding tuple inflated to the sites of the whole system.
        The tensors of the caller are not modified.
    magnetic_moment_tensor_list : list of MixedCartesianIwaharaChibotaruSphericalTensor
        The tensors of the magnetic moment, each the tensor given in the
        corresponding tuple multiplied by -mu_B and inflated to the sites
        of the whole system.
    tensor_frame : str
        The coordinate frame label attached to the tensors of the system,
        i.e. 'input axis frame': the tensors are used in the frame they
        were given in and no rotation is applied to them.
    input_frame_rotation : Rotation
        The rotation from the input coordinate frame to the frame of the
        operators of the system, i.e. the identity for this class, since
        the operators are used in the frame the tensors were given in. It
        is carried so that the doublet methods inherited from
        PseudoSpinSystem report the magnetic axes in that frame.
    energy_print_threshold : float or None
        The energy threshold above which eigenvalues are omitted from the
        printed output; see the corresponding optional argument.

    Private methods
    ---------------
    __error(message)
        Report an error and stop.
    __warning(message)
        Report a warning and continue.
    __construct_hamiltonian()
        Construct the field-independent terms of the Hamiltonian, i.e. the ZFS
        and exchange terms, and store the operator as an attribute. The
        Hamiltonian is diagonalized upon construction.
    __construct_magnetic_moment_operators()
        Construct the operators of the three Cartesian components of the magnetic
        moment and store them as attributes. Construction of the operator matrices
        is requested upon their construction but they are not diagonalized.
    __inflation_list(tensor_tuple) : list of int
        Return the inflation list of a tensor tuple and check its site
        indices.

    Public methods
    --------------
    hamiltonian_operator() : PseudoSpinOperator
        Return the field-free pseudospin Hamiltonian of the system.
    magnetic_moment_operator() : PseudoSpinVectorOperator
        Return the pseudospin magnetic moment operator of the system.
    hamiltonian_tensor() : IwaharaChibotaruSphericalTensor
        Construct and return the irreducible tensor representation of the
        whole field-free Hamiltonian, i.e. the sum of the given tensors.
    magnetic_moment_tensor() : MixedCartesianIwaharaChibotaruSphericalTensor
        Construct and return the mixed Cartesian--spherical tensor
        representation of the whole magnetic moment operator.

    The analysis that depends only on the pseudospin operators is inherited
    from PseudoSpinSystem: static_transition_magnetic_moments,
    static_magnetic_properties, quantization_axis,
    ground_doublet_magnetic_axis, pseudospin_doublet,
    pseudospin_doublet_index_list, pseudospin_doublet_list,
    pseudospin_doublet_summary_table and pseudospin_doublet_table. See the
    base class for their documentation. The tensors of this class are used
    in the frame they were given in, so the quantization axis is by
    construction the z axis of that frame; the principal magnetic axis of
    the ground doublet, which is where the pseudospin would be quantized in
    the conventional frame, is returned by ground_doublet_magnetic_axis and
    is in general a different vector.

    Class methods
    -------------
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if all
        tests passed.
    """

    def __error(self, message):
        """Report an error and stop. The message is printed one line at a
        time, so that every line of a message of several lines is marked as
        an error, and the header names the class of the instance, so that an
        error raised in a base class points at the class that was built.
        """
        print("ERROR in " + type(self).__name__ + ".")
        for line in message.split("\n"):
            print("ERROR: " + line)
        print("Error termination.")
        sys.exit(1)


    def __warning(self, message):
        """Report a warning and continue. The message is printed the way the
        one of __error is, and the calculation goes on.
        """
        print("WARNING in " + type(self).__name__ + ".")
        for line in message.split("\n"):
            print("Warning: " + line)
        print()


    def __inflation_list(self, tensor_tuple):
        """Return the inflation list of the given tensor tuple, i.e. the
        list of ones and zeros that inflate_dimension takes: a one at the
        position of every site the tensor acts on and a zero at every other
        site of the system.

        The site indices of the tuple are checked here, since an index that
        does not name a site of the system, or a site named twice, would
        otherwise produce an inflation list of the wrong number of ones and
        fail inside the tensor with a message that does not say which
        tensor was at fault.
        """
        site_list = list(tensor_tuple[1:])

        for site in site_list:
            if not isinstance(site,(int,np.integer)):
                self.__error("The site indices of a tensor tuple must be integers,\n"
                                    "but one of them is " + repr(site) + ".")

            if site < 0 or site >= self.n_sites:
                self.__error("A tensor tuple names the site " + str(site)
                             + ", but the system holds\n"
                             "the sites 0 to " + str(self.n_sites - 1) + ".")

        if not len(set(site_list)) == len(site_list):
            self.__error("A tensor tuple names the same site more than once: "
                                + str(site_list) + ".")

        if len(site_list) == 0:
            self.__error("A tensor tuple names no site at all. Each tuple must\n"
                                "hold the tensor followed by the sites it acts on.")

        return [1 if i in site_list else 0 for i in range(0,self.n_sites)]


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

            inflation_list = self.__inflation_list(tensor_tuple)

            # inflate_dimension modifies the tensor in place, so the tensor
            # of the caller is copied before it is inflated. Inflating it
            # directly would leave the caller holding a tensor of a grown
            # number of sites: the same tuple list could then not be used
            # to construct a second system (the second inflation fails on
            # the number of sites), and one tensor object could not be
            # reused for several site tuples.
            tensor = deepcopy(tensor_tuple[0])
            tensor.inflate_dimension(inflation_list)
            tensor.frame = self.tensor_frame
            tensor_list.append(tensor)

        self.hamiltonian_tensor_list = tensor_list

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
            inflation_list = self.__inflation_list(tensor_tuple)

            # The multiplication returns a new instance, so the inflation does
            # not modify the mixed tensor stored in the tuple list.
            mixed_tensor = -self.units.mu_B * tensor_tuple[0]
            mixed_tensor.inflate_dimension(inflation_list)

            mixed_tensor.frame = self.tensor_frame
            for component in mixed_tensor.component_list:
                component.frame = self.tensor_frame

            mixed_tensor_list.append(mixed_tensor)

        self.magnetic_moment_tensor_list = mixed_tensor_list

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

    
    def hamiltonian_operator(self):
        """Return the field-free pseudospin Hamiltonian of the system as a
        PseudoSpinOperator instance, carrying its matrix representation and
        its eigenvalues, the lowest of which is zero.

        The operator is built once when the system is constructed and is
        the hamiltonian attribute; this method returns it so that the
        system offers the same interface as
        AbInitioElectronExchangeSystem, which constructs its operators on
        request. The returned instance is the one the system holds, not a
        copy, so it must not be modified in place.
        """
        return self.hamiltonian


    def magnetic_moment_operator(self):
        """Return the pseudospin magnetic moment operator of the system as
        a PseudoSpinVectorOperator instance, carrying the matrix
        representations of its three Cartesian components.

        The operator already includes the factor -mu_B, i.e. it is the
        magnetic moment itself and not the g-tensor that was given to the
        constructor. It is built once when the system is constructed and is
        the magnetic_moment attribute; this method returns it so that the
        system offers the same interface as
        AbInitioElectronExchangeSystem. The returned instance is the one
        the system holds, not a copy, so it must not be modified in place.
        """
        return self.magnetic_moment


    def hamiltonian_tensor(self):
        """Construct and return the irreducible tensor representation of
        the whole field-free Hamiltonian of the system, i.e. the sum of the
        tensors given to the constructor, each inflated to the sites of the
        whole system.

        The tensors given to the constructor describe the system one term
        at a time, each acting on its own sites; the tensor returned here
        is the single tensor of the whole system, which is what the ITO
        table and the tensor algebra of the library take. The returned
        tensor is a new instance and carries the coordinate frame label of
        the system (see the tensor_frame attribute), so it can be modified
        freely without affecting the system.
        """
        total_tensor = deepcopy(self.hamiltonian_tensor_list[0])

        for tensor in self.hamiltonian_tensor_list[1:]:
            total_tensor = total_tensor + tensor

        total_tensor.frame = self.tensor_frame

        return total_tensor


    def magnetic_moment_tensor(self):
        """Construct and return the mixed Cartesian--spherical tensor
        representation of the whole magnetic moment operator of the system,
        i.e. the sum of the magnetic moment tensors given to the
        constructor, each multiplied by -mu_B and inflated to the sites of
        the whole system.

        NOTE that the tensors given to the constructor imply the g-tensor,
        whereas the tensor returned here is the magnetic moment itself: the
        multiplication by the negative Bohr magneton has been carried out,
        as it has in the magnetic_moment attribute.

        The returned tensor is a new instance and carries the coordinate
        frame label of the system, so it can be modified freely without
        affecting the system.
        """
        total_tensor = deepcopy(self.magnetic_moment_tensor_list[0])

        for tensor in self.magnetic_moment_tensor_list[1:]:
            total_tensor = total_tensor + tensor

        total_tensor.frame = self.tensor_frame
        for component in total_tensor.component_list:
            component.frame = self.tensor_frame

        return total_tensor


    def __repr__(self):
        """Return a human-readable summary of the system: what the system
        holds and the lowest of its energy levels.

        A susceptibility or a magnetization attached to the instance by
        external code is appended, as it has always been; this class does
        not itself store such instances.
        """
        tmp_str  = "    ELECTRON EXCHANGE SYSTEM\n\n"
        tmp_str += "    Number of spin sites          {0:12d}\n".format(self.n_sites)
        tmp_str += "    Pseudospins (2S)              " \
                   + " ".join(str(pseudospin)
                              for pseudospin in self.pseudospin_list) + "\n"
        tmp_str += "    Dimension of the basis        {0:12d}\n".format(self.n_basis)
        tmp_str += "    Kramers system                {0:>12}\n".format(
            "yes" if self.kramers_system else "no")
        tmp_str += "    Coordinate frame              " + self.tensor_frame + "\n\n"

        # The lowest levels, which say what kind of a system it is at a
        # glance. The whole spectrum is available through the Hamiltonian
        # operator itself, so only the beginning of it is stated here.
        n_printed = min(8,len(self.hamiltonian.eigenvalues))

        tmp_str += "    Lowest energy levels / " + self.units.energy_unit_str + "\n"
        for i in range(0,n_printed):
            tmp_str += "      {0:4d}  {1:16.6f}\n".format(
                i,float(self.hamiltonian.eigenvalues[i]))

        if n_printed < len(self.hamiltonian.eigenvalues):
            tmp_str += "      ... {0} further states\n".format(
                len(self.hamiltonian.eigenvalues) - n_printed)

        tmp_str += "\n"

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

        if self.n_sites == 0:
            self.__error("The pseudospin list is empty, so the system holds no\n"
                                "spin sites at all.")

        # All tensors of this class are given in one common coordinate
        # frame and no rotation is applied to them, so the frame of the
        # operators IS the input frame and the rotation relating the two is
        # the identity. The doublet methods inherited from PseudoSpinSystem
        # therefore report the magnetic axes in the frame the tensors were
        # given in, which is what the user of this class expects.
        self.tensor_frame         = 'input axis frame'
        self.input_frame_rotation = tensors.Rotation(np.identity(3, dtype=np.float64))

        tikk = time.time()
        if self.print_output:
            print(self.units)
            print("    Constructing basis ...")
        self.basis = pseudospin_operators.PseudoSpinBasis(self.pseudospin_list)

        self.n_basis = self.basis.n_basis

        # Whether the system is a Kramers system, i.e. one of an odd
        # number of electrons, is decided by the parity of the SUM of the
        # pseudospins and not by the parity of the dimension of the basis;
        # see kramers_system_from_basis.
        self.kramers_system = self.kramers_system_from_basis(self.basis)

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

        # The tensors of the caller must not be modified by the
        # construction: the tensors are inflated to the sites of the whole
        # system, which grows their number of sites, so the system has to
        # work on copies. Otherwise the same tuple list could not be used
        # twice, which is what the second construction below checks.
        one_site_zfs = tensors.IwaharaChibotaruSphericalTensor\
                              .from_one_site_cartesian_operator(-5.0,'z',1)
        zfs_rank_list = deepcopy(one_site_zfs.rank_list)

        tuple_list = [(heisenberg,0,1),(one_site_zfs,0)]

        first_system  = cls([1,1],tuple_list,moment_tensor_tuple_list,tmp_units)

        check('the tensors of the caller are not inflated in place',
              one_site_zfs.rank_list == zfs_rank_list)

        second_system = cls([1,1],tuple_list,moment_tensor_tuple_list,tmp_units)

        check('the same tuple list can be used to build a second system',
              np.allclose(first_system.hamiltonian.eigenvalues,
                          second_system.hamiltonian.eigenvalues))

        # The site indices of a tensor tuple are checked. The errors stop
        # the interpreter, so only the accepted case is exercised here.
        check('a tensor tuple naming every site of the system is accepted',
              first_system.n_sites == 2)

        # Whether the system is a Kramers system follows the sum of the
        # pseudospins and not the parity of the dimension of the basis. Two
        # S = 1/2 sites span four states but hold two electrons, so the
        # system is NOT a Kramers system; its states are a singlet and a
        # triplet.
        check('two S = 1/2 sites do not make a Kramers system',
              (system.n_basis == 4) and (system.kramers_system is False))
        check('one half-integer site makes a Kramers system',
              cls.kramers_system_from_basis(
                  pseudospin_operators.PseudoSpinBasis([3])) is True)
        check('three half-integer sites make a Kramers system',
              cls.kramers_system_from_basis(
                  pseudospin_operators.PseudoSpinBasis([1,1,1])) is True)
        check('a half-integer and an integer site make a Kramers system',
              cls.kramers_system_from_basis(
                  pseudospin_operators.PseudoSpinBasis([1,2])) is True)
        check('two integer sites do not make a Kramers system',
              cls.kramers_system_from_basis(
                  pseudospin_operators.PseudoSpinBasis([2,2])) is False)

        # The operators of the system are returned by the methods that the
        # base class calls, so that the two systems of the module offer the
        # same interface.
        check('hamiltonian_operator returns the Hamiltonian of the system',
              system.hamiltonian_operator() is system.hamiltonian)
        check('magnetic_moment_operator returns the moment of the system',
              system.magnetic_moment_operator() is system.magnetic_moment)

        # The tensors of the whole system, i.e. the sum of the tensors of
        # the tuples. The Hamiltonian tensor must reproduce the operator
        # matrix of the system.
        total_tensor = system.hamiltonian_tensor()

        check('the Hamiltonian tensor carries the frame of the system',
              total_tensor.frame == system.tensor_frame)
        check('the Hamiltonian tensor reproduces the operator matrix',
              np.allclose(debug_output.operator_matrix_from_tensor(total_tensor,
                                                                   [1,1]),
                          system.hamiltonian.matrix))

        # The tensor is a new instance, so purging it must not disturb the
        # system.
        total_tensor.purge_ranks()
        check('the returned tensor is independent of the system',
              np.allclose(system.hamiltonian.eigenvalues,[0.0,J,J,J]))

        moment_tensor_total = system.magnetic_moment_tensor()
        moment_operator     = pseudospin_operators\
                              .PseudoSpinVectorOperator(system.basis,
                                                        [moment_tensor_total],
                                                        tmp_units,
                                                        diagonalize_operator_matrix=False,
                                                        store_operator_matrix=True,
                                                        translate_eigenvalues=False)

        check('the magnetic moment tensor reproduces the moment operators',
              all(np.allclose(moment_operator.operator_list[alpha].matrix,
                              moment_matrix_list[alpha],atol=1.0e-10)
                  for alpha in range(0,3)))

        # The analysis inherited from PseudoSpinSystem. The tensors are
        # used in the frame they were given in, so the quantization axis is
        # the z axis of that frame.
        check('the quantization axis is the z axis of the given frame',
              np.allclose(system.quantization_axis(),[0.0,0.0,1.0]))

        # An axial single-site system, i.e. an S = 3/2 ion of an easy-axis
        # zero-field splitting H = D*S_z^2 with D < 0, whose ground doublet
        # |+-3/2> has its magnetic axis along z and which IS a Kramers
        # system. The ZFS is given as the Cartesian tensor diag(0,0,D); the
        # from_one_site_cartesian_operator constructor would give a term
        # linear in S_z, i.e. an axial Zeeman splitting, whose states are
        # equally spaced and form no doublets at all.
        axial_zfs = tensors.IwaharaChibotaruSphericalTensor\
                           .from_one_site_cartesian_tensor(
                               np.diag([0.0,0.0,-10.0]),3)
        axial_moment = tensors.MixedCartesianIwaharaChibotaruSphericalTensor\
                              .from_one_site_isotropic_operator(2.0,3)

        axial_system = cls([3],[(axial_zfs,0)],[(axial_moment,0)],tmp_units)

        check('the axial system is a Kramers system',
              axial_system.kramers_system is True)
        check('the ground doublet magnetic axis of an axial system is z',
              np.allclose(axial_system.ground_doublet_magnetic_axis(),
                          [0.0,0.0,1.0],atol=1.0e-8))
        check('the quantization axis agrees with it in the given frame',
              np.allclose(axial_system.quantization_axis(),
                          axial_system.ground_doublet_magnetic_axis(),
                          atol=1.0e-8))

        # The ground doublet |+-3/2> of the axial system has g = (0,0,6).
        ground_doublet = axial_system.pseudospin_doublet((0,1))
        check('the ground doublet g values are those of the |+-3/2> doublet',
              np.allclose(sorted(abs(value) for value
                                 in ground_doublet.g_tensor.eigenvalues),
                          [0.0,0.0,6.0],atol=1.0e-6))

        # The grouping of the multiplet into doublets and the tabulations.
        index_list = axial_system.pseudospin_doublet_index_list(3)
        check('the multiplet is grouped into doublets',
              index_list == [(0,1),(2,3)])

        doublet_list = axial_system.pseudospin_doublet_list(index_list)
        check('the doublets are constructed from the index list',
              len(doublet_list) == 2)
        check('an already constructed doublet is passed through',
              axial_system.pseudospin_doublet_list(doublet_list)[0]
              is doublet_list[0])

        summary_table = axial_system.pseudospin_doublet_summary_table(doublet_list)
        check('the compound doublet table is built',
              len(summary_table.string_table()) > 0)
        check('the full doublet tabulation is built',
              "DOUBLET" in axial_system.pseudospin_doublet_table(doublet_list))

        # The transition magnetic moments of the axial system: the states
        # are |+-3/2> and |+-1/2> with <mu_z> = -+ g*mu_B*M.
        transition_moments = axial_system.static_transition_magnetic_moments()
        check('the transition magnetic moments are evaluated',
              np.allclose(np.sort(np.abs(transition_moments.expectation_values)),
                          [1.0,1.0,3.0,3.0]))

        # The static magnetic properties, i.e. the main purpose of the
        # class. The method is inherited from PseudoSpinSystem, so it works
        # on the operators of the system whichever way they were built. The
        # chi*T product of an isolated S = 1/2 with g = 2 is
        # 0.375 cm^3 K mol^-1 at any temperature.
        from ouluspin import integration

        free_spin = cls([1],[(tensors.IwaharaChibotaruSphericalTensor
                                     .from_one_site_cartesian_operator(0.0,'z',1),0)],
                        [(tensors.MixedCartesianIwaharaChibotaruSphericalTensor
                                 .from_one_site_isotropic_operator(2.0,1),0)],
                        tmp_units)

        susceptibility = properties.StaticMagneticSusceptibility([2.0,10.0])
        magnetic_properties = free_spin.static_magnetic_properties(
            integration.LebedevLaikovGrid(7),susceptibility=susceptibility)

        magnetic_properties.calculate_susceptibility()

        # The susceptibility is evaluated from the magnetization in a
        # small but finite field, so the chiT product deviates from the
        # exact value by a little more than the numerical accuracy.
        check('static_magnetic_properties evaluates the chiT product',
              np.allclose(magnetic_properties.susceptibility.susceptibility,
                          [0.375,0.375],atol=1.0e-3))

        # The representation states what the system holds.
        check('the representation states the content of the system',
              ("Number of spin sites" in repr(system))
              and ("Lowest energy levels" in repr(system)))

        return debug_output.test_summary('ElectronExchangeSystem',
                                         result_list,print_output)



class AbInitioElectronExchangeSystem(PseudoSpinSystem):
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

    The pseudospin is quantized along the quantization axis of the
    system, which is the z axis of the frame the pseudospin operators
    and the spherical tensors are written in and is returned by the
    quantization_axis method. By default the quantization axis is
    chosen as the principal magnetic axis of the ground
    Kramers/Ising/pseudo doublet, which is the usual choice; an
    alternative axis can be chosen by passing the rotation matrix R as
    an optional argument, in which case the quantization axis is the
    one the user chose and need not be a magnetic axis of the system.

    Arguments
    ---------
    hamiltonian : GeneralOperatorMatrix
        A matrix representation of the Hamiltonian. The instance must
        contain the eigenvalues and eigenvectors as an attribute.
    magnetic_moment : GeneralVectorOperatorMatrix
        Matrix representations of the vector components of the
        magnetic moment vector operator, written in the basis of the
        eigenstates of the Hamiltonian and in the same order as its
        eigenvalues. This is the form in which the matrices appear in a
        SINGLE_ANISO datafile, where they are given together with the
        spin-orbit energies. The class uses these matrices as they are,
        both in the construction of the pseudospin operators and in the
        determination of the frame rotation.
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
        magnetic moment vector operator, written in the basis of the
        eigenstates of the Hamiltonian and in the same order as its
        eigenvalues. This is the form in which the matrices appear in a
        SINGLE_ANISO datafile, where they are given together with the
        spin-orbit energies. The class uses these matrices as they are,
        both in the construction of the pseudospin operators and in the
        determination of the frame rotation.
    basis : PseudoSpinBasis
        The basis defining the pseudospin system.
    units : EnergyUnitSystem
        The energy unit system.
    kramers_system : boolean
        Whether this system is a Kramers (True) or a non-Kramers (False)
        system, i.e. whether the sum of the pseudospins of the sites is
        half-integer; see PseudoSpinSystem.kramers_system_from_basis.
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
        determined from the ground doublet (R was not given),
        'user-defined axis frame' when an explicit R was passed to the
        constructor, and 'quantization axis frame' for a system built by
        from_kuiva_data, whose quantization axis is the axis the states of
        the file are labelled by M along.
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

    The analysis that depends only on the pseudospin operators is inherited
    from PseudoSpinSystem: static_transition_magnetic_moments,
    static_magnetic_properties, quantization_axis,
    ground_doublet_magnetic_axis, pseudospin_doublet,
    pseudospin_doublet_index_list, pseudospin_doublet_list,
    pseudospin_doublet_summary_table and pseudospin_doublet_table. See the
    base class for their documentation. With no explicit R given to the
    constructor the quantization axis is the principal magnetic axis of the
    ground doublet, so quantization_axis and ground_doublet_magnetic_axis
    then return the same vector; with an explicit R, and for a system built
    by from_kuiva_data, they differ.

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
    from_kuiva_data(filename,units,...)
        Construct the system from a pseudospin file (a .psd file) written by
        a Kuiva calculation. The pseudospin basis, the transformation into
        it and the quantization axis all come from the file, so no
        pseudospin structure is given as an argument and neither the mu_z
        diagonalization nor the reordering of the states of the constructor
        is carried out; only the phase correction and the check of the
        behavior under time reversal are.
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if all
        tests passed.

    Private methods
    ---------------
    __error(message)
        Report an error and stop.
    __warning(message)
        Report a warning and continue.
    __calculate_rotation() : void
        Calculate g-tensor of the ground Kramers/Ising/pseudo doublet
        and store the transformation into its principal axis system
        as an attribute. The doublet is built from the magnetic moment
        matrices as they were given to the class, i.e. in the basis of
        the Hamiltonian eigenstates.
    __oriented_unit_vector(vector,name) : array of float64
        Normalize a vector representing an axis to unit
        length and giving it the sign convention of the library, i.e.
        making the first component that is not numerically zero positive.
    __quantization_frame_rotation(axis) : array of float64
        Return the rotation matrix into a coordinate frame whose z axis lies
        along the axis given as an argument. Used by from_kuiva_data, where
        the quantization axis is fixed by the file instead of being
        determined from the ground doublet.
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
    __correct_and_check_phases()
        Correct the arbitrary phases of the pseudospin basis states and
        check the behavior of the resulting operators under time reversal,
        warning when the behavior is not the correct one. This is the last
        step of the construction and is shared by the constructor and the
        from_kuiva_data class method.
    """
    
    def __error(self, message):
        """Report an error and stop. The message is printed one line at a
        time, so that every line of a message of several lines is marked as
        an error, and the header names the class of the instance, so that an
        error raised in a base class points at the class that was built.
        """
        print("ERROR in " + type(self).__name__ + ".")
        for line in message.split("\n"):
            print("ERROR: " + line)
        print("Error termination.")
        sys.exit(1)


    def __warning(self, message):
        """Report a warning and continue. The message is printed the way the
        one of __error is, and the calculation goes on.
        """
        print("WARNING in " + type(self).__name__ + ".")
        for line in message.split("\n"):
            print("Warning: " + line)
        print()


    def __calculate_rotation(self):
        """Calculate g-tensor of the ground Kramers/Ising/pseudo doublet
        and store the transformation into its principal axis system
        as an attribute. The parity of the basis dimension determines
        whether the ground doublet is a Kramers doublet or a non-Kramers
        (quasi-)doublet, and this is passed on to the doublet explicitly.

        The doublet is constructed from the magnetic moment matrices as
        they were given to the class, i.e. in the basis of the Hamiltonian
        eigenstates, which is the convention of the class and the one
        __setup_numerical_matrices follows as well. The doublet is
        therefore built directly rather than through the
        from_general_operator_matrix class method of PseudoSpinDoublet,
        which transforms the moment matrices by the eigenvectors of the
        Hamiltonian and so expects them in the basis the Hamiltonian was
        given in. Going through that method would transform the matrices
        once too often whenever the Hamiltonian handed to the class is not
        already diagonal, which would pair the wrong states into the ground
        doublet and leave the frame rotation pointing along an axis that is
        not a magnetic axis of the system at all. The energies of the
        states are the eigenvalues of the Hamiltonian, in the same order.
        """
        ground_kd = properties.PseudoSpinDoublet(
            (0,1),
            [self.magnetic_moment.operator_list[alpha].matrix
             for alpha in range(0,3)],
            self.units,
            energies=getattr(self.hamiltonian,'eigenvalues',None),
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


    @classmethod
    def __quantization_frame_rotation(cls, axis):
        """Return the rotation matrix into a coordinate frame whose z axis
        lies along the axis given as an argument, i.e. the matrix R for
        which R*axis = (0,0,1).

        The rotation is used by from_kuiva_data, where the quantization axis
        is not determined from the ground doublet but is fixed beforehand:
        the states of a Kuiva pseudospin file are labelled by the projection
        M along an axis recorded in the file, and the pseudospin operators
        of the library are written with that axis as the z axis.

        Only the z axis of the frame is fixed by this requirement. The x and
        y axes are completed into a right-handed triad in an arbitrary way,
        which is legitimate: turning them about z is a rotation of the
        transverse components of the operators that is compensated by the
        phases of the basis states, and the phases are arbitrary to begin
        with and are fixed afterwards by __correct_phases. The construction
        returns the identity matrix for an axis that already is the z axis.
        """
        axis = np.array(axis, dtype=np.float64)
        norm = la.norm(axis)

        if norm <= 0.0:
            instance = cls.__new__(cls)
            instance.__error("The quantization axis given for the construction of the\n"
                             "coordinate frame is a vector of zero length.")

        z_axis = axis/norm

        # A vector that is not parallel to the axis, from which the
        # transverse axes are built by orthogonalization.
        if abs(z_axis[0]) < 0.9:
            reference_vector = np.array([1.0,0.0,0.0], dtype=np.float64)
        else:
            reference_vector = np.array([0.0,1.0,0.0], dtype=np.float64)

        x_axis = reference_vector - np.dot(reference_vector,z_axis)*z_axis
        x_axis = x_axis/la.norm(x_axis)
        y_axis = np.cross(z_axis,x_axis)

        return np.array([x_axis,y_axis,z_axis], dtype=np.float64)


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
            self.__error("Reorder matrix is not orthogonal.")

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

        and in the review:

            L. F. Chibotaru. Ab Initio Methodology for Pseudospin
            Hamiltonians of Anisotropic Magnetic Complexes, in Advances in
            Chemical Physics, Vol. 153, Eds. S. A. Rice and A. R. Dinner,
            Wiley, 2013.

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


    def __correct_and_check_phases(self):
        """Correct the arbitrary phases of the pseudospin basis states and
        check the behavior of the resulting operators under time reversal,
        warning when the behavior is not the correct one.

        This is the last step of the construction of the pseudospin
        operators and is shared by the class constructor and the
        from_kuiva_data class method. The two build the pseudospin operator
        matrices in different ways -- the constructor by diagonalizing the
        mu_z component of the magnetic moment itself, the class method by
        taking the basis and the transformation into it from the file it
        reads -- but the phases of the basis states are arbitrary in both
        cases and are fixed in the same way.
        """
        self.__correct_phases()

        result, result_str = self.__check_time_reversal()

        if not result:
            self.__warning("Pseudospin operators have incorrect behavior under "
                           "time reversal.\n" + result_str)


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

        label_list          = ['mu_x','mu_y','mu_z','H']
        correct_result_list = [-1,-1,-1,1]

        rows = []
        for i in range(0,4):
            rows.append([label_list[i],result_list[i],correct_result_list[i]])

        tmp_str = str(result_table.ResultTable(
            rows,
            column_headers=['Operator','Result','Correct result'],
            title="EFFECT OF TIME REVERSAL ON THE PSEUDOSPIN OPERATORS",
            notes=["A result of 1 stands for a time-even and -1 for a time-odd",
                   "operator; 0 means that the operator has no definite behaviour",
                   "under time reversal."],
            indent=4))

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

        # Whether the system is a Kramers system, i.e. one of an odd
        # number of electrons, is decided by the parity of the SUM of the
        # pseudospins and not by the parity of the dimension of the basis;
        # the two agree for a system of one site but not for a system of
        # several. See kramers_system_from_basis.
        self.kramers_system = self.kramers_system_from_basis(self.basis)

        if not self.magnetic_moment.n_basis == self.n_full_basis:
            self.__error("Inconsistent dimensions in input matrices.")

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
                self.__error("The rotation matrix R must be a proper rotation\n"
                             "(orthogonal with determinant +1). The magnetic moment\n"
                             "is an axial vector, and rotating it with an improper\n"
                             "matrix would silently mirror the physics.")
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
        self.__correct_and_check_phases()


    @classmethod
    def __construct_basis(cls, pseudospin):
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
            instance = cls.__new__(cls)
            instance.__error("The pseudospin argument must be a PseudoSpinBasis, a "
                             "single\ninteger pseudospin or a list of integer "
                             "pseudospins (all\npseudospins given as multiples of "
                             "two).")


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
            instance = cls.__new__(cls)
            instance.__error("from_average_aniso_data only supports single-site systems,\n"
                             "as a single Lande g-factor defines the magnetic moment of\n"
                             "one spin site only.")

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
                                                             kramers=cls.kramers_system_from_basis(basis))

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
    def from_kuiva_data(cls, filename, units,
                        include_bohr_magneton=True,
                        print_output=False):
        """Construct the system from a pseudospin file (a .psd file) written
        by a Kuiva calculation.

        A Kuiva pseudospin file differs from a SINGLE_ANISO datafile in that
        it has already identified its matrices with a pseudospin product
        basis: it carries the pseudospin of every site, the ordered listing
        of the basis states, the effective Hamiltonian and the Cartesian
        components of the magnetic moment OVER THAT BASIS, and the unitary
        that maps the ab initio states to it. Everything the constructor of
        this class determines for itself therefore comes from the file
        instead:

          - the pseudospin basis is the one of the file (the reader checks
            that its listing is exactly the ordering of PseudoSpinBasis, so
            the states need no reordering and the reorder_list of the
            constructor has no counterpart here);
          - the transformation into the pseudospin basis is the one of the
            file, so the mu_z component of the magnetic moment is not
            diagonalized and the states are not projected onto a subspace:
            the dimension of the model IS the dimension of the basis;
          - the quantization axis is the axis along which the states of the
            file are labelled by M, and not the principal magnetic axis of
            the ground doublet. The two are usually the same axis for a
            system of one site, as Kuiva labels the states of a site along
            its own principal magnetic axis, but for a system of several
            sites the ground doublet of the coupled system is a property of
            the whole system and says nothing about how the states of the
            individual sites were labelled. The axis of the file is the one
            the basis states refer to and is therefore the one the operators
            are quantized along.

        The Cartesian components of the magnetic moment are rotated from the
        frame of the file into the frame whose z axis is that quantization
        axis; the Hamiltonian is a scalar under this rotation and is used as
        it stands. A file whose sites are labelled along different axes has
        no such frame and is refused: decomposing it would mix tensors
        written in different quantization frames, and the result would be
        silently wrong. Kuiva can write the model with a common axis
        (common_axis=..., rotate_frame=True) when this happens.

        The phases of the states of the file are arbitrary, as Kuiva
        canonicalizes none of them, so the phase correction and the check of
        the behavior under time reversal are applied exactly as they are for
        a system built by the constructor.

        Arguments
        ---------
        filename : str
            Name of the Kuiva pseudospin file.
        units : EnergyUnitSystem
            The unit system.

        Optional arguments
        ------------------
        include_bohr_magneton : boolean
            Whether the magnetic moment operators read from the file are
            multiplied by the Bohr magneton of the unit system (so that they
            are in units of energy per tesla, which is the convention used
            by the property classes of the library). Default is True.
        print_output : boolean
            Whether to print output. The default is False.
        """
        from ouluspin.qc import kuiva

        calculation = kuiva.KuivaPseudospinFile(filename,units)

        basis = calculation.pseudospin_basis()

        # The axis the states of the file are labelled by M along, which is
        # the axis the pseudospin of the system is quantized along.
        axis = calculation.common_axis()

        if axis is None:
            instance = cls.__new__(cls)
            instance.__error("The sites of the file " + filename + " are labelled\n"
                             "along different axes, so the model has no single "
                             "quantization\naxis and cannot be decomposed into "
                             "pseudospin tensors: the tensors\nof the sites would be "
                             "written in different coordinate frames.\nAsk Kuiva for a "
                             "model written with a common axis (the arguments\n"
                             "common_axis=... and rotate_frame=True).")

        rotation_matrix = cls.__quantization_frame_rotation(axis)

        # The operator matrices over the pseudospin product basis, with the
        # ground state of the Hamiltonian at zero and the Cartesian
        # components of the magnetic moment rotated into the frame of the
        # quantization axis. The Hamiltonian is a scalar in the Cartesian
        # indices and is not affected by the rotation.
        pseudospin_hamiltonian_matrix = calculation.hamiltonian()
        moment_matrix_list = calculation.magnetic_moment(
                                 include_bohr_magneton=include_bohr_magneton)

        pseudospin_magnetic_moment_matrix = list(
            fu.matrix_utils.rotate_vector_operator_matrix(moment_matrix_list,
                                                          rotation_matrix))

        # The ab initio operators in the basis of the eigenstates of the
        # Hamiltonian, which is the form the class documents its hamiltonian
        # and magnetic_moment attributes in. The transformation into that
        # basis is the unitary of the file (its columns are the ab initio
        # states over the product basis) and is not constructed here. The
        # moment matrices are stored in the frame of the file, as the R
        # attribute is the rotation that was applied to them.
        unitary = calculation.unitary()

        hamiltonian = pseudospin_operators\
                      .GeneralOperatorMatrix(np.diag(calculation.relative_energies()
                                                     .astype(np.complex128)),
                                             diagonalize_operator_matrix=True,
                                             translate_eigenvalues=True)
        magnetic_moment = pseudospin_operators\
                          .GeneralVectorOperatorMatrix(
                              [fu.matrix_utils.basis_transformation(unitary,matrix,0)
                               for matrix in moment_matrix_list],
                              diagonalize_operator_matrix=True,
                              translate_eigenvalues=False)

        # The construction of the pseudospin operators of the class
        # constructor is replaced by the data of the file, so the instance is
        # built directly and only the steps that still apply are run on it.
        system = cls.__new__(cls)

        system.hamiltonian     = hamiltonian
        system.magnetic_moment = magnetic_moment
        system.basis           = basis
        system.units           = units
        system.print_output    = print_output

        system.n_basis      = basis.n_basis
        system.n_full_basis = calculation.n_basis

        system.kramers_system = cls.kramers_system_from_basis(basis)

        system.R            = rotation_matrix
        system.tensor_frame = 'quantization axis frame'

        # The states of the file are in the order of PseudoSpinBasis, which
        # the reader checks, so no reordering is applied.
        system.reorder_list = list(range(0,basis.n_basis))

        # The rotation from the ab initio input frame to the frame of the
        # pseudospin operators: first the rotation the file records from the
        # input frame into the frame of its own components, then the rotation
        # from that frame into the frame of the quantization axis.
        system.input_frame_rotation = tensors.Rotation(
            np.dot(rotation_matrix,calculation.frame_rotation))

        system.pseudospin_hamiltonian_matrix     = pseudospin_hamiltonian_matrix
        system.pseudospin_magnetic_moment_matrix = pseudospin_magnetic_moment_matrix

        if print_output:
            print(calculation)
            print("    The pseudospin is quantized along the axis the states of the")
            print("    file are labelled along, which is in the input frame of the")
            print("    calculation the axis")
            print()
            print("      ({0:12.8f},{1:12.8f},{2:12.8f}).".format(
                *system.quantization_axis()))
            print()

        system.__correct_and_check_phases()

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

        # With an explicit R the quantization axis is the one the user
        # chose, i.e. the z axis of the identity frame here. The system is
        # an S = 3/2 ion of an easy-plane zero-field splitting (D > 0), so
        # its ground doublet is |+-1/2>, whose main magnetic axis is
        # transverse; the two axes therefore differ, which is exactly the
        # case the two methods are meant to tell apart.
        check('an explicit R fixes the quantization axis',
              np.allclose(system.quantization_axis(),[0.0,0.0,1.0],atol=1.0e-10))
        check('the ground doublet axis is evaluated independently of R',
              abs(np.dot(system.quantization_axis(),
                         system.ground_doublet_magnetic_axis())) < 1.0e-6)

        # ------------------------------------------------------------------
        # The frame rotation determined by the class itself, i.e. with no R
        # given, evaluated for a system whose easy axis is known but does
        # not lie along a coordinate axis.
        #
        # The Hamiltonian handed to the class here is deliberately NOT
        # diagonal, while the magnetic moment matrices are given in its
        # eigenbasis, which is the documented convention of the class. This
        # is the case that separates the two bases: a construction that
        # transformed the moment matrices by the eigenvectors of the
        # Hamiltonian once more would pair the wrong states into the ground
        # doublet and return an axis that is not the easy axis at all.
        # ------------------------------------------------------------------
        tilt_axis = np.array([1.0,2.0,-2.0])
        tilt_axis = tilt_axis/np.linalg.norm(tilt_axis)

        # The proper rotation carrying the z axis onto the tilted axis
        # (Rodrigues' formula; the two axes are not antiparallel).
        cross_product = np.cross([0.0,0.0,1.0],tilt_axis)
        cosine        = float(np.dot([0.0,0.0,1.0],tilt_axis))
        cross_matrix  = np.array([[0.0,-cross_product[2],cross_product[1]],
                                  [cross_product[2],0.0,-cross_product[0]],
                                  [-cross_product[1],cross_product[0],0.0]])
        tilt_rotation = np.identity(3) + cross_matrix \
                        + np.dot(cross_matrix,cross_matrix)/(1.0 + cosine)

        # An easy-axis S = 3/2 ion (D < 0), whose ground doublet |+-3/2> is
        # quantized along the z axis of the spin, with the magnetic moment
        # rotated so that the easy axis points along the tilted axis in the
        # laboratory frame.
        tilt_hamiltonian_matrix = -abs(D)*np.dot(spin_matrix_list[2],
                                                 spin_matrix_list[2])
        tilt_hamiltonian = pseudospin_operators\
                           .GeneralOperatorMatrix(tilt_hamiltonian_matrix,
                                                  diagonalize_operator_matrix=True,
                                                  translate_eigenvalues=True)

        tilt_moment_list = []
        for alpha in range(0,3):
            laboratory_moment = sum(tilt_rotation[alpha][beta]
                                    *(-g*tmp_units.mu_B*spin_matrix_list[beta])
                                    for beta in range(0,3))
            tilt_moment_list.append(
                np.dot(tilt_hamiltonian.eigenvectors.conj().T,
                       np.dot(laboratory_moment,tilt_hamiltonian.eigenvectors)))

        tilt_system = cls(tilt_hamiltonian,
                          pseudospin_operators
                          .GeneralVectorOperatorMatrix(tilt_moment_list),
                          basis,tmp_units)

        check('the frame rotation is determined from the true ground doublet',
              np.allclose(tilt_system.quantization_axis(),tilt_axis,atol=1.0e-6))
        check('the ground doublet axis agrees with the determined frame',
              np.allclose(tilt_system.ground_doublet_magnetic_axis(),tilt_axis,
                          atol=1.0e-6))

        # The ground doublet of an easy-axis S = 3/2 ion is |+-3/2> with
        # g = (0,0,2*3) = (0,0,6); finding it confirms that the states were
        # paired correctly rather than mixed with the excited doublet.
        tilt_g_values = sorted(abs(value) for value
                               in tilt_system.pseudospin_doublet((0,1))
                                             .g_tensor.eigenvalues)
        check('the ground doublet g values are those of the |+-3/2> doublet',
              np.allclose(tilt_g_values,[0.0,0.0,3.0*g],atol=1.0e-6))
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

        # The grouping of the states of a multiplet into doublets. A
        # Kramers multiplet holds an even number of states and every state
        # belongs to a doublet.
        check('a Kramers multiplet is grouped into doublets',
              system.pseudospin_doublet_index_list(pseudospin)
              == [(0,1),(2,3)])

        # A non-Kramers multiplet is left with one singlet, which is placed
        # where it leaves the smallest splitting within the quasi-doublets.
        # The S = 1 ion of an easy-axis zero-field splitting, i.e. a
        # negative D, has its two lowest states close together and the
        # third one far above, so the singlet is the highest state.
        easy_axis_matrix = -D*np.dot(nk_spin_matrix_list[2],nk_spin_matrix_list[2]) \
                           + E*(np.dot(nk_spin_matrix_list[0],nk_spin_matrix_list[0])
                                - np.dot(nk_spin_matrix_list[1],nk_spin_matrix_list[1]))
        easy_axis_hamiltonian = pseudospin_operators\
                                .GeneralOperatorMatrix(easy_axis_matrix,
                                                       diagonalize_operator_matrix=True,
                                                       translate_eigenvalues=True)
        easy_axis_system = cls(easy_axis_hamiltonian,nk_moment,nk_basis,
                               tmp_units,R=np.identity(3))

        easy_axis_index_list = easy_axis_system\
                               .pseudospin_doublet_index_list(nk_pseudospin)

        check('a non-Kramers multiplet is left with one singlet',
              len([states for states in easy_axis_index_list
                   if len(states) == 1]) == 1)
        check('the singlet of an easy-axis non-Kramers ion is the highest state',
              easy_axis_index_list == [(0,1),(2,)])
        check('the singlet is turned into the energy of the state',
              isinstance(easy_axis_system.pseudospin_doublet_list(
                  easy_axis_index_list)[1],float))

        # The singlet is placed where it costs the least, which is not
        # always at an end of the multiplet. The system below holds five
        # states of the energies 0, 0.1, 5, 10 and 10.1, where the pairs
        # (0,1) and (3,4) leave the smallest total splitting and the
        # singlet is therefore the middle state.
        spaced_energies    = [0.0,0.1,5.0,10.0,10.1]
        spaced_hamiltonian = pseudospin_operators\
                             .GeneralOperatorMatrix(
                                 np.diag(spaced_energies).astype(np.complex128),
                                 diagonalize_operator_matrix=True,
                                 translate_eigenvalues=True)
        spaced_spin_matrix_list = debug_output.spin_matrices(4)
        spaced_moment = pseudospin_operators\
                        .GeneralVectorOperatorMatrix(
                            [-g*tmp_units.mu_B*spin_matrix
                             for spin_matrix in spaced_spin_matrix_list])
        spaced_system = cls(spaced_hamiltonian,spaced_moment,
                            pseudospin_operators.PseudoSpinBasis([4]),
                            tmp_units,R=np.identity(3))

        check('the singlet is placed where it costs the least',
              spaced_system.pseudospin_doublet_index_list(4)
              == [(0,1),(2,),(3,4)])

        # A multiplet that does not fit into the basis of the system is an
        # error, so the largest multiplet the basis holds is the one of the
        # basis itself.
        check('the whole basis can be grouped',
              len(system.pseudospin_doublet_index_list(pseudospin))
              == (pseudospin + 1)//2)

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

            # The principal magnetic axis of the ground doublet, reported in
            # the input frame. System A is axial with an easy axis along the
            # z axis of its own input frame, so the ground doublet |+-3/2>
            # is quantized along z.
            axis_a = system_int.ground_doublet_magnetic_axis()

            check('the ground doublet magnetic axis is a unit vector',
                  abs(np.linalg.norm(axis_a) - 1.0) < 1.0e-10)
            check('the ground doublet magnetic axis of an axial system is z',
                  np.allclose(axis_a,[0.0,0.0,1.0],atol=1.0e-8))

            # With no explicit R the quantization axis is chosen as the
            # principal magnetic axis of the ground doublet, so the two
            # methods must return the same vector.
            check('the quantization axis is a unit vector',
                  abs(np.linalg.norm(system_int.quantization_axis()) - 1.0)
                  < 1.0e-10)
            check('without R the quantization axis is the ground doublet axis',
                  np.allclose(system_int.quantization_axis(),axis_a,
                              atol=1.0e-8))

            # System G carries the same physics rotated by 90 degrees about
            # y, i.e. its magnetic moment along the laboratory x axis is the
            # one that was along z (mu_x = -g*S_z), so its easy axis lies
            # along the x axis of its input frame. The axis is therefore
            # read from the operators themselves and not from the frame the
            # tensors happen to be written in.
            system_g = cls.from_aniso_data(filename_g,3,tmp_units)
            axis_g   = system_g.ground_doublet_magnetic_axis()

            check('the ground doublet magnetic axis follows the input frame',
                  np.allclose(axis_g,[1.0,0.0,0.0],atol=1.0e-6))
            check('the quantization axis of a tilted system follows it as well',
                  np.allclose(system_g.quantization_axis(),axis_g,atol=1.0e-6))

            # The axis must agree with the main principal axis of the
            # g-tensor of the ground doublet expressed in the input frame,
            # which is the same quantity evaluated the long way round.
            ground_doublet = system_g.pseudospin_doublet((0,1))
            doublet_g      = np.array(ground_doublet.input_frame_g_tensor
                                                    .eigenvalues)
            doublet_axis   = np.array(ground_doublet.input_frame_g_tensor
                                                    .eigenvectors)\
                             [:,int(np.argmax(doublet_g))]

            check('the ground doublet magnetic axis matches the doublet g-tensor',
                  abs(abs(np.dot(axis_g,doublet_axis)) - 1.0) < 1.0e-6)

            # The sign of a directionless axis is fixed by the convention
            # that the first component that is not numerically zero is
            # positive, so that the same system always gives the same
            # vector.
            first_nonzero = [component for component in axis_g
                             if abs(component) > 1.0e-8][0]
            check('the sign of the magnetic axis follows the convention',
                  first_nonzero > 0.0)

            # The convention must not depend on which of two components of
            # an equal magnitude the noise of the last digits makes the
            # larger. The axis (1,1,-1)/sqrt(3) of a system of a high
            # symmetry is the case in point: it and its negative must give
            # the same vector.
            tie_axis = np.array([1.0,1.0,-1.0])/np.sqrt(3.0)
            oriented = system_int._PseudoSpinSystem__oriented_unit_vector

            check('the sign convention is stable against tied components',
                  np.allclose(oriented(tie_axis,'axis'),
                              oriented(-tie_axis,'axis')))
            check('the sign convention normalizes the vector',
                  abs(np.linalg.norm(oriented(2.5*tie_axis,'axis')) - 1.0)
                  < 1.0e-12)

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

            # The static magnetic properties, through the method inherited
            # from PseudoSpinSystem. System A is an easy-axis S = 3/2 ion
            # of the gap 2|D_f|; at a temperature well below the gap only
            # the |+-3/2> doublet is populated, so the powder chi*T product
            # approaches that of an Ising doublet of g = (0,0,2*g*3/2).
            # The powder chi is the mean of the three principal ones, each
            # proportional to the square of its g value, so the product of
            # an effective S = 1/2 doublet is S(S+1)/24*sum(g_i^2), i.e.
            # sum(g_i^2)/32, which is 36/32 = 1.125 cm^3 K mol^-1 here and
            # the familiar 0.375 for an isotropic g = 2. At a temperature
            # well above the gap the whole S = 3/2 multiplet is populated
            # and the product approaches the spin-only value
            # g^2*S*(S+1)/8 = 1.875 cm^3 K mol^-1.
            from ouluspin import integration

            spin_only = g**2*(3.0/2.0)*(3.0/2.0 + 1.0)/8.0
            ising     = (2.0*g*3.0/2.0)**2/32.0

            aniso_susceptibility = properties.StaticMagneticSusceptibility(
                [0.5,3000.0])
            aniso_properties = system_int.static_magnetic_properties(
                integration.LebedevLaikovGrid(7),
                susceptibility=aniso_susceptibility)
            aniso_properties.calculate_susceptibility()

            chi_T = aniso_properties.susceptibility.susceptibility

            # The low-temperature value approaches the Ising limit but does
            # not reach it exactly: the susceptibility is evaluated from
            # the magnetization in a small but finite field, and a moment
            # of g_z = 6 is already slightly saturated by it at this
            # temperature, while the mixing with the excited doublet adds a
            # small temperature-independent (van Vleck) contribution. The
            # tolerance is set to accommodate both; it is still far tighter
            # than the difference to any other doublet the system could
            # have (an isotropic one would give 0.375).
            check('static_magnetic_properties reaches the Ising limit at low T',
                  abs(chi_T[0] - ising) < 0.05)
            check('static_magnetic_properties reaches the spin-only limit at high T',
                  abs(chi_T[1] - spin_only) < 5.0e-3)

            # The properties are evaluated by integrating over the
            # orientations of the molecule, so they must not depend on the
            # frame the operators are expressed in. System G carries the
            # same physics as A rotated by 90 degrees about y, and the
            # quantization axes of the two differ accordingly.
            rotated_susceptibility = properties.StaticMagneticSusceptibility(
                [0.5,3000.0])
            rotated_properties = cls.from_aniso_data(filename_a,3,tmp_units,
                                                     R=np.identity(3))\
                                    .static_magnetic_properties(
                                        integration.LebedevLaikovGrid(7),
                                        susceptibility=rotated_susceptibility)
            rotated_properties.calculate_susceptibility()

            check('the magnetic properties do not depend on the frame',
                  np.allclose(rotated_properties.susceptibility.susceptibility,
                              chi_T,atol=1.0e-6))

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

            # The doublet list can be constructed once and given to the
            # tabulation methods, which must then not construct the
            # doublets again.
            doublet_list = system_int.pseudospin_doublet_list([(0,1),(2,3)])
            check('pseudospin_doublet_list constructs the doublets',
                  (len(doublet_list) == 2)
                  and all(isinstance(item,properties.PseudoSpinDoublet)
                          for item in doublet_list))
            check('pseudospin_doublet_list passes constructed doublets through',
                  all(new is old for new, old in
                      zip(system_int.pseudospin_doublet_list(doublet_list),
                          doublet_list)))
            check('pseudospin_doublet_table accepts constructed doublets',
                  system_int.pseudospin_doublet_table(doublet_list)
                  .count('DOUBLET (') == 2)

            # The compound summary table: one line per doublet, with the
            # excited doublet at the energy of the zero-field splitting.
            # The ground doublet |+-3/2> is Ising-like with gz = 6 along
            # z, whereas the excited doublet |+-1/2> has g = (2,4,4), so
            # its largest principal g value, and hence its principal
            # magnetic axis, is transverse and perpendicular to that of
            # the ground doublet.
            summary_table = system_int.pseudospin_doublet_summary_table(doublet_list)
            check('pseudospin_doublet_summary_table has one row per doublet',
                  len(summary_table.rows) == 2)
            check('pseudospin_doublet_summary_table gives the doublet energies',
                  (abs(summary_table.rows[0][1]) < 1.0e-8)
                  and (abs(summary_table.rows[1][1] - 2.0*abs(D_f)) < 1.0e-8))
            check('pseudospin_doublet_summary_table gives the g values',
                  np.allclose(sorted(summary_table.rows[0][2:5]),
                              [0.0,0.0,6.0],atol=1.0e-8))
            check('pseudospin_doublet_summary_table gives the axis angles',
                  (abs(summary_table.rows[0][5]) < 1.0e-6)
                  and (abs(summary_table.rows[1][5] - 90.0) < 1.0e-6))
            check('pseudospin_doublet_summary_table does not list the states',
                  not "States" in summary_table.string_table())

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

            # The operators of an averaged system were already rotated
            # before the construction, so its R attribute is the identity
            # and the axis has to be read from input_frame_rotation. The
            # axis must still be the main magnetic axis of the ground
            # doublet in the laboratory frame.
            average_axis = average_fg.ground_doublet_magnetic_axis()

            check('the magnetic axis of an averaged system is in the input frame',
                  abs(abs(np.dot(average_axis,reference_axis)) - 1.0) < 1.0e-6)
            check('the quantization axis of an averaged system follows it',
                  np.allclose(average_fg.quantization_axis(),average_axis,
                              atol=1.0e-6))

            # --------------------------------------------------------------
            # The from_kuiva_data class method, tested with synthetic Kuiva
            # pseudospin files. Everything the constructor determines for
            # itself comes from the file here: the basis, the transformation
            # into it and the quantization axis. The phases of the basis
            # states of every file are scrambled with random phase factors,
            # as a Kuiva file canonicalizes none of them, so that the phase
            # correction and the check of time reversal are exercised.
            # --------------------------------------------------------------
            def kuiva_file_content(hamiltonian_matrix,moment_list,
                                   pseudospin_list,axis,
                                   frame_rotation=np.identity(3)):
                """Synthetic Kuiva pseudospin file of the system given by its
                Hamiltonian (in the energy unit of the tests) and by its
                magnetic moment matrices (in units of the Bohr magneton),
                both written over the pseudospin product basis of the
                pseudospins given as an argument. The states of every site
                are labelled along the axis given as an argument, and the
                file records the given rotation from the ab initio input
                frame into the frame of its own components. The phases of
                the basis states are scrambled."""
                file_basis = pseudospin_operators.PseudoSpinBasis(pseudospin_list)
                dimension  = file_basis.n_basis

                phase_matrix = np.diag(np.exp(2.0j*np.pi*rng.random(dimension)))

                scrambled_hamiltonian = np.dot(phase_matrix.conj().T,
                                               np.dot(hamiltonian_matrix/hartree,
                                                      phase_matrix))
                scrambled_moment_list = [np.dot(phase_matrix.conj().T,
                                                np.dot(matrix,phase_matrix))
                                         for matrix in moment_list]

                energy_list, state_matrix = np.linalg.eigh(scrambled_hamiltonian)

                def matrix_block(tag,matrix):
                    block_str  = "[" + tag + "]\n"
                    block_str += "shape  {0} {1}\n".format(matrix.shape[0],
                                                           matrix.shape[1])
                    block_str += "unit   arbitrary\n"
                    for i in range(0,matrix.shape[0]):
                        for j in range(0,matrix.shape[1]):
                            block_str += "{0:6d} {1:6d} {2:+.16e} {3:+.16e}\n".format(
                                i,j,matrix[i][j].real,matrix[i][j].imag)
                    block_str += "[END]\n\n"
                    return block_str

                content  = "[HEADER]\n"
                content += "format          KUIVA_PSEUDOSPIN\n"
                content += "format_version  1\n"
                content += "n_sites         {0}\n".format(file_basis.n_sites)
                content += "model_dim       {0}\n".format(dimension)
                content += "energy_unit     Eh\n"
                content += "moment_unit     mu_B\n"
                content += "energy_shift    +0.0000000000000000e+00\n"
                content += "basis_order     site 0 slowest; within a site M ascending\n"
                content += "frame           input frame\n"
                content += "[END]\n\n"
                content += "[PROVENANCE]\n{\"code\": \"kuiva\"}\n[END]\n\n"
                content += "[FRAME]\n"
                for row in np.array(frame_rotation, dtype=np.float64):
                    content += "  {0:+.14f} {1:+.14f} {2:+.14f}\n".format(*row)
                content += "[END]\n\n"
                content += "[SITES]\n"
                for site in range(0,file_basis.n_sites):
                    content += ("{0:6d} {1:4d} {2:4d}"
                                " {3:+.10f} {4:+.10f} {5:+.10f}"
                                "  principal_magnetic_axis | 1 | 0\n").format(
                                    site,pseudospin_list[site],
                                    pseudospin_list[site] + 1,*axis)
                content += "[END]\n\n"
                content += "[BASIS]\n"
                for i in range(0,dimension):
                    content += "{0:6d}  ".format(i)
                    content += " ".join("{0:+d}".format(file_basis
                                                        .basis_state_list[i][site][1])
                                        for site in range(0,file_basis.n_sites))
                    content += "\n"
                content += "[END]\n\n"
                content += "[ENERGIES]\n"
                for i, value in enumerate(energy_list):
                    content += "{0:6d}  {1:+.16e}  {2:+.8e}\n".format(
                        i,value,(value - energy_list[0])*hartree)
                content += "[END]\n\n"
                content += matrix_block('MATRIX H',scrambled_hamiltonian)
                content += matrix_block('MATRIX mu_x',scrambled_moment_list[0])
                content += matrix_block('MATRIX mu_y',scrambled_moment_list[1])
                content += matrix_block('MATRIX mu_z',scrambled_moment_list[2])
                content += matrix_block('MATRIX U',state_matrix)

                return content

            # A single S = 3/2 site with the rhombic zero-field splitting
            # D*S_z^2 + E*(S_x^2 - S_y^2), the states labelled along the z
            # axis of the file, which is the frame of the ab initio
            # calculation as well.
            kuiva_hamiltonian_matrix = D*np.dot(Sz,Sz) \
                                       + E*(np.dot(Sx,Sx) - np.dot(Sy,Sy))
            kuiva_moment_list = [-g*Sx,-g*Sy,-g*Sz]

            filename_kuiva = os.path.join(tmp_dir,'single_site.psd')
            with open(filename_kuiva,'w') as f:
                f.write(kuiva_file_content(kuiva_hamiltonian_matrix,
                                           kuiva_moment_list,[3],
                                           [0.0,0.0,1.0]))

            kuiva_system = cls.from_kuiva_data(filename_kuiva,tmp_units)

            check('from_kuiva_data takes the basis from the file',
                  kuiva_system.basis.pseudospin_list == [3]
                  and kuiva_system.basis.n_basis == 4
                  and kuiva_system.n_basis == 4
                  and kuiva_system.n_full_basis == 4)
            check('from_kuiva_data does not reorder the states of the file',
                  kuiva_system.reorder_list == [0,1,2,3])
            check('from_kuiva_data tensors carry the quantization frame label',
                  kuiva_system.tensor_frame == 'quantization axis frame'
                  and kuiva_system.hamiltonian_tensor().frame
                      == 'quantization axis frame')
            check('from_kuiva_data quantizes along the axis of the file',
                  np.allclose(kuiva_system.quantization_axis(),[0.0,0.0,1.0],
                              atol=1.0e-10))
            check('from_kuiva_data reproduces the spectrum of the file',
                  np.allclose(np.linalg.eigvalsh(kuiva_system
                                                 .pseudospin_hamiltonian_matrix),
                              [0.0,0.0,gap,gap]))

            # The phases of the basis states of the file are arbitrary and
            # must be corrected: the pseudospin Hamiltonian is then even
            # under time reversal, its ITO decomposition has no odd ranks,
            # and the matrix itself reproduces the input Hamiltonian up to
            # the constant shift of the eigenvalues.
            U = kuiva_system.basis.unitary_part_of_time_reversal_operator()
            H_ps = kuiva_system.pseudospin_hamiltonian_matrix
            check('from_kuiva_data corrects the phases of the basis states',
                  np.allclose(np.dot(U,np.dot(H_ps.conj(),U.T)),H_ps,atol=1.0e-8))

            kuiva_tensor = kuiva_system.hamiltonian_tensor()
            odd_rank_maximum = max(abs(parameter)
                                   for rank,parameter in zip(kuiva_tensor.rank_list,
                                                             kuiva_tensor.parameter_list)
                                   if (rank[0]//2) % 2 == 1)
            check('no odd ranks in the Hamiltonian tensor of a Kuiva system',
                  odd_rank_maximum < 1.0e-8)

            shift = (np.trace(H_ps).real
                     - np.trace(kuiva_hamiltonian_matrix).real)/4.0
            check('from_kuiva_data reproduces the input Hamiltonian matrix',
                  np.allclose(H_ps - shift*np.identity(4),
                              kuiva_hamiltonian_matrix,atol=1.0e-8))

            # The magnetic moment of the file is in units of the Bohr
            # magneton and the Bohr magneton is multiplied in by default.
            plain_system = cls.from_kuiva_data(filename_kuiva,tmp_units,
                                               include_bohr_magneton=False)
            check('from_kuiva_data includes the Bohr magneton by default',
                  np.allclose(kuiva_system.pseudospin_magnetic_moment_matrix[2],
                              tmp_units.mu_B
                              *plain_system.pseudospin_magnetic_moment_matrix[2]))

            # The same physics read from a Kuiva file and projected by the
            # constructor itself must give the same ground doublet: the
            # system built at the beginning of these tests carries the same
            # zero-field splitting and the same magnetic moment, quantized
            # along the same axis.
            check('from_kuiva_data ground doublet matches the projected system',
                  np.allclose(np.sort(np.abs(kuiva_system.pseudospin_doublet((0,1))
                                             .g_tensor.eigenvalues)),
                              np.sort(np.abs(system.pseudospin_doublet((0,1))
                                             .g_tensor.eigenvalues)),
                              atol=1.0e-6))

            # The same system with the states labelled along a tilted axis
            # and the Cartesian components of the magnetic moment written in
            # the frame of the file, i.e. the case a real calculation of a
            # molecule of no symmetry produces. The zero-field splitting is
            # purely axial here, as the phase correction fixes the
            # transverse axes only up to a rotation about the quantization
            # axis, under which the rhombic term is not invariant.
            axial_hamiltonian_matrix = D*np.dot(Sz,Sz)
            tilted_moment_list = []
            for alpha in range(0,3):
                tilted_moment_list.append(sum(tilt_rotation[alpha][beta]
                                              *kuiva_moment_list[beta]
                                              for beta in range(0,3)))

            filename_tilted = os.path.join(tmp_dir,'tilted.psd')
            with open(filename_tilted,'w') as f:
                f.write(kuiva_file_content(axial_hamiltonian_matrix,
                                           tilted_moment_list,[3],tilt_axis))

            tilted_system = cls.from_kuiva_data(filename_tilted,tmp_units)
            tilted_tensor = tilted_system.hamiltonian_tensor()
            tilted_tensor.purge_ranks()

            check('from_kuiva_data follows a tilted labelling axis',
                  np.allclose(tilted_system.quantization_axis(),tilt_axis,
                              atol=1.0e-8))
            check('from_kuiva_data recovers the axial ZFS of a tilted system',
                  ([4,0] in tilted_tensor.rank_list)
                  and abs(tilted_tensor.parameter_list[tilted_tensor.rank_list
                                                       .index([4,0])] - D) < 1.0e-8)

            # A file written in a frame of its own: the states are labelled
            # along the z axis of the file, but the file records a rotation
            # from the ab initio input frame into its frame. The quantization
            # axis must then be reported in the input frame, i.e. as the
            # tilted axis again.
            filename_rotated = os.path.join(tmp_dir,'rotated_frame.psd')
            with open(filename_rotated,'w') as f:
                f.write(kuiva_file_content(axial_hamiltonian_matrix,
                                           kuiva_moment_list,[3],[0.0,0.0,1.0],
                                           frame_rotation=tilt_rotation.T))

            rotated_system = cls.from_kuiva_data(filename_rotated,tmp_units)

            check('from_kuiva_data composes the frame rotation of the file',
                  np.allclose(rotated_system.quantization_axis(),tilt_axis,
                              atol=1.0e-8))
            check('the frame of the file does not change the tensors',
                  np.allclose(rotated_system.pseudospin_hamiltonian_matrix,
                              tilted_system.pseudospin_hamiltonian_matrix,
                              atol=1.0e-8))

            # A genuine two-site file: the S = 3/2 ion exchange-coupled to
            # the S = 1/2 radical of the two-site test above. The product
            # basis of the file is the coupled basis, which exercises the
            # time-reversal-based phase correction of the second half of the
            # basis on data that come from a file.
            two_site_moment_list = []
            for alpha in range(0,3):
                two_site_moment_list.append(
                    -(g*np.kron(spin_matrix_list[alpha],radical_identity)
                      + g_radical*np.kron(ion_identity,radical_matrix_list[alpha])))

            filename_two_site = os.path.join(tmp_dir,'two_site.psd')
            with open(filename_two_site,'w') as f:
                f.write(kuiva_file_content(two_site_hamiltonian_matrix,
                                           two_site_moment_list,pseudospin_list,
                                           [0.0,0.0,1.0]))

            two_site_kuiva = cls.from_kuiva_data(filename_two_site,tmp_units)

            check('from_kuiva_data takes a multi-site basis from the file',
                  two_site_kuiva.basis.pseudospin_list == pseudospin_list
                  and two_site_kuiva.basis.n_sites == 2)

            H_ps = two_site_kuiva.pseudospin_hamiltonian_matrix
            U = two_site_kuiva.basis.unitary_part_of_time_reversal_operator()
            check('two-site Kuiva Hamiltonian is even under time reversal',
                  np.allclose(np.dot(U,np.dot(H_ps.conj(),U.T)),H_ps,atol=1.0e-8))

            shift = (np.trace(H_ps).real
                     - np.trace(two_site_hamiltonian_matrix).real)/n_two_site
            check('from_kuiva_data reproduces the two-site input Hamiltonian',
                  np.allclose(H_ps - shift*np.identity(n_two_site),
                              two_site_hamiltonian_matrix,atol=1.0e-8))

            two_site_kuiva_tensor = two_site_kuiva.hamiltonian_tensor()
            exchange_ok = True
            for rank,parameter in zip(reference_tensor.rank_list,
                                      reference_tensor.parameter_list):
                i = two_site_kuiva_tensor.rank_list.index(rank)
                if abs(two_site_kuiva_tensor.parameter_list[i] - parameter) > 1.0e-8:
                    exchange_ok = False
            check('isotropic exchange parameters recovered from a Kuiva file',
                  exchange_ok)

        return debug_output.test_summary('AbInitioElectronExchangeSystem',
                                         result_list,print_output)
            




        
