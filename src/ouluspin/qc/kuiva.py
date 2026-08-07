# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import json

import numpy as np

from ouluspin import pseudospin_operators


class KuivaPseudospinFile:
    """A class to read operator matrices and the pseudospin basis from a pseudospin
    file (.psd) produced by a Kuiva calculation.

    Kuiva writes its multi-site local-multiplet models into a plain-text
    file that, unlike an Orca .anisofile, already identifies the matrices
    with a specific pseudospin product basis: the file carries the
    pseudospin of each site, the ordered

        |S0,M0> (x) |S1,M1> (x) ...

    basis listing, the effective Hamiltonian and the three Cartesian
    components of the magnetic moment over that basis, and the unitary that
    maps the ab initio states (the eigenstates of the Hamiltonian) to it.
    The basis is stored in exactly the order in which PseudoSpinBasis
    constructs it (site 0 slowest; within a site M = -S..+S ascending),
    which this class verifies upon initiation, so that the matrices can be
    used with the pseudospin machinery of the library directly, with no
    reordering of the states.

    NOTE!!! All angular momentum quantum numbers are given as multiplets of two in order to treat
            half-integer angular momenta as integer numbers.

    The conventions of the file are the following.
      - The energies and the Hamiltonian are stored in hartree and are
        converted to the energy unit of the unit system on reading.
      - The magnetic moment matrices are stored in units of the Bohr
        magneton, and the Bohr magneton of the unit system is multiplied in
        on request, following OrcaAnisoFile.
      - The Hamiltonian is NOT diagonal: it is the effective Hamiltonian
        over the pseudospin product basis. Its eigenvalues are listed in the
        [ENERGIES] section of the file and the diagonalizing unitary in the
        [MATRIX U] section (the columns of the unitary are the ab initio
        states in the product basis).
      - The energies of the file are the energies of the ab initio model
        without the energy shift (e.g. the inactive-core energy) recorded in
        the header. The Hamiltonian returned by the hamiltonian method is by
        default translated so that the ground state is at zero, which is the
        form the rest of the library works with and the form OrcaAnisoFile
        returns; the untranslated matrix is available on request.
      - The phases of the states are arbitrary, as the file canonicalizes
        none of them. The time-reversal-proper phase fixing of the library
        applies to these matrices unchanged.
      - The axis along which the states of each site are labelled by M is
        recorded for every site, and the frame of the Cartesian components
        together with the rotation from the ab initio input frame into it
        are recorded in the header and in the [FRAME] section. A file
        written in the quantization-axis frame has its z axis along the
        common labelling axis of the sites, which is the frame the
        pseudospin operators of the library are conventionally written in.

    Arguments
    ---------
    filename : str
        Name of the Kuiva pseudospin file.
    units : EnergyUnitSystem
        The energy unit system.

    Optional arguments
    ------------------

    Attributes
    ----------
    filename : str
        Name of the Kuiva pseudospin file.
    units : EnergyUnitSystem
        The energy unit system.
    version : int
        Format version of the file.
    header : dict
        The whole header of the file as it was read, the keywords as keys
        and the rest of each line as a string value.
    n_sites : int
        The number of pseudospin sites.
    n_basis : int
        The dimension of the pseudospin product basis, i.e. of every matrix
        of the file.
    pseudospin_list : list of int
        The pseudospins of the sites as multiplets of two, in the order of
        the sites. This is the list PseudoSpinBasis takes as an argument.
    site_list : list of dict
        The data of each site, with the keys 'index', 'twice_s', 'dim',
        'axis' (array of float64), 'axis_choice' (str), 'n_electrons' (int
        or None) and 'orbitals' (tuple of int).
    frame : str
        The label of the coordinate frame of the Cartesian components, e.g.
        'input frame' or a quantization-axis frame.
    frame_rotation : array of float64
        The 3 by 3 rotation matrix from the ab initio input frame to the
        frame of the stored Cartesian components, the identity matrix when
        nothing was rotated.
    basis_order : str
        The description of the ordering of the basis states given in the
        header of the file, or an empty string when the file gives none.
    m_convention : str
        The description of the convention by which the states of a site are
        labelled by M, given in the header of the file, or an empty string
        when the file gives none.
    phase_convention : str
        The description of the phase convention given in the header of the
        file, or an empty string when the file gives none.
    energy_shift : float
        The energy shift (e.g. the inactive-core energy) that is NOT
        included in the stored energies, in hartree as it was read. It is
        added to the energies to obtain total energies.
    provenance : dict
        The provenance block of the file, i.e. the record of the Hamiltonian
        the numbers were produced with (the screening and the decoupling of
        the Kuiva calculation), the active space, the system and the
        generator.

    Private methods
    ---------------
    __error(message)
        Report an error and stop.
    __warning(message)
        Report a warning and continue.
    __parse()
        Parse the whole file once and store the contents of its sections as
        attributes.
    __check_basis_order()
        Check that the basis listing of the file is exactly the order in
        which PseudoSpinBasis constructs the basis of the pseudospins of the
        file.
    __check_matrices()
        Check the internal consistency of the matrices of the file, i.e. the
        Hermiticity of the Hamiltonian and of the magnetic moment, the
        unitarity of the state transformation and the trace of the
        Hamiltonian against the listed eigenvalues.

    Public methods
    --------------
    hamiltonian(subtract_ground_energy=True) : array of complex128
        A matrix representation of the effective Hamiltonian over the
        pseudospin product basis. The matrix is NOT diagonal; see the
        description of the conventions above.
    energies(include_shift=False) : array of float64
        The eigenvalues of the effective Hamiltonian in ascending order. The
        optional argument is used to set whether the energy shift of the
        file is added to them, i.e. whether total energies are returned.
    relative_energies() : array of float64
        The eigenvalues of the effective Hamiltonian relative to the lowest
        one, i.e. the excitation energies of the states.
    magnetic_moment(include_bohr_magneton=True) : list of array of complex128
        A list of the matrix representations of the three Cartesian
        components of the magnetic moment over the pseudospin product basis.
        The optional argument is used to set whether the Bohr magneton is
        included in the operator matrices or not.
    unitary() : array of complex128
        The unitary matrix that maps the ab initio states to the pseudospin
        product basis, the columns being the ab initio states in the product
        basis.
    pseudospin_basis() : PseudoSpinBasis
        The basis of the file as an instance of PseudoSpinBasis. The
        identity of the listing of the file with the ordering of the
        instance was checked upon initiation, so that the matrices of this
        class can be used with the returned basis directly.
    site_moment(site,include_bohr_magneton=True) : list of array of complex128
        A list of the matrix representations of the three Cartesian
        components of the magnetic moment of a single site projected onto
        the local states of that site, in the order in which the local
        states are labelled by M.
    common_axis(tolerance=1.0e-3) : array of float64 or None
        The common axis along which the states of every site are labelled by
        M, when the sites share one within the given tolerance, and None
        otherwise.

    Class methods
    -------------
    run_tests(print_output=True) : boolean
        Initiate the class using a synthetic pseudospin file and run a set
        of internal tests. Return True if all tests passed.
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


    def __parse(self):
        """Parse the whole file once and store the contents of its sections
        as attributes.

        The format is line oriented: lines starting with a hash are
        comments, the sections are opened by a [SECTION] marker and closed
        by an [END] marker, and the elements of the matrices are given one
        per line as the record 'i j Re Im'.
        """
        header = {}
        provenance_lines = []
        frame_rows = []
        site_list = []
        basis_rows = []
        energy_list = []
        matrix_dict = {}
        site_matrix_dict = {}

        section = None
        key = None
        current_matrix = None

        f = open(self.filename)

        for raw_line in f:
            line = raw_line.rstrip()

            if line.strip() == '' or line.lstrip().startswith('#'):
                continue

            # A section marker either opens a section, in which case the
            # tags of the matrix sections carry the name of the matrix, or
            # closes the one that is open.
            if line.lstrip().startswith('['):
                tag = line.strip()[1:-1]

                if tag == 'END':
                    if section == 'PROVENANCE':
                        try:
                            self.provenance = json.loads("\n".join(provenance_lines))
                        except json.JSONDecodeError:
                            self.__error("The [PROVENANCE] section of the file "
                                         + self.filename + "\ndoes not contain valid JSON.")
                    section = None
                    key = None
                    current_matrix = None
                    continue

                field_list = tag.split()
                section = field_list[0]

                if section == 'MATRIX':
                    key = field_list[1]
                elif section == 'SITE_MATRIX':
                    key = (int(field_list[1]),field_list[2])

                continue

            if section == 'HEADER':
                keyword, _, value = line.strip().partition(' ')
                header[keyword] = value.strip()

            elif section == 'PROVENANCE':
                provenance_lines.append(line)

            elif section == 'FRAME':
                frame_rows.append([float(x) for x in line.split()])

            elif section == 'SITES':
                # The fields of a site line are separated into three groups
                # by vertical bars: the site itself and the labelling axis,
                # the number of electrons and the orbitals of the site.
                head, _, tail = line.partition('|')
                field_list = head.split()
                electron_str = tail.partition('|')[0].strip()
                orbital_str = tail.partition('|')[2].strip()

                site_list.append({
                    'index': int(field_list[0]),
                    'twice_s': int(field_list[1]),
                    'dim': int(field_list[2]),
                    'axis': np.array([float(x) for x in field_list[3:6]],
                                     dtype=np.float64),
                    'axis_choice': field_list[6].replace('_',' '),
                    'n_electrons': None if electron_str == '?' else int(electron_str),
                    'orbitals': tuple(int(x) for x in orbital_str.split())})

            elif section == 'BASIS':
                field_list = line.split()
                basis_rows.append([int(x) for x in field_list[1:]])

            elif section == 'ENERGIES':
                energy_list.append(float(line.split()[1]))

            elif section == 'MATRIX' or section == 'SITE_MATRIX':
                field_list = line.split()

                if field_list[0] == 'shape':
                    current_matrix = np.zeros((int(field_list[1]),int(field_list[2])),
                                              dtype=np.complex128)
                    if section == 'MATRIX':
                        matrix_dict[key] = current_matrix
                    else:
                        site_matrix_dict[key] = current_matrix

                elif field_list[0] == 'unit':
                    continue

                elif current_matrix is None:
                    self.__error("An element record precedes the shape line of the matrix "
                                 + str(key) + "\nin the file " + self.filename
                                 + "; the file is malformed.")

                else:
                    i = int(field_list[0])
                    j = int(field_list[1])
                    current_matrix[i][j] = complex(float(field_list[2]),float(field_list[3]))

        f.close()

        # The format and the version of the format are checked before
        # anything is read from the header, so that a file of an unknown
        # version is refused rather than misinterpreted.
        if not header.get('format','') == 'KUIVA_PSEUDOSPIN':
            self.__error("The file " + self.filename + " does not declare the\n"
                         "KUIVA_PSEUDOSPIN format.")

        try:
            self.version = int(header.get('format_version','-1'))
        except ValueError:
            self.version = -1

        # A file of an unknown version is refused rather than
        # misinterpreted, which is what the version of the format exists for.
        allowed_versions = [1]

        if not self.version in allowed_versions:
            self.__error("Unsupported version of the KUIVA_PSEUDOSPIN format: "
                         + header.get('format_version','none') + ".\n"
                         "This reader knows the versions "
                         + str(allowed_versions) + " and refuses to guess.")

        for keyword in ('n_sites','model_dim'):
            if not keyword in header:
                self.__error("The header of the file " + self.filename
                             + " does not\ncontain the keyword " + keyword + ".")

        self.header = header
        self.n_sites = int(header['n_sites'])
        self.n_basis = int(header['model_dim'])
        self.energy_shift = float(header.get('energy_shift','0.0'))
        self.frame = header.get('frame','input frame')
        self.basis_order = header.get('basis_order','')
        self.m_convention = header.get('m_convention','')
        self.phase_convention = header.get('phase_convention','')
        self.site_list = site_list
        self.pseudospin_list = [site['twice_s'] for site in site_list]

        self.__basis_rows = basis_rows
        self.__energy_array = np.array(energy_list, dtype=np.float64)
        self.__matrix_dict = matrix_dict
        self.__site_matrix_dict = site_matrix_dict

        if len(frame_rows) == 3:
            self.frame_rotation = np.array(frame_rows, dtype=np.float64)
        else:
            self.frame_rotation = np.identity(3, dtype=np.float64)

        # The units of the stored quantities are fixed by the format, and a
        # file that declares any other units is not read, as the conversions
        # of the class assume these.
        energy_unit = header.get('energy_unit','Eh')
        moment_unit = header.get('moment_unit','mu_B')

        if not energy_unit == 'Eh':
            self.__error("The file " + self.filename + " stores its energies in the\n"
                         "unit " + energy_unit + "; this reader expects them in hartree (Eh).")

        if not moment_unit == 'mu_B':
            self.__error("The file " + self.filename + " stores its magnetic moments in\n"
                         "the unit " + moment_unit + "; this reader expects them in Bohr\n"
                         "magnetons (mu_B).")

        # The dimensions of everything that was read are checked against the
        # header, so that a truncated or an inconsistent file is caught here
        # rather than in the middle of an analysis.
        if not len(site_list) == self.n_sites:
            self.__error("The [SITES] section of the file " + self.filename + " lists "
                         + str(len(site_list)) + " sites\nbut the header declares "
                         + str(self.n_sites) + ".")

        n_expected = 1
        for site in site_list:
            if not site['dim'] == site['twice_s'] + 1:
                self.__error("Site " + str(site['index']) + " of the file " + self.filename
                             + " has the pseudospin\n2S = " + str(site['twice_s'])
                             + " but a dimension of " + str(site['dim']) + ".")
            n_expected = n_expected * site['dim']

        if not n_expected == self.n_basis:
            self.__error("The product of the dimensions of the sites of the file\n"
                         + self.filename + " is " + str(n_expected)
                         + " but the header declares a model\ndimension of "
                         + str(self.n_basis) + ".")

        if not len(basis_rows) == self.n_basis:
            self.__error("The [BASIS] section of the file " + self.filename + " lists "
                         + str(len(basis_rows)) + " states\nbut the header declares a model "
                         "dimension of " + str(self.n_basis) + ".")

        for i in range(0,self.n_basis):
            if not len(basis_rows[i]) == self.n_sites:
                self.__error("The basis state " + str(i) + " of the file " + self.filename
                             + " is labelled by\n" + str(len(basis_rows[i]))
                             + " projections but the file has " + str(self.n_sites)
                             + " sites.")

        for tag in ('H','mu_x','mu_y','mu_z','U'):
            if not tag in matrix_dict:
                self.__error("The matrix " + tag + " was not found in the file "
                             + self.filename + ".")
            if not matrix_dict[tag].shape == (self.n_basis,self.n_basis):
                self.__error("The matrix " + tag + " of the file " + self.filename
                             + " has the dimensions\n" + str(matrix_dict[tag].shape)
                             + " instead of the model dimension "
                             + str(self.n_basis) + ".")

        for site, tag in iter(site_matrix_dict):
            if site < 0 or site >= self.n_sites:
                self.__error("The file " + self.filename + " contains the matrix " + tag
                             + " of site\n" + str(site) + " but the file has only "
                             + str(self.n_sites) + " sites.")
            dim = site_list[site]['dim']
            if not site_matrix_dict[(site,tag)].shape == (dim,dim):
                self.__error("The matrix " + tag + " of site " + str(site) + " of the file\n"
                             + self.filename + " has the dimensions "
                             + str(site_matrix_dict[(site,tag)].shape)
                             + " instead of\nthe dimensions of the site, "
                             + str((dim,dim)) + ".")

        if not self.__energy_array.shape[0] == self.n_basis:
            self.__error("The [ENERGIES] section of the file " + self.filename + " lists "
                         + str(self.__energy_array.shape[0]) + "\nenergies but the header "
                         "declares a model dimension of " + str(self.n_basis) + ".")

        if np.any(np.diff(self.__energy_array) < 0.0):
            self.__warning("The energies of the file " + self.filename + " are not listed "
                           "in\nascending order, which the format expects. The order of the "
                           "states\nof the transformation matrix follows the order of the "
                           "energies.")


    def __check_basis_order(self):
        """Check that the basis listing of the file is exactly the order in
        which PseudoSpinBasis constructs the basis of the pseudospins of the
        file. It is this property that lets the matrices of the file be used
        with the basis of the library directly, and it is therefore checked
        rather than trusted. A deviation is an error and not a reordering,
        as a silent reordering would defeat the check.
        """
        basis = pseudospin_operators.PseudoSpinBasis(self.pseudospin_list)

        if not basis.n_basis == self.n_basis:
            self.__error("PseudoSpinBasis constructs " + str(basis.n_basis)
                         + " states for the pseudospins\nof the file " + self.filename
                         + ", which contradicts the model dimension "
                         + str(self.n_basis) + ".")

        for i in range(0,self.n_basis):
            expected_row = [basis.basis_state_list[i][site][1]
                            for site in range(0,self.n_sites)]

            if not list(self.__basis_rows[i]) == expected_row:
                message = ("The basis listing of the file " + self.filename + " deviates\n"
                           "from the order of PseudoSpinBasis at the state " + str(i) + ":\n"
                           "the file gives the projections " + str(list(self.__basis_rows[i]))
                           + " where the basis\nof the library has " + str(expected_row)
                           + " (as multiples of two).")
                if not self.basis_order == '':
                    message += ("\nThe file declares the ordering as:\n" + self.basis_order)
                message += ("\nThe file is not a valid KUIVA_PSEUDOSPIN file of the version "
                            + str(self.version) + ".")
                self.__error(message)


    def __check_matrices(self):
        """Check the internal consistency of the matrices of the file: the
        Hamiltonian and the components of the magnetic moment are Hermitean,
        the transformation of the states is unitary, and the trace of the
        Hamiltonian equals the sum of the listed eigenvalues. An
        inconsistency is reported as a warning and not as an error, as the
        matrices can still be read and analyzed.
        """
        for tag in ('H','mu_x','mu_y','mu_z'):
            matrix = self.__matrix_dict[tag]
            if not np.allclose(matrix,matrix.conj().T):
                self.__warning("The matrix " + tag + " of the file " + self.filename
                               + " is not Hermitean.")

        unitary = self.__matrix_dict['U']
        if not np.allclose(unitary.conj().T @ unitary,np.identity(self.n_basis)):
            self.__warning("The transformation matrix U of the file " + self.filename
                           + "\nis not unitary.")

        trace = np.trace(self.__matrix_dict['H']).real
        if not np.allclose(trace,np.sum(self.__energy_array)):
            self.__warning("The trace of the Hamiltonian of the file " + self.filename
                           + "\ndoes not equal the sum of the energies listed in the file.")


    def hamiltonian(self, subtract_ground_energy=True):
        """A matrix representation of the effective Hamiltonian over the
        pseudospin product basis, in the current energy unit. The matrix is
        NOT diagonal, as it is the Hamiltonian in the product basis and not
        in the basis of its own eigenstates; its eigenvalues are returned by
        the energies method and the transformation into its eigenstates by
        the unitary method.

        Optional arguments
        ------------------
        subtract_ground_energy : boolean
            Whether the energy of the ground state is subtracted from the
            diagonal, so that the ground state of the returned Hamiltonian
            lies at zero. This is the form the rest of the library works
            with, and the default is therefore True. With False the
            Hamiltonian is returned as it is stored in the file, i.e. with
            the absolute energies of the ab initio model (still without the
            energy shift of the file).
        """
        matrix = np.array(self.__matrix_dict['H'], dtype=np.complex128)

        # The translation is done in hartree, i.e. in the unit the values
        # were stored in, before the conversion of the unit.
        if subtract_ground_energy:
            matrix = matrix - np.min(self.__energy_array)*np.identity(self.n_basis)

        return self.units.convert_energy_unit(matrix,'hartree')


    def energies(self, include_shift=False):
        """The eigenvalues of the effective Hamiltonian in ascending order,
        in the current energy unit.

        Optional arguments
        ------------------
        include_shift : boolean
            Whether the energy shift of the file (e.g. the inactive-core
            energy of the ab initio calculation) is added to the energies,
            i.e. whether total energies are returned. Default is False.
        """
        value_array = self.__energy_array.copy()

        if include_shift:
            value_array = value_array + self.energy_shift

        return self.units.convert_energy_unit(value_array,'hartree')


    def relative_energies(self):
        """The eigenvalues of the effective Hamiltonian relative to the
        lowest one, i.e. the excitation energies of the states, in the
        current energy unit. These are the eigenvalues of the Hamiltonian
        returned by the hamiltonian method with its default arguments.
        """
        value_array = self.__energy_array - np.min(self.__energy_array)

        return self.units.convert_energy_unit(value_array,'hartree')


    def magnetic_moment(self, include_bohr_magneton=True):
        """A list of the matrix representations of the three Cartesian
        components of the magnetic moment over the pseudospin product basis.
        The optional argument is used to set whether the Bohr magneton is
        included in the operator matrices or not.
        """
        matrix_list = []

        for tag in ('mu_x','mu_y','mu_z'):
            matrix = np.array(self.__matrix_dict[tag], dtype=np.complex128)

            if include_bohr_magneton:
                matrix = self.units.mu_B * matrix

            matrix_list.append(matrix)

        return matrix_list


    def unitary(self):
        """The unitary matrix that maps the ab initio states, i.e. the
        eigenstates of the effective Hamiltonian in the order of ascending
        energy, to the pseudospin product basis: the columns of the matrix
        are the ab initio states in the product basis. The phases of the
        states are arbitrary.
        """
        return np.array(self.__matrix_dict['U'], dtype=np.complex128)


    def pseudospin_basis(self):
        """The basis of the file as an instance of PseudoSpinBasis. The
        ordering of the states of the instance was checked to be identical
        with the listing of the file upon initiation, so that the matrices
        of this class can be used with the returned basis directly.
        """
        return pseudospin_operators.PseudoSpinBasis(self.pseudospin_list)


    def site_moment(self, site, include_bohr_magneton=True):
        """A list of the matrix representations of the three Cartesian
        components of the magnetic moment of a single site projected onto
        the local states of that site, in the order in which the local
        states are labelled by M. These are the matrices the g values of the
        individual sites are calculated from.

        Arguments
        ---------
        site : int
            The index of the site.

        Optional arguments
        ------------------
        include_bohr_magneton : boolean
            Whether the Bohr magneton is included in the operator matrices
            or not. Default is True.
        """
        if site < 0 or site >= self.n_sites:
            self.__error("The site " + str(site) + " was requested of the file "
                         + self.filename + ",\nwhich has " + str(self.n_sites) + " sites.")

        matrix_list = []

        for tag in ('mu_x','mu_y','mu_z'):
            key = (int(site),tag)

            if not key in self.__site_matrix_dict:
                self.__error("The file " + self.filename + " does not contain the magnetic\n"
                             "moment of the site " + str(site)
                             + ". The file was written without the\nmoments of the "
                             "individual sites.")

            matrix = np.array(self.__site_matrix_dict[key], dtype=np.complex128)

            if include_bohr_magneton:
                matrix = self.units.mu_B * matrix

            matrix_list.append(matrix)

        return matrix_list


    def common_axis(self, tolerance=1.0e-3):
        """The common axis along which the states of every site are labelled
        by M, returned as the normalized mean of the axes of the sites, when
        the sites share an axis within the given tolerance, and None
        otherwise.

        The tensor analysis of a multi-site system is done in a single
        quantization frame, and a file whose sites are labelled along
        different axes therefore cannot be analyzed as it stands. A caller
        that receives None should ask Kuiva for a file written with a common
        axis instead.

        Optional arguments
        ------------------
        tolerance : float
            The largest deviation allowed between the components of the axes
            of two sites. Default is 1.0e-3, which is loose enough for axes
            that were determined numerically and tight enough to separate
            genuinely different axes. Note that the axes are directed, i.e.
            two antiparallel axes are not a common axis, as the sign of the
            axis fixes the sign of M.
        """
        if len(self.site_list) == 0:
            return None

        axis = self.site_list[0]['axis']

        for site in self.site_list[1:]:
            if np.max(np.abs(site['axis'] - axis)) > tolerance:
                return None

        mean_axis = np.mean([site['axis'] for site in self.site_list], axis=0)
        norm = np.linalg.norm(mean_axis)

        if norm < tolerance:
            self.__error("The mean of the labelling axes of the sites of the file\n"
                         + self.filename + " vanishes; the axes of the file are broken.")

        return mean_axis / norm


    def __repr__(self):
        """Return some basic info on the file."""
        tmp_str = ""
        tmp_str += "KUIVA PSEUDOSPIN FILE\n\n"
        tmp_str += "  File name:           " + self.filename + "\n"
        tmp_str += "  File format version: " + str(self.version) + "\n"
        tmp_str += "  Number of sites:     " + str(self.n_sites) + "\n"
        tmp_str += "  Pseudospins (2S):    " + str(self.pseudospin_list) + "\n"
        tmp_str += "  Basis dimension:     " + str(self.n_basis) + "\n"
        tmp_str += "  Coordinate frame:    " + self.frame + "\n\n"

        return tmp_str


    def __init__(self, filename, units):
        """Upon class initiation, check that the file exists, parse it whole
        and check that its basis listing and its matrices are consistent
        with the pseudospin basis of the library.
        """
        self.filename = filename
        self.units    = units

        try:
            f = open(self.filename)
        except FileNotFoundError:
            self.__error("file " + self.filename + " not found.")
        else:
            f.close()

        self.provenance = {}

        self.__parse()
        self.__check_basis_order()
        self.__check_matrices()

        if not self.provenance:
            self.__warning("The file " + self.filename + " carries no provenance of the\n"
                           "Hamiltonian; the screening and decoupling records of the Kuiva\n"
                           "calculation are normally present.")


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class using a synthetic pseudospin file and run a
        set of tests on the class constructor and the reading methods.
        Return True if all tests passed and False otherwise.

        The synthetic file describes a two-site system of the pseudospins
        1/2 and 1, i.e. a basis of six states, so that the check of the
        ordering of the basis runs on a genuinely mixed product basis. The
        Hamiltonian is constructed from a set of known eigenvalues and a
        known unitary matrix, so that the listed energies and the
        transformation matrix can be checked against a direct
        diagonalization of the matrix that was read.

        Optional arguments
        ------------------
        print_output : boolean
            Whether to print the outcome of each test. Default is True.
        """
        import os
        import tempfile

        from ouluspin import _debug as debug_output
        from ouluspin import units as units_module

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('KuivaPseudospinFile',
                                                       test_name,condition,print_output))

        tmp_units = units_module.EnergyUnitSystem('wavenumber')
        hartree = tmp_units.convert_energy_unit(1.0,'hartree')

        rng = np.random.default_rng(7)

        n_basis = 6
        pseudospin_list = [1,2]
        site_dimensions = [2,3]

        # A Hermitean Hamiltonian constructed from a set of known
        # eigenvalues and a random unitary matrix.
        eigenvalues = np.array([-1.0,-1.0 + 1.0e-4,-1.0 + 2.0e-4,
                                -1.0 + 5.0e-4,-1.0 + 6.0e-4,-1.0 + 9.0e-4])
        random_matrix = rng.standard_normal((n_basis,n_basis)) \
                        + 1.0j*rng.standard_normal((n_basis,n_basis))
        unitary, _ = np.linalg.qr(random_matrix)
        hamiltonian = unitary @ np.diag(eigenvalues) @ unitary.conj().T

        moment_list = []
        for i in range(0,3):
            matrix = rng.standard_normal((n_basis,n_basis)) \
                     + 1.0j*rng.standard_normal((n_basis,n_basis))
            moment_list.append(0.5*(matrix + matrix.conj().T))

        site_moment_list = []
        for dim in iter(site_dimensions):
            component_list = []
            for i in range(0,3):
                matrix = rng.standard_normal((dim,dim)) + 1.0j*rng.standard_normal((dim,dim))
                component_list.append(0.5*(matrix + matrix.conj().T))
            site_moment_list.append(component_list)

        def matrix_block(tag,matrix):
            block_str  = "[" + tag + "]\n"
            block_str += "shape      {0} {1}\n".format(matrix.shape[0],matrix.shape[1])
            block_str += "unit       mu_B\n"
            for i in range(0,matrix.shape[0]):
                for j in range(0,matrix.shape[1]):
                    block_str += "{0:6d} {1:6d}  {2:+.16e} {3:+.16e}\n".format(
                        i,j,matrix[i][j].real,matrix[i][j].imag)
            block_str += "[END]\n\n"
            return block_str

        # The basis listing in the order of PseudoSpinBasis: the first site
        # is the slowest and within a site M runs from -S to +S.
        basis_row_list = []
        for m_0 in (-1,1):
            for m_1 in (-2,0,2):
                basis_row_list.append((m_0,m_1))

        content  = "# A synthetic Kuiva pseudospin file.\n"
        content += "[HEADER]\n"
        content += "format                           KUIVA_PSEUDOSPIN\n"
        content += "format_version                   1\n"
        content += "n_sites                          2\n"
        content += "model_dim                        6\n"
        content += "energy_unit                      Eh\n"
        content += "moment_unit                      mu_B\n"
        content += "energy_shift                     +1.5000000000000000e+00\n"
        content += "hamiltonian_is_diagonal          no\n"
        content += "basis_order                      site 0 slowest (C order); "
        content += "within a site M = -S .. +S ascending\n"
        content += "m_convention                     descending <mu . axis> "
        content += "labelled M = -S .. +S\n"
        content += "frame                            input frame\n"
        content += "phase_convention                 arbitrary (not canonicalized)\n"
        content += "[END]\n\n"
        content += "[PROVENANCE]\n{\"code\": \"kuiva\", \"system\": \"synthetic\"}\n[END]\n\n"
        content += "[FRAME]\n"
        content += "  +1.0 +0.0 +0.0\n  +0.0 +1.0 +0.0\n  +0.0 +0.0 +1.0\n"
        content += "[END]\n\n"
        content += "[SITES]\n"
        content += "# site   2S  dim   axis_x   axis_y   axis_z   axis_choice | N | orbitals\n"
        content += "    0    1    2  +0.0000000000 +0.0000000000 +1.0000000000"
        content += "  principal_magnetic_axis | 1 | 0 1\n"
        content += "    1    2    3  +0.0000000000 +0.0000000000 +1.0000000000"
        content += "  principal_magnetic_axis | 2 | 2 3\n"
        content += "[END]\n\n"
        content += "[BASIS]\n"
        for i, row in enumerate(basis_row_list):
            content += "{0:6d}  {1:+d} {2:+d}\n".format(i,row[0],row[1])
        content += "[END]\n\n"
        content += "[ENERGIES]\n"
        for i, value in enumerate(eigenvalues):
            content += "{0:6d}  {1:+.16e}  {2:+.8e}\n".format(
                i,value,(value - eigenvalues[0])*hartree)
        content += "[END]\n\n"
        content += matrix_block('MATRIX H',hamiltonian)
        content += matrix_block('MATRIX mu_x',moment_list[0])
        content += matrix_block('MATRIX mu_y',moment_list[1])
        content += matrix_block('MATRIX mu_z',moment_list[2])
        content += matrix_block('MATRIX U',unitary)
        for site in range(0,2):
            for i, component in enumerate(('mu_x','mu_y','mu_z')):
                content += matrix_block('SITE_MATRIX ' + str(site) + ' ' + component,
                                        site_moment_list[site][i])

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir,'model.psd')
            with open(filename,'w') as f:
                f.write(content)

            calculation = cls(filename,tmp_units)

            check('format version read', calculation.version == 1)

            check('header read',
                  calculation.n_sites == 2
                  and calculation.n_basis == n_basis
                  and calculation.frame == 'input frame'
                  and calculation.phase_convention == 'arbitrary (not canonicalized)')

            check('pseudospins of the sites read',
                  calculation.pseudospin_list == pseudospin_list)

            check('data of the sites read',
                  calculation.site_list[1]['dim'] == 3
                  and calculation.site_list[1]['n_electrons'] == 2
                  and calculation.site_list[1]['orbitals'] == (2,3)
                  and calculation.site_list[0]['axis_choice'] == 'principal magnetic axis')

            check('provenance read',
                  calculation.provenance['code'] == 'kuiva')

            check('basis ordering checked against PseudoSpinBasis',
                  calculation.pseudospin_basis().n_basis == n_basis
                  and calculation.pseudospin_basis().pseudospin_list == pseudospin_list)

            # The Hamiltonian is by default translated so that its ground
            # state lies at zero, and the untranslated matrix is the one
            # stored in the file, converted to the current unit.
            check('Hamiltonian read and converted from hartree',
                  np.allclose(calculation.hamiltonian(subtract_ground_energy=False),
                              hartree*hamiltonian))

            check('ground state of the Hamiltonian translated to zero',
                  np.allclose(calculation.hamiltonian(),
                              hartree*(hamiltonian
                                       - eigenvalues[0]*np.identity(n_basis))))

            check('energy listing matches a direct diagonalization',
                  np.allclose(np.linalg.eigvalsh(
                                  calculation.hamiltonian(subtract_ground_energy=False)),
                              calculation.energies()))

            check('relative energies given from the ground state',
                  np.allclose(calculation.relative_energies(),
                              hartree*(eigenvalues - eigenvalues[0]))
                  and np.allclose(np.linalg.eigvalsh(calculation.hamiltonian()),
                                  calculation.relative_energies()))

            check('energy shift added on request',
                  np.allclose(calculation.energies(include_shift=True)
                              - calculation.energies(),
                              hartree*1.5))

            check('magnetic moment matrices read',
                  all(np.allclose(matrix,reference) for matrix, reference in
                      zip(calculation.magnetic_moment(include_bohr_magneton=False),
                          moment_list)))

            check('Bohr magneton included on request',
                  np.allclose(calculation.magnetic_moment()[2],
                              tmp_units.mu_B*moment_list[2]))

            check('transformation matrix read and unitary',
                  np.allclose(calculation.unitary().conj().T @ calculation.unitary(),
                              np.identity(n_basis)))

            check('transformation matrix diagonalizes the Hamiltonian',
                  np.allclose(calculation.unitary().conj().T
                              @ calculation.hamiltonian(subtract_ground_energy=False)
                              @ calculation.unitary(),
                              np.diag(calculation.energies())))

            check('magnetic moments of the sites read',
                  all(np.allclose(matrix,reference) for matrix, reference in
                      zip(calculation.site_moment(1,include_bohr_magneton=False),
                          site_moment_list[1])))

            check('Bohr magneton included in the moments of the sites on request',
                  np.allclose(calculation.site_moment(0)[1],
                              tmp_units.mu_B*site_moment_list[0][1]))

            check('common labelling axis of the sites found',
                  np.allclose(calculation.common_axis(),[0.0,0.0,1.0]))

            check('rotation of the frame read as the identity',
                  np.allclose(calculation.frame_rotation,np.identity(3)))

            check('string representation of the file rendered',
                  len(str(calculation)) > 0)

            # A file whose sites are labelled along different axes has no
            # common axis, and a multi-site analysis of it in a single
            # quantization frame is not possible.
            tilted_content = content.replace(
                "    1    2    3  +0.0000000000 +0.0000000000 +1.0000000000",
                "    1    2    3  +1.0000000000 +0.0000000000 +0.0000000000")
            tilted_filename = os.path.join(tmp_dir,'tilted.psd')
            with open(tilted_filename,'w') as f:
                f.write(tilted_content)

            tilted_calculation = cls(tilted_filename,tmp_units)

            check('sites labelled along different axes have no common axis',
                  tilted_calculation.common_axis() is None)

        return debug_output.test_summary('KuivaPseudospinFile',result_list,print_output)
