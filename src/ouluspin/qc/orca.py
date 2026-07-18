# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

import sys

import numpy as np

from copy import deepcopy

from ouluspin import units
from ouluspin import tensors
from ouluspin.qc import molcas


class OrcaCalculation(molcas.AnisoCalculation):
    """A class to read data from an output file of an Orca calculation.

    NOTE!!! All angular momentum quantum numbers are given as multiplets of two in order to treat
            half-integer angular momenta as integer numbers.

    Arguments
    ---------
    filename : str
       Name of the SINGLE_ANISO output file containing the data.
    units : EnergyUnitSystem
       Instance of EnergyUnitSystem containing the necessary tools to convert the energy units.

    Optional arguments
    ------------------
    correlation : str
        The correlation treatment used in an Orca multireference calculations. The default
        is 'NEVPT2'. The other options are 'CASSCF' and 'QD-NEVPT2'.
    aniso : boolean
        Whether the Orca output contains SINGLE_ANISO output. Default is True. It would make more
        sense to have the default as False, but now this is consistent with the
        OpenMolcasCalculation class.
    debug : boolean
        Whether to print debug information. Default is False.

    Attributes
    ----------
    filename : str
       Name of the ADF output file containing the data.
    units : EnergyUnitSystem
        The unit system.
    version : str
        Orca version.
    debug : boolean
        Whether to print debug information.
    pseudospin : int
        Value of the pseudospin used in the SINGLE_ANISO output.
    single_aniso_start_str : str
        A string indicating the start of the the SINGLE_ANISO output in the Orca output
        file.
    correlation : str
        The correlation treatment used in an Orca multireference calculations. The default
        is 'NEVPT2'. The other options are 'CASSCF' and 'QD-NEVPT2'.
    output_instance : int
        Assuming that the output file read contains outputs of several SINGLE_ANISO
        or POLY_ANISO runs, this parameter determines which of these will be read. The
        first output corresponds to value of zero..
    input_frame_rotation : Rotation or None
        The rotation from the input axis frame (the frame of the ab initio
        data) to the principal magnetic axis frame of the ground multiplet,
        as recorded by read_crystal_field. None before read_crystal_field
        has been called or when the rotation block was not found in the
        output. Rotations of the library are always stored as rotations
        FROM the input frame.

    Private methods
    ---------------

    Public methods
    --------------
    read_version()
        Read the version of Orca used to produce the output and store it as an attribute.
        Also define the single_aniso_start_str variable.

    Class methods
    -------------
    run_tests(print_output=True) : boolean
        Initiate the class using a synthetic output file and run a set of
        internal tests. Return True if all tests passed.
    """

    def read_version(self):
        """Read the version of Orca used to produce the output and store it as an attribute.
        Also define the single_aniso_start_str variable.
        """
        f = open(self.filename)

        while True:
            line = f.readline()
            if line.startswith('                         Program Version 5.0.1'):
                self.version = '5.0.1'
                self.single_aniso_start_str = '  Calling the SINGLE_ANISO program'
                break
            elif line.startswith('                         Program Version 5.0.2'):
                self.version = '5.0.2'
                self.single_aniso_start_str = '  Calling the SINGLE_ANISO program'
                break
            elif line.startswith('                         Program Version 5.0.4'):
                self.version = '5.0.4'
                self.single_aniso_start_str = '  Calling the SINGLE_ANISO program'
                break
            elif line == '':
                print("ERROR in OrcaCalculation.:")
                print("Error: Unsupported Orca version or could not read version.")
                print("Error termination.")
                sys.exit(1)

        f.close()
        

    def __repr__(self):
        """Return some basis info on the calculation."""
        tmp_str = ""
        tmp_str += "ORCA CALCULATION\n\n"
        tmp_str += "  Orca ouput file: " + self.filename + "\n"
        tmp_str += "  Version:         " + self.version + "\n\n"

        return tmp_str

    
    def __init__(self, filename, units,
                 correlation='NEVPT2',
                 aniso=True,
                 debug=False):
        """Upon class initiation, check that the Orca output file exists."""
        self.filename    = filename
        self.units       = units
        self.correlation = correlation
        self.debug       = debug

        allowed_correlations = ['CASSCF','NEVPT2','QD-NEVPT2']
        if not self.correlation in allowed_correlations:
            print("ERROR in OrcaCalculation.")
            print("ERROR: Uknown multireference correlation method: " + self.correlation + ".")
            print("Error termination.")
            sys.exit(1)

        try:
            f = open(self.filename)
        except FileNotFoundError:
            print("ERROR in OrcaCalculation.")
            print("ERROR: file " + self.filename + " not found.")
            print("Error termination.")
            sys.exit(1)
        else:
            f.close()

        self.read_version()

        # The rotation from the input axis frame to the principal magnetic
        # axis frame; recorded by read_crystal_field.
        self.input_frame_rotation = None

        # Set a default so that the attribute always exists even when
        # aniso=False.
        self.output_instance = 0

        if aniso:
            if self.correlation == 'CASSCF':
                self.output_instance = 0
            elif self.correlation == 'NEVPT2':
                self.output_instance = 1
            elif self.correlation == 'QD-NEVPT2':
                print("ERROR in OrcaCalculation.")
                print("ERROR: SINGLE_ANISO output in conjunction with QD-NEVPT2 is not yet implemented.")
                print("Error termination.")
                sys.exit(1)

            self.pseudospin = self.read_pseudospin()


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class using a synthetic output file and run a set of
        tests on the class constructor and the reading methods. Return True
        if all tests passed and False otherwise.

        The synthetic file reproduces the parts of the Orca output format
        that the reading methods rely on. The reading methods inherited from
        AnisoCalculation are tested in more detail in
        OpenMolcasCalculation.run_tests; here the tests focus on the Orca
        version detection and on the selection of the correct SINGLE_ANISO
        output instance based on the correlation treatment.

        Optional arguments
        ------------------
        print_output : boolean
            Whether to print the outcome of each test. Default is True.
        """
        import os
        import tempfile

        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('OrcaCalculation',
                                                       test_name,condition,print_output))

        tmp_units = units.EnergyUnitSystem('wavenumber')

        multiplet_line = "     CALCULATION OF PSEUDOSPIN HAMILTONIAN TENSORS FOR THE" \
                         " MULTIPLET 1 ( EFFECTIVE SPIN = {0})\n"

        # A synthetic output with two SINGLE_ANISO instances: the first from
        # the CASSCF states (S = 1/2) and the second from the NEVPT2 states
        # (S = 3/2).
        content  = "                         Program Version 5.0.4 -  RELEASE  -\n\n"
        content += "  Calling the SINGLE_ANISO program\n\n"
        content += multiplet_line.format("1/2") + "\n"
        content += "  Calling the SINGLE_ANISO program\n\n"
        content += multiplet_line.format("3/2") + "\n"

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir,'orca.out')
            with open(filename,'w') as f:
                f.write(content)

            calculation = cls(filename,tmp_units,correlation='CASSCF')
            check('version read', calculation.version == '5.0.4')
            check('CASSCF reads the first SINGLE_ANISO instance',
                  calculation.pseudospin == 1)

            calculation = cls(filename,tmp_units,correlation='NEVPT2')
            check('NEVPT2 reads the second SINGLE_ANISO instance',
                  calculation.pseudospin == 3)

            calculation = cls(filename,tmp_units,aniso=False)
            check('initiation without SINGLE_ANISO output',
                  calculation.output_instance == 0)

        return debug_output.test_summary('OrcaCalculation',result_list,print_output)



class OrcaAnisoOutput(molcas.AnisoCalculation):
    """A class to read data from a standalone output file of the SINGLE_ANISO
    program as called by Orca.

    Unlike OrcaCalculation, which reads the SINGLE_ANISO output embedded in a
    full Orca output file, this class works with a file containing just the
    output of the SINGLE_ANISO module itself (the module prints its own
    banner and version instead of the Orca program version).

    The read_crystal_field method is overridden here: the standalone
    SINGLE_ANISO program prints the ab initio crystal field as separate real
    B(k,q) and C(k,q) parameters of the Hermitian and anti-Hermitian
    equivalent operators in the Chibotaru--Ungur convention (defined in
    L. F. Chibotaru, L. Ungur, J. Chem. Phys. 2012, 137, 064112) instead of
    the complex a(k,q) table read by the parent class. The parameters are
    read from the full parametrization block (all ranks up to 2J, without
    the Stevens alpha(k) prefactors) and converted to the
    Iwahara--Chibotaru convention. The other reading methods inherited from
    AnisoCalculation have not been verified against the standalone output
    format and should be checked before use.

    NOTE!!! All angular momentum quantum numbers are given as multiplets of two in order to treat
            half-integer angular momenta as integer numbers.

    Arguments
    ---------
    filename : str
       Name of the standalone SINGLE_ANISO output file containing the data.
    units : EnergyUnitSystem
       Instance of EnergyUnitSystem containing the necessary tools to convert the energy units.

    Optional arguments
    ------------------
    output_instance : int
        Assuming that the output file read contains outputs of several
        SINGLE_ANISO runs, this parameter determines which of these will be
        read. The first output corresponds to value of zero. Default is 0.
    debug : boolean
        Whether to print debug information. Default is False.

    Attributes
    ----------
    filename : str
       Name of the standalone SINGLE_ANISO output file containing the data.
    units : EnergyUnitSystem
        The unit system.
    version : str
        SINGLE_ANISO program version.
    debug : boolean
        Whether to print debug information.
    pseudospin : int
        Value of the pseudospin of the ground multiplet in the SINGLE_ANISO
        output.
    single_aniso_start_str : str
        A string indicating the start of a SINGLE_ANISO output in the file.
    output_instance : int
        Which SINGLE_ANISO output of the file is read; see the optional
        argument of the same name.
    input_frame_rotation : Rotation or None
        The rotation from the input axis frame (the frame of the ab initio
        data) to the principal magnetic axis frame of the ground multiplet,
        as recorded by read_crystal_field. None before read_crystal_field
        has been called or when the rotation block was not found in the
        output. Rotations of the library are always stored as rotations
        FROM the input frame.

    Private methods
    ---------------

    Public methods
    --------------
    read_version()
        Read the version of the SINGLE_ANISO program used to produce the
        output and store it as an attribute. Also define the
        single_aniso_start_str variable.
    read_crystal_field() : IwaharaChibotaruSphericalTensor
        Read the ITO expansion of the ab initio crystal field of the ground
        atomic multiplet J and return it as an instance of
        IwaharaChibotaruSphericalTensor. Overrides the parent method (see
        above).

    Class methods
    -------------
    run_tests(print_output=True) : boolean
        Initiate the class using a synthetic output file and run a set of
        internal tests. Return True if all tests passed.
    """

    def read_version(self):
        """Read the version of the SINGLE_ANISO program used to produce the
        output and store it as an attribute. Also define the
        single_aniso_start_str variable.
        """
        f = open(self.filename)

        while True:
            line = f.readline()
            if line.startswith('!   Program version :'):
                self.version = line.split(':')[1].split()[0]
                break
            elif line == '':
                print("ERROR in OrcaAnisoOutput.")
                print("Error: Could not read the SINGLE_ANISO program version. The file")
                print("       does not look like a standalone SINGLE_ANISO output.")
                print("Error termination.")
                sys.exit(1)

        f.close()

        if self.version == 'v1.0.0':
            self.single_aniso_start_str = '!   Program version :'
        else:
            print("ERROR in OrcaAnisoOutput.")
            print("Error: Unsupported SINGLE_ANISO version: " + self.version + ".")
            print("Error termination.")
            sys.exit(1)


    def read_crystal_field(self):
        """Read the ITO expansion of the ab initio crystal field of the
        ground atomic multiplet J and return it as an instance of
        IwaharaChibotaruSphericalTensor.

        The parameters are read from the full parametrization block

            Hcf = SUM_{k,q} [ B(k,q)*O(k,q) + C(k,q)*W(k,q) ]

        (all ranks up to 2J, without the Stevens alpha(k) prefactors), first
        stored as an instance of ChibotaruUngurSphericalTensor and then
        converted to the Iwahara--Chibotaru convention. The pseudospin used
        in the conversion is the 2J of the ground atomic multiplet as
        printed in the crystal-field section, which does not necessarily
        correspond to the pseudospin of the magnetic moment and ZFS
        pseudospin operators.

        A fatal error is produced if the SINGLE_ANISO run did not compute
        the crystal field (which the program reports at the beginning of
        the output when it does not recognize the metal ion label in its
        input data).

        The crystal-field parameters are printed by SINGLE_ANISO, and are
        returned by this method, in the main magnetic axes frame of the
        ground multiplet, which is the principal magnetic axis frame in
        which the spherical tensors of the library are conventionally
        expressed. The rotation from the input axis frame to this frame is
        parsed from the crystal-field section and recorded in the
        input_frame_rotation attribute of this instance (rotations of the
        library are always stored as rotations FROM the input frame); the
        attribute is None when the rotation block was not found.
        """
        rank_list           = []
        real_parameter_list = []
        imag_parameter_list = []

        self.input_frame_rotation = None
        axes_rows = {}

        f = open(self.filename)
        instance_counter = 0
        while True:
            line = f.readline()
            if line.startswith(self.single_aniso_start_str):
                if instance_counter >= self.output_instance:
                    break
                else:
                    instance_counter += 1
            elif line == '':
                print("ERROR in OrcaAnisoOutput.")
                print("Error: The file " + self.filename + " does not contain SINGLE_ANISO output.")
                print("Error termination.")
                sys.exit(1)

        # Find the crystal-field section of the ground atomic multiplet and
        # read the value of J. The SINGLE_ANISO program reports early in the
        # output if the crystal field was not computed at all; detect this
        # explicitly to give an understandable error.
        while True:
            line = f.readline()
            if line.startswith('     CALCULATION OF CRYSTAL-FIELD PARAMETERS OF THE GROUND ATOMIC MULTIPLET J'):
                J_str = line.split()[11]
                if len(J_str.split('/')) == 1:
                    J = 2 * int(J_str.replace('.',''))
                else:
                    J = int(J_str.split('/')[0])
                break
            elif line.startswith('Crystal field will not be computed'):
                print("ERROR in OrcaAnisoOutput.")
                print("Error: The SINGLE_ANISO run in the file " + self.filename)
                print("       did not compute the ab initio crystal field (the output")
                print("       states: 'Crystal field will not be computed'; usually the")
                print("       metal ion label was not recognized in the SINGLE_ANISO")
                print("       input data). No crystal-field parameters can be read.")
                print("Error termination.")
                sys.exit(1)
            elif line == '':
                print("ERROR in OrcaAnisoOutput.")
                print("Error: The ab initio crystal-field section was not found in the")
                print("       SINGLE_ANISO output.")
                print("Error termination.")
                sys.exit(1)

        # Find the full parametrization block (without the Stevens alpha(k)
        # prefactors; the leading block with the alpha(k) prefactors only
        # contains the ranks k = 2, 4, 6 and is skipped).
        while True:
            line = f.readline()
            if line.startswith('   Hcf = SUM_{k,q} * [ B(k,q) * O(k,q) + C(k,q) * W(k,q) ]'):
                break
            # The rotation from the initial (input) coordinate system to the
            # magnetic axes frame of the crystal field is printed in this
            # section as rows labeled Xm, Ym and Zm.
            parts = line.replace('R =',' ').replace('|',' ').split()
            if len(parts) == 4 and parts[0] in ('Xm','Ym','Zm'):
                axes_rows[parts[0]] = [float(x) for x in parts[1:4]]
            if line == '':
                print("ERROR in OrcaAnisoOutput.")
                print("Error: End of file reached while searching for the full crystal-field")
                print("       parametrization block.")
                print("Error termination.")
                sys.exit(1)

        while True:
            line = f.readline()
            if line.startswith('  k  |  q  |         B(k,q)        |         C(k,q)'):
                break
            elif line == '':
                print("ERROR in OrcaAnisoOutput.")
                print("Error: End of file reached while searching for the crystal-field")
                print("       parameter listing.")
                print("Error termination.")
                sys.exit(1)

        while True:
            line = f.readline()

            if len(line.split()) == 8:
                tmp_k = 2 * int(line.split()[0])
                tmp_q = 2 * int(line.split()[2])
                tmp_B = self.units.convert_energy_unit(float(line.split()[4]),'wavenumber')
                tmp_C = self.units.convert_energy_unit(float(line.split()[6]),'wavenumber')

                rank_list.append((tmp_k,tmp_q))
                real_parameter_list.append(tmp_B)
                imag_parameter_list.append(tmp_C)

            if line.startswith('*****'):
                break
            if line == '':
                print("ERROR in OrcaAnisoOutput.")
                print("Error: End of file reached while reading the crystal-field")
                print("       parameter listing.")
                print("Error termination.")
                sys.exit(1)

        f.close()

        if len(rank_list) == 0:
            print("ERROR in OrcaAnisoOutput.")
            print("Error: No crystal-field parameters read.")
            print("Error termination.")
            sys.exit(1)

        # Convert from Chibotaru--Ungur definition to Iwahara--Chibotaru definition.
        CU_decomposition = tensors.ChibotaruUngurSphericalTensor(rank_list,
                                                                 real_parameter_list,
                                                                 imag_parameter_list,
                                                                 J)

        IC_decomposition = CU_decomposition.iwahara_chibotaru_spherical_tensor()
        IC_decomposition.frame = 'principal magnetic axis frame'

        if len(axes_rows) == 3:
            # SINGLE_ANISO fixes the signs of the printed magnetic axes
            # arbitrarily, so the axes may form an improper set. All
            # operators of this library are built on angular momenta and
            # are therefore parity even, so an improper frame map acts on
            # them identically to its proper counterpart -R; the overall
            # sign is flipped so that a proper rotation is always stored.
            matrix = np.array([axes_rows['Xm'],axes_rows['Ym'],axes_rows['Zm']],
                              dtype=np.float64)
            if np.linalg.det(matrix) < 0.0:
                matrix = -matrix
            self.input_frame_rotation = tensors.Rotation(matrix)

        if self.debug:
            print("Decomposition of the crystal field using Chibotaru--Ungur definition of ITOs:")
            print()
            print(CU_decomposition)

        if self.debug:
            print("Decomposition of the crystal field using Iwahara--Chibotaru definition of ITOs:")
            print()
            print(IC_decomposition)

        return IC_decomposition


    def __repr__(self):
        """Return some basic info on the calculation."""
        tmp_str = ""
        tmp_str += "ORCA SINGLE_ANISO OUTPUT\n\n"
        tmp_str += "  SINGLE_ANISO output file: " + self.filename + "\n"
        tmp_str += "  Version:                  " + self.version + "\n\n"

        return tmp_str


    def __init__(self, filename, units,
                 output_instance=0,
                 debug=False):
        """Upon class initiation, check that the output file exists, read the
        program version and the pseudospin of the ground multiplet."""
        self.filename        = filename
        self.units           = units
        self.debug           = debug
        self.output_instance = output_instance

        try:
            f = open(self.filename)
        except FileNotFoundError:
            print("ERROR in OrcaAnisoOutput.")
            print("ERROR: file " + self.filename + " not found.")
            print("Error termination.")
            sys.exit(1)
        else:
            f.close()

        self.read_version()

        # The rotation from the input axis frame to the principal magnetic
        # axis frame; recorded by read_crystal_field.
        self.input_frame_rotation = None

        self.pseudospin = self.read_pseudospin()


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class using a synthetic output file and run a set of
        tests on the class constructor and the reading methods. Return True
        if all tests passed and False otherwise.

        The synthetic file reproduces the parts of the standalone
        SINGLE_ANISO output format that the reading methods rely on: the
        program banner with the version, the pseudospin multiplet header and
        the crystal-field section with both parametrization blocks (the
        block with the Stevens alpha(k) prefactors must be skipped and the
        full block read). The crystal field read through the class is
        compared with the same parameters converted directly with
        ChibotaruUngurSphericalTensor.

        Optional arguments
        ------------------
        print_output : boolean
            Whether to print the outcome of each test. Default is True.
        """
        import os
        import tempfile

        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('OrcaAnisoOutput',
                                                       test_name,condition,print_output))

        tmp_units = units.EnergyUnitSystem('wavenumber')

        content  = "!------------------------------------------------------------------!\n"
        content += "!                        SINGLE_ANISO Program                       !\n"
        content += "!   Program version :  v1.0.0                                       !\n"
        content += "!------------------------------------------------------------------!\n"
        content += "\n"
        content += "     CALCULATION OF PSEUDOSPIN HAMILTONIAN TENSORS FOR THE" \
                   " MULTIPLET 1 ( effective S =  1/2)\n"
        content += "\n"
        content += "     CALCULATION OF CRYSTAL-FIELD PARAMETERS OF THE GROUND" \
                   " ATOMIC MULTIPLET J = 15/2.\n"
        content += "\n"
        # The rotation from the initial coordinate system to the magnetic
        # axes frame of the crystal field.
        content += "      | Xm |  0.00000000000000  1.00000000000000  0.00000000000000 |\n"
        content += " R =  | Ym | -1.00000000000000  0.00000000000000  0.00000000000000 |\n"
        content += "      | Zm |  0.00000000000000  0.00000000000000  1.00000000000000 |\n"
        content += "\n"
        # The block with the Stevens prefactors, which must be skipped.
        content += "The Crystal-Field Hamiltonian:\n"
        content += "   Hcf = SUM_{k,q} alpha(k) * [ B(k,q) * O(k,q) +  C(k,q) * W(k,q) ];\n"
        content += "  k  |  q  |    1/alpha(k)  |         B(k,q)        |         C(k,q)        |\n"
        content += "  2  |  0  |    -157.50000  |  0.99000000000000E+02 |  0.00000000000000E+00 |\n"
        content += "\n"
        # The full parametrization block that is actually read.
        content += "The Crystal-Field Hamiltonian:\n"
        content += "   Hcf = SUM_{k,q} * [ B(k,q) * O(k,q) + C(k,q) * W(k,q) ];\n"
        content += "-----------------------------------------------------------|\n"
        content += "  k  |  q  |         B(k,q)        |         C(k,q)        |\n"
        content += "-----|-----|-----------------------|-----------------------|\n"
        content += "  2  |  0  |  0.10000000000000E+01 |  0.00000000000000E+00 |\n"
        content += "  2  |  2  | -0.20000000000000E+01 |  0.30000000000000E+01 |\n"
        content += "-----|-----|-----------------------|-----------------------|\n"
        content += "  4  |  0  |  0.50000000000000E+00 |  0.00000000000000E+00 |\n"
        content += "-----------------------------------------------------------|\n"
        content += "\n"
        content += "********************************************************************************\n"

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir,'single_aniso.out')
            with open(filename,'w') as f:
                f.write(content)

            calculation = cls(filename,tmp_units)
            check('version read', calculation.version == 'v1.0.0')
            check('pseudospin read', calculation.pseudospin == 1)

            crystal_field = calculation.read_crystal_field()

            reference = tensors.ChibotaruUngurSphericalTensor(
                [(4,0),(4,4),(8,0)],[1.0,-2.0,0.5],[0.0,3.0,0.0],15)\
                .iwahara_chibotaru_spherical_tensor()

            check('crystal-field frame label',
                  crystal_field.frame == 'principal magnetic axis frame')
            check('crystal-field frame rotation recorded',
                  calculation.input_frame_rotation is not None and
                  np.allclose(calculation.input_frame_rotation.rotation_matrix,
                              [[0.0,1.0,0.0],[-1.0,0.0,0.0],[0.0,0.0,1.0]]))

            check('crystal-field ranks read',
                  crystal_field.rank_list == reference.rank_list)
            check('crystal-field parameters read and converted',
                  np.allclose(crystal_field.parameter_list,
                              reference.parameter_list))
            check('Stevens-prefactor block skipped',
                  not np.allclose(np.abs(crystal_field.parameter_list[0]),99.0))

            check('string representation renders', len(str(calculation)) > 0)

        return debug_output.test_summary('OrcaAnisoOutput',result_list,print_output)



class OrcaAnisoFile:
    """A class to read operator matrices from a .anisofile produced by an Orca calculation.

    The operator matrices stored in the datafile are expressed in the input
    axis frame (the coordinate frame of the ab initio calculation); the
    datafile defines the input frame of all quantities derived from it.

    Arguments
    ---------
    filename : str
        Name of the anisofile.
    units : EnergyUnitSystem
        The energy unit system.

    Optional arguments
    ------------------

    Attributes
    ----------
    filename : str
        Name of the anisofile.
    units : EnergyUnitSystem
        The energy unit system.
    n_basis : int
        The number of basis functions.
    version : str
        Version of the anisofile.

    Private methods
    ---------------
    __read_version()
        Read version of the anisofile and store it as an attribute.
    __read_n_basis()
        Read the number of basis states and store it as an attribute.
    __read_matrix(tag,convert=False) : array of float64
        Read a single matrix (i.e., either the real or imaginary part) from the anisofile
        indicated by the tag given as an argument, and return it.
    __read_vector(tag,output='vector',convert=False) : array of float64
        Read a single vector from the anisofile indicated by the the given as an argument.
        This is mostly intended for reading the eigenvalues of the SOC operator given as
        a vector in the anisofile. The results can be given either as a vector
        (output='vector', default) or as a diagonal matrix (output='matrix').

    Public methods
    --------------
    hamiltonian() : array of complex128
        A matrix representation of the spin-orbit coupled Hamiltonian.
    magnetic_moment(include_bohr_magneton=True) : list of array of complex128
        A list of the matrix representation of the three Cartesian components of the
        magnetic moment. The optional argument is used to set whether the Bohr
        magneton is included in the operator matrices or not.
    spin() : list of array of complex128
        A list of the matrix representation of the three Cartesian components of the
        spin.

    Class methods
    -------------
    run_tests(print_output=True) : boolean
        Initiate the class using a synthetic anisofile and run a set of
        internal tests. Return True if all tests passed.
    """

    def __read_version(self):
        """Read version of the anisofile and store it as an attribute."""
        f = open(self.filename)

        while True:
            line = f.readline()
            if line.startswith('$format'):
                line = f.readline()
                self.version = line.rstrip()
                break
            elif line == '':
                print("ERROR in OrcaAnisoFile:")
                print("Error: File format version not found.")
                print("Error termination.")
                sys.exit(1)

        f.close()

    
    def __read_n_basis(self):
        """Read the number of basis states and store it as an attribute."""
        f = open(self.filename)

        while True:
            line = f.readline()
            if line.startswith('$nss'):
                line = f.readline()
                self.n_basis = int(line)
                break
            elif line == '':
                print("ERROR in OrcaAnisoFile:")
                print("Error: Number of basis states not found.")
                print("Error termination.")
                sys.exit(1)

        f.close()

    
    def __read_matrix(self,tag,convert=False):
        """Read a single matrix (i.e., either the real or imaginary part) from the anisofile
        indicated by the tag given as an argument, and return it. If the optional argument
        convert is set to True, the values read will be converted to the current energy unit
        as defined in the units instance stored as an attribute.
        """
        f = open(self.filename)

        while True:
            line = f.readline()
            if line.startswith(tag):
                line = f.readline()
                tmp_n_basis = int(line.split()[0])
                break
            elif line == '':
                print("ERROR in OrcaAnisoFile:")
                print("Error: Matrix with tag " + tag + " not found.")
                print("Error termination.")
                sys.exit(1)

        if not self.n_basis == tmp_n_basis:
            print("ERROR in OrcaAnisoFile:")
            print("Error: The dimension of matrix does not match the number of basis states.")
            print("Error termination.")
            sys.exit(1)

        if self.n_basis % 5 == 0:
            n_block_lines = self.n_basis // 5
        else:
            n_block_lines = self.n_basis // 5 + 1

        matrix = np.zeros((self.n_basis,self.n_basis), dtype=np.float64)

        for i in range(0,self.n_basis):
            for block_line in range(0,n_block_lines):
                line = f.readline()

                # The last line of a block contains n_basis % 5 values,
                # except when n_basis is divisible by five, in which case
                # it contains a full five values.
                if (block_line == n_block_lines - 1) and (self.n_basis % 5 != 0):
                    max_k = self.n_basis % 5
                else:
                    max_k = 5

                for k in range(0,max_k):
                    value = float(line.split()[k])
                    j = 5*block_line + k

                    if convert:
                        matrix[i][j] = self.units.convert_energy_unit(value,'hartree')
                    else:
                        matrix[i][j] = value
                
        f.close()
        return matrix

    
    def __read_vector(self,tag,output='vector',convert=False):
        """Read a single vector from the anisofile indicated by the the given as an
        argument. This is mostly intended for reading the eigenvalues of the SOC
        operator given as a vector in the anisofile. The results can be given either
        as a vector (output='vector', default) or as a diagonal matrix (output='matrix').
        """
        f = open(self.filename)

        while True:
            line = f.readline()
            if line.startswith(tag):
                line = f.readline()
                tmp_n_basis = int(line.split()[0])
                break
            elif line == '':
                print("ERROR in OrcaAnisoFile:")
                print("Error: Vector with tag " + tag + " not found.")
                print("Error termination.")
                sys.exit(1)

        if not self.n_basis == tmp_n_basis:
            print("ERROR in OrcaAnisoFile:")
            print("Error: The dimension of vector does not match the number of basis states.")
            print("Error termination.")
            sys.exit(1)

        if self.n_basis % 5 == 0:
            n_block_lines = self.n_basis // 5
        else:
            n_block_lines = self.n_basis // 5 + 1
            
        vector = np.zeros(self.n_basis, dtype=np.float64)

        for block_line in range(0,n_block_lines):
            line = f.readline()

            # The last line contains n_basis % 5 values, except when n_basis
            # is divisible by five, in which case it contains a full five values.
            if (block_line == n_block_lines - 1) and (self.n_basis % 5 != 0):
                max_k = self.n_basis % 5
            else:
                max_k = 5

            for k in range(0,max_k):
                value = float(line.split()[k])
                j = 5*block_line + k

                if convert:
                    vector[j] = self.units.convert_energy_unit(value,'hartree')
                else:
                    vector[j] = value
                
        f.close()

        if output == 'vector':
            return vector
        
        elif output == 'matrix':
            matrix = np.diag(vector)
            return matrix
        
        else:
            print("ERROR in OrcaAnisoFile:")
            print("Error: Unrecognized option for output: " + output)
            print("Error termination.")
            sys.exit(1)
    
        
    def hamiltonian(self):
        """A matrix representation of the spin-orbit coupled Hamiltonian."""

        matrix_real = self.__read_vector('$eso',output='matrix',convert=True)
        matrix_imag = np.zeros((self.n_basis,self.n_basis), dtype=np.float64)

        lowest_diagonal_value = min(np.diag(matrix_real))

        for i in range(0,matrix_real.shape[0]):
            matrix_real[i][i] -= lowest_diagonal_value

        return matrix_real + 1j*matrix_imag


    
    def magnetic_moment(self, include_bohr_magneton=True):
        """A list of the matrix representation of the three Cartesian components of the
        magnetic moment. The optional argument is used to set whether the Bohr
        magneton is included in the operator matrices or not.
        """
        matrix_list = []
        tag_prefix_list = ['$magn_x','$magn_y','$magn_z']
        for i in range(0,3):
            matrix_real = self.__read_matrix(tag_prefix_list[i] + 'r')
            matrix_imag = self.__read_matrix(tag_prefix_list[i] + 'i')

            matrix_full = matrix_real + 1j*matrix_imag
            if include_bohr_magneton:
                matrix_full = self.units.mu_B * matrix_full

            matrix_list.append(deepcopy(matrix_full))

        return matrix_list

    
    def spin(self):
        """A list of the matrix representation of the three Cartesian components of the
        spin.
        """
        matrix_list = []
        tag_prefix_list = ['$spin_x','$spin_y','$spin_z']
        for i in range(0,3):
            matrix_real = self.__read_matrix(tag_prefix_list[i] + 'r')
            matrix_imag = self.__read_matrix(tag_prefix_list[i] + 'i')

            matrix_full = matrix_real + 1j*matrix_imag
            matrix_list.append(deepcopy(matrix_full))

        return matrix_list

    
    def __repr__(self):
        """Return some basis info on the calculation."""
        tmp_str = ""
        tmp_str += "ORCA ANISOFILE\n\n"
        tmp_str += "  File name:           " + self.filename + "\n"
        tmp_str += "  File format version: " + self.version + "\n\n"

        return tmp_str

    
    def __init__(self,filename,units):
        """Upon class initiation, just check that the file exists and that it is of the
        correct version.
        """
        self.filename = filename
        self.units    = units

        try:
            f = open(self.filename)
        except FileNotFoundError:
            print("ERROR in OrcaAnisoFile.")
            print("ERROR: file " + self.filename + " not found.")
            print("Error termination.")
            sys.exit(1)
        else:
            f.close()

        self.__read_version()
        allowed_versions = ['2020']

        if not self.version in allowed_versions:
            print("ERROR in OrcaAnisoFile.:")
            print("Error: Unsupported format version.")
            print("Error termination.")
            sys.exit(1)

        self.__read_n_basis()


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class using a synthetic anisofile and run a set of
        tests on the class constructor and the reading methods. Return True
        if all tests passed and False otherwise.

        The synthetic file uses five basis states so that each matrix row
        fills a complete line of five values, which tests the handling of
        matrix dimensions divisible by five in the block reader.

        Optional arguments
        ------------------
        print_output : boolean
            Whether to print the outcome of each test. Default is True.
        """
        import os
        import tempfile

        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('OrcaAnisoFile',
                                                       test_name,condition,print_output))

        tmp_units = units.EnergyUnitSystem('wavenumber')

        n_basis = 5
        rng = np.random.default_rng(1)

        eso = np.array([0.0,1.0e-4,2.0e-4,3.0e-4,4.0e-4])

        # Random real and imaginary parts for the magnetic moment and spin
        # components. Hermiticity is not required by the reader.
        magn_parts = {}
        spin_parts = {}
        for component in ('x','y','z'):
            for part in ('r','i'):
                magn_parts[component + part] = rng.standard_normal((n_basis,n_basis))
                spin_parts[component + part] = rng.standard_normal((n_basis,n_basis))

        def matrix_block(matrix):
            block_str = "{0}\n".format(n_basis)
            for i in range(0,n_basis):
                block_str += " ".join("{0:22.14e}".format(value)
                                      for value in matrix[i]) + "\n"
            return block_str

        content  = "$format\n2020\n"
        content += "$nss\n{0}\n".format(n_basis)
        content += "$eso\n{0}\n".format(n_basis)
        content += " ".join("{0:22.14e}".format(value) for value in eso) + "\n"
        for component in ('x','y','z'):
            for part in ('r','i'):
                content += "$magn_" + component + part + "\n"
                content += matrix_block(magn_parts[component + part])
        for component in ('x','y','z'):
            for part in ('r','i'):
                content += "$spin_" + component + part + "\n"
                content += matrix_block(spin_parts[component + part])

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir,'orca.anisofile')
            with open(filename,'w') as f:
                f.write(content)

            anisofile = cls(filename,tmp_units)

            check('format version read', anisofile.version == '2020')
            check('number of basis states read', anisofile.n_basis == n_basis)

            # The Hamiltonian is diagonal with the eigenvalues converted from
            # hartree and translated so that the lowest one is zero.
            hamiltonian = anisofile.hamiltonian()
            hartree = tmp_units.convert_energy_unit(1.0,'hartree')
            check('Hamiltonian eigenvalues converted from hartree',
                  np.allclose(np.diag(hamiltonian).real,(eso - eso[0])*hartree)
                  and np.allclose(hamiltonian - np.diag(np.diag(hamiltonian)),0.0))

            moment_list = anisofile.magnetic_moment(include_bohr_magneton=False)
            moment_ok = all(np.allclose(moment_list[i],
                                        magn_parts[component + 'r']
                                        + 1.0j*magn_parts[component + 'i'])
                            for i, component in enumerate(('x','y','z')))
            check('magnetic moment matrices read completely', moment_ok)

            moment_list = anisofile.magnetic_moment(include_bohr_magneton=True)
            check('Bohr magneton included on request',
                  np.allclose(moment_list[2],
                              tmp_units.mu_B*(magn_parts['zr'] + 1.0j*magn_parts['zi'])))

            spin_list = anisofile.spin()
            spin_ok = all(np.allclose(spin_list[i],
                                      spin_parts[component + 'r']
                                      + 1.0j*spin_parts[component + 'i'])
                          for i, component in enumerate(('x','y','z')))
            check('spin matrices read completely', spin_ok)

        return debug_output.test_summary('OrcaAnisoFile',result_list,print_output)
