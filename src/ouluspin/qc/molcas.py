# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

import sys

import numpy as np

from ouluspin import units
from ouluspin import tensors


class AnisoCalculation:
    """A class to read data from an output file of a SINGLE_ANISO or POLY_ANISO calculation.
    Currently only functionality for SINGLE_ANISO has been implemented. This class is
    intended to be inherited by the respective classes of OpenMolcas and Orca calculations,
    where the methods are to be called. The attributes listed below define the interface
    that the reading methods rely on; they are set by the constructors of the inheriting
    classes.

    NOTE!!! All angular momentum quantum numbers are given as multiplets of two in order to treat
            half-integer angular momenta as integer numbers.

    Arguments
    ---------

    Optional arguments
    ------------------

    Attributes
    ----------
    filename : str
        Name of the ADF output file containing the data.
    debug : boolean
        Whether to print debug information. Default is False.
    pseudospin : int
        Value of the pseudospin.
    single_aniso_start_str : str
        A string indicating the start of the the SINGLE_ANISO output in the OpenMolcas output
        file.
    output_instance : int
        Assuming that the output file read contains outputs of several SINGLE_ANISO
        or POLY_ANISO runs, this parameter determines which of these will be read. The
        first output corresponds to value of zero.
    units : EnergyUnitSystem
       Instance of EnergyUnitSystem containing the necessary tools to convert the energy units.
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
    read_pseudospin(multiplet_index=1) : int
        Read and return the magnitude of the pseudospin.
    read_magnetic_moment(multiplet_index=1,allow_negative_gzz=False) : list of IwaharaChibotaruSphericalTensor
        Read and return the ITO expansions of the magnetic moment for each Cartesian component.
    read_zfs_tensor(multiplet_index=1) : IwaharaChibotaruSphericalTensor
        Read the ITO expansions of the ZFS operator and return it as an instances of
        IwaharaChibotaruSphericalTensor.
    read_crystal_field() : IwaharaChibotaruSphericalTensor
        Read the ITO expansion of the ab initio crystal-field operaotr and return it as an
        instance of IwaharaChibotaruSphericalTensor. This method is only available if for
        lanthanides and only if the ab initio crystal-field calculation has been requested.
        The pseudospin is read as the pseudospin corresponding to the lanthanide ground
        multiplet as printed in the SINGLE_ANISO output and does not necessarily correspond
        to the pseudospin of the magnetic moment and ZFS pseudospin operators.
    read_rotation(self,multiplet_index=1) : Rotation
        Read the rotation matrix, which takes the system from input
        coordinate system to the principal magnetic axis coordinate system and
        return an instance of Rotation based on the matrix.
    old_read_zfs_operator_matrix(multiplet_index=1) : array of complex128
        Read a matrix representation of the ZFS oparator for a given multiplet from an
        output file produced using OpenMolcas version of SINGLE_ANISO. Note that
        SINGLE_ANISO prints a rather modest number of digits in the output; thus, the
        numerical precision of the operator matrix is not high.
    new_read_zfs_operator_matrix(multiplet_index=1) : array of complex128
        Read a matrix representation of the ZFS oparator for a given multiplet from an
        output file produced using Orca version of SINGLE_ANISO.
    """

    def read_pseudospin(self, multiplet_index=1):
        """Read and return the magnitude of the pseudospin.

        The optional argument multiplet_index determines from which pseudospin multiplet the
        pseudospin is read if several ones have been calculated. The default value is 1, which
        indicates the ground multiplet pseudospin.
        """
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
                print("ERROR in OpenMolcasCalculation.:")
                print("Error: The file " + self.filename + " does not contain SINGLE_ANISO output.")
                print("Error termination.")
                sys.exit(1)
        
        while True:
            line = f.readline()
            if line.startswith('     CALCULATION OF PSEUDOSPIN HAMILTONIAN TENSORS FOR THE'):
                if int(line.split()[8]) == multiplet_index:
                    pseudospin_string = line.split()[13]
                    pseudospin_string = pseudospin_string.replace(')','')
                    if "/" in pseudospin_string:
                        pseudospin = int(pseudospin_string.split("/")[0])
                    else:
                        pseudospin = 2*int(pseudospin_string)
                    break
            if line == "":
                print("ERROR in OpenMolcasCalculation.")
                print("Error: End of file reached when looking for pseudospin.")
                print("Error termination.")
                sys.exit()
        f.close()
        return pseudospin

    
    def read_magnetic_moment(self, multiplet_index=1, allow_negative_gzz=False):
        """Read and return the ITO expansions of the magnetic moment. The parameters are
        returned as a list with each list item corresponding to a single Cartesian component.
        The parameters are first read and stored as an instances of
        ChibotaruUngurSphericalTensor and then converted to the Iwahara--Chibotaru format.

        The optional argument multiplet_index determines from which pseudospin multiplet the
        ITO expansions are read if more than one multiplet is included in the SINGLE_ANISO
        calculation. The default value is 1 indicating the ground pseudospin multiplet. If
        the optional argument allow_negative_gzz is set to true, the phase of the ITO
        decomposition parameters will not be corrected to provide a positive g_zz element.

        SINGLE_ANISO prints the parameters in the main magnetic axes frame
        of the multiplet in question, which is different for every
        multiplet. Following the convention of the library, the returned
        components are rotated to the INPUT axis frame (the frame in which
        the ab initio data was given): the magnetic moment is a
        Cartesian-indexed quantity, so both the Cartesian components and
        the ITO structure are rotated (a mixed Cartesian--spherical
        rotation).
        """
        index_to_cartesian = {0: "X", 1: "Y", 2: "Z"}
        cartesian_to_index = {"X": 0, "Y": 1, "Z": 2}

        rank_list = []
        real_parameter_list = []
        imag_parameter_list = []
        for i in range(0,3):
            rank_list.append([])
            real_parameter_list.append([])
            imag_parameter_list.append([])

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
                print("ERROR in OpenMolcasCalculation.:")
                print("Error: The file " + self.filename + " does not contain SINGLE_ANISO output.")
                print("Error termination.")
                sys.exit(1)

        # Find the correct multiplet.
        while True:
            line = f.readline()
            if line.startswith("     CALCULATION OF PSEUDOSPIN HAMILTONIAN TENSORS FOR THE MULTIPLET"):
                if int(line.split()[8]) == multiplet_index:
                    break
            if line == "":
                print("ERROR in OpenMolcasCalculation.")
                print("Error: End of file reached when looking for pseudospin multiplet " + str(multiplet_index) + ".")
                print("Error termination.")
                sys.exit(1)

        # Find the ITO listing.
        while True:
            line = f.readline()
            if line.startswith("DECOMPOSITION OF THE MAGNETIC MOMENT Mu_i IN IRREDUCIBLE TENSOR OPERATORS (ITO):"):
                break

            if line == "":
                print("ERROR in OpenMolcasCalculation.")
                print("Error: End of file reached when looking for ITO decomposition of magnetic moment.")
                print("Error termination.")
                sys.exit(1)

        # Find start of the parameter listing.
        while True:
            line = f.readline()
            if line.startswith("  n  |  m  | i |        B(i,n,m)       |        C(i,n,m)"):
                break
            if line == "":
                print("ERROR in OpenMolcasCalculation.")
                print("Error: End of file reached when looking for the start of magnetic moment ITO listing.")
                print("Error termination.")
                sys.exit(1)

        # Read the parameters.
        g_zz = None

        while True:
            line = f.readline()

            if len(line.split()) == 10:
                k = 2*int(line.split()[0])
                q = 2*int(line.split()[2])
                A = float(line.split()[6])
                C = float(line.split()[8])

                cartesian_index = cartesian_to_index[line.split()[4]]

                if (k == 2) and (q == 0):
                    g_zz = A
                
                rank_list[cartesian_index].append((k,q))
                real_parameter_list[cartesian_index].append(A)
                imag_parameter_list[cartesian_index].append(C)
            
            if line == "\n":
                break
            if line == "":
                print("ERROR in OpenMolcasCalculation.")
                print("Error: End of file reached while reading magnetic moment ITO listing.")
                print("Error termination.")
                sys.exit(1)

        f.close()

        # If the g_10^z element corresponding to the Cartesian g_zz element is negative,
        # change the phases of all parameters so that it is positive. If no rank (1,0)
        # parameter was present in the listing, no phase correction can be made.
        if (not allow_negative_gzz) and (g_zz is not None) and (g_zz < 0.0):
            for i in range(0,3):
                for j in range(0,len(rank_list[i])):
                    real_parameter_list[i][j] *= -1.0
                    imag_parameter_list[i][j] *= -1.0

        # Convert from Chibotaru--Ungur definition to Iwahara--Chibotaru definition.
        CU_decomposition_list = []
        for i in range(0,3):
            CU_decomposition_list.append(tensors.ChibotaruUngurSphericalTensor(rank_list[i],
                                                                               real_parameter_list[i],
                                                                               imag_parameter_list[i],
                                                                               self.pseudospin))

        IC_decomposition_list = []
        for i in range(0,3):
            IC_decomposition_list.append(CU_decomposition_list[i].iwahara_chibotaru_spherical_tensor())

        # The parameters were printed in the main magnetic axes frame of
        # this multiplet; rotate the full mixed Cartesian--spherical moment
        # back to the input frame (the frame rotations of the library are
        # always FROM the input frame, so the inverse rotation is applied
        # here).
        mixed_moment = tensors.MixedCartesianIwaharaChibotaruSphericalTensor(IC_decomposition_list)
        rotation = self.read_rotation(multiplet_index)
        mixed_moment.rotate(rotation.inverse())
        mixed_moment.frame = 'input axis frame'
        for component in mixed_moment.component_list:
            component.frame = 'input axis frame'

        return mixed_moment.component_list

    
    def read_zfs_tensor(self, multiplet_index=1):
        """Read the ITO expansions of the ZFS operator and return it as an  instances of
        IwaharaChibotaruSphericalTensor. The parameters are first read and stored as an instances of
        ChibotaruUngurITOExpansion and then converted to the Iwahara--Chibotaru format.

        The optional argument multiplet_index determines from which pseudospin multiplet the
        ITO expansions are read if more than one multiplet is included in the SINGLE_ANISO
        calculation. The default value is 1 indicating the ground pseudospin multiplet.

        SINGLE_ANISO prints the parameters in the main magnetic axes frame
        of the multiplet in question, which is different for every
        multiplet. Following the convention of the library, the returned
        tensor is rotated to the INPUT axis frame (the frame in which the
        ab initio data was given), so that tensors read for different
        multiplets or from different calculations are directly comparable.
        """
        rank_list           = []
        real_parameter_list = []
        imag_parameter_list = []

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
                print("ERROR in OpenMolcasCalculation.:")
                print("Error: The file " + self.filename + " does not contain SINGLE_ANISO output.")
                print("Error termination.")
                sys.exit(1)

        # Find the correct multiplet
        while True:
            line = f.readline()
            if line.startswith("     CALCULATION OF PSEUDOSPIN HAMILTONIAN TENSORS FOR THE MULTIPLET"):
                if int(line.split()[8]) == multiplet_index:
                    break
            if line == "":
                print("ERROR in OpenMolcasCalculation.")
                print("Error: End of file reached when looking for pseudospin multiplet " + str(multiplet_index) + ".")
                print("Error termination.")
                sys.exit(1)

        # Find the ZFS ITO listing.
        while True:
            line = f.readline()
            if line.startswith("   ZFS = SUM_{n,m}: [ E(n,m) * O(n,m) +  F(n,m) * W(n,m) ]"):
                break

            if line == "":
                print("ERROR in OpenMolcasCalculation.")
                print("Error: End of file reached when looking for ITO decomposition of ZFS.")
                print("Error termination.")
                sys.exit(1)

        while True:
            line = f.readline()
            if line.startswith("  n  |  m  |         E(n,m)        |         F(n,m)"):
                break
            if line == "":
                print("ERROR in OpenMolcasCalculation.")
                print("Error: End of file reached when looking for the start of ZFS ITO listing.")
                print("Error termination.")
                sys.exit(1)

        while True:
            line = f.readline()

            if len(line.split()) == 8:
                k = 2*int(line.split()[0])
                q = 2*int(line.split()[2])
                E = self.units.convert_energy_unit(float(line.split()[4]),'wavenumber')
                F = self.units.convert_energy_unit(float(line.split()[6]),'wavenumber')
                
                rank_list.append((k,q))
                real_parameter_list.append(E)
                imag_parameter_list.append(F)
            
            if line.startswith("**************************"):
                break
            if line == "":
                print("ERROR in OpenMolcasCalculation.")
                print("Error: End of file reached while reading ZFS ITO listing.")
                print("Error termination.")
                sys.exit(1)

        f.close()

        # Convert from Chibotaru--Ungur definition to Iwahara--Chibotaru definition.
        CU_decomposition = tensors.ChibotaruUngurSphericalTensor(rank_list,
                                                                 real_parameter_list,
                                                                 imag_parameter_list,
                                                                 self.pseudospin)

        IC_decomposition = CU_decomposition.iwahara_chibotaru_spherical_tensor()

        # The parameters were printed in the main magnetic axes frame of
        # this multiplet; rotate the tensor back to the input frame (the
        # frame rotations of the library are always FROM the input frame,
        # so the inverse rotation is applied here).
        rotation = self.read_rotation(multiplet_index)
        IC_decomposition.rotate(rotation.inverse())
        IC_decomposition.frame = 'input axis frame'

        if self.debug:
            print("Decomposition of the ZFS using Chibotaru--Ungur definition of ITOs:")
            print()
            print(CU_decomposition)

        if self.debug:
            print("Decomposition of the ZFS  using Iwahara--Chibotaru definition of ITOs:")
            print()
            print(IC_decomposition)

        return IC_decomposition


    def read_crystal_field(self):
        """Read the ITO expansion of the ab initio crystal-field operaotr and return it as an
        instance of IwaharaChibotaruSphericalTensor. This method is only available if for
        lanthanides and only if the ab initio crystal-field calculation has been requested.
        The pseudospin is read as the pseudospin corresponding to the lanthanide ground
        multiplet as printed in the SINGLE_ANSIO output and does not necessarily correspond
        to the pseudospin of the magnetic moment and ZFS pseudospin operators.

        The crystal-field parameters are printed by SINGLE_ANISO, and are
        returned by this method, in the main magnetic axes frame of the
        ground multiplet, which is the principal magnetic axis frame in
        which the spherical tensors of the library are conventionally
        expressed. The rotation from the input axis frame to this frame is
        parsed from the crystal-field section, when present, and recorded
        in the input_frame_rotation attribute of this instance (rotations
        of the library are always stored as rotations FROM the input
        frame); the attribute is None when the rotation block was not
        found.
        """
        rank_list = []
        parameter_list = []

        self.input_frame_rotation = None
        axes_rows = {}

        f = open(self.filename, 'r')
        instance_counter = 0
        while True:
            line = f.readline()
            if line.startswith(self.single_aniso_start_str):
                if instance_counter >= self.output_instance:
                    break
                else:
                    instance_counter += 1
            elif line == '':
                print("ERROR in OpenMolcasCalculation.:")
                print("Error: The file " + self.filename + " does not contain SINGLE_ANISO output.")
                print("Error termination.")
                sys.exit(1)

        while True:
            line = f.readline()
            if line.startswith('     CALCULATION OF CRYSTAL-FIELD PARAMETERS OF THE GROUND ATOMIC MULTIPLET J'):
                J_str = line.split()[11]
                if len(J_str.split('/')) == 1:
                    J = 2 * int(J_str.replace('.',''))
                else:
                    J = int(J_str.split('/')[0])
                break
            elif line == '':
                print("ERROR in OpenMolcasCalculation.")
                print("Error: Value of J not found in the SINGLE_ANISO output.")
                print("Error termination.")
                sys.exit(1)
                
        while True:
            line = f.readline()
            if line.lower().startswith('  k |  q  |           complex parameter  a(k,q)'):
                break
            # The rotation from the initial (input) coordinate system to the
            # magnetic axes frame of the crystal field is printed in this
            # section as rows labeled Xm, Ym and Zm.
            parts = line.replace('R =',' ').replace('|',' ').split()
            if len(parts) == 4 and parts[0] in ('Xm','Ym','Zm'):
                axes_rows[parts[0]] = [float(x) for x in parts[1:4]]
            if line == '':
                print("ERROR in OpenMolcasCalculation.:")
                print("Error: End of file reached while searching ab intio crystal-field parameters.")
                print("Error termination.")
                sys.exit(1)

        while True:
            line = f.readline()
            if line.startswith('---'):
                pass
            elif line == '\n':
                break
            elif line.split()[6] == '*I':
                tmp_k = 2 * int(line.split()[0])
                tmp_q = 2 * int(line.split()[2])

                tmp_real = self.units.convert_energy_unit(float(line.split()[4]),'wavenumber')
                tmp_imag = self.units.convert_energy_unit(float(line.split()[5]),'wavenumber')

                rank_list.append((tmp_k,tmp_q))
                parameter_list.append(complex(tmp_real,tmp_imag))

        if len(rank_list) == 0:
            print("ERROR in OpenMolcasCalculation.")
            print("Error: No crystal-field parameters read.")
            print("Error termination.")
            sys.exit(1)

        if not len(rank_list) == len(parameter_list):
            print("ERROR in OpenMolcasCalculation.")
            print("Error: Inconsistent number of crystal-field parameters and ranks.")
            print("Error termination.")
            sys.exit(1)

        f.close()

        if len(axes_rows) == 3:
            # See read_rotation for the reasoning behind making the frame
            # rotation proper.
            matrix = np.array([axes_rows['Xm'],axes_rows['Ym'],axes_rows['Zm']],
                              dtype=np.float64)
            if np.linalg.det(matrix) < 0.0:
                matrix = -matrix
            self.input_frame_rotation = tensors.Rotation(matrix)

        crystal_field = tensors.IwaharaChibotaruSphericalTensor(rank_list,parameter_list)
        crystal_field.frame = 'principal magnetic axis frame'

        return crystal_field



    def read_rotation(self,multiplet_index=1):
        """Read the rotation matrix, which takes the system from input
        coordinate system to the principal magnetic axis coordinate system and
        return an instance of Rotation based on the matrix.

        The rotation matrix is taken as the set of principal magnetic axes. The matrix
        is printed in SINGLE_ANISO in such a way that the principal magnetic axes
        (i.e., the eigenvectors) are the horizontal rows. Thus, the listed matrix is
        the R^-1 matrix in the diagonalizing transformation R^-1*g*R. Since elsewhere
        in the code we will transform the nuclear tensors using a transformation of
        the form U*A*U^-1, the listed R^-1 = U matrix is the rotation matrix we are
        looking for.

        The optional argument multiplet_index determines from which pseudospin multiplet the
        ITO expansions are read if more than one multiplet is included in the SINGLE_ANISO
        calculation. The default value is 1 indicating the ground pseudospin multiplet.
        """
        matrix = np.zeros((3,3), dtype=np.float64)
        
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
                print("ERROR in OpenMolcasCalculation.:")
                print("Error: The file " + self.filename + " does not contain SINGLE_ANISO output.")
                print("Error termination.")
                sys.exit(1)

        # Find the correct multiplet
        while True:
            line = f.readline()
            if line.startswith("     CALCULATION OF PSEUDOSPIN HAMILTONIAN TENSORS FOR THE MULTIPLET"):
                if int(line.split()[8]) == multiplet_index:
                    break
            if line == "":
                print("ERROR in OpenMolcasCalculation.")
                print("Error: End of file reached when looking for pseudospin multiplet " + str(multiplet_index) + ".")
                print("Error termination.")
                sys.exit(1)

        while True:
            line = f.readline()
            if line.startswith('    MAIN VALUES    |             MAIN MAGNETIC AXES     |'):
                break
            if line == "":
                print("ERROR in OpenMolcasCalculation.")
                print("Error: End of file reached when looking for transformation matrix.")
                print("Error termination.")
                sys.exit(1)

        f.readline()
        for i in range(0,3):
            line = f.readline()
            for j in range(0,3):
                matrix[i][j] = float(line.split()[6+j])

        f.close()

        # SINGLE_ANISO fixes the signs of the printed magnetic axes
        # arbitrarily, so the axes may form an improper set (the output even
        # reports the sign of the product gX*gY*gZ). All operators of this
        # library are built on angular momenta and are therefore parity
        # even, so an improper frame map acts on them identically to its
        # proper counterpart -R; the overall sign is flipped so that a
        # proper rotation is always returned.
        if np.linalg.det(matrix) < 0.0:
            matrix = -matrix

        return tensors.Rotation(matrix)


    def old_read_zfs_operator_matrix(self,multiplet_index=1):
        """Read a matrix representation of the ZFS oparator for a given multiplet from an
        output file produced using OpenMolcas version of SINGLE_ANISO. Note that
        SINGLE_ANISO prints a rather modest number of digits in the output; thus, the
        numerical precision of the operator matrix is not high.
        """
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
                print("ERROR in OpenMolcasCalculation.:")
                print("Error: The file " + self.filename + " does not contain SINGLE_ANISO output.")
                print("Error termination.")
                sys.exit(1)

        # Find the correct multiplet
        while True:
            line = f.readline()
            if line.startswith("     CALCULATION OF PSEUDOSPIN HAMILTONIAN TENSORS FOR THE MULTIPLET"):
                if int(line.split()[8]) == multiplet_index:
                    break
            if line == "":
                print("ERROR in OpenMolcasCalculation.")
                print("Error: End of file reached when looking for pseudospin multiplet " + str(multiplet_index) + ".")
                print("Error termination.")
                sys.exit(1)

        # Find the matrix listing
        while True:
            line = f.readline()
            if line.startswith("Ab Initio Calculated Zero-Field Splitting Matrix written in the basis of Pseudospin Eigenfunctions"):
                break
            if line == "":
                print("ERROR in OpenMolcasCalculation.")
                print("Error: End of file reached when looking for ZFS operator matrix.")
                print("Error termination.")
                sys.exit(1)

        f.readline()
        f.readline()
        f.readline()

        n_basis = self.pseudospin + 1

        matrix = np.zeros((n_basis,n_basis), dtype=np.complex128)
        
        for i in range(0,n_basis):
            line = f.readline()
            modified_line = line[8:].replace('|',' ')
            
            for j in range(0,n_basis):
                real_value = float(modified_line.split()[2*j])
                imag_value = float(modified_line.split()[2*j + 1])

                real_value = self.units.convert_energy_unit(real_value,'wavenumber')
                imag_value = self.units.convert_energy_unit(imag_value,'wavenumber')

                matrix[i][j] = complex(real_value,imag_value)

        f.close()
        return matrix


    def new_read_zfs_operator_matrix(self,multiplet_index=1):
        """Read a matrix representation of the ZFS oparator for a given multiplet from an
        output file produced using Orca version of SINGLE_ANISO.
        """
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
                print("ERROR in OpenMolcasCalculation.:")
                print("Error: The file " + self.filename + " does not contain SINGLE_ANISO output.")
                print("Error termination.")
                sys.exit(1)

        # Find the correct multiplet
        while True:
            line = f.readline()
            if line.startswith("     CALCULATION OF PSEUDOSPIN HAMILTONIAN TENSORS FOR THE MULTIPLET"):
                if int(line.split()[8]) == multiplet_index:
                    break
            if line == "":
                print("ERROR in OpenMolcasCalculation.")
                print("Error: End of file reached when looking for pseudospin multiplet " + str(multiplet_index) + ".")
                print("Error termination.")
                sys.exit(1)

        # Find the matrix listing
        while True:
            line = f.readline()
            if line.startswith("      Ab Initio Calculated ZFS Matrix written in the basis of Pseudospin Eigenfunctions"):
                break
            if line == "":
                print("ERROR in OpenMolcasCalculation.")
                print("Error: End of file reached when looking for ZFS operator matrix.")
                print("Error termination.")
                sys.exit(1)

        n_basis = self.pseudospin + 1
        if n_basis % 2 == 0:
            n_blocks = n_basis // 2
            even     = True
        else:
            n_blocks = (n_basis + 1) // 2
            even     = False

        matrix = np.zeros((n_basis,n_basis), dtype=np.complex128)

        for block in range(0,n_blocks):
            # For an odd basis dimension (a non-Kramers system) the last
            # block contains only a single column.
            if (not even) and (block == n_blocks - 1):
                n_columns_in_block = 1
            else:
                n_columns_in_block = 2

            f.readline()
            f.readline()
            f.readline()
            for basis_state in range(0,n_basis):
                line = f.readline()
                trimmed_line = line[11:].replace('|','')

                for column in range(0,n_columns_in_block):
                    real_value = float(trimmed_line.split()[2*column])
                    imag_value = float(trimmed_line.split()[2*column + 1])

                    real_value = self.units.convert_energy_unit(real_value,'wavenumber')
                    imag_value = self.units.convert_energy_unit(imag_value,'wavenumber')

                    matrix[basis_state][2*block + column] = complex(real_value,imag_value)
            f.readline()

        f.close()
        return matrix

            
    
class OpenMolcasCalculation(AnisoCalculation):
    """A class to read data from an output file of a SINGLE_ANISO calculations of OpenMolcas.

    NOTE!!! All angular momentum quantum numbers are given as multiplets of two in order to treat
            half-integer angular momenta as integer numbers.

    Arguments
    ---------
    filename : str
       Name of the SINGLE_ANISO output file containing the data.
    units : EnergyUnitSystem
       Instance of EnergyUnitSystem containing the necessary tools to convert the energy units.

    Optional arugments
    ------------------
    aniso : boolean
        Whether the output contains SINGLE_ANISO output. Default is True.
    output_instance : int
        Assuming that the output file read contains outputs of several SINGLE_ANISO
        or POLY_ANISO runs, this parameter determines which of these will be read. The
        first output corresponds to value of zero. Default is 0.
    debug : boolean
        Whether to print debug information. Default is False.

    Attributes
    ----------
    filename : str
       Name of the ADF output file containing the data.
    version : str
        OpenMolcas version..
    debug : boolean
        Whether to print debug information. Default is False.
    pseudospin : int
        Value of the pseudospin.
    single_aniso_start_str : str
        A string indicating the start of the the SINGLE_ANISO output in the OpenMolcas output
        file.
    output_instance : int
        Assuming that the output file read contains outputs of several SINGLE_ANISO
        or POLY_ANISO runs, this parameter determines which of these will be read. The
        first output corresponds to value of zero.
    units : EnergyUnitSystem
       Instance of EnergyUnitSystem containing the necessary tools to convert the energy units.
    input_frame_rotation : Rotation or None
        The rotation from the input axis frame (the frame of the ab initio
        data) to the principal magnetic axis frame of the ground multiplet,
        as recorded by read_crystal_field. None before read_crystal_field
        has been called or when the rotation block was not found in the
        output. Rotations of the library are always stored as rotations
        FROM the input frame.

    Public methods
    --------------
    read_version()
        Read the version of OpenMolcas used to produce the output and store it as an attribute.
        Also define the single_aniso_start_str variable.
    read_coordinates() : tuple of float
        Read coordinates and return them as a (x,y,z) tuple. The coordinates are read as
        the origin of the angular momentum used in the RASSI calculations. If the RASSI
        output is not included in the Molcas output file, the coordinates will not be
        avalaible and a None is returned. NOTE!!! If the angular momentum origin does
        not correspond to the actual coordinate of the spin site, this will lead to 
        incorrect coordinates. While this will most likely not be an issue in the ab initio
        calculation, it will certainly give the wrong results if used in the calculation
        of hyperfine couplings within the point-dipole approximation.

    Private methods
    ---------------

    Class methods
    -------------
    run_tests(print_output=True) : boolean
        Initiate the class using a synthetic output file and run a set of
        internal tests. Return True if all tests passed.
    """

    def read_version(self):
        """Read the version of OpenMolcas used to produce the output and store it as an attribute.
        Also define the single_aniso_start_str variable.
        """
        f = open(self.filename)

        while True:
            line = f.readline()

            if line.startswith('               OPENMOLCASOPENMOLCA        version: 20.10'):
                self.version = "20.10"
                self.single_aniso_start_str = "                                         &SINGLE_ANISO_OPEN"
                break
            elif line.startswith('               OPENMOLCASOPENMOLCA        version: 21.02'):
                self.version = "21.02"
                self.single_aniso_start_str = "                                           &SINGLE_ANISO"
                break
            elif line == '':
                print("ERROR in OpenMolcasCalculation.:")
                print("Error: Unsupported OpenMolcas version or could not read version.")
                print("Error termination.")
                sys.exit(1)
                

        f.close()


    def read_coordinates(self):
        """Read coordinates and return them as a (x,y,z) tuple. The coordinates are read as
        the origin of the angular momentum used in the RASSI calculations. If the RASSI
        output is not included in the Molcas output file, the coordinates will not be
        avalaible and a None is returned. NOTE!!! If the angular momentum origin does
        not correspond to the actual coordinate of the spin site, this will lead to 
        incorrect coordinates. While this will most likely not be an issue in the ab initio
        calculation, it will certainly give the wrong results if used in the calculation
        of hyperfine couplings within the point-dipole approximation.
        """
        f = open(self.filename)
        while True:
            line = f.readline()
            if line.startswith(self.single_aniso_start_str):
                break
            elif line == '':
                print("ERROR in OpenMolcasCalculation.:")
                print("Error: The file " + self.filename + " does not contain SINGLE_ANISO output.")
                print("Error termination.")
                sys.exit(1)

        while True:
            line = f.readline()
            if line.startswith(' PROPERTY: ANGMOM     COMPONENT:   1'):
                line = f.readline()
                x = self.units.bohr_to_angstrom * float(line.split()[1])
                y = self.units.bohr_to_angstrom * float(line.split()[2])
                z = self.units.bohr_to_angstrom * float(line.split()[3])
                return (x,y,z)
            elif line == "":
                return None

        
    def __repr__(self):
        """Return some basis info on the calculation."""
        tmp_str = ""
        tmp_str += "OPENMOLCAS CALCULATION\n\n"
        tmp_str += "  OpenMolcas ouput file: " + self.filename + "\n"
        tmp_str += "  Version:               " + self.version + "\n\n"

        return tmp_str

    
    def __init__(self, filename, units,
                 aniso=True,
                 output_instance=0,
                 debug=False):
        """Upon class initiation, check that the OpenMolcas output file exists."""
        self.filename        = filename
        self.units           = units
        self.output_instance = output_instance
        self.debug           = debug

        try:
            f = open(self.filename)
        except FileNotFoundError:
            print("ERROR in OpenMolcasCalculation.")
            print("ERROR: file " + self.filename + " not found.")
            print("Error termination.")
            sys.exit(1)
        else:
            f.close()

        self.read_version()

        # The rotation from the input axis frame to the principal magnetic
        # axis frame; recorded by read_crystal_field.
        self.input_frame_rotation = None

        if aniso:
            self.pseudospin = self.read_pseudospin()


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class using a synthetic output file and run a set of
        tests on the class constructor and the reading methods. Return True
        if all tests passed and False otherwise.

        The synthetic file reproduces the parts of the SINGLE_ANISO output
        format that the reading methods rely on, so these tests check that
        the parsing logic has not been broken by modifications; they cannot
        detect changes in the OpenMolcas output format itself.

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
            result_list.append(debug_output.test_check('OpenMolcasCalculation',
                                                       test_name,condition,print_output))

        tmp_units = units.EnergyUnitSystem('wavenumber')

        version_line   = "               OPENMOLCASOPENMOLCA        version: 21.02\n"
        start_line     = "                                           &SINGLE_ANISO\n"
        multiplet_line = "     CALCULATION OF PSEUDOSPIN HAMILTONIAN TENSORS FOR THE" \
                         " MULTIPLET 1 ( EFFECTIVE SPIN = {0})\n"

        # A synthetic output for an S = 3/2 multiplet.
        content  = version_line + "\n" + start_line + "\n"
        content += multiplet_line.format("3/2") + "\n"
        content += "DECOMPOSITION OF THE MAGNETIC MOMENT Mu_i IN IRREDUCIBLE TENSOR OPERATORS (ITO):\n"
        content += "  n  |  m  | i |        B(i,n,m)       |        C(i,n,m)\n"
        content += "  1  |  0  | Z |   2.000000  |   0.000000  |\n"
        content += "  1  |  1  | X |   2.000000  |   0.000000  |\n"
        content += "  1  |  1  | Y |   0.000000  |   2.000000  |\n"
        content += "\n"
        content += "   ZFS = SUM_{n,m}: [ E(n,m) * O(n,m) +  F(n,m) * W(n,m) ]\n"
        content += "  n  |  m  |         E(n,m)        |         F(n,m)\n"
        content += "  2  |  0  |   1.500000  |   0.000000  |\n"
        content += "**************************\n"
        content += "\n"
        content += "    MAIN VALUES    |             MAIN MAGNETIC AXES     |\n"
        content += "\n"
        content += "  X: gX = 2.000000 | x,y,z:   1.0   0.0   0.0\n"
        content += "  Y: gY = 2.000000 | x,y,z:   0.0   1.0   0.0\n"
        content += "  Z: gZ = 2.000000 | x,y,z:   0.0   0.0   1.0\n"
        content += "\n"
        content += "      Ab Initio Calculated ZFS Matrix written in the basis of Pseudospin Eigenfunctions\n"
        # Two column blocks of the 4 x 4 matrix diag(1,2,3,4); each block has
        # three header lines, four data rows with an 11-character prefix, and
        # one trailing line.
        block_rows = [[(1.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0)],
                      [(0.0,0.0),(2.0,0.0),(0.0,0.0),(0.0,0.0)],
                      [(0.0,0.0),(0.0,0.0),(3.0,0.0),(0.0,0.0)],
                      [(0.0,0.0),(0.0,0.0),(0.0,0.0),(4.0,0.0)]]
        for block in range(0,2):
            content += "\n\n\n"
            for i in range(0,4):
                row = block_rows[i][2*block:2*block+2]
                content += 11*" " + "  ".join("{0:10.6f} {1:10.6f}".format(v[0],v[1])
                                              for v in row) + "\n"
            content += "\n"

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir,'molcas.out')
            with open(filename,'w') as f:
                f.write(content)

            calculation = cls(filename,tmp_units)

            check('version read', calculation.version == '21.02')
            check('pseudospin read', calculation.pseudospin == 3)

            moment_list = calculation.read_magnetic_moment()
            reference = tensors.IwaharaChibotaruSphericalTensor\
                               .from_one_site_cartesian_operator(2.0,'z',3)
            check('magnetic moment z component converted correctly',
                  moment_list[2] == reference)
            B_plus  = moment_list[0].parameter_list[moment_list[0].rank_list.index([2,2])]
            B_minus = moment_list[0].parameter_list[moment_list[0].rank_list.index([2,-2])]
            check('magnetic moment x component Hermiticity',
                  abs(B_minus + B_plus.conjugate()) < 1.0e-12)

            # <SS|O_2^0|SS> = 3 for S = 3/2, so E(2,0) = 1.5 gives the IC
            # parameter 4.5.
            zfs = calculation.read_zfs_tensor()
            check('ZFS tensor converted correctly',
                  ([4,0] in zfs.rank_list) and
                  abs(zfs.parameter_list[zfs.rank_list.index([4,0])] - 4.5) < 1.0e-12)

            rotation = calculation.read_rotation()
            check('rotation matrix read',
                  np.allclose(rotation.rotation_matrix,np.identity(3)))

            matrix = calculation.new_read_zfs_operator_matrix()
            check('ZFS operator matrix read (even dimension)',
                  np.allclose(matrix,np.diag([1.0,2.0,3.0,4.0])))

            # A second file with an integer pseudospin (S = 1) to test the
            # single-column final block of the ZFS matrix reader.
            content  = version_line + "\n" + start_line + "\n"
            content += multiplet_line.format("1") + "\n"
            content += "      Ab Initio Calculated ZFS Matrix written in the basis of Pseudospin Eigenfunctions\n"
            odd_rows = [[(1.0,0.0),(0.0,0.0),(0.0,0.0)],
                        [(0.0,0.0),(2.0,0.0),(0.0,0.0)],
                        [(0.0,0.0),(0.0,0.0),(3.0,0.0)]]
            for block, n_columns in ((0,2),(1,1)):
                content += "\n\n\n"
                for i in range(0,3):
                    row = odd_rows[i][2*block:2*block+n_columns]
                    content += 11*" " + "  ".join("{0:10.6f} {1:10.6f}".format(v[0],v[1])
                                                  for v in row) + "\n"
                content += "\n"

            filename = os.path.join(tmp_dir,'molcas_odd.out')
            with open(filename,'w') as f:
                f.write(content)

            calculation = cls(filename,tmp_units)
            check('integer pseudospin read', calculation.pseudospin == 2)
            matrix = calculation.new_read_zfs_operator_matrix()
            check('ZFS operator matrix read (odd dimension)',
                  np.allclose(matrix,np.diag([1.0,2.0,3.0])))

            # A third file with non-identity magnetic axes: SINGLE_ANISO
            # prints the ITO parameters in the multiplet's own magnetic
            # axes frame and the readers must rotate them back to the
            # input frame. The printed axes are rotated by 90 degrees
            # about z, and the Zm axis is flipped so that the axes form an
            # improper set, which read_rotation must convert to a proper
            # rotation.
            content  = version_line + "\n" + start_line + "\n"
            content += multiplet_line.format("3/2") + "\n"
            content += "DECOMPOSITION OF THE MAGNETIC MOMENT Mu_i IN IRREDUCIBLE TENSOR OPERATORS (ITO):\n"
            content += "  n  |  m  | i |        B(i,n,m)       |        C(i,n,m)\n"
            content += "  1  |  0  | Z |   2.000000  |   0.000000  |\n"
            content += "  1  |  1  | X |   2.000000  |   0.000000  |\n"
            content += "  1  |  1  | Y |   0.000000  |   2.000000  |\n"
            content += "\n"
            content += "   ZFS = SUM_{n,m}: [ E(n,m) * O(n,m) +  F(n,m) * W(n,m) ]\n"
            content += "  n  |  m  |         E(n,m)        |         F(n,m)\n"
            content += "  2  |  0  |   1.500000  |   0.000000  |\n"
            content += "  2  |  2  |   0.400000  |   0.000000  |\n"
            content += "**************************\n"
            content += "\n"
            content += "    MAIN VALUES    |             MAIN MAGNETIC AXES     |\n"
            content += "\n"
            content += "  X: gX = 2.000000 | x,y,z:   0.0   1.0   0.0\n"
            content += "  Y: gY = 2.000000 | x,y,z:  -1.0   0.0   0.0\n"
            content += "  Z: gZ = 2.000000 | x,y,z:   0.0   0.0  -1.0\n"
            content += "\n"

            filename = os.path.join(tmp_dir,'molcas3.out')
            with open(filename,'w') as f:
                f.write(content)

            calculation = cls(filename,tmp_units)

            rotation = calculation.read_rotation()
            check('improper printed axes converted to a proper rotation',
                  abs(np.linalg.det(rotation.rotation_matrix) - 1.0) < 1.0e-10
                  and np.allclose(rotation.rotation_matrix,
                                  [[0.0,-1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,1.0]]))

            # The reference tensors are the printed multiplet-frame
            # parameters rotated to the input frame with the inverse of the
            # parsed frame rotation.
            zfs_reference = tensors.ChibotaruUngurSphericalTensor(
                [(4,0),(4,4)],[1.5,0.4],[0.0,0.0],3)\
                .iwahara_chibotaru_spherical_tensor()
            zfs_reference.rotate(rotation.inverse())
            zfs = calculation.read_zfs_tensor()
            check('ZFS tensor rotated to the input frame', zfs == zfs_reference)
            check('ZFS tensor frame label', zfs.frame == 'input axis frame')

            moment_components = []
            for ranks, reals, imags in ([(2,0)],[2.0],[0.0]), \
                                       ([(2,2)],[2.0],[0.0]), \
                                       ([(2,2)],[0.0],[2.0]):
                moment_components.append(tensors.ChibotaruUngurSphericalTensor(
                    ranks,reals,imags,3).iwahara_chibotaru_spherical_tensor())
            # The CU-to-IC conversion of the moment listing orders the
            # components as (z, x, y) rows in the synthetic file above but
            # the reader returns (x, y, z); build the reference in the
            # reader's order.
            moment_reference = tensors.MixedCartesianIwaharaChibotaruSphericalTensor(
                [moment_components[1],moment_components[2],moment_components[0]])
            moment_reference.rotate(rotation.inverse())
            moment_list = calculation.read_magnetic_moment()
            check('magnetic moment rotated to the input frame',
                  all(moment_list[alpha] == moment_reference.component_list[alpha]
                      for alpha in range(0,3)))
            check('magnetic moment frame labels',
                  all(component.frame == 'input axis frame'
                      for component in moment_list))

        return debug_output.test_summary('OpenMolcasCalculation',
                                         result_list,print_output)


