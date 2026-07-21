# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import time

from math import sqrt, ceil
from copy import deepcopy

import numpy as np
import numpy.linalg as la

from ouluspin._fortran import fortran_utils as fu

from ouluspin import tensors
from ouluspin import pseudospin_operators
from ouluspin import result_table

from ouluspin import _debug as output


# The lowest temperature, in kelvins, the static magnetic properties are
# evaluated at. The properties follow from the Boltzmann populations of the
# states, i.e. from the factors exp(-E / (k_B*T)), which are not defined at
# T = 0: the factor of the lowest state, whose energy is zero, is the
# indeterminate exp(-0/0) and comes out of a floating-point evaluation as a
# NaN that spreads to the whole result. A temperature point at (or below)
# this limit is therefore raised to it by IsothermalStaticMagnetization and
# StaticMagneticSusceptibility, so that a temperature range given as, say,
# numpy.linspace(0,300,301) can be used as it stands.
#
# The value is deliberately a conservative one. Any temperature above zero
# is in fact well-defined numerically, since the Boltzmann factors of the
# excited states merely underflow to zero, which is the correct zero
# temperature limit; the limit is set well clear of that regime rather than
# at the smallest temperature that still happens to evaluate. At 1.0e-3 K
# the thermal energy k_B*T is about 7.0e-4 cm^-1, which is far below the
# splittings the pseudospin Hamiltonians of the library describe, so the
# properties are those of the zero temperature limit. A system whose
# splitting is smaller still, such as the tunneling gap of a very axial
# single-molecule magnet, is the one case where the difference shows, and
# there the limit can be lowered through the MINIMUM_TEMPERATURE attribute
# of the two classes.
MINIMUM_TEMPERATURE = 1.0e-3



class IsothermalStaticMagnetization:
    """A class to store and print the values of isothermal magnetization calculated
    at different temperatures and different field strenghts.

    This class is not used for the actual calculation of the magnetization. This
    should rather be considered as a container for storing the magnetization.

    Arguments
    ---------
    T_list : list of float
        A list of the different temperatures the isothermal magnetization is
        evaluated in. A temperature at or below MINIMUM_TEMPERATURE is
        raised to it, since the magnetization is not defined at zero
        temperature; see the attribute below.
    B_list : list of float or list of list of float
        A list of the different field values (in tesla) the magnetization is
        evaluated in. The field list can also be a list of lists where the
        outer list index corresponds to different temperatures that have
        different field values. If different B lists are used for different
        temperatures, they must all have the same number of values.

    Optional arguments
    ------------------
    magnetization : array of float64
        Values of the magnetization. The first index corresponds to the
        temperature point and the second to the field point. The default
        is None in which case an empty (zero) magnetization array will
        be constructed upon class initation.

    Attributes
    ----------
    T_list : list of float
        A list of the different temperatures the isothermal magnetization is
        evaluated in, with the temperatures at or below MINIMUM_TEMPERATURE
        raised to it.
    B_list : list of float or list of list of float
        A list of the different field values (in tesla) the magnetization is
        evaluated in. The field list can also be a list of lists where the
        outer list index corresponds to different temperatures that have
        different field values.
    magnetization : array of float64
        Values of the magnetization. The first index corresponds to the
        temperature point and the second to the field point.
    n_T_points : int
        The number of temperature points.
    n_B_points : int
        The number of field points.
    single_field_range : boolean
        Whether a single range of field values is used for all temperature
        points or not.
    MINIMUM_TEMPERATURE : float
        The lowest temperature, in kelvins, the magnetization is evaluated
        at. The magnetization follows from the Boltzmann populations of the
        states and is not defined at zero temperature, so a temperature
        point at or below this limit is raised to it. The limit is a
        conservative one, and low enough that the magnetization at it is
        that of the zero temperature limit; see the module constant
        MINIMUM_TEMPERATURE, which sets it.

    Private methods
    ---------------
    __checked_temperature_list(T_list) : list of float
        Return the temperature list with the temperatures at or below
        MINIMUM_TEMPERATURE raised to it.
    __single_field_range_table() : ResultTable
        Return a human-readable table of the magnetization with a format used
        for a single field range for all temperature points.
    __multiple_field_range_table() : ResultTable
        Return a human-readable table of the magnetization with a format used
        for different field ranges for all temperature points.

    Public methods
    --------------
    data_table() : ResultTable
        Return a human-readable table of the magnetization. This is mostly a
        wrapper for the two different private table methods.
    data_file(filename)
        Write the data to a file.
    comparison_table(comparison,labels=('M','M')) : ResultTable
        Return a human-readable table of the magnetization stored in this
        this instance and in the instance given as an argument along with the
        deviations. The T_list and B_list attributes of this instance and the other
        instance must be equal.
    rms_error(comparison) : float
        Compare the magnetization stored in this instance and that store in the
        instace given as an argument, and return the root-mean-square deviation
        between them.

    Class methods
    -------------
    from_data_file(filename, single_B_range=True, oersteds=False)
        Read the data from a formatted text file.
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if all
        tests passed.
    """

    __UNIT_NOTE = "The product of Bohr magneton and the Avogadro constant is used as the unit."

    # The module constant is published as an attribute of the class, so
    # that the limit can be read, and lowered, through the class itself.
    MINIMUM_TEMPERATURE = MINIMUM_TEMPERATURE

    def __single_field_range_table(self):
        """Return the table of the magnetization, as an instance of
        ResultTable, in the format used when the same field range is used
        for all the temperature points. The first column contains the
        field and the remaining columns the magnetization at each
        temperature.
        """
        column_headers = ['B / T'] \
                         + ["{0:.3f} K".format(T) for T in self.T_list]
        formats        = ['.3f'] + self.n_T_points*['.6f']

        rows = []
        for i in range(0,self.n_B_points):
            rows.append([self.B_list[i]]
                        + [self.magnetization[j][i]
                           for j in range(0,self.n_T_points)])

        return result_table.ResultTable(rows,
                                        column_headers=column_headers,
                                        title="MOLAR MAGNETIZATION",
                                        notes=self.__UNIT_NOTE,
                                        formats=formats,
                                        table_type='magnetization')


    def __multiple_field_range_table(self):
        """Return the table of the magnetization, as an instance of
        ResultTable, in the format used when a different field range is
        used for the different temperature points. Each temperature point
        contributes a pair of columns containing the field and the
        magnetization.
        """
        temperature_headers = []
        field_headers       = []
        formats             = []

        for T in self.T_list:
            temperature_headers.extend(["{0:.3f} K".format(T),""])
            field_headers.extend(['B / T','M'])
            formats.extend(['.6f','.6f'])

        rows = []
        for i in range(0,self.n_B_points):
            row = []
            for j in range(0,self.n_T_points):
                row.extend([self.B_list[j][i],self.magnetization[j][i]])
            rows.append(row)

        return result_table.ResultTable(rows,
                                        column_headers=[temperature_headers,
                                                        field_headers],
                                        title="MOLAR MAGNETIZATION",
                                        notes=self.__UNIT_NOTE,
                                        formats=formats,
                                        table_type='magnetization')


    def data_table(self):
        """Return a table of the magnetization as an instance of
        ResultTable. This is mostly a wrapper for the two different private
        table methods, which are used depending on whether the same field
        range was used for all the temperature points.
        """
        if self.single_field_range:
            return self.__single_field_range_table()
        else:
            return self.__multiple_field_range_table()


    def data_file(self,filename):
        """Write the data to a file in the bare, machine-readable form of
        the data table.
        """
        self.data_table().data_file(filename)

    
    def comparison_table(self,comparison,labels=('M','M')):
        """Return a table, as an instance of ResultTable, of the
        magnetization stored in this instance and in the instance given as
        an argument along with the deviations. The temperature points form
        the sections of the table and the root-mean-square deviation is
        reported below it. The T_list and B_list attributes of this
        instance and the other instance must be equal.

        Arguments
        ---------
        comparison : IsothermalStaticMagnetization
            The instance the magnetization of this instance is compared to.

        Optional arguments
        ------------------
        labels : tuple of str
            The column headers of the magnetization of this instance and of
            the compared instance. Default is ('M','M').
        """
        if not isinstance(comparison, IsothermalStaticMagnetization):
            print("ERROR in IsothermalStaticMagnetization.")
            print("ERROR: Arguments to comparison_table must be an instance of IsothermalStaticMagnetization.")
            print("Error termination.")
            sys.exit(1)

        if not np.allclose(self.T_list,comparison.T_list):
            print("ERROR in IsothermalStaticMagnetization.")
            print("ERROR: Cannot compare magnetization between two instances with different temperature points.")
            print("Error termination.")
            sys.exit(1)
            
        if not np.allclose(self.B_list,comparison.B_list):
            print("ERROR in IsothermalStaticMagnetization.")
            print("ERROR: Cannot compare magnetization between two instances with different field points.")
            print("Error termination.")
            sys.exit(1)

        rows = []

        for i in range(0,self.n_T_points):
            rows.append("Temperature: {0:.3f} K".format(self.T_list[i]))

            for j in range(0,self.n_B_points):
                if self.single_field_range:
                    B = self.B_list[j]
                else:
                    B = self.B_list[i][j]

                M_this  = self.magnetization[i][j]
                M_other = comparison.magnetization[i][j]

                rows.append([B,M_this,M_other,M_this-M_other])

        summary = ["RMS error: {0:.6f}".format(self.rms_error(comparison))]

        return result_table.ResultTable(rows,
                                        column_headers=['B / T',labels[0],labels[1],
                                                        'Deviation'],
                                        title="COMPARISON OF MOLAR MAGNETIZATIONS",
                                        notes=self.__UNIT_NOTE,
                                        summary=summary,
                                        formats=4*['.6f'],
                                        table_type='magnetization')

    
    def rms_error(self,comparison):
        """Compare the magnetization stored in this instance and that store in the
        instace given as an argument, and return the root-mean-square deviation
        between them.
        """
        if not isinstance(comparison, IsothermalStaticMagnetization):
            print("ERROR in IsothermalStaticMagnetization.")
            print("ERROR: Arguments to rms_error must be an instance of IsothermalStaticMagnetization.")
            print("Error termination.")
            sys.exit(1)

        if not self.magnetization.shape == comparison.magnetization.shape:
            print("ERROR in IsothermalStaticMagnetization.")
            print("ERROR: Cannot compare magnetizations between two instances with different numbers of data points..")
            print("Error termination.")
            sys.exit(1)

        square_sum = 0.0

        for i in range(0,self.n_T_points):
            for j in range(0,self.n_B_points):
                square_sum += (self.magnetization[i][j] - comparison.magnetization[i][j])**2

        return sqrt(square_sum / (self.n_T_points * self.n_B_points))


    def __repr__(self):
        """Return a human-readable table of the calculated magnetization."""
        return str(self.data_table())

    
    def __checked_temperature_list(self, T_list):
        """Return the temperature list with every temperature at or below
        MINIMUM_TEMPERATURE raised to it, so that the Boltzmann populations
        stay defined. A negative temperature is an error rather than
        something to be corrected quietly.
        """
        for T in T_list:
            if float(T) < 0.0:
                print("ERROR in IsothermalStaticMagnetization.")
                print("ERROR: Negative temperature: " + str(float(T)) + ".")
                print("Error termination.")
                sys.exit(1)

        checked_list = [max(float(T),self.MINIMUM_TEMPERATURE) for T in T_list]

        # A temperature range is usually given as a numpy array, e.g. as
        # numpy.linspace(0,300,301), and stays one.
        if isinstance(T_list,np.ndarray):
            return np.array(checked_list, dtype=np.float64)

        return checked_list


    def __init__(self, T_list, B_list,
                 magnetization=None):
        """Just set up the arguments as attributes. The temperatures too
        close to zero are raised to MINIMUM_TEMPERATURE.
        """
        self.T_list = self.__checked_temperature_list(T_list)
        self.B_list = B_list

        self.n_T_points = len(self.T_list)

        if isinstance(self.B_list[0], list):
            self.single_field_range = False
            self.n_B_points = len(self.B_list[0])
            for B_set in self.B_list:
                if not len(B_set) == self.n_B_points:
                    print("ERROR in IsothermalStaticMagnetization.")
                    print("ERROR: Inconsistent number of field values for different temperatures.")
                    print("Error termination.")
                    sys.exit(1)
        else:
            self.single_field_range = True
            self.n_B_points = len(self.B_list)

        if magnetization is None:
            self.magnetization = np.zeros((self.n_T_points,self.n_B_points), dtype=np.float64)
        else:
            self.magnetization = magnetization


    @classmethod
    def from_data_file(cls, filename, single_B_range=True, oersteds=False):
        """Read the data from a formatted text file."""
        try:
            f = open(filename)
        except FileNotFoundError:
            print("ERROR in IsothermalStaticMagnetization.")
            print("ERROR: file " + filename + " not found.")
            print("Error termination.")
            sys.exit(1)

        # Read the temperature points from the header.
        line = f.readline()
        n_T_points = len(line.split())
        T_list = []
        for T_str in line.split():
            T_list.append(float(T_str))

        # Read through the file once to set the B values.
        B_list = []
        if not single_B_range:
            for i in range(0,n_T_points):
                B_list.append([])

        while True:
            line = f.readline()
            if line == "":
                break
            else:
                if single_B_range:
                    if oersteds:
                        B_list.append(float(line.split()[0]) / 10000.0)
                    else:
                        B_list.append(float(line.split()[0]))
                else:
                    for i in range(0,n_T_points):
                        tmp_B = float(line.split()[2*i])
                        if oersteds:
                            B_list[i].append(tmp_B / 10000.0)
                        else:
                            B_list[i].append(tmp_B)
                            
        if single_B_range:
            n_B_points = len(B_list)
        else:
            n_B_points = len(B_list[0])

        # Read through the file again to read the magnetization data.
        magnetization = np.zeros((n_T_points,n_B_points), dtype=np.float64)
        f.seek(0)
        line = f.readline()

        for i in range(0,n_B_points):
            line = f.readline()
            for j in range(0,n_T_points):
                if single_B_range:
                    magnetization[j][i] = float(line.split()[j+1])
                else:
                    magnetization[j][i] = float(line.split()[2*j+1])

        f.close()

        return cls(T_list,B_list,magnetization=magnetization)


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
        import os
        import tempfile

        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('IsothermalStaticMagnetization',
                                                       test_name,condition,print_output))

        magnetization_a = cls([1.0,2.0],[0.5,1.0],
                              magnetization=np.array([[1.0,2.0],[3.0,4.0]]))
        magnetization_b = cls([1.0,2.0],[0.5,1.0],
                              magnetization=np.array([[1.5,2.0],[3.0,5.0]]))

        check('dimensions',
              (magnetization_a.n_T_points == 2) and (magnetization_a.n_B_points == 2)
              and magnetization_a.single_field_range)

        rms_reference = sqrt(((magnetization_a.magnetization
                               - magnetization_b.magnetization)**2).mean())
        check('root-mean-square error',
              abs(magnetization_a.rms_error(magnetization_b) - rms_reference) < 1.0e-12)

        check('data table renders', len(str(magnetization_a.data_table())) > 0)
        check('comparison table renders',
              len(str(magnetization_a.comparison_table(magnetization_b))) > 0)

        # A multiple-field-range instance.
        magnetization_c = cls([1.0,2.0],[[0.5,1.0],[0.6,1.1]],
                              magnetization=np.array([[1.0,2.0],[3.0,4.0]]))
        check('multiple field ranges recognized',
              not magnetization_c.single_field_range)

        # The magnetization is not defined at zero temperature, so a
        # temperature point at or below the limit is raised to it and the
        # rest of the range is left as it is.
        zero_T = cls([0.0,0.5*cls.MINIMUM_TEMPERATURE,1.8],[1.0])
        check('a zero temperature is raised to the limit',
              zero_T.T_list[0] == cls.MINIMUM_TEMPERATURE)
        check('a temperature below the limit is raised to it',
              zero_T.T_list[1] == cls.MINIMUM_TEMPERATURE)
        check('the temperatures above the limit are left as they are',
              zero_T.T_list[2] == 1.8)

        # A temperature range is usually given as a numpy array and stays
        # one, so that it can be used as such afterwards.
        array_T = cls(np.linspace(0.0,10.0,3),[1.0])
        check('a temperature range given as an array stays an array',
              isinstance(array_T.T_list,np.ndarray)
              and np.allclose(array_T.T_list,
                              [cls.MINIMUM_TEMPERATURE,5.0,10.0]))

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir,'magnetization.dat')
            magnetization_a.data_file(filename)
            check('single-range data file written', os.path.getsize(filename) > 0)
            magnetization_c.data_file(filename)
            check('multiple-range data file written', os.path.getsize(filename) > 0)

            # Read a data file in the format expected by from_data_file:
            # a header line of temperatures followed by rows of B and M values.
            filename = os.path.join(tmp_dir,'input.dat')
            with open(filename,'w') as f:
                f.write("1.0 10.0\n")
                f.write("0.5 0.1 0.2\n")
                f.write("1.0 0.3 0.4\n")
            magnetization_d = cls.from_data_file(filename)
            check('from_data_file',
                  np.allclose(magnetization_d.T_list,[1.0,10.0])
                  and np.allclose(magnetization_d.B_list,[0.5,1.0])
                  and np.allclose(magnetization_d.magnetization,
                                  [[0.1,0.3],[0.2,0.4]]))
            magnetization_e = cls.from_data_file(filename,oersteds=True)
            check('from_data_file with oersteds',
                  np.allclose(magnetization_e.B_list,[0.5e-4,1.0e-4]))

        return debug_output.test_summary('IsothermalStaticMagnetization',
                                         result_list,print_output)



class StaticMagneticSusceptibility:
    """A class to store and print the values of the product of the magnetic
    susceptibility and temperature as a function of temperature; i.e.,
    the chiT plot.

    This class is not used for the actual calculation of the susceptibility.
    This should rather be considered as a container for storing the
    susceptibility.

    Arguments
    ---------
    T_list : list of float
        List of the temperature value used in evaluation of the
        susceptibility. A temperature at or below MINIMUM_TEMPERATURE is
        raised to it, since the susceptibility is not defined at zero
        temperature; see the attribute below.

    Optional arguments
    ------------------
    susceptibility : list of float
        List of the values of susceptibility. The default is None in which
        case an empty (zero) list will be constructed upon initiation.
    B : float
        Magnitude of the measurement field in teslas. The default is 0.1.

    Attributes
    ----------
    T_list : list of float
        List of the temperature value used in evaluation of the
        susceptibility, with the temperatures at or below
        MINIMUM_TEMPERATURE raised to it.
    susceptibility : list of float
        List of the values of susceptibility.
    B : float
        Magnitude of the measurement field in teslas. The default is 0.1.
    n_T_points : int
        The number of temperature points.
    MINIMUM_TEMPERATURE : float
        The lowest temperature, in kelvins, the susceptibility is evaluated
        at. What is measured is the magnetization, which follows from the
        Boltzmann populations of the states and is not defined at zero
        temperature, so a temperature point at or below this limit is
        raised to it. Note that the chiT product itself tends to zero as
        the temperature does, so the first point of a range starting at
        zero is essentially zero either way; see the module constant
        MINIMUM_TEMPERATURE, which sets the limit.

    Private methods
    ---------------
    __checked_temperature_list(T_list) : list of float
        Return the temperature list with the temperatures at or below
        MINIMUM_TEMPERATURE raised to it.

    Public methods
    --------------
    data_table() : ResultTable
        Return a human-readable table of the susceptibility.
    data_file(filename)
        Write the data to a file.
    comparison_table(comparison,labels=('chiT','chiT')) : ResultTable
        Return a human-readable table of the susceptibility stored in this
        this instance and in the instance given as an argument along with the
        deviations. The T_list attributes of this instance and the other
        instance must be equal.
    rms_error(comparison) : float
        Compare the susceptibility stored in this instance and that store in the
        instace given as an argument, and return the root-mean-square deviation
        between them.

    Class methods
    -------------
    from_data_file(filename,B=0.1)
        Read the data from a formatted text file.
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if all
        tests passed.
    """

    __UNIT_NOTE = "The listed susceptibility is the chiT product in units chiT / cm^3 K / mol."

    # The module constant is published as an attribute of the class, so
    # that the limit can be read, and lowered, through the class itself.
    MINIMUM_TEMPERATURE = MINIMUM_TEMPERATURE

    def data_table(self):
        """Return a table of the susceptibility as an instance of
        ResultTable. Each row contains a temperature point and the chiT
        product at that temperature.
        """
        rows = []
        for i in range(0,self.n_T_points):
            rows.append([self.T_list[i],self.susceptibility[i]])

        return result_table.ResultTable(rows,
                                        column_headers=['T / K','chiT'],
                                        title="MOLAR MAGNETIC SUSCEPTIBILITY",
                                        notes=self.__UNIT_NOTE,
                                        formats=['.3f','.6f'],
                                        table_type='susceptibility')


    def data_file(self,filename):
        """Write the data to a file in the bare, machine-readable form of
        the data table.
        """
        self.data_table().data_file(filename)

    
    def comparison_table(self,comparison,labels=('chiT','chiT')):
        """Return a table, as an instance of ResultTable, of the
        susceptibility stored in this instance and in the instance given as
        an argument along with the deviations. The root-mean-square
        deviation is reported below the table. The T_list attributes of
        this instance and the other instance must be equal.

        Arguments
        ---------
        comparison : StaticMagneticSusceptibility
            The instance the susceptibility of this instance is compared
            to.

        Optional arguments
        ------------------
        labels : tuple of str
            The column headers of the susceptibility of this instance and
            of the compared instance. Default is ('chiT','chiT').
        """
        if not isinstance(comparison, StaticMagneticSusceptibility):
            print("ERROR in StaticMagneticSusceptibility.")
            print("ERROR: Arguments to comparison_table must be an instance of StaticMagneticSusceptibility.")
            print("Error termination.")
            sys.exit(1)

        if not np.allclose(self.T_list,comparison.T_list):
            print("ERROR in StaticMagneticSusceptibility.")
            print("ERROR: Cannot compare susceptibilities between two instances with different temperature points.")
            print("Error termination.")
            sys.exit(1)
        
        rows = []
        for i in range(0,self.n_T_points):
            rows.append([self.T_list[i],
                         self.susceptibility[i],
                         comparison.susceptibility[i],
                         self.susceptibility[i] - comparison.susceptibility[i]])

        summary = ["RMS error: {0:.6f}".format(self.rms_error(comparison))]

        return result_table.ResultTable(rows,
                                        column_headers=['T / K',labels[0],labels[1],
                                                        'Deviation'],
                                        title="COMPARISON OF MOLAR MAGNETIC SUSCEPTIBILITIES",
                                        notes=self.__UNIT_NOTE,
                                        summary=summary,
                                        formats=['.3f','.6f','.6f','.6f'],
                                        table_type='susceptibility')

    
    def rms_error(self,comparison):
        """Compare the susceptibility stored in this instance and that store in the
        instace given as an argument, and return the root-mean-square deviation
        between them.
        """
        if not isinstance(comparison, StaticMagneticSusceptibility):
            print("ERROR in IsothermalStaticMagnetization.")
            print("ERROR: Arguments to rms_error must be an instance of StaticMagneticSusceptibility.")
            print("Error termination.")
            sys.exit(1)

        if not len(self.susceptibility) == len(comparison.susceptibility):
            print("ERROR in IsothermalStaticMagnetization.")
            print("ERROR: Cannot compare susceptibilities between two instances with different numbers of data points.")
            print("Error termination.")
            sys.exit(1)

        square_sum = 0.0
        for i in range(0,self.n_T_points):
            square_sum += (self.susceptibility[i] - comparison.susceptibility[i])**2

        return sqrt(square_sum / float(self.n_T_points))


    def __repr__(self):
        """Return a human-readable table of the calculated susceptibility."""
        return str(self.data_table())

    
    def __checked_temperature_list(self, T_list):
        """Return the temperature list with every temperature at or below
        MINIMUM_TEMPERATURE raised to it, so that the Boltzmann populations
        stay defined. A negative temperature is an error rather than
        something to be corrected quietly.
        """
        for T in T_list:
            if float(T) < 0.0:
                print("ERROR in StaticMagneticSusceptibility.")
                print("ERROR: Negative temperature: " + str(float(T)) + ".")
                print("Error termination.")
                sys.exit(1)

        checked_list = [max(float(T),self.MINIMUM_TEMPERATURE) for T in T_list]

        # A temperature range is usually given as a numpy array, e.g. as
        # numpy.linspace(0,300,301), and stays one.
        if isinstance(T_list,np.ndarray):
            return np.array(checked_list, dtype=np.float64)

        return checked_list


    def __init__(self, T_list,
                 susceptibility=None,
                 B=0.1):
        """Just set up the arguments as attributes. The temperatures too
        close to zero are raised to MINIMUM_TEMPERATURE.
        """
        self.T_list = self.__checked_temperature_list(T_list)
        self.B      = B

        self.n_T_points = len(self.T_list)

        if susceptibility is None:
            self.susceptibility = []
            for i in range(0,self.n_T_points):
                self.susceptibility.append(0.0)
        else:
            self.susceptibility = susceptibility


    @classmethod
    def from_data_file(cls,filename,B=0.1):
        """Read the data from a formatted text file."""
        try:
            f = open(filename)
        except FileNotFoundError:
            print("ERROR in StaticMagneticSusceptibility.")
            print("ERROR: file " + filename + " not found.")
            print("Error termination.")
            sys.exit(1)

        T_list         = []
        susceptibility = []
        
        for line in iter(f):
            T_list.append(float(line.split()[0]))
            susceptibility.append(float(line.split()[1]))

        f.close()

        return cls(T_list,susceptibility,B=B)


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
        import os
        import tempfile

        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('StaticMagneticSusceptibility',
                                                       test_name,condition,print_output))

        susceptibility_a = cls([1.0,10.0,100.0],susceptibility=[0.4,0.38,0.375])
        susceptibility_b = cls([1.0,10.0,100.0],susceptibility=[0.4,0.39,0.375])

        check('dimensions and default field',
              (susceptibility_a.n_T_points == 3) and (susceptibility_a.B == 0.1))
        check('empty susceptibility initialization',
              cls([1.0,2.0]).susceptibility == [0.0,0.0])

        rms_reference = sqrt(np.mean((np.array(susceptibility_a.susceptibility)
                                      - np.array(susceptibility_b.susceptibility))**2))
        check('root-mean-square error',
              abs(susceptibility_a.rms_error(susceptibility_b) - rms_reference) < 1.0e-12)

        check('data table renders', len(str(susceptibility_a.data_table())) > 0)
        check('comparison table renders',
              len(str(susceptibility_a.comparison_table(susceptibility_b))) > 0)

        # The susceptibility is not defined at zero temperature, so a
        # temperature point at or below the limit is raised to it and the
        # rest of the range is left as it is. A range given as a numpy
        # array stays one.
        zero_T = cls([0.0,0.5*cls.MINIMUM_TEMPERATURE,2.0])
        check('a zero temperature is raised to the limit',
              zero_T.T_list[0] == cls.MINIMUM_TEMPERATURE)
        check('a temperature below the limit is raised to it',
              zero_T.T_list[1] == cls.MINIMUM_TEMPERATURE)
        check('the temperatures above the limit are left as they are',
              zero_T.T_list[2] == 2.0)

        array_T = cls(np.linspace(0.0,300.0,301))
        check('a temperature range given as an array stays an array',
              isinstance(array_T.T_list,np.ndarray)
              and array_T.T_list[0] == cls.MINIMUM_TEMPERATURE
              and array_T.T_list[-1] == 300.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir,'susceptibility.dat')
            susceptibility_a.data_file(filename)
            check('data file written', os.path.getsize(filename) > 0)

            # Read a data file in the format expected by from_data_file:
            # rows of T and chiT values without a header.
            filename = os.path.join(tmp_dir,'input.dat')
            with open(filename,'w') as f:
                f.write("1.0 0.40\n")
                f.write("10.0 0.38\n")
            susceptibility_c = cls.from_data_file(filename)
            check('from_data_file',
                  np.allclose(susceptibility_c.T_list,[1.0,10.0])
                  and np.allclose(susceptibility_c.susceptibility,[0.40,0.38]))

        return debug_output.test_summary('StaticMagneticSusceptibility',
                                         result_list,print_output)



class StaticMagneticProperties:
    """A class to evaluate static magnetic properties of an electron system
    from the field-free Hamiltonian and the magnetic moment operator of the
    system given as input.

    The properties are those measured of a static sample. What an experiment
    measures is in both cases the magnetization: the magnetization itself is
    reported as such, and the magnetic susceptibility is obtained by dividing
    the powder magnetization measured in a small field by the strength of
    that field. The susceptibility is reported as the chi*T product (see
    StaticMagneticSusceptibility).

    Arguments
    ---------
    hamiltonian : PseudoSpinOperator or GeneralOperatorMatrix
        The field-free Hamiltonian of the system. The operator must carry
        its matrix representation, i.e. it must have been constructed with
        store_operator_matrix set to True.
    magnetic_moment : PseudoSpinVectorOperator or GeneralVectorOperatorMatrix
        The magnetic moment operator of the system, holding the three
        Cartesian components. The components must carry their matrix
        representations.
    grid : ZCWGrid or LebedevLaikovGrid or SimpleGrid
        Grid used for powder integration. See the note on the accuracy of
        the grid below.
    units : EnergyUnitSystem
        The energy unit system.

    Optional arguments
    ------------------
    susceptibility : StaticMagneticSusceptibility
        Used for defining the specs of the magnetic susceptibility calculation and storing
        the calculated susceptibility. Default is a None, in which case susceptibility is
        not calculated.
    magnetization : IsothermalStaticMagnetization
        Used for defining the specs of the magnetization calculation and storing
        the calculated mangetization. Default is a None, in which case magnetization is
        not calculated.
    print_output : boolean
        Whether to print output during the calculation. Default is False.
    sample_orientation : str
        How the orientations of the individual molecules will be taken into account in
        evaluation of the macroscopic magnetization. The possbile options are 'powder'
        (default) in which case a standard powder integration over the grid supplied as
        an argument will be carried out, 'free' in which case the molecules are allowed
        to freely rotate in the sample and their directions are determined by Boltzmann
        statistics or 'maximal' in which case the grid is searched over and the
        magnetization is taken along the direction at which it has a maximal value at
        each temperature.
    n : float
        The number of mangetic subsystems per one mole of sample. I.e., the magnetization
        and magnetic susceptibility will be multiplied by this number. The default value
        is 1.0.

    Attributes
    ----------
    hamiltonian : PseudoSpinOperator or GeneralOperatorMatrix
        The field-free Hamiltonian of the system.
    magnetic_moment : PseudoSpinVectorOperator or GeneralVectorOperatorMatrix
        The magnetic moment operator of the system.
    hamiltonian_matrix : array of complex128
        The matrix representation of the Hamiltonian, taken from the
        Hamiltonian operator upon class initiation.
    magnetic_moment_matrix : array of complex128
        The matrix representations of the three Cartesian components of the
        magnetic moment operator, stored as one n_basis x n_basis x 3 array
        in the memory order the Fortran routines expect.
    grid : ZCWGrid or LebedevLaikovGrid or SimpleGrid
        Grid used for powder integration.
    units : EnergyUnitSystem
        The energy unit system.
    susceptibility : StaticMagneticSusceptibility
        Used for defining the specs of the magnetic susceptibility calculation and storing
        the calculated susceptibility.
    magnetization : IsothermalStaticMagnetization
        Used for defining the specs of the magnetization calculation and storing
        the calculated mangetization.
    print_output : boolean
        Whether to print output during the calculation.
    sample_orientation : str
        How the orientations of the individual molecules are taken into account in the
        evaluation of the macroscopic magnetization; see the corresponding optional
        argument.
    n_basis : int
        Dimension of the operator matrices.
    n : float
        The number of mangetic subsystems per one mole of sample. I.e., the magnetization
        and magnetic susceptibility will be multiplied by this number.

    Private methods
    ---------------
    __calculate_magnetization_at_fixed_field(T_list,B) : list of float
        Calculate the integrated powder magnetization at a fixed field strength B
        at temperatures listed in T_list and return the list of scalar magnetization
        values.

    Public methods
    --------------
    calculate_susceptibility()
        Calculate the magnetic susceptibility and store it as an attribute.
    calculate_magnetization()
        Calculate the magnetization and store it as an attribute.

    Class methods
    -------------
    from_electron_exchange_system(electron_exchange_system, grid,
                                  susceptibility=None,
                                  magnetization=None,
                                  print_output=False,
                                  sample_orientation='powder',
                                  n=1.0)
        Instead of providing the operators, provide an instance of
        ElectronExchangeSystem that carries them. The main purpose of this
        class method is to simplify the interface.
    from_ab_initio_system(ab_initio_system, grid,
                          susceptibility=None,
                          magnetization=None,
                          print_output=False,
                          sample_orientation='powder',
                          n=1.0)
        The counterpart of from_electron_exchange_system for an
        AbInitioElectronExchangeSystem, which constructs its operators on
        request instead of storing them.
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if all
        tests passed.

    Note on the accuracy of the powder integration
    ----------------------------------------------
    The cost of the calculation is directly proportional to the number of
    grid points: the Hamiltonian is diagonalized once per grid point and per
    field strength, and that diagonalization dominates the calculation.
    Choosing the grid is therefore the one decision that sets the cost of a
    calculation of the static magnetic properties.

    The two grids behave quite differently, and which one is the better
    depends on the property:

      - The susceptibility is measured in a small field, where the
        magnetization is linear in the field. The powder average is then an
        integral of a quadratic form over the sphere, which the
        Lebedev--Laikov grid integrates exactly: it reaches machine
        precision already at grid_quality = 7 (86 points), whereas the ZCW
        grid converges only as a power of the number of points.

      - The magnetization is measured in a field strong enough to saturate
        the moment. The magnetization of a strongly anisotropic system is
        then close to the modulus of the projection of the easy axis on the
        field direction, i.e. a function with a cusp rather than a smooth
        one. A quadrature exact for polynomials gains little from that
        exactness: the Lebedev--Laikov grid converges slowly and
        erratically, and the uniformly distributed ZCW grid is roughly an
        order of magnitude cheaper at a given accuracy.

    Since the magnetization is the harder of the two and the susceptibility
    is accurate on either grid, the ZCW grid is the better choice overall
    and the recommended one for a calculation of both properties. The
    default settings of both grid classes, ZCWGrid(5) (233 points) and
    LebedevLaikovGrid(17) (590 points), integrate both properties of the
    strongly anisotropic Dy(III) benchmark system to better than 1.0e-3 of
    the converged value; the benchmark is tabulated in the class
    documentation of the two grid classes. A calculation of the
    susceptibility alone is well converged with a much smaller
    Lebedev--Laikov grid, LebedevLaikovGrid(7) or so.
    """

    def __calculate_magnetization_at_fixed_field(self,T_list,B):
        """Calculate the integrated powder magnetization at a fixed field strength B
        at temperatures listed in T_list and return the list of scalar magnetization
        values.
        """

        if self.sample_orientation == 'powder':
            M_value_list = fu.powder_magnetization_utils.\
                           powder_magnetization(self.hamiltonian_matrix,
                                                self.magnetic_moment_matrix,
                                                T_list,B,
                                                self.grid.vectors,
                                                self.grid.weights,
                                                self.units.k_B)
        elif self.sample_orientation == 'free':
            M_value_list = fu.powder_magnetization_utils.\
                           free_rotation_magnetization(self.hamiltonian_matrix,
                                                       self.magnetic_moment_matrix,
                                                       T_list,B,
                                                       self.grid.vectors,
                                                       self.grid.weights,
                                                       self.units.k_B)
        elif self.sample_orientation == 'maximal':
            M_value_list, vectors = fu.powder_magnetization_utils.\
                                    maximum_magnetization(self.hamiltonian_matrix,
                                                          self.magnetic_moment_matrix,
                                                          T_list,B,
                                                          self.grid.vectors,
                                                          self.units.k_B)

        for i in range(0,len(M_value_list)):
            M_value_list[i] = M_value_list[i] / self.units.mu_B

        return M_value_list

        
    def calculate_susceptibility(self):
        """Calculate the magnetic susceptibility and store it as an attribute."""
        if self.susceptibility is None:
            print("ERROR in MicroscopicElectronSystem.")
            print("ERROR: Susceptibility calculation requested, but no StaticMagneticSusceptibility instance given.")
            print("Error termination.")
            sys.exit(1)

        # Conversion factor from units of Bohr magneton times Avogadro constant to cgsemu
        # (note that the field unit is in teslas):
        conversion_factor = 0.5584939449309
            
        M_value_list = self.__calculate_magnetization_at_fixed_field(self.susceptibility.T_list,
                                                                     self.susceptibility.B)

        B = self.susceptibility.B

        for i in range(0,self.susceptibility.n_T_points):
            T = self.susceptibility.T_list[i]
            M = M_value_list[i]

            self.susceptibility.susceptibility[i] = conversion_factor * self.n * M * T / B

    
    def calculate_magnetization(self):
        """Calculate the magnetization and store it as an attribute."""
        if self.magnetization is None:
            print("ERROR in MicroscopicElectronSystem.")
            print("ERROR: Magnetization calculation requested, but no IsothermalStaticMagnetization instance given.")
            print("Error termination.")
            sys.exit(1)

        if self.print_output:
            print("    Calculation of magnetization ...")
        for j in range(0,self.magnetization.n_B_points):
            if self.print_output:
                print("      Field strength {0:4} out of {1:4}".format(j+1,self.magnetization.n_B_points))
            B = self.magnetization.B_list[j]
            M_value_list = self.__calculate_magnetization_at_fixed_field(self.magnetization.T_list,B)
            
            for i in range(0,self.magnetization.n_T_points):
                self.magnetization.magnetization[i][j] = self.n * M_value_list[i]

        if self.print_output:
            print()

    
    def __init__(self, hamiltonian, magnetic_moment, grid, units,
                 susceptibility=None,
                 magnetization=None,
                 print_output=False,
                 sample_orientation='powder',
                 n=1.0):
        """Store the operators as attributes and take their matrix
        representations into the form the Fortran routines expect. Depending
        on the optional arguments, calculate the magnetic susceptibility
        and/or magnetization.
        """

        self.hamiltonian = hamiltonian
        self.magnetic_moment = magnetic_moment
        self.grid = grid
        self.units = units
        self.print_output = print_output
        self.sample_orientation = sample_orientation
        self.n = n

        allowed_sample_orientations = ['powder','free','maximal']

        if not self.sample_orientation in allowed_sample_orientations:
            print("ERROR in StaticMagneticProperties.")
            print("ERROR: Unrecognized sample orientation:" + self.sample_orientation)
            print("Error termination.")
            sys.exit(1)

        moment_matrix_list = self.magnetic_moment.matrix_list()

        if not len(moment_matrix_list) == 3:
            print("ERROR in StaticMagneticProperties.")
            print("ERROR: Inconsistent number of magnetic moment operators.")
            print("Error termination.")
            sys.exit(1)

        self.n_basis = self.hamiltonian.matrix.shape[0]

        for i in range(0,3):
            if not moment_matrix_list[i].shape[0] == self.n_basis:
                print("ERROR in StaticMagneticProperties.")
                print("ERROR: Inconsistent operator dimensions.")
                print("Error termination.")
                sys.exit(1)

        # The matrices are taken into the layout the Fortran routines expect
        # once here rather than on every call into Fortran. The Cartesian
        # component is the last index of the magnetic moment array, so that
        # each component is contiguous in memory and can be handed to BLAS
        # as it stands.
        self.hamiltonian_matrix = np.asfortranarray(self.hamiltonian.matrix,
                                                    dtype=np.complex128)
        self.magnetic_moment_matrix = np.asfortranarray(
            np.stack(moment_matrix_list,axis=-1), dtype=np.complex128)

        self.susceptibility = susceptibility
        self.magnetization = magnetization

        if not self.susceptibility is None:
            self.calculate_susceptibility()

        if not self.magnetization is None:
            self.calculate_magnetization()


    @classmethod
    def from_electron_exchange_system(cls, electron_exchange_system, grid,
                                      susceptibility=None,
                                      magnetization=None,
                                      print_output=False,
                                      sample_orientation='powder',
                                      n=1.0):
        """Instead of providing the operators, provide an instance of
        ElectronExchangeSystem that carries them. The main purpose of this
        class method is to simplify the interface.
        """
        return cls(electron_exchange_system.hamiltonian,
                   electron_exchange_system.magnetic_moment,
                   grid,
                   electron_exchange_system.units,
                   susceptibility=susceptibility,
                   magnetization=magnetization,
                   print_output=print_output,
                   sample_orientation=sample_orientation,
                   n=n)


    @classmethod
    def from_ab_initio_system(cls, ab_initio_system, grid,
                              susceptibility=None,
                              magnetization=None,
                              print_output=False,
                              sample_orientation='powder',
                              n=1.0):
        """The counterpart of from_electron_exchange_system for an instance
        of AbInitioElectronExchangeSystem, which constructs its pseudospin
        operators on request instead of storing them. The operators are
        those of the pseudospin multiplet the ab initio states have been
        projected onto, expressed in the coordinate frame of the tensors of
        the system.
        """
        return cls(ab_initio_system.hamiltonian_operator(),
                   ab_initio_system.magnetic_moment_operator(),
                   grid,
                   ab_initio_system.units,
                   susceptibility=susceptibility,
                   magnetization=magnetization,
                   print_output=print_output,
                   sample_orientation=sample_orientation,
                   n=n)


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class and run a set of tests on the class constructor
        and the class methods. Return True if all tests passed and False
        otherwise.

        The tests are based on an isotropic spin-1/2 system with g = 2, for
        which the Curie law gives chi*T = 0.375 cm^3 K mol^-1 and the
        magnetization saturates to g*S = 1 Bohr magneton.

        Optional arguments
        ------------------
        print_output : boolean
            Whether to print the outcome of each test. Default is True.
        """
        from ouluspin import units, integration
        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('StaticMagneticProperties',
                                                       test_name,condition,print_output))

        tmp_units = units.EnergyUnitSystem('wavenumber')
        grid      = integration.LebedevLaikovGrid(3)

        # An isotropic spin-1/2 with g = 2: the field-free Hamiltonian is
        # zero and the magnetic moment operator is mu = g*mu_B*S. The
        # operators are handed over as the operator instances the class
        # takes, i.e. as a GeneralOperatorMatrix and a
        # GeneralVectorOperatorMatrix built on the bare matrices.
        g = 2.0
        spin_matrix_list = debug_output.spin_matrices(1)
        hamiltonian      = pseudospin_operators\
                           .GeneralOperatorMatrix(np.zeros((2,2),
                                                           dtype=np.complex128))
        magnetic_moment  = pseudospin_operators\
                           .GeneralVectorOperatorMatrix(
                               [g*tmp_units.mu_B*spin_matrix for spin_matrix
                                in spin_matrix_list])

        susceptibility = StaticMagneticSusceptibility([2.0,20.0,200.0],B=0.1)
        smp = cls(hamiltonian,magnetic_moment,grid,tmp_units,
                  susceptibility=susceptibility)

        check('Curie law chi*T = 0.375 cm^3 K mol^-1 for S = 1/2, g = 2',
              np.allclose(susceptibility.susceptibility,0.375,rtol=2.0e-3))

        magnetization = IsothermalStaticMagnetization([1.0],[1.0,3.0,7.0])
        smp = cls(hamiltonian,magnetic_moment,grid,tmp_units,
                  magnetization=magnetization)

        # The exact powder magnetization of an isotropic two-level system is
        # g*S*tanh(g*S*mu_B*B / (k_B*T)).
        B_values = np.array([1.0,3.0,7.0])
        M_reference = np.tanh(tmp_units.mu_B*B_values/(tmp_units.k_B*1.0))
        check('magnetization matches the analytical two-level result',
              np.allclose(magnetization.magnetization[0],M_reference,rtol=1.0e-6))
        check('magnetization saturates to g*S',
              abs(magnetization.magnetization[0][-1] - 1.0) < 0.01)

        # Doubling the number of magnetic subsystems per mole doubles chi.
        susceptibility_2 = StaticMagneticSusceptibility([2.0],B=0.1)
        smp = cls(hamiltonian,magnetic_moment,grid,tmp_units,
                  susceptibility=susceptibility_2,n=2.0)
        check('scaling with the number of subsystems',
              abs(susceptibility_2.susceptibility[0]
                  - 2.0*susceptibility.susceptibility[0]) < 1.0e-8)

        # A temperature range starting at zero is usable as it stands: the
        # zero point is raised to the lowest temperature the Boltzmann
        # populations are defined at, and the properties come out finite.
        # The two-level system is fully polarized there, so the
        # magnetization is the saturation value g*S = 1.
        zero_T_magnetization = IsothermalStaticMagnetization([0.0,1.0],[7.0])
        zero_T_susceptibility = StaticMagneticSusceptibility([0.0,2.0],B=0.1)
        smp = cls(hamiltonian,magnetic_moment,grid,tmp_units,
                  susceptibility=zero_T_susceptibility,
                  magnetization=zero_T_magnetization)

        check('a magnetization requested at zero temperature is finite',
              np.all(np.isfinite(zero_T_magnetization.magnetization)))
        check('a susceptibility requested at zero temperature is finite',
              np.all(np.isfinite(zero_T_susceptibility.susceptibility)))
        check('the magnetization at zero temperature is the saturation value',
              abs(zero_T_magnetization.magnetization[0][0] - 1.0) < 1.0e-12)

        # The class methods taking a system instead of the operators. Both
        # systems below are the same isotropic S = 1/2 with g = 2 as above,
        # so both must reproduce the Curie law.
        from ouluspin.systems import electron_exchange_system

        # The Hamiltonian of the free spin is zero, which is given to the
        # system as a vanishing axial term.
        zero_tensor = tensors.IwaharaChibotaruSphericalTensor\
                             .from_one_site_cartesian_operator(0.0,'z',1)
        moment_tensor = tensors.MixedCartesianIwaharaChibotaruSphericalTensor\
                               .from_one_site_isotropic_operator(g,1)
        exchange_system = electron_exchange_system\
                          .ElectronExchangeSystem([1],[(zero_tensor,0)],
                                                  [(moment_tensor,0)],
                                                  tmp_units)

        susceptibility_3 = StaticMagneticSusceptibility([2.0,20.0,200.0],B=0.1)
        smp = cls.from_electron_exchange_system(exchange_system,grid,
                                                susceptibility=susceptibility_3)
        check('from_electron_exchange_system reproduces the Curie law',
              np.allclose(susceptibility_3.susceptibility,0.375,rtol=2.0e-3))

        # The ab initio system takes the Hamiltonian diagonalized, as it
        # comes out of a quantum-chemical calculation.
        basis = pseudospin_operators.PseudoSpinBasis([1])
        diagonalized_hamiltonian = pseudospin_operators\
                                   .GeneralOperatorMatrix(hamiltonian.matrix,
                                                          diagonalize_operator_matrix=True,
                                                          translate_eigenvalues=True)
        ab_initio_system = electron_exchange_system\
                           .AbInitioElectronExchangeSystem(diagonalized_hamiltonian,
                                                           magnetic_moment,
                                                           basis,tmp_units,
                                                           R=np.identity(3))

        susceptibility_4 = StaticMagneticSusceptibility([2.0,20.0,200.0],B=0.1)
        smp = cls.from_ab_initio_system(ab_initio_system,grid,
                                        susceptibility=susceptibility_4)
        check('from_ab_initio_system reproduces the Curie law',
              np.allclose(susceptibility_4.susceptibility,0.375,rtol=2.0e-3))

        # The operators are stored as the instances given, and their matrix
        # representations in the layout the Fortran routines expect, i.e.
        # with the Cartesian component as the last index.
        smp = cls(hamiltonian,magnetic_moment,grid,tmp_units)
        check('the operator instances are stored as attributes',
              (smp.hamiltonian is hamiltonian)
              and (smp.magnetic_moment is magnetic_moment))
        check('the magnetic moment matrices are stacked component last',
              smp.magnetic_moment_matrix.shape == (2,2,3)
              and all(np.allclose(smp.magnetic_moment_matrix[:,:,k],
                                  magnetic_moment.matrix_list()[k])
                      for k in range(0,3)))

        return debug_output.test_summary('StaticMagneticProperties',
                                         result_list,print_output)



class StaticTransitionMagneticMoments:
    """A class to store and print tables of static transition magnetic moment
    matrix elements. This class is not used for the calculation of the matrix
    elements but rather to store them and to print human-readable output. The
    real transition magnetic moment between states I and J is defined as
    
        ( |<I|mu_x|J>| + |<I|mu_y|J>| + |<I|mu_z|J>| ) / 3.

    The magnetic moment is plotted as a function of the projection mu_z of
    the magnetic moment to construct an effective barrier for the
    relaxation of magnetization in single-molecule magnets as discussed in

        L. Ungur, M. Thewissen, J.-P. Costes, W. Wernsdorfer, L. F.
        Chibotaru. Inorg. Chem. 2013, 52, 6328--6337.

    The matrix elements are evaluated in the basis in which the projection
    of the magnetic moment is diagonal within each group of degenerate
    states, not in the basis the diagonalization of the Hamiltonian
    happens to return. The distinction matters: a diagonalization returns
    an arbitrary basis of a degenerate subspace, so the two states of a
    Kramers doublet come out as arbitrary combinations of the states of a
    definite moment, and both the expectation values and the matrix
    elements between the states of two doublets depend on that arbitrary
    choice. In the quantized basis the expectation value of the projection
    of an axial doublet is the g_z/2 of its g-tensor, as it should be, and
    more generally

        max |<mu_z>| = ( sum_i ( g_i * (u_i . z) )^2 )^(1/2) / 2,

    where g_i and u_i are the principal values and the principal axes of
    the g-tensor of the doublet in the frame of the magnetic moment
    operators. The states of a degenerate group are ordered by a
    descending projection, i.e. the state of the largest moment first.

    Arguments
    ---------
    magnetic_moment : PseudoSpinVectorOperator
        Magnetic moment operators. The instances of PseudoSpinOperator should
        contain the matrix representation of the operators.
    hamiltonian : PseudoSpinOperator
        The Hamiltonian operator instance containing the eigenvalues and
        eigenvectors. The Hamiltonian must be diagonalized before passing.
    n_states : int
        The number of lowest-energy states for which the transition magnetic
        moments will be considered.
    units : EnergyUnitSystem
        The unit system containing the definition of the energy units and
        the Bohr Magneton.

    Optional arguments
    ------------------
    print_output : boolean
        Whether to print output. Default is False.
    degeneracy_tolerance : float or None
        The energy separation, in the energy unit of the unit system, below
        which two states are considered degenerate and their moment is
        quantized (see above). Default is None, in which case the tolerance
        follows the numerical accuracy of the energies, i.e. 1.0e-6 times
        the spread of the tabulated spectrum, with 1.0e-6 times the Bohr
        magneton of the unit system (the tolerance of PseudoSpinDoublet) as
        the floor. The tolerance must stay below the physically meaningful
        gaps of the spectrum, in particular below the tunneling gap of a
        non-Kramers system, whose states are split for a physical reason
        and must not be recombined.

    Attributes
    ----------
    transition_magnetic_moment : array of float64
        An array containing the values of the transition magnetic moment.
    eigenvalues : list of float
        A list of the eigenvalues needed for the plotting of the effective
        barrier.
    n_states : int
        The number of lowest-energy states for which the transition magnetic
        moments will be considered.
    expectation_values : list of float
        A list of the expectation values of mu_z for the n_states lowest
        states, evaluated in the quantized basis.
    degeneracy_tolerance : float
        The degeneracy tolerance actually used.
    print_output : boolean
        Whether to print output. Default is False.
    units : EnergyUnitSystem
        The unit system containing the definition of the energy units and
        the Bohr Magneton.

    Public methods
    --------------
    transition_magnetic_moment_table() : ResultTable
        Construct and return a human-readable table of the transition magnetic
        moments.

    The effective barrier of the reversal of the magnetization is plotted by
    handing the instance to ResultPlot, which draws the states by their
    magnetic moment projection and their energy and the transitions between
    them as arrows.

    Private methods
    ---------------
    __quantization_transformation(mu_z)
        Return the block-diagonal unitary transformation that diagonalizes
        the projection of the magnetic moment within each group of
        degenerate states.

    Class methods
    -------------
    from_electron_exchange_system(magnetic_moment, hamiltonian, n_states, units,
                                  print_output=False)
        Instead of providing the operator matrices, provide an instance of
        ElectronExchangeSystem that contains the information necessary
        for the generation of the required matrices. The main purpose of this
        class method is to simplify the interface.
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if all
        tests passed.
    """

    def transition_magnetic_moment_table(self):
        """Construct and return a table of the transition magnetic moments
        as an instance of ResultTable. Each row contains a pair of states,
        given by their indices, energies and expectation values of the
        magnetic moment, followed by the magnitude of the transition
        magnetic moment between them.
        """
        rows = []

        for i in range(0,self.n_states):
            E_i  = self.eigenvalues[i]
            mu_i = self.expectation_values[i]

            for j in range(0,i):
                rows.append([i,E_i,mu_i,
                             j,self.eigenvalues[j],self.expectation_values[j],
                             self.transition_magnetic_moment[i][j]])

        return result_table.ResultTable(rows,
                                        column_headers=["i","E_i","mu_z,i",
                                                        "f","E_f","mu_z,f",
                                                        "|mu_if|"],
                                        title="TRANSITION MAGNETIC MOMENTS",
                                        formats=[None,'.2f','.3f',
                                                 None,'.2f','.3f',
                                                 '.8f'],
                                        table_type='transition_moments')

    
    def __repr__(self):
        """Return a human-readable table of the expectation values and transition moments."""
        return str(self.transition_magnetic_moment_table())


    def __quantization_transformation(self, mu_z):
        """Return the unitary transformation that quantizes the magnetic
        moment within each group of degenerate states.

        A diagonalization returns an arbitrary basis of a degenerate
        subspace, and the projection of the magnetic moment is not
        diagonal in such a basis: the two states of a Kramers doublet come
        out as arbitrary combinations of the states of a definite moment.
        The expectation value of the projection is then not the moment of
        the doublet at all, and neither are the matrix elements between the
        states of two doublets. The states of each degenerate group are
        therefore recombined into the states that diagonalize the
        projection, which are the ones the relaxation of the magnetization
        is described in.

        The transformation is block diagonal, one block per group of
        degenerate states, so it leaves the energies untouched. The states
        of a group are ordered by a descending projection, i.e. the state
        of the largest moment first.

        Arguments
        ---------
        mu_z : array of complex128
            The projection of the magnetic moment in the eigenbasis of the
            Hamiltonian.
        """
        transformation = np.identity(self.n_states,dtype=np.complex128)

        group       = []
        group_start = 0

        def close_group(group_list):
            """Diagonalize the projection within one group of degenerate
            states and store the result in the transformation.
            """
            if len(group_list) < 2:
                return

            block = mu_z[np.ix_(group_list,group_list)]

            # The projection is Hermitian, so its eigenvectors within the
            # group are orthonormal and the transformation stays unitary.
            eigenvalues, eigenvectors = np.linalg.eigh(block)

            order        = np.argsort(-eigenvalues)
            eigenvectors = eigenvectors[:,order]

            transformation[np.ix_(group_list,group_list)] = eigenvectors

        for i in range(0,self.n_states):
            if len(group) == 0:
                group = [i]
                group_start = i
                continue

            if abs(self.eigenvalues[i] - self.eigenvalues[group_start]) \
               <= self.degeneracy_tolerance:
                group.append(i)
            else:
                close_group(group)
                group = [i]
                group_start = i

        close_group(group)

        return transformation


    def __init__(self, magnetic_moment, hamiltonian, n_states, units,
                 print_output=False,
                 degeneracy_tolerance=None):
        """Construct the transition magnetic moments and store them as an array."""
        self.n_states     = n_states
        self.units        = units
        self.print_output = print_output

        tikk = time.time()
        if self.print_output:
            print("    Calculating transition magnetic moments ...")

        self.transition_magnetic_moment = np.zeros((self.n_states,self.n_states), dtype=np.float64)
        self.expectation_values         = []
        self.eigenvalues                = hamiltonian.eigenvalues[:self.n_states]

        # The states whose energies differ by less than the tolerance are
        # treated as degenerate. The default follows the numerical accuracy
        # of the energies, which is set by the accuracy of the operator
        # matrices the calculation started from and therefore scales with
        # the spread of the spectrum; the Zeeman energy in a field of about
        # one microtesla, i.e. the tolerance used by PseudoSpinDoublet, is
        # used as the floor for a spectrum that is nearly degenerate as a
        # whole.
        if degeneracy_tolerance is None:
            if len(self.eigenvalues) > 0:
                energy_span = max(self.eigenvalues) - min(self.eigenvalues)
            else:
                energy_span = 0.0

            self.degeneracy_tolerance = max(1.0e-6*self.units.mu_B,
                                            1.0e-6*energy_span)
        else:
            self.degeneracy_tolerance = degeneracy_tolerance

        C = hamiltonian.eigenvectors

        # The magnetic moment in the eigenbasis of the Hamiltonian, one
        # component at a time.
        mu_list = []
        for k in range(0,3):
            if self.print_output:
                print("      Cartesian component " + str(k) + " ...")

            mu = fu.matrix_utils.basis_transformation(
                C,magnetic_moment.operator_list[k].matrix,0)

            mu_list.append(np.asarray(mu)[:self.n_states,:self.n_states])

        # Within a degenerate group the eigenvectors are arbitrary, so the
        # states are recombined into the ones that carry a definite
        # projection of the magnetic moment before anything is read off
        # them.
        transformation = self.__quantization_transformation(mu_list[2])

        for k in range(0,3):
            mu_list[k] = np.conjugate(transformation.T) @ mu_list[k] \
                         @ transformation

        for i in range(0,self.n_states):
            if abs(mu_list[2][i][i].imag) > 1.0e-9:
                print("ERROR in StaticTransitionMagneticMoments.")
                print("ERROR: Complex-valued expectation value of magnetic moment projection.")
                print("Error termination.")
                sys.exit(1)

            self.expectation_values.append(mu_list[2][i][i].real
                                           / self.units.mu_B)

            for j in range(0,self.n_states):
                for k in range(0,3):
                    self.transition_magnetic_moment[i][j] += \
                        abs(mu_list[k][i][j]) / 3.0

        tokk = time.time()
        if self.print_output:
            print("    Done.")
            print("    Time spent {0:12.3f}".format(tokk-tikk))
            print()


    @classmethod
    def from_electron_exchange_system(cls, electron_exchange_system, n_states,
                                      print_output=False):
        """Instead of providing the operator matrices, provide an instance of
        ElectronExchangeSystem that contains the information necessary
        for the generation of the required matrices. The main purpose of this
        class method is to simplify the interface.
        """
        return cls(electron_exchange_system.magnetic_moment,
                   electron_exchange_system.hamiltonian,
                   n_states,
                   electron_exchange_system.units,
                   print_output=print_output)


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
        import os
        import tempfile

        from ouluspin import units
        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('StaticTransitionMagneticMoments',
                                                       test_name,condition,print_output))

        tmp_units = units.EnergyUnitSystem('wavenumber')

        # A spin-1/2 split by an axial term: H = 10*S_z, mu = -2*mu_B*S.
        # The ground state is |S,-1/2> with <mu_z> = +1 (in units of mu_B).
        hamiltonian_tensor = tensors.IwaharaChibotaruSphericalTensor\
                                    .from_one_site_cartesian_operator(10.0,'z',1)
        basis = pseudospin_operators.PseudoSpinBasis([1])
        hamiltonian = pseudospin_operators.PseudoSpinOperator(basis,
                                                              [hamiltonian_tensor],
                                                              tmp_units,
                                                              translate_eigenvalues=True)

        moment_tensor = tensors.MixedCartesianIwaharaChibotaruSphericalTensor\
                               .from_one_site_isotropic_operator(-2.0*tmp_units.mu_B,1)
        magnetic_moment = pseudospin_operators\
                          .PseudoSpinVectorOperator(basis,[moment_tensor],
                                                    tmp_units,
                                                    diagonalize_operator_matrix=False,
                                                    store_operator_matrix=True)

        transition_moments = cls(magnetic_moment,hamiltonian,2,tmp_units)

        check('expectation values of mu_z in units of mu_B',
              np.allclose(transition_moments.expectation_values,[1.0,-1.0]))

        # |<0|mu_x|1>| = |<0|mu_y|1>| = mu_B and <0|mu_z|1> = 0, so the
        # averaged transition moment is 2*mu_B/3.
        check('transition magnetic moment between the Zeeman states',
              abs(transition_moments.transition_magnetic_moment[1][0]
                  - 2.0*tmp_units.mu_B/3.0) < 1.0e-10)
        check('eigenvalues stored', np.allclose(transition_moments.eigenvalues,
                                                [0.0,10.0]))
        check('transition moment table renders',
              len(str(transition_moments.transition_magnetic_moment_table())) > 0)

        # A degenerate Kramers doublet. The two states come out of the
        # diagonalization as arbitrary combinations of the states of a
        # definite moment, so the projection has to be quantized within the
        # doublet before it is read off. The doublet is the ground doublet
        # of an axial system with no splitting at all: H = 0, mu = -2*mu_B*S,
        # for which the projection of the doublet is g_z/2 = 1 in units of
        # the Bohr magneton.
        zero_tensor = tensors.IwaharaChibotaruSphericalTensor\
                             .from_one_site_cartesian_operator(0.0,'z',1)
        degenerate_hamiltonian = pseudospin_operators.PseudoSpinOperator(
            basis,[zero_tensor],tmp_units,translate_eigenvalues=True)

        degenerate_moments = cls(magnetic_moment,degenerate_hamiltonian,2,
                                 tmp_units)

        check('the states of a degenerate doublet are degenerate',
              abs(degenerate_moments.eigenvalues[1]
                  - degenerate_moments.eigenvalues[0]) < 1.0e-10)
        check('the moment is quantized within a degenerate doublet',
              np.allclose(np.abs(degenerate_moments.expectation_values),
                          [1.0,1.0]))
        check('the states of a degenerate doublet carry opposite moments',
              abs(degenerate_moments.expectation_values[0]
                  + degenerate_moments.expectation_values[1]) < 1.0e-10)
        check('the state of the largest moment comes first',
              degenerate_moments.expectation_values[0]
              > degenerate_moments.expectation_values[1])

        # The projection of an axial doublet is the g_z/2 of its g-tensor;
        # more generally the largest projection follows from the g-tensor
        # of the doublet in the frame of the magnetic moment operators.
        doublet = PseudoSpinDoublet((0,1),
                                    [magnetic_moment.operator_list[k].matrix
                                     for k in range(0,3)],
                                    tmp_units)

        g_values = np.asarray(doublet.g_tensor.eigenvalues,dtype=np.float64)
        g_axes   = np.asarray(doublet.g_tensor.eigenvectors,dtype=np.float64)

        largest_projection = 0.5*np.sqrt(sum((g_values[k]*g_axes[2][k])**2
                                             for k in range(0,3)))

        check('the projection of the doublet follows from its g-tensor',
              abs(abs(degenerate_moments.expectation_values[0])
                  - largest_projection) < 1.0e-8)

        # A doublet whose easy axis is tilted away from the quantization
        # axis. The magnetic moment is of the Ising kind, mu = -2*mu_B*S_u
        # along the axis u lying at 45 degrees from z in the xz plane, so
        # that the projection is not diagonal in the basis returned by the
        # diagonalization: its diagonal elements are cos(45 deg) = 0.7071
        # of the moment of the doublet and the rest of the moment sits in
        # the off-diagonal element. The doublet itself is the same doublet
        # whichever way it is looked at, so the largest projection is still
        # cos(45 deg); reading the diagonal without quantizing the
        # projection first happens to give the same number here, so the
        # test below compares the two against each other.
        angle = 0.25*np.pi

        # The Cartesian tensor of an Ising moment along the tilted axis,
        # i.e. -2*mu_B times the projector on that axis.
        axis          = np.array([np.sin(angle),0.0,np.cos(angle)])
        tilted_matrix = -2.0*tmp_units.mu_B*np.outer(axis,axis)

        tilted_tensor = tensors.MixedCartesianIwaharaChibotaruSphericalTensor\
                               .from_one_site_cartesian_tensor(tilted_matrix,1)

        tilted_moment = pseudospin_operators.PseudoSpinVectorOperator(
            basis,[tilted_tensor],tmp_units,
            diagonalize_operator_matrix=False,
            store_operator_matrix=True)

        tilted = cls(tilted_moment,degenerate_hamiltonian,2,tmp_units)

        check('the quantized projection follows the tilted axis',
              abs(abs(tilted.expectation_values[0]) - np.cos(angle)) < 1.0e-8)

        # The same doublet with the quantization switched off, which a
        # negative tolerance does since two exactly degenerate states are
        # still within a vanishing one. The projection is then the part of
        # the moment that happens to be diagonal in the basis returned by
        # the diagonalization, i.e. cos^2(45 deg) = 0.5 instead of the
        # cos(45 deg) = 0.7071 the doublet actually carries. The two differ,
        # so the quantization is what makes the projection the moment of the
        # doublet rather than an artefact of the basis.
        unquantized = cls(tilted_moment,degenerate_hamiltonian,2,tmp_units,
                          degeneracy_tolerance=-1.0)

        check('the degeneracy tolerance is stored',
              unquantized.degeneracy_tolerance == -1.0)
        check('the unquantized projection is the basis-dependent one',
              abs(abs(unquantized.expectation_values[0])
                  - np.cos(angle)**2) < 1.0e-8)
        check('the quantization increases the projection of the doublet',
              abs(tilted.expectation_values[0])
              > abs(unquantized.expectation_values[0]) + 1.0e-8)

        return debug_output.test_summary('StaticTransitionMagneticMoments',
                                         result_list,print_output)



class PseudoSpinDoublet:
    """A class to calculate properties related to a doublet of two states
    described by an effective pseudospin S = 1/2. The doublet can be a
    Kramers doublet, an exactly degenerate non-Kramers (Ising) doublet or a
    non-degenerate quasi-doublet formed by two singlet states of a
    non-Kramers system. This class generalizes and replaces the earlier
    KramersDoublet class.

    The g-tensor is evaluated following Section III.A of

        L. F. Chibotaru and L. Ungur. J. Chem. Phys. 2012, 137, 064112.

    The construction uses only the matrix elements of the magnetic moment
    within the two-state subspace and is invariant under unitary mixing of
    the two states, so it applies equally to degenerate doublets and to
    quasi-doublets. For a quasi-doublet of a time-reversal-even Hamiltonian
    the transverse principal g values vanish identically and the doublet is
    of the Ising type; the g-tensor alone does not define its low-energy
    physics, so the tunneling gap between the two states is stored as a
    separate attribute (the effective S = 1/2 Hamiltonian of a
    quasi-doublet contains the gap as a pseudospin term in addition to the
    Zeeman interaction).

    The g values are reported as dimensionless quantities: the magnetic
    moment operators (which include the Bohr magneton, i.e. are in units of
    energy per tesla of the unit system) are divided by the Bohr magneton
    of the unit system.

    Time-reversal symmetry of the input is checked: a component of the
    magnetic moment proportional to the identity within the doublet (which
    is forbidden by time-reversal symmetry and would contaminate the
    g-tensor) is projected out and a warning is printed if it is
    significant. For quasi-doublets the diagonal moment matrix elements,
    which also vanish under time reversal, are checked in the same way.

    Arguments
    ---------
    states : tuple of int
        The indices of the two states spanning the doublet.
    magnetic_moment : list of complex128
        A list of the matrix representation of the three Cartesian components
        of the magnetic moment operator. The magnetic moment operators need
        to be in the same basis the states in the states argument are defined.
        Usually this means that the magnetic moment is transformed to the
        Hamiltonian eigenbasis.
    units : EnergyUnitSystem
        The unit system.

    Optional arguments
    ------------------
    energies : list of float or None
        The eigenvalues of the Hamiltonian, in the energy unit of the unit
        system, in the same basis and order as the magnetic moment
        operators. These are needed for the tunneling gap and for the
        automatic Kramers classification. Default is None, in which case
        the tunneling gap is not available (None) and the classification
        relies on the structure of the g-tensor alone.
    kramers : boolean or None
        Whether the doublet is a Kramers doublet. The default None means
        that the nature of the doublet is determined automatically: a
        doublet split by more than the degeneracy tolerance is classified
        as a non-Kramers quasi-doublet; a degenerate doublet with more than
        one nonvanishing principal g value must be a Kramers doublet; a
        degenerate doublet with a single nonvanishing principal g value
        (axial doublet) cannot be classified from the magnetic data alone
        and the attribute is left as None. Pass True or False explicitly
        when the electron-count parity of the system is known.
    degeneracy_tolerance : float or None
        The energy separation, in the energy unit of the unit system, below
        which the two states are considered degenerate. The default None
        uses 1.0e-6 times the Bohr magneton of the unit system (the Zeeman
        energy in a field of about one microtesla), which is above typical
        numerical noise in the splitting of an ab initio Kramers doublet
        but below physically meaningful tunneling gaps.
    rotation : Rotation or None
        The rotation from the input coordinate frame (the frame in which
        the original, unrotated problem was defined, e.g. the frame of the
        operator matrices of an ab initio calculation) to the frame in
        which the magnetic moment operators passed to this class are
        expressed: the tensor components of the two frames are related by
        v_current = R * v_input. When given, the doublet "knows" how its
        frame is oriented with respect to the input frame, the g-tensor
        expressed in the input frame is available in the
        input_frame_g_tensor attribute, and the string representation
        prints the g-tensor and its principal magnetic axes with respect to
        the input frame. The default None corresponds to the identity (the
        operators are given in the input frame).
    print_output : boolean
        Whether to print output. Default is False.

    Attributes
    ----------
    states : tuple of int
        The indices of the two states spanning the doublet.
    rotation : Rotation
        The rotation from the input coordinate frame to the frame of the
        magnetic moment operators (see the corresponding optional
        argument). The identity if none was given.
    input_frame_g_tensor : CartesianTensor
        The g-tensor expressed in the input coordinate frame,
        g_input = R^T * g * R. Identical to g_tensor when no rotation was
        given.
    reduced_magnetic_moment : list of array of complex128
        A list of three two-by-two matrices containing the matrix representations of the
        three Cartesian components of the magnetic moment operator in a basis of the
        two states spanning the doublet (as given, before the removal of the
        identity component).
    identity_magnetic_moment : array of float64
        The three coefficients of the identity component of the magnetic
        moment within the doublet, in the units of the magnetic moment
        operators. These vanish for time-reversal-symmetric input; a
        significant value indicates broken time-reversal symmetry and is
        reported with a warning. The component is projected out before the
        g-tensor is evaluated.
    g_tensor : CartesianTensor
        The Cartesian g-tensor describing the Zeeman interaction of the
        doublet, expressed in the frame of the magnetic moment operators
        passed to the class. Dimensionless.
    state_energies : tuple of float or None
        The eigenvalues of the two states spanning the doublet, in the
        energy unit of the unit system and in the order of the states
        attribute, or None if the energies were not provided.
    tunneling_gap : float or None
        The energy separation of the two states in the energy unit of the
        unit system, or None if the energies were not provided. For a
        Kramers doublet this is zero up to numerical noise; for a
        quasi-doublet it is the tunneling gap of the effective S = 1/2
        Hamiltonian.
    kramers : boolean or None
        True for a Kramers doublet, False for a non-Kramers doublet or
        quasi-doublet, None if the nature of the doublet cannot be decided
        from the available data (see the kramers argument).
    degeneracy_tolerance : float
        The degeneracy tolerance actually used (see the corresponding
        optional argument).
    units : EnergyUnitSystem
        The unit system.
    print_output : boolean
        Whether to print output. Default is False.

    Public methods
    --------------

    Private methods
    ---------------
    __remove_identity_component()
        Project the identity component out of the reduced magnetic moment
        and store it as an attribute; warn if it is significant.
    __calculate_g_tensor()
        Calculate the Cartesian g-tensor and store it as an attribute.
    __classify_doublet(energies,kramers)
        Determine the state energies, the tunneling gap and the Kramers
        classification and store them as attributes.
    __check_quasi_doublet_time_reversal()
        For a genuinely split quasi-doublet, check that the diagonal
        moments vanish as required by time-reversal symmetry and warn if
        they do not.

    Class methods
    -------------
    from_pseudospin_operator(states,hamiltonian,magnetic_moment,units)
        Initiate the class with the Hamiltonian and magnetic moment as instances
        of PseudoSpinOperator. The Hamiltonian is needed as the magnetic moment
        operators need to be transformed from the pseudospin eigenbasis to the
        Hamiltonian eigenbasis; its eigenvalues also provide the energies.
    from_general_operator_matrix(states,magnetic_moment,units)
        Initiate the class with the Hamiltonian and magnetic moment as instances
        of GeneralOperatorMatrix.
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if all
        tests passed.
    """

    def __remove_identity_component(self):
        """Split the reduced magnetic moment into its identity and traceless
        parts within the doublet. The identity coefficients are stored in
        the identity_magnetic_moment attribute and the traceless parts are
        returned as a list of three two-by-two matrices.

        Time-reversal symmetry forbids an identity component: for a Kramers
        doublet the two states are time-reversal partners (the diagonal
        moments are opposite), and for a quasi-doublet of a non-degenerate
        time-even Hamiltonian the diagonal moments vanish individually. A
        significant identity component therefore indicates broken
        time-reversal symmetry in the input (e.g. a magnetic field folded
        into the Hamiltonian); it cannot be represented by an S = 1/2
        Zeeman term and would contaminate the g-tensor, so it is projected
        out with a warning.
        """
        identity_component = np.zeros(3, dtype=np.float64)
        traceless_moment   = []
        moment_scale       = 0.0

        for alpha in range(0,3):
            trace_half = 0.5*np.trace(self.reduced_magnetic_moment[alpha])
            identity_component[alpha] = trace_half.real
            traceless_moment.append(self.reduced_magnetic_moment[alpha]
                                    - trace_half*np.identity(2, dtype=np.complex128))
            moment_scale = max(moment_scale,
                               np.max(np.abs(self.reduced_magnetic_moment[alpha])))

        self.identity_magnetic_moment = identity_component

        if moment_scale > 0.0 and \
           np.max(np.abs(identity_component)) > 1.0e-6*moment_scale:
            print("WARNING in PseudoSpinDoublet.")
            print("Warning: The magnetic moment has a significant component")
            print("         proportional to the identity within the doublet,")
            print("         which is forbidden by time-reversal symmetry. The")
            print("         input may break time reversal (e.g. a magnetic")
            print("         field folded into the Hamiltonian). The component")
            print("         has been projected out of the g-tensor.")
            print("         Identity component:", identity_component)
            print()

        return traceless_moment


    def __calculate_g_tensor(self):
        """Calculate the Cartesian g-tensor and store it as an attribute. The calculation
        follows Section III.A in

            L. F. Chibotaru and L. Ungur. J. Chem. Phys. 2012, 137, 064112.

        The identity component of the moment (forbidden by time-reversal
        symmetry) is projected out first, and the moments are divided by
        the Bohr magneton of the unit system so that the g values are
        dimensionless. The construction is invariant under unitary mixing
        of the two states and therefore applies to degenerate doublets and
        quasi-doublets alike.
        """
        traceless_moment = self.__remove_identity_component()

        A = np.zeros((3,3), dtype=np.complex128)

        for alpha in range(0,3):
            for beta in range(0,3):
                for i in range(0,2):
                    for j in range(0,2):
                        A[alpha][beta] += 0.5 * traceless_moment[alpha][i][j]\
                                          * traceless_moment[beta][j][i]

        A /= self.units.mu_B**2

        # Check that A is Hermitian and diagonalize it.
        if not np.allclose(A,A.conj().T):
            print("ERROR in PseudoSpinDoublet.")
            print("Error: A matrix is not Hermitian.")
            print(A)
            print("Error termination.")
            sys.exit(1)

        A_eig, R = la.eigh(A)

        # A is positive semidefinite up to numerical noise. Set eigenvalues
        # that are zero relative to the largest one to exactly zero to avoid
        # a domain error in the square root below; the threshold is relative,
        # so it does not depend on the unit system. A significantly negative
        # eigenvalue indicates an internal inconsistency.
        A_scale = np.max(np.abs(A_eig))
        for i in range(0,3):
            if abs(A_eig[i]) < 1.0e-12*A_scale:
                A_eig[i] = 0.0
            elif A_eig[i] < 0.0:
                print("ERROR in PseudoSpinDoublet.")
                print("Error: The A matrix has a negative eigenvalue.")
                print(A_eig)
                print("Error termination.")
                sys.exit(1)

        R_inv    = la.inv(R)
        g_diag   = np.zeros((3,3), dtype=np.float64)

        for alpha in range(0,3):
            g_diag[alpha][alpha] = 2.0*sqrt(A_eig[alpha])

        g = np.dot(R,np.dot(g_diag,R_inv))

        # Check that the g tensor is real at this point.
        if not np.allclose(g.imag,np.zeros((3,3), dtype=np.float64)):
            print("ERROR in PseudoSpinDoublet.")
            print("Error: Complex g tensor.")
            print(g)
            print("Error termination.")
            sys.exit(1)

        # Disregard the imaginary part of g that should be close to zero.
        self.g_tensor = tensors.CartesianTensor(g.real)


    def __classify_doublet(self,energies,kramers):
        """Determine the energies of the two states, the tunneling gap and
        whether the doublet is a Kramers doublet, and store them in the
        state_energies, tunneling_gap and kramers attributes. See the class
        documentation for the classification rules. Called after the
        g-tensor has been evaluated.
        """
        # The tunneling gap from the state energies, when available.
        if energies is None:
            self.state_energies = None
            self.tunneling_gap  = None
        else:
            if len(energies) <= max(self.states):
                print("ERROR in PseudoSpinDoublet.")
                print("Error: The energies list is shorter than the largest state index.")
                print("Error termination.")
                sys.exit(1)
            self.state_energies = (energies[self.states[0]],
                                   energies[self.states[1]])
            self.tunneling_gap  = abs(energies[self.states[1]]
                                      - energies[self.states[0]])

        degenerate = None
        if self.tunneling_gap is not None:
            degenerate = self.tunneling_gap <= self.degeneracy_tolerance

        # The number of nonvanishing principal g values. A doublet with more
        # than one nonvanishing principal value must be a Kramers doublet
        # (a non-Kramers doublet carries an Ising moment only); an axial
        # doublet cannot be classified from the magnetic data alone.
        g_values = np.abs(np.array(self.g_tensor.eigenvalues))
        if np.max(g_values) > 0.0:
            n_magnetic_axes = int(np.sum(g_values > 1.0e-6*np.max(g_values)))
        else:
            n_magnetic_axes = 0

        if kramers is not None:
            # An explicit classification from the caller takes precedence,
            # but a Kramers doublet split by more than the degeneracy
            # tolerance violates Kramers' theorem and is flagged.
            self.kramers = kramers
            if kramers and degenerate is not None and not degenerate:
                print("WARNING in PseudoSpinDoublet.")
                print("Warning: The doublet was declared a Kramers doublet but the")
                print("         two states are split by more than the degeneracy")
                print("         tolerance. Kramers doublets are exactly degenerate")
                print("         in zero field; check the input or the tolerance.")
                print("         Splitting: {0} {1}".format(self.tunneling_gap,
                                                           self.units.energy_unit_str))
                print()
        elif degenerate is not None and not degenerate:
            # A split doublet cannot be a Kramers doublet.
            self.kramers = False
        elif n_magnetic_axes >= 2:
            # More than one magnetic axis requires a Kramers doublet.
            self.kramers = True
        else:
            # Degenerate (or unknown energies) and axial: undecidable from
            # the available data.
            self.kramers = None


    def __check_quasi_doublet_time_reversal(self):
        """For a quasi-doublet, check that the diagonal (traceless) part of
        the magnetic moment within the doublet vanishes, as required by
        time-reversal symmetry for non-degenerate states of a time-even
        Hamiltonian, and print a warning if it does not. The g-tensor is
        evaluated as usual, but its physical interpretation is questionable
        when this check fails.
        """
        moment_scale  = 0.0
        largest_diag  = 0.0
        for alpha in range(0,3):
            moment = self.reduced_magnetic_moment[alpha]
            moment_scale = max(moment_scale,np.max(np.abs(moment)))
            largest_diag = max(largest_diag,
                               0.5*abs(moment[0][0] - moment[1][1]))

        if moment_scale > 0.0 and largest_diag > 1.0e-6*moment_scale:
            print("WARNING in PseudoSpinDoublet.")
            print("Warning: The doublet is a non-degenerate quasi-doublet, but the")
            print("         magnetic moment has significant diagonal matrix elements")
            print("         within it. Time-reversal symmetry requires the diagonal")
            print("         moments of non-degenerate states to vanish; the input")
            print("         may break time reversal, or the two states may not form")
            print("         a physically meaningful quasi-doublet.")
            print()


    def __repr__(self):
        """Return a human-readable summary of the doublet: its type, tunneling gap and g-tensor in the input axis frame."""
        if self.kramers is True:
            tmp_str = "      Doublet type: Kramers doublet\n"
        elif self.kramers is False:
            tmp_str = "      Doublet type: non-Kramers (quasi-)doublet\n"
        else:
            tmp_str = "      Doublet type: undetermined (axial doublet)\n"

        if self.tunneling_gap is None:
            tmp_str += "      Tunneling gap: not available\n\n"
        else:
            tmp_str += "      Tunneling gap: {0:16.9e} {1}\n\n"\
                       .format(self.tunneling_gap,self.units.energy_unit_str)

        return tmp_str + str(self.input_frame_g_tensor)


    def __init__(self, states, magnetic_moment, units,
                 energies=None,
                 kramers=None,
                 degeneracy_tolerance=None,
                 rotation=None,
                 print_output=False):
        """Upon class initiation calculate the g-tensor, the tunneling gap
        and the Kramers classification."""
        self.states       = states
        self.units        = units
        self.print_output = print_output

        if not len(self.states) == 2:
            print("ERROR in PseudoSpinDoublet.")
            print("Error: The states tuple must contain the indices of two states.")
            print("Error termination.")
            sys.exit(1)

        # The rotation from the input coordinate frame to the frame of the
        # given magnetic moment operators; the identity if none was given.
        if rotation is None:
            self.rotation = tensors.Rotation(np.identity(3, dtype=np.float64))
        elif isinstance(rotation,tensors.Rotation):
            self.rotation = rotation
        else:
            print("ERROR in PseudoSpinDoublet.")
            print("Error: The rotation argument must be an instance of Rotation")
            print("       (or None for the identity).")
            print("Error termination.")
            sys.exit(1)

        # The default degeneracy tolerance is the Zeeman energy in a field
        # of about one microtesla, expressed in the current energy unit
        # through the Bohr magneton of the unit system.
        if degeneracy_tolerance is None:
            self.degeneracy_tolerance = 1.0e-6*self.units.mu_B
        else:
            self.degeneracy_tolerance = degeneracy_tolerance

        tikk = time.time()

        if self.print_output:
            print('    Evaluating properties of a pseudospin doublet ...')
            print('        Transforming magnetic moment operators ...')

        self.reduced_magnetic_moment = []
        for alpha in range(0,3):
            tmp_magnetic_moment = np.zeros((2,2), dtype=np.complex128)

            i = self.states[0]
            j = self.states[1]

            tmp_magnetic_moment[0][0] = magnetic_moment[alpha][i][i]
            tmp_magnetic_moment[1][1] = magnetic_moment[alpha][j][j]
            tmp_magnetic_moment[1][0] = magnetic_moment[alpha][j][i]
            tmp_magnetic_moment[0][1] = magnetic_moment[alpha][i][j]

            self.reduced_magnetic_moment.append(tmp_magnetic_moment)

        if self.print_output:
            print('        Evaluating g-tensor ...')
        self.__calculate_g_tensor()

        # The g-tensor expressed in the input coordinate frame. The tensor
        # components of the two frames are related by g = R g_input R^T,
        # so g_input = R^T g R. With the default identity rotation this is
        # identical to the g_tensor attribute.
        rotation_matrix = np.asarray(self.rotation.rotation_matrix, dtype=np.float64)
        self.input_frame_g_tensor = tensors.CartesianTensor(
            np.dot(rotation_matrix.T,np.dot(self.g_tensor.tensor,rotation_matrix)))
        self.g_tensor.frame = 'frame of the given magnetic moment operators'
        self.input_frame_g_tensor.frame = 'input axis frame'

        if self.print_output:
            print('        Classifying the doublet ...')
        self.__classify_doublet(energies,kramers)

        # The diagonal-moment check applies only to genuinely split
        # quasi-doublets: an exactly degenerate non-Kramers (Ising) doublet
        # may legitimately carry diagonal moments in the basis in which it
        # is given (e.g. the |+-M> basis).
        if self.kramers is False and self.tunneling_gap is not None \
           and self.tunneling_gap > self.degeneracy_tolerance:
            self.__check_quasi_doublet_time_reversal()

        tokk = time.time()
        if self.print_output:
            print("    Done.")
            print("    Time spent {0:12.3f}".format(tokk-tikk))
            print()


    @classmethod
    def from_pseudospin_operator(cls,states,hamiltonian,magnetic_moment,units,
                                 kramers=None,degeneracy_tolerance=None,
                                 rotation=None,
                                 print_output=False):
        """Initiate the class with the Hamiltonian and magnetic moment as
        instances of PseudoSpinOperator. The Hamiltonian is needed as the magnetic
        moment operators need to be transformed from the pseudospin eigenbasis to
        the Hamiltonian eigenbasis; its eigenvalues also provide the state
        energies for the tunneling gap and the Kramers classification.

        Arguments
        ---------
        states : tuple of int
            The indices of the two states spanning the doublet.
        hamiltonian : PseudoSpinOperator
            The instance containing the Hamiltonian. The instance must contain
            the eigenvectors (and eigenvalues) as attributes.
        magnetic_moment : PseudoSpinVectorOperator
            A list of the instances containing the three Cartesian component of
            the magnetic moment operator. Each instance must contain the matrix
            representation of the operator.
        units : EnergyUnitSystem
            The unit system.

        Optional arguments
        ------------------
        kramers : boolean or None
            Whether the doublet is a Kramers doublet; None (default) means
            automatic classification. See the class documentation.
        degeneracy_tolerance : float or None
            The degeneracy tolerance; see the class documentation.
        rotation : Rotation or None
            The rotation from the input coordinate frame to the frame of
            the magnetic moment operators; see the class documentation.
        print_output : boolean
            Whether to print output.
        """

        transformed_magnetic_moment_list = []
        for alpha in range(0,3):
            tmp_operator = fu.matrix_utils.basis_transformation(hamiltonian.eigenvectors,
                                                                magnetic_moment.operator_list[alpha].matrix,0)

            transformed_magnetic_moment_list.append(deepcopy(tmp_operator))

        return cls(states,transformed_magnetic_moment_list,units,
                   energies=getattr(hamiltonian,'eigenvalues',None),
                   kramers=kramers,
                   degeneracy_tolerance=degeneracy_tolerance,
                   rotation=rotation,
                   print_output=print_output)


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class and run a set of tests on the class constructor
        and the class methods. Return True if all tests passed and False
        otherwise.

        The tests cover the absolute scale (dimensionless g values), the
        isotropy and principal axes of the g-tensor, the tunneling gap, the
        Kramers classification of degenerate, axial and quasi-doublets, and
        the removal of a time-reversal-forbidden identity component of the
        moment. The quasi-doublet test uses the ground doublet of a spin-1
        zero-field-splitting model H = D*Sz^2 + E*(Sx^2 - Sy^2) with D < 0,
        whose exact ground quasi-doublet has a gap of 2E and an Ising
        g-tensor with gz = 4 along z.

        Optional arguments
        ------------------
        print_output : boolean
            Whether to print the outcome of each test. Default is True.
        """
        from ouluspin import units
        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('PseudoSpinDoublet',test_name,
                                                       condition,print_output))

        tmp_units = units.EnergyUnitSystem('wavenumber')

        # An isotropic spin-1/2 doublet with mu = -2*mu_B*S must give an
        # isotropic and dimensionless g-tensor with g = 2, be classified as
        # a Kramers doublet (three magnetic axes) and have a vanishing
        # tunneling gap.
        spin_matrix_list = debug_output.spin_matrices(1)
        magnetic_moment  = [-2.0*tmp_units.mu_B*spin_matrix for spin_matrix
                            in spin_matrix_list]

        doublet = cls((0,1),magnetic_moment,tmp_units,energies=[0.0,0.0])

        g = doublet.g_tensor.tensor
        check('g-tensor is symmetric', doublet.g_tensor.symmetric)
        check('g-tensor of an isotropic doublet is isotropic',
              np.allclose(g,g[0][0]*np.identity(3)))
        check('isotropic spin-1/2 doublet has g = 2',
              np.allclose(g,2.0*np.identity(3)))
        check('isotropic degenerate doublet classified as Kramers',
              doublet.kramers is True)
        check('degenerate doublet has zero tunneling gap',
              doublet.tunneling_gap == 0.0)
        check('identity moment of a Kramers doublet vanishes',
              np.allclose(doublet.identity_magnetic_moment,0.0))

        # An Ising-type doublet with mu = -2*mu_B*S_z only: the g-tensor must
        # be axial with vanishing transverse components and gz = 2, and the
        # largest principal value must be along z. A degenerate axial doublet
        # cannot be classified from the magnetic data alone (kramers None),
        # but an explicit classification must be respected.
        magnetic_moment = [0.0*spin_matrix_list[0],
                           0.0*spin_matrix_list[1],
                           -2.0*tmp_units.mu_B*spin_matrix_list[2]]
        doublet = cls((0,1),magnetic_moment,tmp_units,energies=[0.0,0.0])
        eigenvalues  = np.array(doublet.g_tensor.eigenvalues)
        eigenvectors = doublet.g_tensor.eigenvectors
        largest      = np.argmax(eigenvalues)
        check('axial doublet has two vanishing principal g values',
              (np.sum(np.abs(eigenvalues) < 1.0e-8) == 2))
        check('axial spin-1/2 doublet has gz = 2',
              abs(np.max(eigenvalues) - 2.0) < 1.0e-8)
        check('main magnetic axis of an axial doublet is along z',
              abs(abs(eigenvectors[2][largest]) - 1.0) < 1.0e-8)
        check('degenerate axial doublet is left unclassified',
              doublet.kramers is None)
        doublet = cls((0,1),magnetic_moment,tmp_units,energies=[0.0,0.0],
                      kramers=True)
        check('explicit Kramers classification is respected',
              doublet.kramers is True)

        # Without a rotation argument the input frame coincides with the
        # frame of the moment operators.
        check('input-frame g-tensor defaults to the operator frame',
              np.allclose(doublet.input_frame_g_tensor.tensor,
                          doublet.g_tensor.tensor))

        # With an attached rotation the g-tensor is also expressed in the
        # input frame: for the axial doublet above and a 90-degree rotation
        # about y (v_current = R*v_input with the current z axis
        # corresponding to the input-frame -x axis), the main magnetic axis
        # in the input frame must lie along x while the principal g values
        # are unchanged.
        rotation_matrix = np.array([[0.0,0.0,1.0],
                                    [0.0,1.0,0.0],
                                    [-1.0,0.0,0.0]])
        doublet = cls((0,1),magnetic_moment,tmp_units,energies=[0.0,0.0],
                      kramers=True,
                      rotation=tensors.Rotation(rotation_matrix))
        input_eigenvalues  = np.array(doublet.input_frame_g_tensor.eigenvalues)
        input_eigenvectors = np.array(doublet.input_frame_g_tensor.eigenvectors)
        largest            = int(np.argmax(input_eigenvalues))
        check('rotation preserves the principal g values',
              np.allclose(np.sort(input_eigenvalues),
                          np.sort(doublet.g_tensor.eigenvalues)))
        check('main magnetic axis expressed in the input frame',
              abs(abs(input_eigenvectors[0][largest]) - 1.0) < 1.0e-8)

        # A non-Kramers quasi-doublet: the ground doublet of a spin-1 system
        # with H = D*Sz^2 + E*(Sx^2 - Sy^2), D < 0. The exact ground states
        # are (|1> +- |-1>)/sqrt(2) split by 2E, with an Ising g-tensor
        # gz = 2*g_e*|<1|Sz|2>| = 4 along z regardless of the gap.
        D_zfs, E_zfs = -10.0, 0.5
        spin_matrix_list = debug_output.spin_matrices(2)
        Sx, Sy, Sz = spin_matrix_list
        H = D_zfs*np.dot(Sz,Sz) + E_zfs*(np.dot(Sx,Sx) - np.dot(Sy,Sy))
        eigenvalues_h, eigenvectors_h = la.eigh(H)
        magnetic_moment = [-2.0*tmp_units.mu_B
                           *np.dot(eigenvectors_h.conj().T,np.dot(S,eigenvectors_h))
                           for S in spin_matrix_list]

        doublet = cls((0,1),magnetic_moment,tmp_units,energies=list(eigenvalues_h))
        eigenvalues  = np.array(doublet.g_tensor.eigenvalues)
        eigenvectors = doublet.g_tensor.eigenvectors
        largest      = np.argmax(eigenvalues)
        check('quasi-doublet tunneling gap equals 2E',
              abs(doublet.tunneling_gap - 2.0*E_zfs) < 1.0e-10)
        check('quasi-doublet classified as non-Kramers',
              doublet.kramers is False)
        check('quasi-doublet g-tensor is of Ising type',
              (np.sum(np.abs(eigenvalues) < 1.0e-8) == 2))
        check('quasi-doublet has gz = 4',
              abs(np.max(eigenvalues) - 4.0) < 1.0e-8)
        check('main magnetic axis of the quasi-doublet is along z',
              abs(abs(eigenvectors[2][largest]) - 1.0) < 1.0e-8)

        # A time-reversal-forbidden identity component of the moment must be
        # projected out of the g-tensor (and reported; the warning printed
        # by the next test is expected).
        g_reference = doublet.g_tensor.tensor
        magnetic_moment_broken = deepcopy(magnetic_moment)
        magnetic_moment_broken[2] = magnetic_moment_broken[2]\
                                    + 0.3*tmp_units.mu_B*np.identity(3, dtype=np.complex128)
        doublet = cls((0,1),magnetic_moment_broken,tmp_units,
                      energies=list(eigenvalues_h))
        check('identity moment component is detected',
              abs(doublet.identity_magnetic_moment[2] - 0.3*tmp_units.mu_B) < 1.0e-10)
        check('identity moment component is projected out of g',
              np.allclose(doublet.g_tensor.tensor,g_reference))

        check('string representation renders in the input frame',
              'input axis frame' in str(doublet))

        return debug_output.test_summary('PseudoSpinDoublet',result_list,print_output)


    @classmethod
    def from_general_operator_matrix(cls,states,hamiltonian,magnetic_moment,units,
                                     kramers=None,degeneracy_tolerance=None,
                                     rotation=None,
                                     print_output=False):
        """Initiate the class with the Hamiltonian and magnetic moment as instances
        of GeneralOperatorMatrix. The Hamiltonian is needed as the magnetic
        moment operators need to be transformed from the pseudospin eigenbasis to
        the Hamiltonian eigenbasis; its eigenvalues also provide the state
        energies for the tunneling gap and the Kramers classification.

        states : tuple of int
            The indices of the two states spanning the doublet.
        hamiltonian : GeneralOperatorMatrix
            The instance containing the Hamiltonian. The instance must contain
            the eigenvectors (and eigenvalues) as attributes.
        magnetic_moment : GeneralVectorOperatorMatrix
            A list of the instances containing the three Cartesian component of
            the magnetic moment operator. Each instance must contain the matrix
            representation of the operator.
        units : EnergyUnitSystem
            The unit system.

        Optional arguments
        ------------------
        kramers : boolean or None
            Whether the doublet is a Kramers doublet; None (default) means
            automatic classification. See the class documentation.
        degeneracy_tolerance : float or None
            The degeneracy tolerance; see the class documentation.
        rotation : Rotation or None
            The rotation from the input coordinate frame to the frame of
            the magnetic moment operators; see the class documentation.
        print_output : boolean
            Whether to print output.
        """

        transformed_magnetic_moment_list = []
        for alpha in range(0,3):
            tmp_operator = fu.matrix_utils.basis_transformation(hamiltonian.eigenvectors,
                                                                magnetic_moment.operator_list[alpha].matrix,0)

            transformed_magnetic_moment_list.append(deepcopy(tmp_operator))

        return cls(states,transformed_magnetic_moment_list,units,
                   energies=getattr(hamiltonian,'eigenvalues',None),
                   kramers=kramers,
                   degeneracy_tolerance=degeneracy_tolerance,
                   rotation=rotation,
                   print_output=print_output)
