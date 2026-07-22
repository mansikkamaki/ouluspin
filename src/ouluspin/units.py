# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import math

import scipy
import scipy.constants as const


class EnergyUnitSystem:
    """A class used to define the energy unit, to carry out unit conversions,
    and to store the values of various natural constants.

    The energy unit will be chosen by the user. All other units are given in the
    SI, with the Boltzmann constant, Planck constant and the Bohr magneton
    converted to the user-chosen energy unit.

    This class will also contain units that are independent of unit systems,
    namely the free-electron g-factor and the Avogadro constant, as well as the
    conversion factors for converting between Bohr and Angstrom.

    The natural constants stored as attributes and used in the conversions
    are obtained from the scipy.constants module.

    Arguments
    ---------
    energy_unit : str
        The energy unit to be used. Allowed values are:
          - joule
          - wavenumber (cm^-1)
          - kelvin
          - millielectronvolt
          - electronvolt

    Optional arguments
    ------------------

    Attributes
    ----------
    energy_unit : str
        The energy unit to be used.
    pi : float
        Value of pi.
    g_e : float
        The free-electron g-factor.
    c : float
        The speed of light in SI.
    e : float
        The (positive) elementary charge in SI.
    k_B : float
        The Boltzmann constant in the chocen energy unit.
    h : float
        The Planck constant in the chosen energy unit.
    hbar : float
        The reduced Placnk constant (h/(2*pi)) in the chosen energy unit.
    mu_B : float
        The Bohr magneton in the chosen energy unit.
    mu_N : float
        The nuclear magneton in the chosen energy unit.
    k_B_si : float
        The Boltzmann constant in SI.
    h_si : float
        The Planck constant in Si.
    hbar_si : float
        The reduced Placnk constant in SI.
    mu_B_si : float
        The Bohr magneton in SI.
    mu_N_si : float
        The nuclear magneton in SI.
    mu_0_si : float
        Permeability of vacuumin in SI.
    E_h_si : float
        The Hartree energy in SI.
    N_A : float
        The Avogadro constant.
    energy_unit_str : str
        String of the energy unit abbreviation.
    k_B_str : str
        Unit of the Boltzmann constant.
    h_str : str
        Unit of the Planck constant.
    mu_B_str : str
        Unit of the Borh magneton.
    mu_N_str : str
        Unit of the nuclear magneton.
    si_to_energy : float
        A conversion factor from the SI energy unit (joule) to the energy
        unit of the system.
    bohr_to_angstrom : float
        A conversion factor for converting Bohrs to Angstrom.

    Private methods
    ---------------
    __error(message)
        Report an error and stop.
    __warning(message)
        Report a warning and continue.

    Public methods
    --------------
    convert_energy_unit(value,unit) : float
       Takes an energy value and its unit as arguments and converts the value to the
       chosen energy unit.

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


    def convert_energy_unit(self, value, unit):
        """Takes an energy value and its unit as arguments and converts the value to the
        chosen energy unit.
        """
        if unit == self.energy_unit:
            return value
        
        if unit == 'joule':
            unit_to_si = 1.0
        elif unit == 'wavenumber':
            unit_to_si = 100.0 * self.h_si * self.c
        elif unit == 'millielectronvolt':
            unit_to_si = self.e / 1000.0
        elif unit == 'electronvolt':
            unit_to_si = self.e
        elif unit == 'kelvin':
            unit_to_si = self.k_B_si
        elif unit == 'megahertz':
            unit_to_si = 1.0e6 * self.h_si
        elif unit == 'millitesla':
            unit_to_si = self.mu_B_si / 1000.0
        elif unit == 'tesla':
            unit_to_si = self.mu_B_si
        elif unit == 'hartree':
            unit_to_si = self.E_h_si
        else:
            self.__error("Unknown energy unit in conversion: " + unit + ".")

        return value * unit_to_si * self.si_to_energy

    
    def __repr__(self):
        """Return a human-readable table of the unit system."""
        tmp_str = "    SYSTEM OF UNITS\n\n"
        tmp_str += "      Units provided by the physical_constants method in SciPy version " + scipy.__version__ + "\n\n"
        tmp_str += "      Energy unit: " + self.energy_unit + " (" + self.energy_unit_str + ")\n\n"

        tmp_str += "      Electron g-factor:                 {0:24.16e} \n\n".format(self.g_e)
        tmp_str += "      Speed of light in SI:              {0:24.16e} m s^-1\n".format(self.c)
        tmp_str += "      Boltzmann constant in SI:          {0:24.16e} J K^-1\n".format(self.k_B_si)
        tmp_str += "      Planck constant in SI:             {0:24.16e} J s\n".format(self.h_si)
        tmp_str += "      Reduced Planck constant in SI:     {0:24.16e} J s\n".format(self.hbar_si)
        tmp_str += "      Bohr magneton in SI:               {0:24.16e} J T^-1\n".format(self.mu_B_si)
        tmp_str += "      Nuclear magneton in SI:            {0:24.16e} J T^-1\n".format(self.mu_N_si)
        tmp_str += "      Vacuum permeability in SI:         {0:24.16e} H m^-1\n".format(self.mu_0_si)
        tmp_str += "      Hartree energy in SI:              {0:24.16e} J\n\n".format(self.E_h_si)

        tmp_str += "      Boltzmann constant in energy unit: {0:24.16e} ".format(self.k_B)  + self.k_B_str  + "\n"
        tmp_str += "      Planck constant in energy unit:    {0:24.16e} ".format(self.h)    + self.h_str    + "\n"
        tmp_str += "      Bohr magneton in energ unit:       {0:24.16e} ".format(self.mu_B) + self.mu_B_str + "\n"
        tmp_str += "      Nuclear magneton in energy unit:   {0:24.16e} ".format(self.mu_N) + self.mu_N_str + "\n\n"

        return tmp_str


    def __init__(self, energy_unit):
        """Upon class initation set natural constants in correct units as attributes."""

        self.energy_unit = energy_unit

        # Define natural constants in SI.
        self.pi   = math.pi
        self.g_e  = -const.physical_constants['electron g factor'][0]
        self.c    = const.c
        self.e    = const.e
        
        self.k_B_si  = const.k
        self.h_si    = const.h
        self.hbar_si = const.hbar
        self.mu_B_si = const.physical_constants['Bohr magneton'][0]
        self.mu_N_si = const.physical_constants['nuclear magneton'][0]
        self.mu_0_si = 4*self.pi * 1.00000000054e-7
        self.E_h_si  = const.physical_constants['Hartree energy'][0]
        self.N_A     = const.N_A

        # Convert natural constants to the chocen energy unit.
        if self.energy_unit == 'joule':
            self.si_to_energy    = 1.0
            self.energy_unit_str = 'J'
            self.k_B_str         = 'J K^-1'
            self.h_str           = 'J s'
            self.mu_B_str        = 'J T^-1'
            self.mu_N_str        = 'J T^-1'
            
        elif self.energy_unit == 'wavenumber':
            self.si_to_energy    = 0.01 / (self.h_si * self.c)
            self.energy_unit_str = 'cm^-1'
            self.k_B_str         = 'cm^-1 K^-1'
            self.h_str           = 'cm^-1 s'
            self.mu_B_str        = 'cm^-1 T^-1'
            self.mu_N_str        = 'cm^-1 T^-1'
            
        elif self.energy_unit == 'millielectronvolt':
            self.si_to_energy    = 1000.0 / self.e
            self.energy_unit_str = 'meV'
            self.k_B_str         = 'meV K^-1'
            self.h_str           = 'meV s'
            self.mu_B_str        = 'meV T^-1'
            self.mu_N_str        = 'meV T^-1'
            
        elif self.energy_unit == 'electronvolt':
            self.si_to_energy    = 1.0 / self.e
            self.energy_unit_str = 'eV'
            self.k_B_str         = 'eV K^-1'
            self.h_str           = 'eV s'
            self.mu_B_str        = 'eV T^-1'
            self.mu_N_str        = 'eV T^-1'
            
        elif self.energy_unit == 'kelvin':
            self.si_to_energy    = 1.0 / self.k_B_si
            self.energy_unit_str = 'K'
            self.k_B_str         = ''
            self.h_str           = 'K s'
            self.mu_B_str        = 'K T^-1'
            self.mu_N_str        = 'K T^-1'

        else:
            self.__error("Unknown energy unit: " + energy_unit + ".")

        self.k_B  = self.k_B_si  * self.si_to_energy
        self.h    = self.h_si    * self.si_to_energy
        self.hbar = self.hbar_si * self.si_to_energy
        self.mu_B = self.mu_B_si * self.si_to_energy
        self.mu_N = self.mu_N_si * self.si_to_energy

        # Set other units.
        self.bohr_to_angstrom = 1.0e10 * const.physical_constants['Bohr radius'][0]


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
            result_list.append(debug_output.test_check('EnergyUnitSystem',test_name,
                                                       condition,print_output))

        u = cls('wavenumber')
        check('Boltzmann constant in cm^-1 K^-1',
              abs(u.k_B - 0.69503) < 1.0e-4)
        check('Bohr magneton in cm^-1 T^-1',
              abs(u.mu_B - 0.466864) < 1.0e-5)
        check('identity conversion',
              u.convert_energy_unit(3.14,'wavenumber') == 3.14)
        check('kelvin to wavenumber conversion',
              abs(u.convert_energy_unit(1.0,'kelvin') - u.k_B) < 1.0e-12)
        check('hartree to wavenumber conversion',
              abs(u.convert_energy_unit(1.0,'hartree') - 219474.63) < 0.1)
        check('millielectronvolt to wavenumber conversion',
              abs(u.convert_energy_unit(1.0,'millielectronvolt') - 8.06554) < 1.0e-4)

        u_K = cls('kelvin')
        check('consistency between wavenumber and kelvin unit systems',
              abs(u_K.convert_energy_unit(1.0,'wavenumber')*u.k_B - 1.0) < 1.0e-10)
        check('string representation renders',
              len(str(u_K)) > 0)

        return debug_output.test_summary('EnergyUnitSystem',result_list,print_output)


        
