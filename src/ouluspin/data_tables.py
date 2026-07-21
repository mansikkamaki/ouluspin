# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tabulated and calculated data of the ions treated by the library.

The module collects the data that belong to a chemical species rather than
to a particular calculation: the electron configuration of an ion, the term
structure of its open shell, the quantum numbers of its ground multiplet
and the quantities that follow from them. The data are reached through the
IonData class, which is constructed from the name of the ion:

    dy = ouluspin.IonData('Dy(III)')
    dy.J                            # 15, i.e. J = 15/2 in the doubled form
    dy.lande_g_factor               # 4/3
    print(dy.data_table().string_table())

What can be calculated is calculated rather than tabulated: the terms of
the open shell come from the cfp_utils module of the Fortran extension, and
the ground term is picked out of them by Hund's rules. What cannot be
derived, i.e. which oxidation states of an element are chemically common
and which of them is the ordinary one, is tabulated in the class.

Angular momenta are stored as attributes in the doubled form used
throughout the library, i.e. J = 15 stands for J = 15/2, and are printed as
the true angular momenta in the tables.
"""

import sys

import numpy as np


class IonData:
    """The data of a chemical ion, e.g. Dy(III).

    The class is constructed from the name of the ion and evaluates the
    electron configuration of the ion, the terms of its open shell, the
    quantum numbers of its Hund's-rule ground multiplet and the Lande
    g-factor of that multiplet. The values are stored as attributes and are
    also available as tables through the table methods.

    The name of the ion is given as a string that begins with the chemical
    symbol and states the oxidation state either as an Arabic or as a Roman
    numeral. The parsing is case-insensitive and tolerates the usual
    punctuation, so all of

        'Dy(III)'   'dy iii'   'Dy-III'   'Dy3'   'dy 3'   'Dy3+'   'Dy(3)'

    name the same ion. A Roman numeral must be separated from the chemical
    symbol by a space or by punctuation, since 'Siii' would otherwise be
    both Si(II) and S(III); an Arabic numeral needs no separator, as digits
    cannot be read as part of a symbol. The oxidation state may also be left
    out entirely, in which case the ordinary oxidation state of the element
    is used, e.g. 'Dy' for Dy(III) and 'Fe' for Fe(III).

    Only ions of a single open shell are treated, i.e. the d-block and
    f-block ions whose electrons outside the noble-gas core all belong to
    one d or one f shell. The oxidation states that are too low for this
    (the neutral d-block atoms and the mono-positive f-block ions, which
    carry electrons in the outer s shell as well) and those that are too
    high (which would break into the core) are a fatal error, and the
    message states which of the two is the case. An oxidation state that is
    possible but not chemically common is accepted with a warning, since
    the assignment of the whole valence to the d or f shell is then not
    certain; La(II), for instance, is taken here as 4f1 whereas it is in
    fact 5d1.

    Arguments
    ---------
    ion : str
        The name of the ion, e.g. 'Dy(III)' (see above).

    Optional arguments
    ------------------
    print_output : boolean
        Whether to print the data of the ion upon construction. Default is
        False.

    Attributes
    ----------
    ion : str
        The name of the ion in the canonical form, e.g. 'Dy(III)'.
    element : str
        The chemical symbol of the element in its canonical form, e.g.
        'Dy'.
    atomic_number : int
        The atomic number of the element.
    oxidation_state : int
        The oxidation state of the ion, e.g. 3 for Dy(III).
    block : str
        The block of the periodic table the element belongs to, i.e. '3d',
        '4d', '5d', '4f' or '5f'.
    shell : str
        The open shell of the ion, e.g. '4f'.
    l : int
        The orbital angular momentum of the open shell, i.e. 2 for a d
        shell and 3 for an f shell. NOTE that this is the true value and
        not the doubled one, as it is an integer by nature.
    n_electrons : int
        The number of electrons in the open shell, e.g. 9 for Dy(III).
    electron_configuration : str
        The full electron configuration of the ion, e.g.
        '1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f9'.
    valence_configuration : str
        The configuration of the open shell alone, e.g. '4f9'.
    core_configuration : str
        The noble-gas core in the bracket notation, e.g. '[Xe]'.
    L : int
        The total orbital angular momentum of the ground term, in the
        DOUBLED form, i.e. 10 for the L = 5 of Dy(III).
    S : int
        The total spin of the ground term, in the DOUBLED form, i.e. 5 for
        the S = 5/2 of Dy(III).
    J : int
        The total angular momentum of the ground multiplet, in the DOUBLED
        form, i.e. 15 for the J = 15/2 of Dy(III). This is the value that
        is passed to PseudoSpinBasis as the pseudospin of the multiplet.
    term_symbol : str
        The term symbol of the ground term without the total angular
        momentum, e.g. '6H'.
    ground_multiplet_symbol : str
        The term symbol of the ground multiplet, e.g. '6H15/2'.
    lande_g_factor : float
        The Lande g-factor of the ground multiplet, evaluated with the
        g-factor of the free electron (the CODATA value 2.00231930...),
        e.g. 1.33410643 for Dy(III). This is the accurate value, meant for
        quantitative work. Zero when the ground multiplet has J = 0, where
        the g-factor is not defined.
    lande_g_factor_simple : float
        The Lande g-factor of the ground multiplet in the approximation
        g_e = 2 of the textbooks, e.g. 1.33333333 for Dy(III). In this
        approximation the factor is a rational number, which is what makes
        it convenient for a derivation by hand; it is also the value the
        tables of magnetochemistry are computed with.
    lande_g_factor_str : str
        The simple Lande g-factor as the exact fraction it is, e.g. '4/3'
        for Dy(III), '2' for Gd(III) and '0' for a multiplet of J = 0. It
        is evaluated in exact arithmetic and not by rounding a float.
    terms : list of str
        The term symbols of all the terms of the open shell, e.g.
        ['1S', '3P', ...]. A term symbol appears as many times as there are
        terms carrying it, which are told apart by their seniority; the
        term_data attribute holds the full labels.
    term_data : list of dict
        The terms of the open shell with their complete labels, each a
        dictionary with the items 'symbol', 'S' and 'L' (both DOUBLED),
        'seniority', 'w' (the ordinal among the terms of the same S and L)
        and 'g2_casimir_x12' (twelve times the G2 Casimir eigenvalue, which
        labels the G2 representation of an f-shell term and is zero for the
        other shells).
    n_terms : int
        The number of terms of the open shell.
    n_states : int
        The number of states of the open shell, i.e. the sum of
        (2S+1)(2L+1) over its terms. It is the number of ways the electrons
        are placed in the spin orbitals of the shell, i.e. the binomial
        coefficient of 2(2l+1) over n.
    n_spin_states : int
        The number of spin states of the open shell, i.e. the sum of
        (2L+1) over its terms: the states of ONE spin component of each
        term, which is the number of states of the term divided by its
        multiplicity. A 6H term holds 6*11 = 66 states and 11 spin states.
        These are the states a spin-free calculation carries, the
        spin-orbit coupling turning each of them into the (2S+1) states of
        its multiplicity.
    n_total_electrons : int
        The number of electrons of the ion, i.e. the atomic number less the
        oxidation state.
    n_valence_electrons : int
        The number of electrons of the valence, i.e. those of the open
        shell, which is the same as the n_electrons attribute. The class
        treats only the ions whose electrons outside the noble-gas core all
        belong to that one shell.
    degeneracy : int
        The number of states of the ground multiplet, i.e. 2J + 1.
    print_output : boolean
        Whether the data are printed upon construction.

    Public methods
    --------------
    curie_susceptibility(simple_g_factor=False) : float
        Return the chi*T product of the free ion given by the Curie law for
        the ground multiplet, in cm^3 K mol^-1, with either the accurate or
        the simple Lande g-factor.
    states_by_multiplicity() : list of dict
        Return the terms and the states of the open shell grouped by spin
        multiplicity.
    coefficients_of_fractional_parentage() : dict
        Return the coefficients of fractional parentage of the open shell.
        They are calculated on the first call and stored afterwards.
    data_table() : ResultTable
        Return the full data of the ion as a table, i.e. the ground
        multiplet with its Lande g-factor and chiT product in both
        conventions, the terms of the open shell, and the number of terms,
        of spin states and of states of each spin multiplicity together
        with their totals.
    ground_multiplet_table() : ResultTable
        Return the properties of the ground multiplet alone as a table,
        i.e. the data table without the terms and the state counts.
    cfp_table(threshold=1.0e-10) : ResultTable
        Return the coefficients of fractional parentage as a table.
    term_table() : ResultTable
        Return the terms of the open shell as a table.

    Private methods
    ---------------
    __error(message)
        Report an error and stop.
    __warning(message)
        Report a warning and continue.
    __resolve_ion(ion)
        Parse the name of the ion and store the element and the oxidation
        state as attributes.
    __resolve_configuration()
        Determine the open shell and the electron configuration of the ion
        and store them as attributes.
    __resolve_ground_multiplet()
        Determine the terms of the open shell and the quantum numbers of
        the ground multiplet and store them as attributes.
    __valence_configuration_string() : str
        Return the configuration of the open shell as a string, e.g. '4f9'.
    __lande_g_factor(two_L,two_S,two_J,electron_g_factor) : float
        Return the Lande g-factor of a multiplet, evaluated with the given
        g-factor of the free electron.
    __lande_g_factor_fraction(two_L,two_S,two_J) : str
        Return the Lande g-factor of g_e = 2 as an exact fraction.
    __momentum_string(doubled_value) : str
        Return a doubled angular momentum as the true one, e.g. '15/2'.
    __roman_numeral(value) : str
        Return the Roman numeral of an oxidation state.
    __canonical_symbol(symbol) : str
        Return a chemical symbol in its canonical form and check it.
    __term_label(term) : str
        Return the label of a term as the tables print it.
    __ground_multiplet_rows() : list
        Return the rows shared by the two data tables.
    __table_notes(spin_states=False) : list of str
        Return the explanatory texts of the tables.

    Class methods
    -------------
    parse_ion_name(ion) : (str, int or None)
        Parse the name of an ion into the chemical symbol and the oxidation
        state, the latter being None when the name states none.
    supported_elements() : list of str
        Return the chemical symbols of the elements the class treats.
    common_oxidation_states(element) : list of int
        Return the chemically common oxidation states of the element.
    default_oxidation_state(element) : int
        Return the oxidation state used when the name of an ion states
        none.
    term_symbol_of(two_S, two_L) : str
        Return the term symbol of the given total spin and orbital angular
        momentum, both in the doubled form.
    electron_g_factor() : float
        Return the g-factor of the free electron, i.e. the CODATA value the
        accurate Lande g-factor is evaluated with.
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if
        all tests passed.

    Note on the sources of the data
    -------------------------------
    The term structure of the open shell, and with it the ground term, is
    calculated by the cfp_utils module of the Fortran extension rather than
    tabulated: cfp_terms returns the terms of l^n, and Hund's rules pick the
    ground term out of them as the term of the largest S and, among those,
    of the largest L. The total angular momentum of the ground multiplet
    follows Hund's third rule, i.e. J = |L - S| for a shell less than half
    full and J = L + S for a shell at least half full, and the Lande
    g-factor follows from L, S and J.

    What is tabulated in the class is the part that does not follow from
    the physics: the chemical symbols and atomic numbers of the elements,
    the blocks of the periodic table with their noble-gas cores, and the
    common and ordinary oxidation states of each element. The electron
    configuration itself is not tabulated: the number of electrons of the
    open shell is the atomic number less the core and less the oxidation
    state, which is exact for every ion the class treats.

    Note on the cost
    ----------------
    The Fortran routine that returns the terms of a shell builds the
    coefficients of fractional parentage of that whole shell on its first
    call, and caches them. Constructing the first ion of an f shell
    therefore takes a few tenths of a second and the first ion of a d shell
    a few milliseconds; every further ion of the same shell, and the
    coefficients themselves, are then free. The
    coefficients_of_fractional_parentage method is nevertheless kept out of
    the constructor, so that an instance built only for its ground
    multiplet does not also assemble and store the coefficient matrix.
    """

    # The chemical symbols by atomic number. The whole periodic table is
    # listed so that the name of any element can be parsed and answered
    # with a message telling that the element is not treated, rather than
    # with a message telling that the symbol is unknown.
    __ELEMENT_SYMBOLS = (
        'H',  'He', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar', 'K',  'Ca',
        'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',  'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I',  'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
        'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
    )

    # The electron configurations of the noble gases, which are the cores
    # of the ions treated by the class.
    __NOBLE_GAS_CONFIGURATIONS = {
        'He': "1s2",
        'Ne': "1s2 2s2 2p6",
        'Ar': "1s2 2s2 2p6 3s2 3p6",
        'Kr': "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6",
        'Xe': "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6",
        'Rn': "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d10 6s2 6p6",
    }

    # The blocks of the periodic table the class treats. Each block is
    # given by the range of the atomic numbers, the noble-gas core the ions
    # of the block are built on, the number of electrons of that core, the
    # open shell and its orbital angular momentum, and the lowest oxidation
    # state for which the whole valence belongs to the open shell.
    #
    # The core of the 5d block is the closed 4f shell on top of xenon, i.e.
    # 68 electrons, so that the valence of a 5d ion is what is left of the
    # 5d shell. The lowest oxidation state is 1 for the d-block ions, whose
    # outer s shell is empty as soon as the atom is ionized, and 2 for the
    # f-block ions, whose mono-positive state still carries the outer s
    # electron.
    __BLOCKS = (
        {'name': '3d', 'first_Z': 21, 'last_Z': 30, 'core': 'Ar',
         'core_electrons': 18, 'core_label': "[Ar]",
         'shell': '3d', 'l': 2, 'lowest_oxidation_state': 1},
        {'name': '4d', 'first_Z': 39, 'last_Z': 48, 'core': 'Kr',
         'core_electrons': 36, 'core_label': "[Kr]",
         'shell': '4d', 'l': 2, 'lowest_oxidation_state': 1},
        {'name': '5d', 'first_Z': 72, 'last_Z': 80, 'core': 'Xe',
         'core_electrons': 68, 'core_label': "[Xe] 4f14",
         'shell': '5d', 'l': 2, 'lowest_oxidation_state': 1},
        {'name': '4f', 'first_Z': 57, 'last_Z': 71, 'core': 'Xe',
         'core_electrons': 54, 'core_label': "[Xe]",
         'shell': '4f', 'l': 3, 'lowest_oxidation_state': 2},
        {'name': '5f', 'first_Z': 89, 'last_Z': 103, 'core': 'Rn',
         'core_electrons': 86, 'core_label': "[Rn]",
         'shell': '5f', 'l': 3, 'lowest_oxidation_state': 2},
    )

    # The chemically common oxidation states of each element treated, and
    # the ordinary one, which is used when the name of an ion states none.
    # The lists are the states met in ordinary chemistry; a state outside
    # them is still accepted, with a warning, as long as it leaves a single
    # open shell.
    #
    # The ordinary state is the one the element is usually met in, which is
    # a chemical convention and not a magnetic one: it is +6 for uranium
    # and +4 for thorium, both of which leave an empty f shell and no
    # magnetism at all. The oxidation state is worth stating explicitly
    # whenever a particular one is meant.
    __OXIDATION_STATES = {
        # The 3d block.
        'Sc': {'common': (3,),            'ordinary': 3},
        'Ti': {'common': (2,3,4),         'ordinary': 4},
        'V':  {'common': (2,3,4,5),       'ordinary': 5},
        'Cr': {'common': (2,3,6),         'ordinary': 3},
        'Mn': {'common': (2,3,4,6,7),     'ordinary': 2},
        'Fe': {'common': (2,3),           'ordinary': 3},
        'Co': {'common': (2,3),           'ordinary': 2},
        'Ni': {'common': (2,3),           'ordinary': 2},
        'Cu': {'common': (1,2),           'ordinary': 2},
        'Zn': {'common': (2,),            'ordinary': 2},
        # The 4d block.
        'Y':  {'common': (3,),            'ordinary': 3},
        'Zr': {'common': (4,),            'ordinary': 4},
        'Nb': {'common': (4,5),           'ordinary': 5},
        'Mo': {'common': (3,4,5,6),       'ordinary': 6},
        'Tc': {'common': (4,7),           'ordinary': 7},
        'Ru': {'common': (2,3,4),         'ordinary': 3},
        'Rh': {'common': (1,3),           'ordinary': 3},
        'Pd': {'common': (2,4),           'ordinary': 2},
        'Ag': {'common': (1,2),           'ordinary': 1},
        'Cd': {'common': (2,),            'ordinary': 2},
        # The 5d block.
        'Hf': {'common': (4,),            'ordinary': 4},
        'Ta': {'common': (5,),            'ordinary': 5},
        'W':  {'common': (4,5,6),         'ordinary': 6},
        'Re': {'common': (4,6,7),         'ordinary': 7},
        'Os': {'common': (3,4,6,8),       'ordinary': 4},
        'Ir': {'common': (1,3,4),         'ordinary': 4},
        'Pt': {'common': (2,4),           'ordinary': 2},
        'Au': {'common': (1,3),           'ordinary': 3},
        'Hg': {'common': (1,2),           'ordinary': 2},
        # The lanthanides. The trivalent state is the one of all of them;
        # the divalent and tetravalent states listed are those met as
        # ordinary compounds.
        'La': {'common': (3,),            'ordinary': 3},
        'Ce': {'common': (3,4),           'ordinary': 3},
        'Pr': {'common': (3,4),           'ordinary': 3},
        'Nd': {'common': (2,3),           'ordinary': 3},
        'Pm': {'common': (3,),            'ordinary': 3},
        'Sm': {'common': (2,3),           'ordinary': 3},
        'Eu': {'common': (2,3),           'ordinary': 3},
        'Gd': {'common': (3,),            'ordinary': 3},
        'Tb': {'common': (3,4),           'ordinary': 3},
        'Dy': {'common': (2,3),           'ordinary': 3},
        'Ho': {'common': (3,),            'ordinary': 3},
        'Er': {'common': (3,),            'ordinary': 3},
        'Tm': {'common': (2,3),           'ordinary': 3},
        'Yb': {'common': (2,3),           'ordinary': 3},
        'Lu': {'common': (3,),            'ordinary': 3},
        # The actinides.
        'Ac': {'common': (3,),            'ordinary': 3},
        'Th': {'common': (3,4),           'ordinary': 4},
        'Pa': {'common': (4,5),           'ordinary': 5},
        'U':  {'common': (3,4,5,6),       'ordinary': 6},
        'Np': {'common': (3,4,5,6),       'ordinary': 5},
        'Pu': {'common': (3,4,5,6),       'ordinary': 4},
        'Am': {'common': (2,3,4,5,6),     'ordinary': 3},
        'Cm': {'common': (3,4),           'ordinary': 3},
        'Bk': {'common': (3,4),           'ordinary': 3},
        'Cf': {'common': (2,3),           'ordinary': 3},
        'Es': {'common': (2,3),           'ordinary': 3},
        'Fm': {'common': (2,3),           'ordinary': 3},
        'Md': {'common': (2,3),           'ordinary': 3},
        'No': {'common': (2,3),           'ordinary': 2},
        'Lr': {'common': (3,),            'ordinary': 3},
    }

    # The letters of the term symbols by the total orbital angular
    # momentum. The letter J is skipped, as is the letter P for L = 12, the
    # standard spectroscopic sequence; the letters cover every term of an
    # s, p, d or f shell, whose largest orbital angular momentum is L = 12.
    __TERM_LETTERS = "SPDFGHIKLMNOQRTUVWXYZ"

    # The g-factor of the free electron, i.e. the CODATA value, which is
    # the one the accurate Lande g-factor is evaluated with. The simple
    # Lande g-factor uses the value 2 instead. It is read from the
    # constants of scipy upon the first use.
    ELECTRON_G_FACTOR = None

    # The g-factor of the free electron in the approximation of the
    # textbooks, i.e. the one that makes the Lande g-factor a rational
    # number.
    SIMPLE_ELECTRON_G_FACTOR = 2.0

    # The Curie constant N_A*mu_B^2/(3*k_B) in cgs emu, i.e. the chi*T
    # product of a multiplet of g = 1 and J(J+1) = 1, in cm^3 K mol^-1. It
    # is evaluated from the fundamental constants upon the first call of
    # curie_susceptibility.
    __CURIE_CONSTANT = None

    # The Roman numerals recognized as oxidation states. Oxidation states
    # beyond +8 do not occur.
    __ROMAN_NUMERALS = {
        'i': 1, 'ii': 2, 'iii': 3, 'iv': 4,
        'v': 5, 'vi': 6, 'vii': 7, 'viii': 8,
    }


    def __error(self, message):
        """Report an error and stop. The data of an ion are ordinarily
        resolved at the beginning of a calculation, so an error is reported
        the way the other errors of the library are.
        """
        print("ERROR in IonData.")
        for line in message.split("\n"):
            print("Error: " + line)
        print("Error termination.")
        sys.exit(1)


    def __warning(self, message):
        """Report a warning and continue."""
        print("WARNING in IonData.")
        for line in message.split("\n"):
            print("Warning: " + line)
        print()


    @classmethod
    def parse_ion_name(cls, ion):
        """Parse the name of an ion and return the chemical symbol in its
        canonical form together with the oxidation state, the latter being
        None when the name states none.

        The parsing is case-insensitive and accepts the oxidation state as
        an Arabic or as a Roman numeral, with the usual punctuation around
        it. A Roman numeral must be separated from the chemical symbol,
        since a name such as 'Siii' would otherwise be both Si(II) and
        S(III); an Arabic numeral needs no separator.

        An unparsable name or an unknown chemical symbol produces a fatal
        error.

        Arguments
        ---------
        ion : str
            The name of the ion, e.g. 'Dy(III)', 'dy 3' or 'Dy'.
        """
        import re

        instance = cls.__new__(cls)

        if not isinstance(ion,str):
            instance.__error("The name of the ion must be given as a string, but an "
                             "instance of\n" + type(ion).__name__ + " was given.")

        text = ion.strip().lower()

        if text == "":
            instance.__error("The name of the ion is empty.")

        # The separators that may stand between the chemical symbol and the
        # oxidation state, and the signs and brackets that may surround the
        # oxidation state itself.
        separator = r'[\s\(\[\{\-_,\.:;]'
        closing   = r'[\s\)\]\}\+]*'

        # An Arabic numeral, which may follow the symbol directly since a
        # digit cannot be read as part of a chemical symbol.
        match = re.match(r'^([a-z]{1,3})' + separator + r'*\+?\s*([0-9]+)'
                         + closing + r'$', text)
        if match is not None:
            return (instance.__canonical_symbol(match.group(1)),
                    int(match.group(2)))

        # A Roman numeral, which must be separated from the symbol.
        match = re.match(r'^([a-z]{1,3})' + separator + r'+([ivx]+)'
                         + closing + r'$', text)
        if match is not None:
            numeral = match.group(2)

            if not numeral in cls.__ROMAN_NUMERALS:
                instance.__error("The oxidation state '" + match.group(2)
                                 + "' of the name '" + ion + "' is not a Roman\n"
                                 "numeral between I and VIII.")

            return (instance.__canonical_symbol(match.group(1)),
                    cls.__ROMAN_NUMERALS[numeral])

        # The chemical symbol alone, i.e. no oxidation state.
        match = re.match(r'^([a-z]{1,3})$', text)
        if match is not None:
            return (instance.__canonical_symbol(match.group(1)), None)

        instance.__error("The name of the ion could not be parsed: '" + ion + "'.\n"
                         "The name must begin with the chemical symbol and may state\n"
                         "the oxidation state as an Arabic or as a Roman numeral, e.g.\n"
                         "'Dy(III)', 'Dy III', 'Dy3', 'Dy 3+' or 'Dy'. A Roman numeral\n"
                         "must be separated from the symbol by a space or by\n"
                         "punctuation, since a name such as 'Siii' would otherwise be\n"
                         "both Si(II) and S(III).")


    def __canonical_symbol(self, symbol):
        """Return the chemical symbol in its canonical form, i.e. with the
        first letter in upper case, and check that it names an element.
        """
        canonical = symbol[0].upper() + symbol[1:].lower()

        if not canonical in self.__ELEMENT_SYMBOLS:
            self.__error("There is no element of the chemical symbol '"
                         + canonical + "'.")

        return canonical


    @classmethod
    def supported_elements(cls):
        """Return the chemical symbols of the elements the class treats,
        i.e. those of the d and f blocks, in the order of the atomic
        number.
        """
        return [symbol for symbol in cls.__ELEMENT_SYMBOLS
                if symbol in cls.__OXIDATION_STATES]


    @classmethod
    def common_oxidation_states(cls, element):
        """Return the chemically common oxidation states of the given
        element as a list of int.

        Arguments
        ---------
        element : str
            The chemical symbol of the element, in any case.
        """
        instance = cls.__new__(cls)
        symbol   = instance.__canonical_symbol(element.strip().lower())

        if not symbol in cls.__OXIDATION_STATES:
            instance.__error("The element " + symbol + " is not treated by the class.\n"
                             "The class treats the d-block and f-block elements.")

        return list(cls.__OXIDATION_STATES[symbol]['common'])


    @classmethod
    def default_oxidation_state(cls, element):
        """Return the oxidation state used for the given element when the
        name of an ion states none, i.e. the state the element is
        ordinarily met in.

        Arguments
        ---------
        element : str
            The chemical symbol of the element, in any case.
        """
        instance = cls.__new__(cls)
        symbol   = instance.__canonical_symbol(element.strip().lower())

        if not symbol in cls.__OXIDATION_STATES:
            instance.__error("The element " + symbol + " is not treated by the class.\n"
                             "The class treats the d-block and f-block elements.")

        return cls.__OXIDATION_STATES[symbol]['ordinary']


    @classmethod
    def term_symbol_of(cls, two_S, two_L):
        """Return the term symbol of the given total spin and total orbital
        angular momentum, e.g. '6H' for 2S = 5 and 2L = 10.

        Arguments
        ---------
        two_S : int
            Twice the total spin.
        two_L : int
            Twice the total orbital angular momentum.
        """
        if two_L % 2 == 1:
            instance = cls.__new__(cls)
            instance.__error("The total orbital angular momentum of a term is an "
                             "integer,\nso its doubled value cannot be the odd number "
                             + str(two_L) + ".")

        orbital = two_L//2

        if orbital >= len(cls.__TERM_LETTERS):
            instance = cls.__new__(cls)
            instance.__error("There is no term letter for L = " + str(orbital) + ".")

        return "{0:d}{1:s}".format(two_S + 1, cls.__TERM_LETTERS[orbital])


    @classmethod
    def __momentum_string(cls, doubled_value):
        """Return the true angular momentum of a doubled value as a string,
        i.e. '15/2' for 15 and '4' for 8.
        """
        if doubled_value % 2 == 0:
            return str(doubled_value//2)

        return "{0:d}/2".format(doubled_value)


    def __resolve_ion(self, ion):
        """Parse the name of the ion and store the element, the atomic
        number and the oxidation state as attributes. The oxidation state
        that the name leaves out is taken as the ordinary state of the
        element.
        """
        symbol, oxidation_state = self.parse_ion_name(ion)

        if not symbol in self.__OXIDATION_STATES:
            self.__error("The element " + symbol + " is not treated by the class.\n"
                         "The class treats the ions of the d block (Sc-Zn, Y-Cd,\n"
                         "Hf-Hg) and of the f block (La-Lu, Ac-Lr), whose electrons\n"
                         "outside the noble-gas core belong to a single open shell.")

        self.element       = symbol
        self.atomic_number = self.__ELEMENT_SYMBOLS.index(symbol) + 1

        if oxidation_state is None:
            self.oxidation_state = self.__OXIDATION_STATES[symbol]['ordinary']
        else:
            self.oxidation_state = oxidation_state

        if self.oxidation_state <= 0:
            self.__error("The oxidation state of the ion must be positive, but "
                         + str(self.oxidation_state) + " was given.\n"
                         "A neutral atom of the d or f block carries electrons in its\n"
                         "outer s shell as well, so it has more than one open shell\n"
                         "and is not treated by the class.")

        self.ion = "{0:s}({1:s})".format(
            self.element,"I"*self.oxidation_state if self.oxidation_state <= 3
            else self.__roman_numeral(self.oxidation_state))


    @staticmethod
    def __roman_numeral(value):
        """Return the Roman numeral of a small positive integer, used to
        write the oxidation state of the canonical name of an ion.
        """
        numeral_list = ((10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I"))
        text         = ""

        for number, numeral in numeral_list:
            while value >= number:
                text  += numeral
                value -= number

        return text


    def __resolve_configuration(self):
        """Determine the open shell of the ion and its electron
        configuration and store them as attributes.

        The number of electrons of the open shell is the atomic number less
        the electrons of the noble-gas core and less the oxidation state,
        which is exact for every ion the class treats: the ion carries no
        electrons outside the core other than those of the open shell. The
        two ways in which that fails are the two errors raised here, i.e.
        an oxidation state so low that the outer s shell is still occupied
        and one so high that the core would have to be broken into.
        """
        block = None

        for candidate in self.__BLOCKS:
            if candidate['first_Z'] <= self.atomic_number <= candidate['last_Z']:
                block = candidate
                break

        if block is None:
            self.__error("The element " + self.element + " is not treated by the class.\n"
                         "The class treats the ions of the d block (Sc-Zn, Y-Cd,\n"
                         "Hf-Hg) and of the f block (La-Lu, Ac-Lr).")

        self.block = block['name']
        self.shell = block['shell']
        self.l     = block['l']

        n_shell_maximum  = 2*(2*self.l + 1)
        self.n_electrons = self.atomic_number - block['core_electrons'] \
                           - self.oxidation_state

        if self.oxidation_state < block['lowest_oxidation_state']:
            self.__error("The ion " + self.ion + " has more than one open shell and is "
                         "not treated\n"
                         "by the class: at this oxidation state the outer s shell of "
                         "the\n"
                         "atom is still occupied beside the " + self.shell + " shell. "
                         "The lowest oxidation\n"
                         "state of a " + self.block + " element that leaves a single "
                         "open shell is +"
                         + str(block['lowest_oxidation_state']) + ".")

        if self.n_electrons > n_shell_maximum:
            self.__error("The ion " + self.ion + " has more than one open shell and is "
                         "not treated\n"
                         "by the class: it carries " + str(self.n_electrons)
                         + " electrons outside the " + block['core_label'] + " core,\n"
                         "whereas the " + self.shell + " shell holds at most "
                         + str(n_shell_maximum) + " of them, so the rest\n"
                         "occupy a further shell.")

        if self.n_electrons < 0:
            self.__error("The oxidation state +" + str(self.oxidation_state) + " of "
                         + self.element + " would remove electrons from the\n"
                         + block['core_label'] + " core. The highest oxidation state "
                         "of " + self.element + " that the class\n"
                         "treats is +"
                         + str(self.atomic_number - block['core_electrons']) + ".")

        # An oxidation state that is not one of the chemically common ones
        # is accepted, but the assignment of the whole valence to the open
        # shell is then not certain, which is what the warning says.
        if not self.oxidation_state in self.__OXIDATION_STATES[self.element]['common']:
            self.__warning(
                "The oxidation state +" + str(self.oxidation_state) + " of "
                + self.element + " is not one of its chemically common\n"
                "states, which are "
                + ", ".join("+" + str(state) for state
                            in self.__OXIDATION_STATES[self.element]['common'])
                + ". The configuration of the ion is taken\n"
                "here as the single open shell " + self.__valence_configuration_string()
                + ", but a configuration with a further\n"
                "open shell may be the correct one for such a state; La(II), for\n"
                "instance, is 5d1 and not 4f1. Check the configuration before "
                "using it.")

        # The electrons of the ion, i.e. those of the neutral atom less the
        # ones the oxidation state has removed, and the electrons of the
        # valence, which are those of the open shell: the class treats only
        # the ions whose electrons outside the noble-gas core all belong to
        # that one shell.
        self.n_total_electrons   = self.atomic_number - self.oxidation_state
        self.n_valence_electrons = self.n_electrons

        self.core_configuration     = block['core_label']
        self.valence_configuration  = self.__valence_configuration_string()
        self.electron_configuration = self.__NOBLE_GAS_CONFIGURATIONS[block['core']]

        # The closed 4f shell of the core of the 5d block stands between
        # the core of xenon and the open shell.
        if self.block == '5d':
            self.electron_configuration += " 4f14"

        if self.n_electrons > 0:
            self.electron_configuration += " " + self.valence_configuration


    def __valence_configuration_string(self):
        """Return the configuration of the open shell as a string, e.g.
        '4f9'. An empty shell is stated as such, e.g. '4f0', so that the
        shell the ion belongs to is always visible.
        """
        return "{0:s}{1:d}".format(self.shell,self.n_electrons)


    def __resolve_ground_multiplet(self):
        """Determine the terms of the open shell and the quantum numbers of
        the ground multiplet and store them as attributes.

        The terms are those returned by the cfp_utils module of the Fortran
        extension for the configuration l^n. Hund's first two rules pick
        the ground term out of them as the term of the largest total spin
        and, among the terms of that spin, of the largest total orbital
        angular momentum; Hund's third rule gives the total angular
        momentum of the ground multiplet as |L - S| for a shell less than
        half full and as L + S for a shell at least half full.
        """
        from ouluspin._fortran import fortran_utils as fu

        two_l = 2*self.l

        self.n_terms = int(fu.cfp_utils.cfp_number_of_terms(two_l,self.n_electrons))

        two_S_array, two_L_array, seniority_array, w_array, g12_array = \
            fu.cfp_utils.cfp_terms(two_l,self.n_electrons,self.n_terms)

        self.term_data = []
        for i in range(0,self.n_terms):
            self.term_data.append(
                {'symbol':         self.term_symbol_of(int(two_S_array[i]),
                                                       int(two_L_array[i])),
                 'S':              int(two_S_array[i]),
                 'L':              int(two_L_array[i]),
                 'seniority':      int(seniority_array[i]),
                 'w':              int(w_array[i]),
                 'g2_casimir_x12': int(g12_array[i])})

        self.terms = [term['symbol'] for term in self.term_data]

        # The number of states of the configuration. A term of the total
        # spin S and the total orbital angular momentum L holds
        # (2S+1)(2L+1) states, and the terms of a configuration account for
        # all of its states, i.e. for the number of ways n electrons are
        # placed in the 2(2l+1) spin orbitals of the shell.
        self.n_states = sum((term['S'] + 1)*(term['L'] + 1)
                            for term in self.term_data)

        # The spin states of the configuration, i.e. the states of one spin
        # component of each term, which is the sum of (2L+1) over the
        # terms. They are the states a spin-free calculation carries, the
        # spin-orbit coupling turning each of them into the (2S+1) states
        # of its multiplicity.
        self.n_spin_states = sum(term['L'] + 1 for term in self.term_data)

        # Hund's first and second rules: the largest spin, and the largest
        # orbital angular momentum among the terms of that spin.
        ground = max(self.term_data, key=lambda term: (term['S'],term['L']))

        self.S = ground['S']
        self.L = ground['L']

        # Hund's third rule. The shell is half full at 2l + 1 electrons,
        # where the two branches agree, since the ground term of the
        # half-filled shell has L = 0.
        if self.n_electrons < 2*self.l + 1:
            self.J = abs(self.L - self.S)
        else:
            self.J = self.L + self.S

        self.term_symbol             = ground['symbol']
        self.ground_multiplet_symbol = self.term_symbol \
                                       + self.__momentum_string(self.J)
        self.degeneracy              = self.J + 1

        # The Lande g-factor of the ground multiplet, in three forms: the
        # accurate one, evaluated with the g-factor of the free electron,
        # the simple one of the textbooks, evaluated with g_e = 2, and the
        # exact fraction of the latter. The factor is not defined for a
        # multiplet of J = 0, i.e. for an empty or a closed shell and for
        # the ground multiplet of a shell one electron short of half full;
        # zero is stored there, and the quantities that would use it, such
        # as the Curie susceptibility, vanish in any case.
        self.lande_g_factor        = self.__lande_g_factor(
            self.L,self.S,self.J,self.electron_g_factor())
        self.lande_g_factor_simple = self.__lande_g_factor(
            self.L,self.S,self.J,self.SIMPLE_ELECTRON_G_FACTOR)
        self.lande_g_factor_str    = self.__lande_g_factor_fraction(
            self.L,self.S,self.J)


    @classmethod
    def __lande_g_factor(cls, two_L, two_S, two_J, electron_g_factor):
        """Return the Lande g-factor of the multiplet of the given angular
        momenta, all in the doubled form:

            g_J = 1 + (g_e - 1) [J(J+1) + S(S+1) - L(L+1)] / [2 J(J+1)],

        where g_e is the g-factor of the free electron. The orbital
        g-factor is one, so g_e is the only one that enters. Setting
        g_e = 2 gives the familiar form of the textbooks,

            g_J = 1 + [J(J+1) + S(S+1) - L(L+1)] / [2 J(J+1)].

        Zero is returned for J = 0, where the factor is not defined.

        Arguments
        ---------
        two_L, two_S, two_J : int
            The angular momenta of the multiplet in the doubled form.
        electron_g_factor : float
            The g-factor of the free electron.
        """
        if two_J == 0:
            return 0.0

        L = two_L/2.0
        S = two_S/2.0
        J = two_J/2.0

        return 1.0 + (electron_g_factor - 1.0) \
               * (J*(J + 1.0) + S*(S + 1.0) - L*(L + 1.0))/(2.0*J*(J + 1.0))


    @staticmethod
    def __lande_g_factor_fraction(two_L, two_S, two_J):
        """Return the Lande g-factor of g_e = 2 as an exact fraction in the
        form of a string, e.g. '4/3' for the ground multiplet of Dy(III)
        and '2' for that of Gd(III).

        With g_e = 2 the factor is a rational number, since the angular
        momenta are integers or half-integers, and it is evaluated here in
        exact arithmetic so that the fraction is the exact one and not the
        result of rounding a float. '0' is returned for J = 0, where the
        factor is not defined.
        """
        from fractions import Fraction

        if two_J == 0:
            return "0"

        L = Fraction(two_L,2)
        S = Fraction(two_S,2)
        J = Fraction(two_J,2)

        g = 1 + (J*(J + 1) + S*(S + 1) - L*(L + 1))/(2*J*(J + 1))

        if g.denominator == 1:
            return str(g.numerator)

        return "{0:d}/{1:d}".format(g.numerator,g.denominator)


    def __init__(self, ion,
                 print_output=False):
        """Upon class initiation parse the name of the ion, determine its
        electron configuration and evaluate the terms of its open shell and
        the quantum numbers of its ground multiplet.
        """
        self.print_output = print_output

        # The coefficients of fractional parentage are not evaluated here:
        # they are the one quantity of the class whose evaluation costs
        # something, and they are wanted far less often than the rest.
        self.__cfp_data = None

        self.__resolve_ion(ion)
        self.__resolve_configuration()
        self.__resolve_ground_multiplet()

        if self.print_output:
            print(self.data_table().string_table())


    @classmethod
    def electron_g_factor(cls):
        """Return the g-factor of the free electron, i.e. the CODATA value
        carried by the constants of scipy, which is the one the accurate
        Lande g-factor is evaluated with. The value is read once and stored
        in the ELECTRON_G_FACTOR class attribute.

        The constant of scipy is negative, as the g-factor of the electron
        is by the sign convention of CODATA; the magnitude is what enters
        the Lande formula.
        """
        if cls.ELECTRON_G_FACTOR is None:
            import scipy.constants as const

            cls.ELECTRON_G_FACTOR = abs(
                const.physical_constants['electron g factor'][0])

        return cls.ELECTRON_G_FACTOR


    def curie_susceptibility(self, simple_g_factor=False):
        """Return the chi*T product of the free ion given by the Curie law
        for its ground multiplet, in cm^3 K mol^-1, i.e.

            chi*T = N_A mu_B^2 g_J^2 J(J+1) / (3 k_B).

        The value is the high-temperature limit of the chi*T product of the
        ion, i.e. the value the product approaches once the whole ground
        multiplet is populated but the excited multiplets are not. It is
        the number the measured chi*T of a compound of the ion is
        ordinarily compared with; for Dy(III) it is 14.19 cm^3 K mol^-1
        with the accurate Lande g-factor and the 14.17 of the tables of the
        textbooks with the simple one.

        The value vanishes for an ion whose ground multiplet has J = 0,
        i.e. for a closed or an empty shell and for e.g. Eu(III) and
        Sm(II).

        Optional arguments
        ------------------
        simple_g_factor : boolean
            Whether the simple Lande g-factor of g_e = 2, i.e. the
            lande_g_factor_simple attribute, is used instead of the
            accurate one. Default is False, i.e. the accurate factor of the
            lande_g_factor attribute. The simple factor reproduces the
            values tabulated in the textbooks of magnetochemistry, which
            are evaluated in the same approximation.
        """
        import scipy.constants as const

        if type(self).__CURIE_CONSTANT is None:
            # The constants are converted to the cgs system, in which the
            # molar susceptibility of the library is reported: the Bohr
            # magneton in erg per gauss and the Boltzmann constant in erg
            # per kelvin.
            mu_B = const.physical_constants['Bohr magneton'][0]*1.0e3
            k_B  = const.k*1.0e7

            type(self).__CURIE_CONSTANT = const.N_A*mu_B**2/(3.0*k_B)

        J = self.J/2.0

        if simple_g_factor:
            g_factor = self.lande_g_factor_simple
        else:
            g_factor = self.lande_g_factor

        return type(self).__CURIE_CONSTANT * g_factor**2 * J*(J + 1.0)


    def coefficients_of_fractional_parentage(self):
        """Return the coefficients of fractional parentage of the open
        shell of the ion, i.e. the coefficients

            ( l^(n-1) (alpha1 S1 L1) l ; S L {| l^n alpha S L )

        connecting the terms of the configuration l^n of the ion to those
        of the configuration l^(n-1) of the ion with one electron less.

        The coefficients are evaluated on the first call and stored
        afterwards, since evaluating them costs considerably more than the
        rest of the data of the class. The whole shell is built at once by
        the Fortran routine, so the first call is the expensive one for
        every ion of the same shell.

        The return value is a dictionary with the items

            'matrix'        the coefficients as an array of float64 of the
                            shape (number of terms of l^n, number of terms
                            of l^(n-1)),
            'terms'         the terms of l^n, in the order of the rows of
                            the matrix, as the dictionaries of the
                            term_data attribute,
            'parent_terms'  the terms of l^(n-1), in the order of the
                            columns.

        An ion of an empty shell has no parent configuration; the matrix is
        then empty and the parent terms are an empty list.

        The sign conventions of the coefficients are those of the cfp_utils
        module; see its documentation, and the entry on the sign
        conventions in to-do.txt, for how they relate to the tables of
        Nielson and Koster.
        """
        if self.__cfp_data is not None:
            return self.__cfp_data

        from ouluspin._fortran import fortran_utils as fu

        two_l = 2*self.l

        if self.n_electrons == 0:
            self.__cfp_data = {'matrix':       np.zeros((0,0), dtype=np.float64),
                               'terms':        list(self.term_data),
                               'parent_terms': []}
            return self.__cfp_data

        n_parent_terms = int(fu.cfp_utils.cfp_number_of_terms(two_l,
                                                              self.n_electrons - 1))

        two_S_array, two_L_array, seniority_array, w_array, g12_array = \
            fu.cfp_utils.cfp_terms(two_l,self.n_electrons - 1,n_parent_terms)

        parent_term_list = []
        for i in range(0,n_parent_terms):
            parent_term_list.append(
                {'symbol':         self.term_symbol_of(int(two_S_array[i]),
                                                       int(two_L_array[i])),
                 'S':              int(two_S_array[i]),
                 'L':              int(two_L_array[i]),
                 'seniority':      int(seniority_array[i]),
                 'w':              int(w_array[i]),
                 'g2_casimir_x12': int(g12_array[i])})

        matrix = fu.cfp_utils.cfp_full_table(two_l,self.n_electrons,
                                             self.n_terms,n_parent_terms)

        self.__cfp_data = {'matrix':       np.array(matrix, dtype=np.float64),
                           'terms':        list(self.term_data),
                           'parent_terms': parent_term_list}

        return self.__cfp_data


    def states_by_multiplicity(self):
        """Return the terms and the states of the open shell grouped by
        spin multiplicity, as a list of dictionaries with the items
        'multiplicity' (2S + 1), 'S' (the total spin in the DOUBLED form),
        'n_terms', 'n_spin_states' and 'n_states', ordered by decreasing
        multiplicity.

        A term of the total spin S and the total orbital angular momentum L
        holds (2S+1)(2L+1) states, so the states of a multiplicity are the
        sum of that over the terms carrying it. The states of all the
        multiplicities together are the states of the configuration, i.e.
        the n_states attribute.

        The SPIN STATES of a multiplicity are the states of one spin
        component alone, i.e. the states divided by the multiplicity, which
        is the sum of (2L+1) over the terms carrying it: a 6H term holds
        6*11 = 66 states and 11 spin states. They are the states a spin-free
        calculation of that multiplicity carries, i.e. the number of roots
        that has to be asked for, whereas the states themselves are what a
        calculation including the spin-orbit coupling carries.
        """
        group_list = []

        for multiplicity in sorted({term['S'] + 1 for term in self.term_data},
                                   reverse=True):
            term_list = [term for term in self.term_data
                         if term['S'] + 1 == multiplicity]

            spin_states = sum(term['L'] + 1 for term in term_list)

            group_list.append(
                {'multiplicity':  multiplicity,
                 'S':             multiplicity - 1,
                 'n_terms':       len(term_list),
                 'n_spin_states': spin_states,
                 'n_states':      multiplicity*spin_states})

        return group_list


    def __term_label(self, term):
        """Return the label of a term as it is printed in the tables, i.e.
        the term symbol followed by the seniority when the configuration
        holds more than one term of that symbol, e.g. '2F' or '2F (v = 5)'.
        """
        if len([other for other in self.term_data
                if other['symbol'] == term['symbol']]) == 1:
            return term['symbol']

        return "{0:s} (v = {1:d})".format(term['symbol'],term['seniority'])


    def __ground_multiplet_rows(self):
        """Return the rows of the table of the ground multiplet, i.e. the
        list of the (quantity, value) pairs that both table methods print.
        """
        return [
            ["Element",                    self.element],
            ["Atomic number",              str(self.atomic_number)],
            ["Number of electrons",        str(self.n_total_electrons)],
            ["Oxidation state",            "+{0:d}".format(self.oxidation_state)],
            ["Block",                      self.block],
            ["Electron configuration",     self.electron_configuration],
            ["Valence configuration",      self.core_configuration + " "
                                           + self.valence_configuration],
            ["Number of valence electrons",str(self.n_valence_electrons)],
            ["Number of terms",            str(self.n_terms)],
            None,
            ["Ground term",                self.term_symbol],
            ["Ground multiplet",           self.ground_multiplet_symbol],
            ["Total spin S",               self.__momentum_string(self.S)],
            ["Total orbital momentum L",   self.__momentum_string(self.L)],
            ["Total angular momentum J",   self.__momentum_string(self.J)],
            ["Degeneracy 2J + 1",          str(self.degeneracy)],
            ["Lande g-factor",             "{0:.6f}".format(self.lande_g_factor)],
            ["Lande g-factor, g_e = 2",     "{0:.6f}".format(self.lande_g_factor_simple)],
            ["Lande g-factor as a fraction", self.lande_g_factor_str],
            ["Curie chiT / cm^3 K mol^-1",  "{0:.4f}".format(self.curie_susceptibility())],
            ["Curie chiT, g_e = 2",         "{0:.4f}".format(
                self.curie_susceptibility(simple_g_factor=True))],
        ]


    def ground_multiplet_table(self):
        """Return the properties of the ground multiplet of the ion as an
        instance of ResultTable, i.e. the electron configuration and the
        quantum numbers of the ground multiplet without the listing of the
        terms of the open shell. The full data, including the terms, are
        given by the data_table method.

        The angular momenta are printed as the true angular momenta, i.e.
        as J = 15/2 for the Dy(III) whose J attribute is 15.
        """
        from ouluspin import result_table

        return result_table.ResultTable(
            self.__ground_multiplet_rows(),
            column_headers=["Quantity","Value"],
            title="ION DATA OF " + self.ion.upper(),
            notes=self.__table_notes(),
            alignments=['l','l'],
            table_type='ion_data')


    def __table_notes(self, spin_states=False):
        """Return the explanatory texts of the tables of the ion, which
        state where the values come from. The note defining the spin states
        is added only for the table that counts them.
        """
        note_list = ["The ground multiplet follows Hund's rules and the terms are\n"
                     "those of the configuration " + self.valence_configuration
                     + ", evaluated from the coefficients of\n"
                     "fractional parentage of the shell.",
                     "The Lande g-factor and the chiT product are given both with the\n"
                     "g-factor of the free electron ({0:.9f}) and in the\n"
                     "approximation g_e = 2 of the textbooks, which is the one the\n"
                     "tabulated values of the literature are evaluated in. The\n"
                     "g-factor of that approximation is a rational number and is\n"
                     "given as the exact fraction it is."
                     .format(self.electron_g_factor())]

        if spin_states:
            note_list.append(
                "The spin states of a multiplicity are the states of one spin\n"
                "component alone, i.e. the states divided by the multiplicity:\n"
                "a 6H term holds 6*11 = 66 states and 11 spin states.")

        return note_list


    def term_table(self):
        """Return the terms of the open shell of the ion as an instance of
        ResultTable, one term per row with its term symbol, its total spin
        and orbital angular momentum, its seniority and, for an f shell,
        the label of its G2 representation.

        The terms are given in the canonical order of the cfp_utils module
        of the Fortran extension, which is the order the coefficients of
        fractional parentage are indexed in.
        """
        from ouluspin import result_table

        f_shell = (self.l == 3)

        row_list = []
        for i in range(0,self.n_terms):
            term = self.term_data[i]

            row = [str(i + 1),
                   term['symbol'],
                   self.__momentum_string(term['S']),
                   self.__momentum_string(term['L']),
                   str(term['seniority']),
                   str(term['w'])]

            if f_shell:
                row.append(str(term['g2_casimir_x12']))

            row_list.append(row)

        column_headers = ["Index","Term","S","L","v","w"]
        alignments     = ['r','l','r','r','r','r']

        if f_shell:
            column_headers.append("12g(G2)")
            alignments.append('r')

        return result_table.ResultTable(
            row_list,
            column_headers=column_headers,
            title="TERMS OF THE CONFIGURATION " + self.valence_configuration,
            notes=["The terms are listed in the canonical order of the library.\n"
                   "v is the seniority and w the ordinal among the terms of the\n"
                   "same S and L."
                   + ("\n12g(G2) is twelve times the Casimir eigenvalue of the G2\n"
                      "group, which labels the G2 representation of the term."
                      if f_shell else "")],
            alignments=alignments,
            table_type='ion_data')


    def data_table(self):
        """Return the full data of the ion as an instance of ResultTable,
        i.e. the properties of the ground multiplet followed by the listing
        of the terms of the open shell.

        The terms are printed as a section of the same table, one line
        holding as many terms as fit, so that the table stays readable for
        a configuration such as f7, which has 119 terms. The terms with
        their complete labels are tabulated one per row by the term_table
        method.
        """
        from ouluspin import result_table

        row_list = list(self.__ground_multiplet_rows())

        # The terms of the shell, gathered onto as few lines as is
        # readable. A term symbol carried by several terms of different
        # seniority is stated once with the number of the terms.
        row_list.append(None)
        row_list.append("Terms of the configuration " + self.valence_configuration)

        symbol_list = []
        for term in self.term_data:
            if len(symbol_list) > 0 and symbol_list[-1][0] == term['symbol']:
                symbol_list[-1][1] += 1
            else:
                symbol_list.append([term['symbol'],1])

        text_list = []
        for symbol, count in symbol_list:
            if count == 1:
                text_list.append(symbol)
            else:
                text_list.append("{0:s} (x{1:d})".format(symbol,count))

        # The terms are wrapped by hand, since the cell of a table is not
        # wrapped by the renderers.
        line_list   = []
        line        = ""
        line_length = 52

        for text in text_list:
            if line == "":
                line = text
            elif len(line) + 2 + len(text) <= line_length:
                line += ", " + text
            else:
                # The comma is left at the end of the line, so that a
                # broken listing reads as one listing.
                line_list.append(line + ",")
                line = text

        if not line == "":
            line_list.append(line)

        for i in range(0,len(line_list)):
            if i == 0:
                row_list.append(["Terms",line_list[i]])
            else:
                row_list.append(["",line_list[i]])

        # The terms and the states of each spin multiplicity, and the whole
        # configuration as their sum. The states of a multiplicity are the
        # states that a magnetic measurement of the free ion sees together
        # when the spin-orbit coupling is small against the separation of
        # the multiplicities.
        group_list = self.states_by_multiplicity()

        row_list.append(None)
        row_list.append("States of the configuration " + self.valence_configuration)

        for group in group_list:
            row_list.append(
                ["Multiplicity {0:d} (S = {1:s})".format(
                    group['multiplicity'],self.__momentum_string(group['S'])),
                 "{0:5d} terms, {1:6d} spin states, {2:6d} states".format(
                     group['n_terms'],group['n_spin_states'],
                     group['n_states'])])

        row_list.append(None)
        row_list.append(["Total",
                         "{0:5d} terms, {1:6d} spin states, {2:6d} states".format(
                             self.n_terms,self.n_spin_states,self.n_states)])

        return result_table.ResultTable(
            row_list,
            column_headers=["Quantity","Value"],
            title="ION DATA OF " + self.ion.upper(),
            notes=self.__table_notes(spin_states=True),
            footnotes=None,
            summary=["Number of terms of " + self.valence_configuration + ": "
                     + str(self.n_terms) + " ("
                     + str(len(symbol_list)) + " distinct term symbols)",
                     "Number of states of " + self.valence_configuration + ": "
                     + str(self.n_states) + " (" + str(self.n_spin_states)
                     + " spin states)"],
            alignments=['l','l'],
            table_type='ion_data')


    def cfp_table(self, threshold=1.0e-10):
        """Return the coefficients of fractional parentage of the open
        shell of the ion as an instance of ResultTable, one coefficient per
        row: the term of the configuration l^n, the parent term of
        l^(n-1) and the value of the coefficient.

        Only the coefficients whose magnitude exceeds the threshold are
        tabulated, since most of the coefficients of a large configuration
        vanish. The table is nevertheless a long one for a shell of many
        terms: the f7 configuration of Gd(III), for instance, has 119 terms
        and 119 parent terms.

        The coefficients are evaluated by the
        coefficients_of_fractional_parentage method, i.e. on the first call
        and not upon the construction of the instance.

        Optional arguments
        ------------------
        threshold : float
            The magnitude below which a coefficient is left out of the
            table. Default is 1.0e-10, which drops the coefficients that
            vanish by the coupling rules.
        """
        from ouluspin import result_table

        data = self.coefficients_of_fractional_parentage()

        if self.n_electrons == 0:
            return result_table.ResultTable(
                [],
                column_headers=["Term","Parent term","CFP"],
                title="COEFFICIENTS OF FRACTIONAL PARENTAGE OF "
                      + self.valence_configuration,
                notes=["The shell is empty, so it has no parent configuration."],
                table_type='ion_data')

        matrix       = data['matrix']
        term_list    = data['terms']
        parent_list  = data['parent_terms']

        parent_label = []
        for parent in parent_list:
            if len([other for other in parent_list
                    if other['symbol'] == parent['symbol']]) == 1:
                parent_label.append(parent['symbol'])
            else:
                parent_label.append("{0:s} (v = {1:d})".format(parent['symbol'],
                                                               parent['seniority']))

        row_list = []
        for i in range(0,len(term_list)):
            for j in range(0,len(parent_list)):
                if abs(matrix[i][j]) <= threshold:
                    continue

                row_list.append([self.__term_label(term_list[i]),
                                 parent_label[j],
                                 float(matrix[i][j])])

        return result_table.ResultTable(
            row_list,
            column_headers=["Term of " + self.valence_configuration,
                            "Parent term of " + self.shell
                            + str(self.n_electrons - 1),
                            "CFP"],
            title="COEFFICIENTS OF FRACTIONAL PARENTAGE OF "
                  + self.valence_configuration,
            notes=["The coefficients ( l^(n-1) (S1 L1) l ; S L {| l^n S L ) of the\n"
                   "shell, with the terms labelled by their seniority v where the\n"
                   "term symbol alone does not identify them.",
                   "Coefficients of a magnitude below {0:.1e} are not listed."
                   .format(threshold)],
            summary=["Number of listed coefficients: " + str(len(row_list))],
            formats=[None,None,'.8f'],
            alignments=['l','l','r'],
            table_type='ion_data')


    def __repr__(self):
        """Return the data of the ion as the plain-text rendering of the
        table given by the data_table method.
        """
        return self.data_table().string_table()


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class and run a set of tests on the class
        constructor and the class methods. Return True if all tests passed
        and False otherwise.

        The ground terms and multiplets of the tests are the literature
        values of the free ions, which are the standard Hund's-rule ground
        states tabulated in every textbook of magnetochemistry.

        Optional arguments
        ------------------
        print_output : boolean
            Whether to print the outcome of each test. Default is True.
        """
        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('IonData',test_name,
                                                       condition,print_output))

        # ------------------------------------------------------------------
        # The parsing of the name of the ion.
        # ------------------------------------------------------------------
        check('a Roman numeral in parentheses is parsed',
              cls.parse_ion_name('Dy(III)') == ('Dy',3))
        check('the parsing is case-insensitive',
              cls.parse_ion_name('dy(iii)') == ('Dy',3))
        check('a Roman numeral separated by a space is parsed',
              cls.parse_ion_name('dy iii') == ('Dy',3))
        check('a Roman numeral separated by a hyphen is parsed',
              cls.parse_ion_name('Dy-III') == ('Dy',3))
        check('an Arabic numeral without a separator is parsed',
              cls.parse_ion_name('Dy3') == ('Dy',3))
        check('an Arabic numeral with a space is parsed',
              cls.parse_ion_name('Dy 3') == ('Dy',3))
        check('a trailing plus sign is accepted',
              cls.parse_ion_name('Dy3+') == ('Dy',3))
        check('a leading plus sign is accepted',
              cls.parse_ion_name('Dy+3') == ('Dy',3))
        check('an Arabic numeral in parentheses is parsed',
              cls.parse_ion_name('Dy(3)') == ('Dy',3))
        check('a bare chemical symbol leaves the oxidation state open',
              cls.parse_ion_name('Dy') == ('Dy',None))
        check('a one-letter symbol is parsed',
              cls.parse_ion_name('V(III)') == ('V',3))
        check('a higher Roman numeral is parsed',
              cls.parse_ion_name('U vi') == ('U',6))
        check('surrounding whitespace is ignored',
              cls.parse_ion_name('  Fe(II)  ') == ('Fe',2))

        # The ordinary oxidation state of the element is used when the name
        # states none.
        check('the ordinary oxidation state of a lanthanide is +3',
              cls.default_oxidation_state('Dy') == 3)
        check('the ordinary oxidation state is used when none is given',
              cls('Dy').oxidation_state == 3)
        check('the ordinary oxidation state of iron is +3',
              cls('Fe').oxidation_state == 3)
        check('the common oxidation states are listed',
              cls.common_oxidation_states('Eu') == [2,3])
        check('the treated elements are listed',
              ('Dy' in cls.supported_elements())
              and ('Fe' in cls.supported_elements())
              and (not 'Si' in cls.supported_elements()))

        # ------------------------------------------------------------------
        # The electron configurations.
        # ------------------------------------------------------------------
        dy = cls('Dy(III)')

        check('the canonical name of the ion is stored',
              dy.ion == 'Dy(III)')
        check('the element and the atomic number are stored',
              (dy.element == 'Dy') and (dy.atomic_number == 66))
        check('the block of the element is recognized',
              dy.block == '4f')
        check('the valence configuration of Dy(III) is 4f9',
              dy.valence_configuration == '4f9')
        check('the core of a lanthanide is xenon',
              dy.core_configuration == '[Xe]')
        check('the full configuration ends in the open shell',
              dy.electron_configuration.endswith('5s2 5p6 4f9'))
        check('the full configuration holds the right number of electrons',
              sum(int(part[2:]) for part
                  in dy.electron_configuration.split()) == 66 - 3)

        # The configurations of the four blocks.
        configuration_list = [('Ce(III)','4f1'), ('Ce(IV)','4f0'),
                              ('Gd(III)','4f7'), ('Eu(II)','4f7'),
                              ('Yb(II)','4f14'), ('Lu(III)','4f14'),
                              ('Fe(III)','3d5'), ('Fe(II)','3d6'),
                              ('Cu(II)','3d9'),  ('Zn(II)','3d10'),
                              ('Ti(IV)','3d0'),  ('Mn(II)','3d5'),
                              ('Ru(III)','4d5'), ('Ag(I)','4d10'),
                              ('Os(IV)','5d4'),  ('Ir(IV)','5d5'),
                              ('Pt(II)','5d8'),  ('Hg(II)','5d10'),
                              ('U(III)','5f3'),  ('U(IV)','5f2'),
                              ('U(VI)','5f0'),   ('Np(V)','5f2'),
                              ('Pu(IV)','5f4'),  ('Am(III)','5f6'),
                              ('Cm(III)','5f7'), ('Th(IV)','5f0')]

        configuration_ok = True
        for name, configuration in configuration_list:
            if not cls(name).valence_configuration == configuration:
                configuration_ok = False
        check('the valence configurations of the four blocks are correct',
              configuration_ok)

        # ------------------------------------------------------------------
        # The ground multiplets, against the literature values of the free
        # ions. The lanthanide series is the standard table of
        # magnetochemistry.
        # ------------------------------------------------------------------
        lanthanide_list = [('La(III)','1S0',   0.0),
                           ('Ce(III)','2F5/2', 6.0/7.0),
                           ('Pr(III)','3H4',   4.0/5.0),
                           ('Nd(III)','4I9/2', 8.0/11.0),
                           ('Pm(III)','5I4',   3.0/5.0),
                           ('Sm(III)','6H5/2', 2.0/7.0),
                           ('Eu(III)','7F0',   0.0),
                           ('Gd(III)','8S7/2', 2.0),
                           ('Tb(III)','7F6',   3.0/2.0),
                           ('Dy(III)','6H15/2',4.0/3.0),
                           ('Ho(III)','5I8',   5.0/4.0),
                           ('Er(III)','4I15/2',6.0/5.0),
                           ('Tm(III)','3H6',   7.0/6.0),
                           ('Yb(III)','2F7/2', 8.0/7.0),
                           ('Lu(III)','1S0',   0.0)]

        multiplet_ok = True
        g_factor_ok  = True
        for name, symbol, g_factor in lanthanide_list:
            ion = cls(name)

            if not ion.ground_multiplet_symbol == symbol:
                multiplet_ok = False
            # The tabulated factors are those of the textbooks, i.e. the
            # ones of the approximation g_e = 2, and they are exact
            # fractions in that approximation.
            if abs(ion.lande_g_factor_simple - g_factor) > 1.0e-12:
                g_factor_ok = False

        check('the ground multiplets of the trivalent lanthanides are correct',
              multiplet_ok)
        check('the simple Lande g-factors of the trivalent lanthanides are correct',
              g_factor_ok)

        # The three forms of the Lande g-factor. The accurate one is
        # evaluated with the g-factor of the free electron, i.e. it differs
        # from the simple one by a term proportional to g_e - 2, and the
        # string is the exact fraction of the simple one.
        g_e = cls.electron_g_factor()

        check('the free-electron g-factor is the CODATA value',
              abs(g_e - 2.00231930436256) < 1.0e-12)
        check('the accurate g-factor uses the free-electron g-factor',
              abs(dy.lande_g_factor - (1.0 + (g_e - 1.0)/3.0)) < 1.0e-12)
        check('the simple g-factor uses g_e = 2',
              dy.lande_g_factor_simple == 4.0/3.0)
        check('the two g-factors differ by a term proportional to g_e - 2',
              abs((dy.lande_g_factor - dy.lande_g_factor_simple)
                  - (g_e - 2.0)/3.0) < 1.0e-15)
        check('the accurate g-factor of a spin-only ion is the electron one',
              abs(cls('Gd(III)').lande_g_factor - g_e) < 1.0e-12)
        check('the simple g-factor of a spin-only ion is exactly two',
              cls('Gd(III)').lande_g_factor_simple == 2.0)

        # The exact fraction of the simple g-factor.
        fraction_list = [('Ce(III)','6/7'),  ('Pr(III)','4/5'),
                         ('Nd(III)','8/11'), ('Sm(III)','2/7'),
                         ('Gd(III)','2'),    ('Tb(III)','3/2'),
                         ('Dy(III)','4/3'),  ('Ho(III)','5/4'),
                         ('Er(III)','6/5'),  ('Tm(III)','7/6'),
                         ('Yb(III)','8/7'),  ('Eu(III)','0')]

        fraction_ok = True
        for name, fraction in fraction_list:
            ion = cls(name)

            if not ion.lande_g_factor_str == fraction:
                fraction_ok = False

            # The fraction must be the simple factor itself.
            if not fraction == '0':
                numerator, _, denominator = fraction.partition('/')
                value = float(numerator)/float(denominator or 1)

                if abs(value - ion.lande_g_factor_simple) > 1.0e-12:
                    fraction_ok = False

        check('the Lande g-factor is given as an exact fraction',fraction_ok)

        # The ground multiplets of the other blocks and of the other
        # oxidation states.
        multiplet_list = [('Ce(IV)','1S0'),   ('Eu(II)','8S7/2'),
                          ('Sm(II)','7F0'),   ('Yb(II)','1S0'),
                          ('Tb(IV)','8S7/2'),
                          # Tm(II) is 4f13, i.e. isoelectronic with Yb(III).
                          ('Tm(II)','2F7/2'),
                          ('Ti(III)','2D3/2'),('V(III)','3F2'),
                          ('Cr(III)','4F3/2'),('Mn(II)','6S5/2'),
                          ('Fe(III)','6S5/2'),('Fe(II)','5D4'),
                          ('Co(II)','4F9/2'), ('Ni(II)','3F4'),
                          ('Cu(II)','2D5/2'), ('Zn(II)','1S0'),
                          ('U(III)','4I9/2'), ('U(IV)','3H4'),
                          ('Np(IV)','4I9/2'), ('Pu(III)','6H5/2'),
                          ('Am(III)','7F0'),  ('Cm(III)','8S7/2')]

        other_multiplet_ok = True
        for name, symbol in multiplet_list:
            if not cls(name).ground_multiplet_symbol == symbol:
                other_multiplet_ok = False
        check('the ground multiplets of the other ions are correct',
              other_multiplet_ok)

        # The angular momenta are stored in the doubled form and printed as
        # the true ones.
        check('the angular momenta are stored in the doubled form',
              (dy.J == 15) and (dy.L == 10) and (dy.S == 5))
        check('the degeneracy of the ground multiplet is 2J + 1',
              dy.degeneracy == 16)
        check('the ground term symbol is stored',
              dy.term_symbol == '6H')

        # Hund's third rule: the total angular momentum is |L - S| below
        # the half-filled shell and L + S above it.
        check("Hund's third rule below the half-filled shell",
              cls('Pr(III)').J == abs(cls('Pr(III)').L - cls('Pr(III)').S))
        check("Hund's third rule above the half-filled shell",
              cls('Dy(III)').J == cls('Dy(III)').L + cls('Dy(III)').S)
        check('the half-filled shell has L = 0 and J = S',
              (cls('Gd(III)').L == 0) and (cls('Gd(III)').J == cls('Gd(III)').S))

        # ------------------------------------------------------------------
        # The terms of the open shell.
        # ------------------------------------------------------------------
        check('the number of terms of f9 is that of f5',
              cls('Dy(III)').n_terms == cls('Sm(III)').n_terms)
        check('the number of terms of d5 is 16',
              cls('Fe(III)').n_terms == 16)
        check('the number of terms of f7 is 119',
              cls('Gd(III)').n_terms == 119)
        check('a closed shell has a single term',
              (cls('Lu(III)').n_terms == 1)
              and (cls('Lu(III)').terms == ['1S']))
        check('an empty shell has a single term',
              (cls('La(III)').n_terms == 1)
              and (cls('La(III)').terms == ['1S']))
        check('the ground term is among the terms of the shell',
              dy.term_symbol in dy.terms)
        check('the terms carry their full labels',
              all(('seniority' in term) and ('S' in term) and ('L' in term)
                  for term in dy.term_data))

        # The terms of d2, which are the textbook example: 1S, 1D, 1G, 3P
        # and 3F.
        check('the terms of d2 are the textbook ones',
              sorted(cls('Ti(II)').terms) == sorted(['1S','1D','1G','3P','3F']))

        # The terms of f2: 1S, 1D, 1G, 1I, 3P, 3F, 3H.
        check('the terms of f2 are the textbook ones',
              sorted(cls('Pr(III)').terms)
              == sorted(['1S','1D','1G','1I','3P','3F','3H']))

        # The number of states of a configuration is the binomial
        # coefficient, i.e. the number of ways the electrons are placed in
        # the spin orbitals of the shell, and the terms must account for
        # all of them.
        import math

        state_count_ok = True
        for name in ('Fe(III)','Dy(III)','Gd(III)','Pr(III)','Ni(II)',
                     'Lu(III)','La(III)','U(III)'):
            ion = cls(name)
            if not ion.n_states == math.comb(2*(2*ion.l + 1),ion.n_electrons):
                state_count_ok = False
        check('the terms account for all the states of the configuration',
              state_count_ok)

        check('the number of states of f9 is 2002',
              cls('Dy(III)').n_states == 2002)
        check('a closed shell has a single state',
              cls('Lu(III)').n_states == 1)

        # The spin states, i.e. the states of one spin component of each
        # term. Every term contributes 2L+1 of them, so the states of a
        # term are its multiplicity times its spin states: the 6H term of
        # f9 holds 11 spin states and 6*11 = 66 states.
        spin_state_ok = True
        for name in ('Fe(III)','Dy(III)','Gd(III)','Pr(III)','Ni(II)',
                     'Lu(III)','U(III)'):
            ion = cls(name)

            if not ion.n_spin_states \
                   == sum(term['L'] + 1 for term in ion.term_data):
                spin_state_ok = False

            # The states of a multiplicity are its multiplicity times its
            # spin states, which is what the division the table performs
            # rests on.
            for group in ion.states_by_multiplicity():
                if not group['n_states'] \
                       == group['multiplicity']*group['n_spin_states']:
                    spin_state_ok = False

            if not sum(group['n_spin_states'] for group
                       in ion.states_by_multiplicity()) == ion.n_spin_states:
                spin_state_ok = False

        check('the spin states are the states of one spin component',
              spin_state_ok)

        # The sextets of f9 are 6P, 6F and 6H, i.e. L = 1, 3 and 5, which
        # give 3 + 7 + 11 = 21 spin states and 6*21 = 126 states.
        check('the spin states of the sextets of f9 are counted',
              (cls('Dy(III)').states_by_multiplicity()[0]['n_spin_states'] == 21)
              and (cls('Dy(III)').n_spin_states == 735))

        # The 8S term of the half-filled f shell has L = 0, so it holds a
        # single spin state and eight states.
        check('an S term holds a single spin state',
              (cls('Gd(III)').states_by_multiplicity()[0]['n_spin_states'] == 1)
              and (cls('Gd(III)').states_by_multiplicity()[0]['n_states'] == 8))

        # The electrons of the ion and of its valence.
        check('the number of electrons is the atomic number less the charge',
              (dy.n_total_electrons == 63) and (cls('Fe(II)').n_total_electrons == 24))
        check('the valence electrons are those of the open shell',
              (dy.n_valence_electrons == 9)
              and (dy.n_valence_electrons == dy.n_electrons))
        check('a closed-shell ion carries a full valence',
              cls('Lu(III)').n_valence_electrons == 14)

        # The states grouped by spin multiplicity. The groups must exhaust
        # the terms and the states of the configuration, and the
        # multiplicities must be those of the terms.
        grouping_ok = True
        for name in ('Fe(III)','Dy(III)','Gd(III)','Pr(III)','Ni(II)'):
            ion   = cls(name)
            group = ion.states_by_multiplicity()

            if not sum(item['n_terms'] for item in group) == ion.n_terms:
                grouping_ok = False
            if not sum(item['n_states'] for item in group) == ion.n_states:
                grouping_ok = False
            if not sorted(item['multiplicity'] for item in group) \
                   == sorted({term['S'] + 1 for term in ion.term_data}):
                grouping_ok = False

            # The groups are ordered by decreasing multiplicity.
            if not [item['multiplicity'] for item in group] \
                   == sorted([item['multiplicity'] for item in group],reverse=True):
                grouping_ok = False

        check('the spin multiplicities account for all the terms and states',
              grouping_ok)

        # The sextet states of f9, i.e. the three terms 6P, 6F and 6H:
        # 6*(3 + 7 + 11) = 126 states.
        dy_group = cls('Dy(III)').states_by_multiplicity()
        check('the states of the highest multiplicity of f9 are counted',
              (dy_group[0]['multiplicity'] == 6)
              and (dy_group[0]['n_terms'] == 3)
              and (dy_group[0]['n_states'] == 126))

        # The half-filled f shell carries a single octet term, 8S, whose
        # eight states are those of the ground multiplet.
        gd_group = cls('Gd(III)').states_by_multiplicity()
        check('the half-filled shell has one term of the highest multiplicity',
              (gd_group[0]['multiplicity'] == 8)
              and (gd_group[0]['n_terms'] == 1)
              and (gd_group[0]['n_states'] == 8))

        # The ground multiplet of every ion the class treats, against an
        # independent evaluation of Hund's rules. The class picks the
        # ground term out of the terms returned by the Fortran extension,
        # whereas the reference here fills the orbitals of the shell by
        # hand: the electrons are placed one per orbital from the largest
        # projection downwards with parallel spins, and the rest are paired
        # from the largest projection again. The two ways of arriving at
        # the ground term share nothing, so the check covers both the term
        # enumeration of the extension and the rules applied to it.
        def reference_ground_multiplet(l, n):
            """Return (2S, 2L, 2J) of the Hund's-rule ground multiplet of
            the configuration l^n, evaluated by filling the orbitals.
            """
            half_shell = 2*l + 1

            if n <= half_shell:
                two_S      = n
                projection = sum(range(l,l - n,-1)) if n > 0 else 0
            else:
                paired     = n - half_shell
                two_S      = half_shell - paired
                projection = sum(range(l,l - paired,-1)) if paired > 0 else 0

            two_L = 2*abs(projection)

            if n < half_shell:
                return (two_S,two_L,abs(two_L - two_S))

            return (two_S,two_L,two_L + two_S)

        import io
        import contextlib

        hund_ok      = True
        checked_ions = 0

        for element in cls.supported_elements():
            for state in range(1,20):
                # The oxidation states that leave no single open shell are
                # a fatal error, which is what is being skipped here; the
                # warning of an uncommon state is not printed either.
                buffer = io.StringIO()

                try:
                    with contextlib.redirect_stdout(buffer):
                        ion = cls("{0:s}({1:d})".format(element,state))
                except SystemExit:
                    continue

                checked_ions += 1

                if not (ion.S,ion.L,ion.J) \
                       == reference_ground_multiplet(ion.l,ion.n_electrons):
                    hund_ok = False

        check('the ground multiplet of every treated ion follows Hund\'s rules',
              hund_ok)
        check('the treated ions cover the d and f blocks',
              checked_ions > 400)

        # ------------------------------------------------------------------
        # The Curie susceptibility, against the literature values of the
        # trivalent lanthanides.
        # ------------------------------------------------------------------
        curie_list = [('Ce(III)',0.80), ('Pr(III)',1.60), ('Nd(III)',1.64),
                      ('Sm(III)',0.09), ('Eu(III)',0.00), ('Gd(III)',7.88),
                      ('Tb(III)',11.82),('Dy(III)',14.17),('Ho(III)',14.07),
                      ('Er(III)',11.48),('Tm(III)',7.15), ('Yb(III)',2.57)]

        # The tabulated values are those of the textbooks, which use the
        # simple g-factor; the accurate factor shifts them by about a part
        # in a thousand, which the check below covers separately.
        curie_ok = True
        for name, value in curie_list:
            if abs(cls(name).curie_susceptibility(simple_g_factor=True)
                   - value) > 5.0e-3:
                curie_ok = False
        check('the Curie chiT products of the lanthanides are correct',
              curie_ok)

        # The default of the method is the accurate g-factor, so the chiT
        # product it gives is larger by the square of the ratio of the two
        # factors.
        ratio_ok = True
        for name, value in curie_list:
            ion = cls(name)

            if ion.J == 0:
                continue

            if not abs(ion.curie_susceptibility()
                       - ion.curie_susceptibility(simple_g_factor=True)
                         *(ion.lande_g_factor/ion.lande_g_factor_simple)**2) \
                   < 1.0e-12:
                ratio_ok = False
        check('the Curie chiT product follows the chosen g-factor',ratio_ok)
        check('the Curie chiT product uses the accurate g-factor by default',
              abs(cls('Dy(III)').curie_susceptibility() - 14.1887) < 5.0e-4)

        check('an ion of J = 0 has a vanishing Curie chiT product',
              abs(cls('Eu(III)').curie_susceptibility()) < 1.0e-12)

        # The Curie law of a spin-only ion: an S = 5/2 ion of g = 2 has
        # chi*T = 4.375 cm^3 K mol^-1. The free Fe(III) ion has L = 0, so
        # its ground multiplet is spin-only and g_J = 2.
        check('the spin-only Curie chiT product is recovered',
              abs(cls('Fe(III)').curie_susceptibility(simple_g_factor=True)
                  - 4.377) < 5.0e-3)

        # ------------------------------------------------------------------
        # The coefficients of fractional parentage.
        # ------------------------------------------------------------------
        # They are not evaluated upon construction.
        pr = cls('Pr(III)')
        check('the coefficients of fractional parentage are not evaluated at once',
              pr._IonData__cfp_data is None)

        cfp_data = pr.coefficients_of_fractional_parentage()

        check('the coefficients are evaluated on request',
              pr._IonData__cfp_data is not None)
        check('the coefficients are stored after the first call',
              pr.coefficients_of_fractional_parentage() is cfp_data)
        check('the shape of the coefficient matrix follows the term counts',
              cfp_data['matrix'].shape
              == (len(cfp_data['terms']),len(cfp_data['parent_terms'])))
        check('the parent configuration of f2 is f1',
              len(cfp_data['parent_terms']) == 1)

        # The rows of the CFP matrix are normalized: the sum of the squares
        # of the coefficients of one daughter term is one.
        norm_ok = True
        for name in ('Pr(III)','Nd(III)','Fe(III)','Ti(II)'):
            ion    = cls(name)
            matrix = ion.coefficients_of_fractional_parentage()['matrix']

            for i in range(0,matrix.shape[0]):
                if abs(np.dot(matrix[i],matrix[i]) - 1.0) > 1.0e-10:
                    norm_ok = False
        check('the coefficients of each term are normalized',norm_ok)

        # An empty shell has no parent configuration.
        empty_cfp = cls('La(III)').coefficients_of_fractional_parentage()
        check('an empty shell has no parent terms',
              (len(empty_cfp['parent_terms']) == 0)
              and (empty_cfp['matrix'].size == 0))

        # ------------------------------------------------------------------
        # The tables.
        # ------------------------------------------------------------------
        from ouluspin import result_table

        data_table = dy.data_table()
        check('the data table is a ResultTable',
              isinstance(data_table,result_table.ResultTable))

        table_string = data_table.string_table()
        check('the data table states the ion',
              'DY(III)' in table_string)
        check('the data table states the configuration',
              '4f9' in table_string)
        check('the data table states the ground multiplet',
              '6H15/2' in table_string)
        check('the data table states the true angular momentum',
              '15/2' in table_string)
        check('the data table lists the terms',
              '6H' in table_string)

        # The states of each spin multiplicity and of the whole
        # configuration.
        check('the data table states the spin multiplicities',
              'Multiplicity 6 (S = 5/2)' in table_string)
        check('the data table counts the states of a multiplicity',
              '126 states' in table_string)
        check('the data table counts the states of the configuration',
              ('2002 states' in table_string)
              and ('Number of states of 4f9: 2002' in table_string))
        check('the data table counts the spin states',
              ('21 spin states' in table_string)
              and ('735 spin states' in table_string))
        check('the data table states the numbers of electrons',
              ('Number of electrons' in table_string)
              and ('Number of valence electrons' in table_string))
        check('the data table gives the electron counts of the ion',
              all(line.split()[-1] == '63' for line in table_string.splitlines()
                  if line.strip().startswith('Number of electrons'))
              and all(line.split()[-1] == '9' for line in table_string.splitlines()
                      if line.strip().startswith('Number of valence electrons')))
        check('the ground multiplet table states the electron counts too',
              ('Number of electrons' in dy.ground_multiplet_table().string_table())
              and ('Number of valence electrons'
                   in dy.ground_multiplet_table().string_table()))
        check('the data table lists every multiplicity of the configuration',
              all("Multiplicity {0:d} ".format(group['multiplicity'])
                  in table_string
                  for group in dy.states_by_multiplicity()))

        ground_table = dy.ground_multiplet_table().string_table()
        check('the ground multiplet table is shorter than the data table',
              len(ground_table) < len(table_string))
        check('the ground multiplet table states the Lande g-factor',
              'Lande' in ground_table)
        check('the tables state all three forms of the Lande g-factor',
              ('1.334106' in ground_table) and ('1.333333' in ground_table)
              and ('4/3' in ground_table))
        check('the tables state the chiT product with both g-factors',
              ('14.1887' in ground_table) and ('14.1723' in ground_table))

        term_table = dy.term_table().string_table()
        check('the term table lists one term per row',
              term_table.count('\n') > dy.n_terms)

        cfp_table = pr.cfp_table().string_table()
        check('the CFP table is built',
              'FRACTIONAL PARENTAGE' in cfp_table)
        check('the CFP table lists the coefficients',
              'Parent term' in cfp_table)

        check('the representation is the data table',
              repr(dy) == dy.data_table().string_table())

        # The table of an f-shell ion carries the G2 labels and that of a
        # d-shell ion does not.
        check('the term table of an f ion carries the G2 labels',
              '12g(G2)' in dy.term_table().string_table())
        check('the term table of a d ion carries no G2 labels',
              not '12g(G2)' in cls('Fe(III)').term_table().string_table())

        return debug_output.test_summary('IonData',result_list,print_output)
