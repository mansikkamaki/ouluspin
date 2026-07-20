# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""Structures used to represent plots of calculated results.

The module does for the plots what the result_table module does for the
tables: it separates the CONTENT of a plot (the data sets, the axis labels
and the legends) from its APPEARANCE (the image written on disk). The
classes of the library that produce plottable results are handed to
ResultPlot, which stores what is to be plotted and writes the image in the
format that is asked for.

Three kinds of plots are recognized:

    standard_plot
        One or more data sets plotted against a common variable, e.g. the
        chiT product as a function of the temperature.
    effective_barrier
        The effective barrier of the reversal of the magnetization of a
        single-molecule magnet: the states are placed by their magnetic
        moment projection and their energy, and the transitions between
        them are drawn as arrows whose width follows the magnitude of the
        transition magnetic moment.
    energy_level_diagram
        One or more energy level structures drawn as horizontal bars on a
        discrete horizontal axis.

The images are written by the internal _images module, which is imported
only when a plot is written, so that the ordinary use of the library does
not need the plotting libraries.
"""

import sys

import numpy as np


class ResultPlot:
    """A plot of calculated results.

    An instance stores the data sets of the plot together with the axis
    labels, the legends and the options that decide how the data are drawn.
    The appearance is produced by the writing methods png_plot(),
    tiff_plot() and pdf_plot(), which write the image on disk in the
    corresponding format.

    The instance is ordinarily constructed from an instance of one of the
    classes of the library that produce plottable results, which decides
    the kind of the plot:

        IsothermalStaticMagnetization
            A standard plot of the magnetization as a function of the
            magnetic field, one curve per temperature.
        StaticMagneticSusceptibility
            A standard plot of the chiT product as a function of the
            temperature.
        StaticTransitionMagneticMoments
            An effective barrier of the reversal of the magnetization.
        A Hamiltonian operator carrying eigenvalues
            An energy level diagram of the eigenvalues. Any instance with
            an eigenvalues attribute is accepted, e.g. PseudoSpinOperator
            or GeneralOperatorMatrix, and the operator must have been
            diagonalized.

    A plot of data that do not come from any of these is constructed with
    the from_data class method.

    All the options are stored as attributes and can be changed afterwards,
    so that the same instance can be written several times with a different
    appearance.

    Arguments
    ---------
    source : object
        The instance the plot is constructed from, one of the classes
        listed above.

    Optional arguments
    ------------------
    style : str
        How the data sets are drawn. The recognized values are 'line' for
        a continuous line, 'points' for the data points alone and
        'line_points' for a line carrying the data points. Default is
        'line'. The value is ignored by the plots that are not drawn as
        curves, i.e. the effective barrier and the energy level diagram.
    label : str or None
        The legend of the data set, used when the source contributes a
        single data set. Default is None, in which case the label of the
        source is used.
    n_states : int or None
        The number of the lowest-energy states drawn in an energy level
        diagram. Default is None, in which case all the states are drawn.
    energy_cutoff : float or None
        The highest energy drawn in an effective barrier plot. The states
        above the cutoff lie above the relaxation pathway and are left out
        together with the transitions involving them. Default is None, in
        which case the cutoff is deduced from the pathway itself (see the
        __barrier_cutoff private method).
    degeneracy_tolerance : float or None
        The energy separation, in the energy unit of the source, below
        which two states of an energy level diagram are considered
        degenerate and their bars are drawn beside each other instead of
        on top of each other. Default is None, in which case the tolerance
        follows the numerical accuracy of the energies, i.e. 1.0e-6 times
        the spread of the drawn spectrum, which groups the states of a
        Kramers doublet without grouping the doublets themselves. Widen it
        to draw states that are close but not degenerate beside each other
        as well.
    title : str or None
        The title written above the plot. Default is None, in which case
        no title is written.

    Attributes
    ----------
    plot_type : str
        The kind of the plot, i.e. 'standard_plot', 'effective_barrier' or
        'energy_level_diagram'.
    data_sets : list of dict
        The data sets of the plot. Each is a dictionary with the items 'x'
        and 'y' holding the values, 'label' holding the legend and 'style'
        holding the drawing style of the set.
    levels : list of dict
        The states of an effective barrier plot, each with the items
        'moment' and 'energy'. Empty for the other kinds of plots.
    transitions : list of dict
        The transitions of an effective barrier plot, each with the items
        'from', 'to' and 'moment', the indices referring to the levels.
        Empty for the other kinds of plots.
    columns : list of dict
        The energy level structures of an energy level diagram, each with
        the items 'energies' and 'label'. Empty for the other kinds of
        plots.
    degeneracy_tolerance : float
        The separation below which two states of an energy level diagram
        are drawn beside each other as a degenerate group.
    x_label : str
        The label of the horizontal axis.
    y_label : str
        The label of the vertical axis.
    title : str or None
        The title written above the plot.
    legend : boolean
        Whether the legends of the data sets are drawn. Set to True when
        the plot holds more than one labelled data set.
    style : str
        The default drawing style of the data sets.

    Public methods
    --------------
    png_plot(filename,...)
        Write the plot into a PNG file.
    tiff_plot(filename,...)
        Write the plot into a TIFF file.
    pdf_plot(filename,...)
        Write the plot into a PDF file, i.e. as vector graphics.

    Private methods
    ---------------
    __from_magnetization(source,...)
        Build the data sets of a magnetization plot.
    __from_susceptibility(source,...)
        Build the data sets of a susceptibility plot.
    __from_transition_moments(source,...)
        Build the levels and the transitions of an effective barrier.
    __from_eigenvalues(source,label,n_states,degeneracy_tolerance)
        Build the column of an energy level diagram.
    __barrier_cutoff(energies,moments)
        Return the highest energy drawn in an effective barrier plot.
    __route_order(levels)
        Return the order the states are passed in along the relaxation
        pathway, used to point the arrows of the barrier plot.
    __plot_content()
        Return the content of the plot in the form used by the writers.
    __write(filename,file_format,...)
        Write the plot in the given format.

    Class methods
    -------------
    from_data(x,y,...) : ResultPlot
        Return a standard plot built from the given values.
    plot_types() : list of str
        Return the names of the recognized kinds of plots.
    styles() : list of str
        Return the names of the recognized drawing styles.
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if
        all tests passed.
    """

    __PLOT_TYPES = ('standard_plot','effective_barrier','energy_level_diagram')
    __STYLES     = ('line','points','line_points')

    # The defaults of the writing methods. The resolution is high enough
    # for the plots of a publication, and the draft copy is legible on the
    # screen without being precise.
    DEFAULT_RESOLUTION = 600
    DRAFT_RESOLUTION   = 150
    DEFAULT_WIDTH      = 6.0
    DEFAULT_SIZE_RATIO = 4.0/3.0

    # The default compression of the raster formats.
    DEFAULT_PNG_COMPRESSION  = 9
    DEFAULT_TIFF_COMPRESSION = 'lzw'

    def __error(self, message):
        """Report an error and stop. The plots are written at the end of a
        calculation, so an error is reported the way the other errors of
        the library are.
        """
        print("ERROR in ResultPlot.")
        print("ERROR: " + message)
        print("Error termination.")
        sys.exit(1)


    def __from_magnetization(self, source, style, label):
        """Build the data sets of a plot of the magnetization as a function
        of the magnetic field, one data set per temperature point.
        """
        self.plot_type = 'standard_plot'
        self.x_label   = "B / T"
        self.y_label   = "M / mu_B"

        for i in range(0,source.n_T_points):
            if source.single_field_range:
                field_list = list(source.B_list)
            else:
                field_list = list(source.B_list[i])

            if label is None:
                set_label = "T = {0:.3f} K".format(source.T_list[i])
            else:
                set_label = label

            self.data_sets.append({'x':     [float(B) for B in field_list],
                                   'y':     [float(M) for M
                                             in source.magnetization[i]],
                                   'label': set_label,
                                   'style': style})


    def __from_susceptibility(self, source, style, label):
        """Build the data set of a plot of the chiT product as a function
        of the temperature.
        """
        self.plot_type = 'standard_plot'
        self.x_label   = "T / K"
        self.y_label   = "chiT / cm^3 K mol^-1"

        if label is None:
            set_label = "chiT"
        else:
            set_label = label

        self.data_sets.append({'x':     [float(T) for T in source.T_list],
                               'y':     [float(chi) for chi
                                         in source.susceptibility],
                               'label': set_label,
                               'style': style})


    def __barrier_cutoff(self, energy_list, moment_list):
        """Return the highest energy drawn in an effective barrier plot.

        The relaxation pathway climbs the barrier through the states whose
        magnetic moment projection is large and crosses it at the states
        whose projection has collapsed towards zero. The states above the
        crossing lie above the pathway, where the transition moments are
        always strong and the arrows carry no information, so the cutoff is
        placed at the first state whose projection has fallen to a small
        fraction of the largest one.
        """
        if len(energy_list) == 0:
            return 0.0

        largest_moment = max([abs(moment) for moment in moment_list])

        if largest_moment <= 0.0:
            return max(energy_list)

        for i in range(0,len(energy_list)):
            if abs(moment_list[i]) < 0.25*largest_moment:
                return energy_list[i]

        return max(energy_list)


    def __from_transition_moments(self, source, style, label, energy_cutoff):
        """Build the levels and the transitions of an effective barrier
        plot of the reversal of the magnetization.
        """
        self.plot_type = 'effective_barrier'
        self.x_label   = "Magnetic moment / mu_B"
        self.y_label   = "E / " + self.__energy_unit(source)

        energy_list = [float(energy) for energy in source.eigenvalues]
        moment_list = [float(moment) for moment in source.expectation_values]

        if energy_cutoff is None:
            cutoff = self.__barrier_cutoff(energy_list,moment_list)
        else:
            cutoff = float(energy_cutoff)

        self.energy_cutoff = cutoff

        # The states of the pathway, i.e. the ones at or below the cutoff.
        # The states of a Kramers doublet are degenerate but are rarely
        # exactly so in a numerical calculation, so the cutoff is given a
        # tolerance that is small against the gaps of the spectrum but
        # large against the splitting of a pair; a pair must not be broken
        # by drawing one of its states and leaving out the other.
        if len(energy_list) > 0:
            energy_span = max(energy_list) - min(energy_list)
        else:
            energy_span = 0.0

        tolerance = max(1.0e-8,1.0e-6*energy_span)

        index_list = [i for i in range(0,len(energy_list))
                      if energy_list[i] <= cutoff + tolerance]

        for i in index_list:
            self.levels.append({'moment': moment_list[i],
                                'energy': energy_list[i]})

        # The transitions between the drawn states, pointed along the
        # relaxation pathway.
        order = self.__route_order(self.levels)

        for a in range(0,len(index_list)):
            for b in range(0,a):
                moment = float(source.transition_magnetic_moment[index_list[a]]
                                                                [index_list[b]])

                if order[a] < order[b]:
                    first, second = a, b
                else:
                    first, second = b, a

                self.transitions.append({'from':   first,
                                         'to':     second,
                                         'moment': moment})


    def __route_order(self, level_list):
        """Return the position of each state along the relaxation pathway.

        The pathway starts at the lowest state on the side of the negative
        magnetic moment projection, climbs the barrier, crosses it and
        comes down on the side of the positive projection. The states are
        therefore ordered by increasing energy on the negative side and by
        decreasing energy on the positive side, and the arrows are drawn
        from the earlier state of the pathway to the later one.
        """
        negative = [i for i in range(0,len(level_list))
                    if level_list[i]['moment'] < 0.0]
        positive = [i for i in range(0,len(level_list))
                    if level_list[i]['moment'] >= 0.0]

        negative.sort(key=lambda i: level_list[i]['energy'])
        positive.sort(key=lambda i: -level_list[i]['energy'])

        order = len(level_list)*[0]
        for position, index in enumerate(negative + positive):
            order[index] = position

        return order


    def __energy_unit(self, source):
        """Return the energy unit of the source, or an empty label when the
        source carries no unit system.
        """
        units = getattr(source,'units',None)

        if units is None:
            return ""

        return getattr(units,'energy_unit_str',"")


    def __from_eigenvalues(self, source, label, n_states, degeneracy_tolerance):
        """Build the column of an energy level diagram out of the
        eigenvalues of an operator.
        """
        self.plot_type = 'energy_level_diagram'
        self.x_label   = ""
        self.y_label   = "E / " + self.__energy_unit(source)

        energy_list = [float(energy) for energy in source.eigenvalues]

        if n_states is not None:
            energy_list = energy_list[:n_states]

        # The states of a degenerate group are drawn beside each other, so
        # the diagram needs to know which states are degenerate. The
        # default tolerance follows the numerical accuracy of the energies,
        # which scales with the spread of the spectrum.
        if degeneracy_tolerance is None:
            if len(energy_list) > 0:
                energy_span = max(energy_list) - min(energy_list)
            else:
                energy_span = 0.0

            self.degeneracy_tolerance = max(1.0e-8,1.0e-6*energy_span)
        else:
            self.degeneracy_tolerance = float(degeneracy_tolerance)

        if label is None:
            column_label = ""
        else:
            column_label = label

        self.columns.append({'energies': energy_list, 'label': column_label})


    def __init__(self, source,
                 style='line',
                 label=None,
                 n_states=None,
                 energy_cutoff=None,
                 degeneracy_tolerance=None,
                 title=None):
        """Upon class initiation store the data of the source and the
        options of the plot. The kind of the plot follows from the class of
        the source.
        """
        from ouluspin import properties

        if not style in self.__STYLES:
            self.__error("Unknown plotting style: " + str(style) + ".\n"
                         "ERROR: The recognized styles are: "
                         + ", ".join(self.styles()) + ".")

        self.data_sets            = []
        self.levels               = []
        self.transitions          = []
        self.columns              = []
        self.style                = style
        self.title                = title
        self.legend               = False
        self.energy_cutoff        = None
        self.degeneracy_tolerance = 0.0
        self.x_label              = ""
        self.y_label              = ""

        if isinstance(source,properties.IsothermalStaticMagnetization):
            self.__from_magnetization(source,style,label)
        elif isinstance(source,properties.StaticMagneticSusceptibility):
            self.__from_susceptibility(source,style,label)
        elif isinstance(source,properties.StaticTransitionMagneticMoments):
            self.__from_transition_moments(source,style,label,energy_cutoff)
        elif hasattr(source,'eigenvalues') and len(source.eigenvalues) > 0:
            self.__from_eigenvalues(source,label,n_states,degeneracy_tolerance)
        else:
            self.__error("Cannot build a plot of an instance of "
                         + type(source).__name__ + ".\n"
                         "ERROR: The recognized sources are the magnetization, "
                         "the susceptibility, the\n"
                         "ERROR: transition magnetic moments and an operator "
                         "carrying eigenvalues.\n"
                         "ERROR: An operator must be diagonalized before it is "
                         "plotted.")

        self.legend = len([data_set for data_set in self.data_sets
                           if not data_set['label'] == ""]) > 1


    @classmethod
    def from_data(cls, x, y,
                  style='line',
                  label=None,
                  x_label="",
                  y_label="",
                  title=None):
        """Return a standard plot built from the given values instead of
        from an instance of one of the classes of the library.

        Arguments
        ---------
        x : list of float
            The values of the horizontal axis.
        y : list of float
            The values of the vertical axis, one per value of x.

        Optional arguments
        ------------------
        style : str
            The drawing style of the data set (see the class docstring).
            Default is 'line'.
        label : str or None
            The legend of the data set. Default is None, in which case the
            set carries no legend.
        x_label : str
            The label of the horizontal axis. Default is an empty string.
        y_label : str
            The label of the vertical axis. Default is an empty string.
        title : str or None
            The title written above the plot. Default is None.
        """
        instance = cls.__new__(cls)

        if not style in cls.__STYLES:
            instance.__error("Unknown plotting style: " + str(style) + ".")

        x_values = [float(value) for value in x]
        y_values = [float(value) for value in y]

        if not len(x_values) == len(y_values):
            instance.__error("The numbers of the x and the y values differ: "
                             + str(len(x_values)) + " and "
                             + str(len(y_values)) + ".")

        instance.plot_type     = 'standard_plot'
        instance.data_sets     = [{'x':     x_values,
                                   'y':     y_values,
                                   'label': "" if label is None else label,
                                   'style': style}]
        instance.levels        = []
        instance.transitions   = []
        instance.columns       = []
        instance.style         = style
        instance.title         = title
        instance.x_label              = x_label
        instance.y_label              = y_label
        instance.legend               = False
        instance.energy_cutoff        = None
        instance.degeneracy_tolerance = 0.0

        return instance


    def __plot_content(self):
        """Return the content of the plot in the form the writers of the
        _images module use, i.e. as a dictionary of the data sets and of
        the labels and options belonging to them.
        """
        return {'plot_type':            self.plot_type,
                'data_sets':            [dict(data_set)
                                         for data_set in self.data_sets],
                'levels':               [dict(level) for level in self.levels],
                'transitions':          [dict(transition)
                                         for transition in self.transitions],
                'columns':              [dict(column)
                                         for column in self.columns],
                'x_label':              self.x_label,
                'y_label':              self.y_label,
                'title':                self.title,
                'legend':               self.legend,
                'degeneracy_tolerance': self.degeneracy_tolerance}


    def __write(self, filename, file_format,
                size_ratio=None,
                resolution=None,
                draft_copy=False,
                overwrite=False,
                compression=None):
        """Write the plot on disk in the given format. The arguments are
        the ones of the public writing methods, which document them.
        """
        import os

        from ouluspin import _images

        if size_ratio is None:
            size_ratio = self.DEFAULT_SIZE_RATIO
        if resolution is None:
            resolution = self.DEFAULT_RESOLUTION

        if size_ratio <= 0.0:
            self.__error("The size ratio must be positive.")
        if resolution <= 0:
            self.__error("The resolution must be positive.")

        name_list = [(filename,resolution)]

        if draft_copy:
            base, extension = os.path.splitext(filename)
            name_list.append((base + "_draft" + extension,
                              self.DRAFT_RESOLUTION))

        for name, name_resolution in name_list:
            if os.path.exists(name) and not overwrite:
                self.__error("The file " + name + " already exists.\n"
                             "ERROR: Give overwrite=True to replace it.")

        content = self.__plot_content()

        for name, name_resolution in name_list:
            _images.write_plot(content,name,file_format,
                               width=self.DEFAULT_WIDTH,
                               size_ratio=size_ratio,
                               resolution=name_resolution,
                               compression=compression)


    def png_plot(self, filename,
                 size_ratio=None,
                 resolution=None,
                 draft_copy=False,
                 overwrite=False,
                 compression=None):
        """Write the plot into a PNG file.

        Arguments
        ---------
        filename : str
            The name of the file to write.

        Optional arguments
        ------------------
        size_ratio : float or None
            The ratio of the width of the plot to its height. Default is
            None, in which case the default ratio of the class is used.
        resolution : int or None
            The resolution of the image in dots per inch. Default is None,
            in which case the default resolution of the class is used,
            which is high enough for the plots of a publication.
        draft_copy : boolean
            Whether a copy of a low resolution is written beside the image,
            with '_draft' appended to the name. The copy is meant to be
            used as a placeholder in the drafts of a manuscript, where the
            full resolution is not needed. Default is False.
        overwrite : boolean
            Whether an existing file is replaced. Default is False, in
            which case writing over an existing file is an error.
        compression : int or None
            The compression level of the PNG file, from 0 to 9. Default is
            None, in which case the level 9 of the class is used.
        """
        if compression is None:
            compression = self.DEFAULT_PNG_COMPRESSION

        self.__write(filename,'png',
                     size_ratio=size_ratio,
                     resolution=resolution,
                     draft_copy=draft_copy,
                     overwrite=overwrite,
                     compression=compression)


    def tiff_plot(self, filename,
                  size_ratio=None,
                  resolution=None,
                  draft_copy=False,
                  overwrite=False,
                  compression=None):
        """Write the plot into a TIFF file.

        The arguments are the ones of the png_plot method, except that the
        compression is the name of the compression of the TIFF file, e.g.
        'lzw' or 'none'. Default is None, in which case the LZW
        compression of the class is used.
        """
        if compression is None:
            compression = self.DEFAULT_TIFF_COMPRESSION

        self.__write(filename,'tiff',
                     size_ratio=size_ratio,
                     resolution=resolution,
                     draft_copy=draft_copy,
                     overwrite=overwrite,
                     compression=compression)


    def pdf_plot(self, filename,
                 size_ratio=None,
                 resolution=None,
                 draft_copy=False,
                 overwrite=False):
        """Write the plot into a PDF file, i.e. as vector graphics.

        The arguments are the ones of the png_plot method. A PDF file
        carries the plot as vector graphics, so the resolution only
        affects the raster parts of the image, if any, and the draft copy
        is rarely needed.
        """
        self.__write(filename,'pdf',
                     size_ratio=size_ratio,
                     resolution=resolution,
                     draft_copy=draft_copy,
                     overwrite=overwrite)


    def __add__(self, other):
        """Return a plot holding the data sets of this plot and of the plot
        given as the argument, e.g. the chiT products of two systems drawn
        in the same plot field or the energy level structures of two
        systems drawn side by side.

        Only truly compatible plots are summed. The two plots must be of
        the same kind and must carry the same axis labels, so that the
        summed plot means something; the effective barrier plots are never
        summed, since two barriers drawn in the same field cannot be told
        apart.
        """
        if not isinstance(other,ResultPlot):
            self.__error("Only two plots can be summed.")

        if not self.plot_type == other.plot_type:
            self.__error("Cannot sum a plot of the type " + self.plot_type
                         + " and a plot of the type " + other.plot_type + ".")

        if self.plot_type == 'effective_barrier':
            self.__error("Two effective barrier plots cannot be summed.\n"
                         "ERROR: The barriers drawn in the same plot field "
                         "cannot be told apart.")

        if not self.x_label == other.x_label:
            self.__error("The horizontal axes of the plots differ: '"
                         + self.x_label + "' and '" + other.x_label + "'.")

        if not self.y_label == other.y_label:
            self.__error("The vertical axes of the plots differ: '"
                         + self.y_label + "' and '" + other.y_label + "'.")

        result = self.__new__(type(self))

        result.plot_type     = self.plot_type
        result.data_sets     = [dict(data_set) for data_set
                                in self.data_sets + other.data_sets]
        result.levels        = []
        result.transitions   = []
        result.columns       = [dict(column) for column
                                in self.columns + other.columns]
        result.style         = self.style
        result.title         = self.title
        result.x_label       = self.x_label
        result.y_label       = self.y_label
        result.energy_cutoff = None

        # The summed diagram holds the levels of both, so the wider of the
        # two tolerances is the one that groups them all.
        result.degeneracy_tolerance = max(self.degeneracy_tolerance,
                                          other.degeneracy_tolerance)

        result.legend = len([data_set for data_set in result.data_sets
                             if not data_set['label'] == ""]) > 1

        return result


    def __repr__(self):
        """Return a short description of the plot. A plot is written on
        disk rather than printed, so the representation only states what
        the instance holds.
        """
        if self.plot_type == 'effective_barrier':
            content_str = "{0} states, {1} transitions".format(
                len(self.levels),len(self.transitions))
        elif self.plot_type == 'energy_level_diagram':
            content_str = "{0} level structures".format(len(self.columns))
        else:
            content_str = "{0} data sets".format(len(self.data_sets))

        return ("ResultPlot(" + self.plot_type + ", " + content_str + ")\n")


    @classmethod
    def plot_types(cls):
        """Return the names of the recognized kinds of plots."""
        return list(cls.__PLOT_TYPES)


    @classmethod
    def styles(cls):
        """Return the names of the recognized drawing styles."""
        return list(cls.__STYLES)


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class and run a set of tests on the class
        constructor and the class methods. Return True if all tests passed
        and False otherwise.

        The tests that write a file are only run when the plotting library
        is available, so that the test suite passes without it.

        Optional arguments
        ------------------
        print_output : boolean
            Whether to print the outcome of each test. Default is True.
        """
        import os
        import tempfile

        from ouluspin import _debug as debug_output
        from ouluspin import properties
        from ouluspin import units as unit_systems

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('ResultPlot',test_name,
                                                       condition,print_output))

        # A plot built from plain values.
        x_values = [0.0,1.0,2.0,3.0]
        y_values = [0.0,1.0,4.0,9.0]

        plot = cls.from_data(x_values,y_values,
                             x_label="T / K",y_label="chiT",
                             label="first",title="TEST PLOT")

        check('a plot is built from the given values',
              plot.plot_type == 'standard_plot')
        check('the values are stored',
              (plot.data_sets[0]['x'] == x_values)
              and (plot.data_sets[0]['y'] == y_values))
        check('the axis labels are stored',
              (plot.x_label == "T / K") and (plot.y_label == "chiT"))
        check('the title is stored', plot.title == "TEST PLOT")
        check('the representation states the content',
              'standard_plot' in repr(plot))
        check('the plot types are listed',
              'effective_barrier' in cls.plot_types())
        check('the drawing styles are listed', 'line_points' in cls.styles())

        # The summing of two compatible plots.
        second_plot = cls.from_data(x_values,[1.0,2.0,3.0,4.0],
                                    x_label="T / K",y_label="chiT",
                                    label="second")
        summed_plot = plot + second_plot

        check('two compatible plots are summed',
              len(summed_plot.data_sets) == 2)
        check('the summed plot keeps the axis labels',
              summed_plot.x_label == "T / K")
        check('the summed plot draws the legends',
              summed_plot.legend is True)
        check('the summands are not changed by the summing',
              (len(plot.data_sets) == 1) and (len(second_plot.data_sets) == 1))

        # The plots built from the classes of the library.
        from ouluspin import pseudospin_operators
        from ouluspin import tensors

        tmp_units = unit_systems.EnergyUnitSystem('wavenumber')

        susceptibility = properties.StaticMagneticSusceptibility(
            [1.0,2.0,3.0],susceptibility=[3.0,2.5,2.0])

        susceptibility_plot = cls(susceptibility)

        check('a susceptibility gives a standard plot',
              susceptibility_plot.plot_type == 'standard_plot')
        check('the susceptibility plot uses the temperature as the variable',
              susceptibility_plot.x_label == "T / K")
        check('the susceptibility plot holds the data',
              len(susceptibility_plot.data_sets[0]['x']) == 3)

        # An energy level diagram of the eigenvalues of an operator. The
        # operator is an axial Zeeman splitting of a pseudospin S = 1/2.
        hamiltonian_tensor = tensors.IwaharaChibotaruSphericalTensor\
                                    .from_one_site_cartesian_operator(10.0,'z',1)
        basis       = pseudospin_operators.PseudoSpinBasis([1])
        hamiltonian = pseudospin_operators.PseudoSpinOperator(
            basis,[hamiltonian_tensor],tmp_units,translate_eigenvalues=True)

        diagram = cls(hamiltonian,label="first")

        check('an operator gives an energy level diagram',
              diagram.plot_type == 'energy_level_diagram')
        check('the energy level diagram holds the eigenvalues',
              len(diagram.columns[0]['energies']) == len(hamiltonian.eigenvalues))
        check('the energy axis carries the unit of the source',
              "cm^-1" in diagram.y_label)

        limited_diagram = cls(hamiltonian,n_states=1)
        check('the number of the drawn states can be limited',
              len(limited_diagram.columns[0]['energies']) == 1)

        summed_diagram = diagram + cls(hamiltonian,label="second")
        check('two energy level diagrams are summed side by side',
              len(summed_diagram.columns) == 2)

        # The states of a degenerate group are drawn beside each other, so
        # the diagram carries the tolerance that tells which states are
        # degenerate.
        check('the energy level diagram carries a degeneracy tolerance',
              diagram.degeneracy_tolerance > 0.0)
        check('the degeneracy tolerance can be given',
              cls(hamiltonian,degeneracy_tolerance=2.5).degeneracy_tolerance
              == 2.5)

        # The grouping itself, which the writer does. A doublet is drawn as
        # two bars beside each other and two states far apart as two groups
        # of one bar.
        from ouluspin import _images

        check('degenerate states form one group',
              [len(group) for group
               in _images.degenerate_groups([0.0,1.0e-9,10.0,10.0],1.0e-6)]
              == [2,2])
        check('states further apart than the tolerance are separate groups',
              [len(group) for group
               in _images.degenerate_groups([0.0,1.0,2.0],1.0e-6)] == [1,1,1])
        check('the groups are ordered by the energy',
              [group[0] for group
               in _images.degenerate_groups([2.0,0.0,1.0],1.0e-6)]
              == [0.0,1.0,2.0])

        # The effective barrier of the transition magnetic moments.
        moment_tensor = tensors.MixedCartesianIwaharaChibotaruSphericalTensor\
                               .from_one_site_isotropic_operator(
                                   -2.0*tmp_units.mu_B,1)
        magnetic_moment = pseudospin_operators.PseudoSpinVectorOperator(
            basis,[moment_tensor],tmp_units,
            diagonalize_operator_matrix=False,
            store_operator_matrix=True)

        transition_moments = properties.StaticTransitionMagneticMoments(
            magnetic_moment,hamiltonian,2,tmp_units)

        barrier = cls(transition_moments)

        check('the transition moments give an effective barrier',
              barrier.plot_type == 'effective_barrier')
        check('the barrier holds the states',
              len(barrier.levels) > 0)
        check('the barrier holds the transitions between the drawn states',
              len(barrier.transitions)
              == len(barrier.levels)*(len(barrier.levels) - 1)//2)
        check('the barrier states carry a moment and an energy',
              ('moment' in barrier.levels[0]) and ('energy' in barrier.levels[0]))
        check('the moment is the horizontal axis of the barrier',
              "Magnetic moment" in barrier.x_label)

        # The arrows of the barrier point along the relaxation pathway,
        # i.e. from the negative side of the barrier to the positive one.
        route_ok = True
        for transition in barrier.transitions:
            first  = barrier.levels[transition['from']]
            second = barrier.levels[transition['to']]

            if first['moment'] < 0.0 and second['moment'] < 0.0:
                # Both on the climbing side: the arrow points upwards.
                if first['energy'] > second['energy']:
                    route_ok = False
            elif first['moment'] >= 0.0 and second['moment'] >= 0.0:
                # Both on the descending side: the arrow points downwards.
                if first['energy'] < second['energy']:
                    route_ok = False

        check('the arrows of the barrier follow the relaxation pathway',route_ok)

        # The writing of the files. The plots are written only when the
        # plotting library is available.
        try:
            import matplotlib
            plotting_available = True
        except ImportError:
            plotting_available = False

        if plotting_available:
            directory = tempfile.mkdtemp(prefix='ouluspin_result_plot_')

            png_name  = os.path.join(directory,'plot.png')
            tiff_name = os.path.join(directory,'plot.tiff')
            pdf_name  = os.path.join(directory,'plot.pdf')

            plot.png_plot(png_name)
            plot.tiff_plot(tiff_name)
            plot.pdf_plot(pdf_name)

            check('the PNG file is written', os.path.getsize(png_name) > 0)
            check('the TIFF file is written', os.path.getsize(tiff_name) > 0)
            check('the PDF file is written', os.path.getsize(pdf_name) > 0)

            check('the PNG file is a PNG file',
                  open(png_name,'rb').read(4) == b'\x89PNG')
            check('the PDF file is a PDF file',
                  open(pdf_name,'rb').read(4) == b'%PDF')

            # An existing file is only replaced when the overwriting was
            # asked for. The error stops the interpreter, so the guard is
            # tested through the check that precedes it.
            check('an existing file is detected',
                  os.path.exists(png_name))

            plot.png_plot(png_name,overwrite=True)
            check('an existing file is replaced when asked for',
                  os.path.getsize(png_name) > 0)

            # The draft copy is written beside the image.
            draft_name = os.path.join(directory,'draft.png')
            plot.png_plot(draft_name,draft_copy=True)
            check('the draft copy is written beside the image',
                  os.path.getsize(os.path.join(directory,'draft_draft.png')) > 0)
            check('the draft copy is smaller than the image',
                  os.path.getsize(os.path.join(directory,'draft_draft.png'))
                  < os.path.getsize(draft_name))

            # The other kinds of plots are written as well.
            barrier_name = os.path.join(directory,'barrier.pdf')
            diagram_name = os.path.join(directory,'diagram.pdf')

            barrier.pdf_plot(barrier_name)
            summed_diagram.pdf_plot(diagram_name)

            check('the effective barrier is written',
                  os.path.getsize(barrier_name) > 0)
            check('the energy level diagram is written',
                  os.path.getsize(diagram_name) > 0)

            for name in os.listdir(directory):
                os.remove(os.path.join(directory,name))
            os.rmdir(directory)

        return debug_output.test_summary('ResultPlot',result_list,print_output)
