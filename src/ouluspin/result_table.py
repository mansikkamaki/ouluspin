# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""Structures used to represent tables of calculated results.

The classes of this module separate the CONTENT of a result table (the
numbers, the headers and the explanatory texts) from its APPEARANCE (the
plain-text layout used in the standard output of the library). The classes
of the library that report results construct an instance of ResultTable and
return it; the caller obtains the printable table simply by printing the
instance or by converting it to a str.

The structured representation is what makes further output formats
possible: the same instance can later be rendered as a LaTeX table or
written into a .docx or .odt document by adding the corresponding rendering
methods to ResultTable, without touching any of the classes that produce
the results.
"""

import sys

import numpy as np


class ResultTable:
    """A table of calculated results.

    An instance stores the table content as rows of cells together with the
    headers, the title and the explanatory texts belonging to the table. The
    appearance of the table is produced by the rendering methods, currently
    the plain-text renderer string_table(), which is also used by the
    __repr__ dunder method. Additional renderers (LaTeX, .docx, .odt) will
    be added later and will use the same stored content.

    The class does not decide what a table contains. The routine that
    produces the results decides which values are tabulated and constructs
    the instance accordingly. The only exceptions are the class methods,
    which build specific compound tables of a fixed structure out of the
    result objects of the library.

    The cells of a row may be numbers (formatted with the format of their
    column), strings (printed as they are) or None (printed as an empty
    cell). In addition to the ordinary rows given as lists of cells, the
    rows argument accepts two special items used to structure long tables:
    a row given as a plain str is printed as a section heading spanning the
    table, and a row given as None is printed as a horizontal rule.

    Arguments
    ---------
    rows : list
        The rows of the table. Each ordinary row is a list (or tuple, or
        array) of the cells of the row, all rows having the same number of
        cells. A row given as a str is a section heading and a row given as
        None is a horizontal rule (see above).

    Optional arguments
    ------------------
    column_headers : list of str or list of list of str or None
        The headers of the columns, one per column of the rows. A list of
        lists gives a multi-line column header, one list per header line.
        The row header column is not covered by these; it is named by the
        row_header_label. Default is None, in which case no column headers
        are printed.
    row_headers : list of str or None
        The headers of the rows, one per ordinary row. When given, the row
        headers are printed as an additional leftmost column. Section
        headings and rules are not counted as rows here. Default is None.
    row_header_label : str
        The column header of the row header column. Default is an empty
        string.
    title : str or None
        The title of the table, printed above it. Default is None, in which
        case no title is printed.
    notes : list of str or str or None
        Explanatory texts printed between the title and the table itself,
        used e.g. to state the units or the coordinate frame of the
        tabulated values. A text containing newlines is printed on several
        lines, each of them indented by the renderer. Default is None.
    footnotes : list of str or str or None
        Explanatory texts printed below the table. A text containing
        newlines is printed on several lines, the continuation lines being
        aligned with the text of the first one. The footnotes are
        labelled with the markers 'a)', 'b)', ... in the order they are
        given; the corresponding marker is added to the header of the
        column it explains by the routine that constructs the table (see
        the footnote_marker class method). Default is None.
    summary : list of str or str or None
        Lines printed below the table and above the footnotes, used for
        quantities that summarize the table, such as the number of grid
        points or a root-mean-square deviation. Default is None.
    formats : list of str or str or None
        The format specifications (as used by the built-in format
        function, without the field width, e.g. '.6f') of the numbers of
        the columns, one per column, or a single specification applied to
        all columns. Default is None, in which case the format of the
        table type is used for the floating-point columns and the integers
        and strings are printed as they are. The field widths are always
        determined by the renderer from the content of the column.
    alignments : list of str or str or None
        The alignment of the columns, one per column, or a single value
        applied to all columns. The recognized values are 'l', 'r' and 'c'.
        Default is None, in which case the numeric columns are aligned to
        the right and the columns of strings to the left.
    table_type : str or None
        The name of a preset that sets the default number format, the
        indentation and the rule character of a specific kind of table (see
        the table_types class method for the recognized names). Default is
        None, which is equivalent to 'generic'. The presets only provide
        defaults; any option given explicitly takes precedence.
    indent : int or None
        The number of spaces the table is indented by in the plain-text
        output. Default is None, in which case the indentation of the table
        type is used.
    column_separator : str
        The string separating two columns in the plain-text output. Default
        is two spaces.
    rule_character : str or None
        The character the horizontal rules of the plain-text table are
        drawn with. Default is None, in which case the rule character of
        the table type is used.

    Attributes
    ----------
    rows : list
        The rows of the table, as given.
    column_headers : list of list of str
        The lines of the column header, each a list with one entry per
        column. An empty list when no column headers were given.
    row_headers : list of str or None
        The row headers, as given.
    row_header_label : str
        The column header of the row header column.
    title : str or None
        The title of the table.
    notes : list of str
        The explanatory lines printed above the table.
    footnotes : list of str
        The explanatory lines printed below the table.
    summary : list of str
        The summarizing lines printed below the table.
    formats : list of str or None
        The number formats of the columns, one per column, or None.
    alignments : list of str or None
        The alignments of the columns, one per column, or None.
    table_type : str
        The name of the preset used.
    indent : int
        The indentation of the plain-text table.
    column_separator : str
        The separator of the columns of the plain-text table.
    rule_character : str
        The character the horizontal rules are drawn with.
    n_columns : int
        The number of columns of the table, including the row header
        column when row headers were given.

    Public methods
    --------------
    string_table(plain=False) : str
        Return the table as a plain-text string. With plain set to True a
        bare, machine-readable form without the title, the rules and the
        explanatory texts is returned, with the column headers on a single
        comment line.
    data_file(filename,plain=True)
        Write the table into a text file.

    Private methods
    ---------------
    __resolve_options(...)
        Fill in the options that were not given from the table type preset.
    __columns()
        Return the content of the table as a list of columns of formatted
        strings together with the header lines and the alignments.
    __format_cell(value,column_format)
        Return the printable string of a single cell.

    Class methods
    -------------
    table_types() : list of str
        Return the names of the recognized table type presets.
    footnote_marker(index) : str
        Return the marker ('a)', 'b)', ...) of the footnote of the given
        index, to be appended to the header of the column the footnote
        explains.
    pseudospin_doublet_compound_table(doublets,...) : ResultTable
        Return a compound table summarizing the properties of a list of
        pseudospin doublets of a system.
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if
        all tests passed.
    """

    # The presets of the recognized table types. Each preset gives the
    # default number format of the floating-point columns, the default
    # indentation and the default rule character of the table.
    __TABLE_TYPE_PRESETS = {
        'generic':             {'float_format': '.6f', 'indent': 6, 'rule_character': '-'},
        'spherical_tensor':    {'float_format': '.6f', 'indent': 6, 'rule_character': '-'},
        'cartesian_tensor':    {'float_format': '.6f', 'indent': 4, 'rule_character': '-'},
        'basis_states':        {'float_format': '.6f', 'indent': 6, 'rule_character': '-'},
        'eigenvalues':         {'float_format': '.3f', 'indent': 6, 'rule_character': '-'},
        'eigenvectors':        {'float_format': '.6f', 'indent': 6, 'rule_character': '-'},
        'magnetization':       {'float_format': '.6f', 'indent': 6, 'rule_character': '-'},
        'susceptibility':      {'float_format': '.6f', 'indent': 6, 'rule_character': '-'},
        'transition_moments':  {'float_format': '.6f', 'indent': 6, 'rule_character': '='},
        'pseudospin_doublets': {'float_format': '.4f', 'indent': 4, 'rule_character': '-'},
        'grid':                {'float_format': '.6f', 'indent': 6, 'rule_character': '-'},
    }

    def __resolve_options(self, table_type, indent, rule_character, formats, alignments):
        """Store the table type preset and fill in the options that were not
        given explicitly from it. The number of columns is deduced from the
        column headers or from the first ordinary row.
        """
        if table_type is None:
            self.table_type = 'generic'
        elif table_type in self.__TABLE_TYPE_PRESETS:
            self.table_type = table_type
        else:
            print("ERROR in ResultTable.")
            print("ERROR: Unknown table type: " + str(table_type) + ".")
            print("ERROR: The recognized table types are: "
                  + ", ".join(self.table_types()) + ".")
            print("Error termination.")
            sys.exit(1)

        preset = self.__TABLE_TYPE_PRESETS[self.table_type]

        self.float_format = preset['float_format']

        if indent is None:
            self.indent = preset['indent']
        else:
            self.indent = indent

        if rule_character is None:
            self.rule_character = preset['rule_character']
        else:
            self.rule_character = rule_character

        # The number of columns. The column headers cover the data columns
        # only, so the row header column, which is named by the
        # row_header_label, is counted separately.
        n_columns = 0
        if len(self.column_headers) > 0:
            n_columns = len(self.column_headers[0])
        else:
            for row in self.rows:
                if not (row is None or isinstance(row,str)):
                    n_columns = len(row)
                    break

        if self.row_headers is not None:
            n_columns += 1

        self.n_columns = n_columns

        # The number of data columns, i.e. the columns of the rows.
        if self.row_headers is None:
            n_data_columns = n_columns
        else:
            n_data_columns = n_columns - 1

        # The formats and the alignments are expanded into one entry per
        # data column. The row header column is a column of strings and is
        # not covered by them.
        if formats is None:
            self.formats = None
        elif isinstance(formats,str):
            self.formats = n_data_columns*[formats]
        else:
            self.formats = list(formats)

        if alignments is None:
            self.alignments = None
        elif isinstance(alignments,str):
            self.alignments = n_data_columns*[alignments]
        else:
            self.alignments = list(alignments)


    def __format_cell(self, value, column_format):
        """Return the printable string of a single cell. The strings are
        printed as they are, None is printed as an empty cell, and the
        numbers are printed with the format of their column, defaulting to
        the number format of the table type for the floating-point values
        and to the plain str for the integers.
        """
        if value is None:
            return ""

        if isinstance(value,str):
            return value

        if isinstance(value,(bool,np.bool_)):
            return str(value)

        if column_format is not None:
            try:
                return format(value,column_format)
            except (TypeError,ValueError):
                return str(value)

        if isinstance(value,(int,np.integer)):
            return str(int(value))

        if isinstance(value,(float,np.floating)):
            return format(float(value),self.float_format)

        if isinstance(value,(complex,np.complexfloating)):
            return "({0},{1})".format(format(value.real,self.float_format),
                                      format(value.imag,self.float_format))

        return str(value)


    def __columns(self):
        """Return the content of the table in a column-wise form as a tuple

            (header_lines, body, alignments, widths)

        where header_lines is a list of lists of the formatted headers of
        the columns, body is the list of the rows with all cells converted
        into strings (the section headings and the rules being kept as they
        are), alignments is the list of the alignments of the columns and
        widths is the list of the field widths of the columns.
        """
        # Convert the cells into strings, keeping the special rows as they
        # are. The row headers are prepended as an additional column.
        body       = []
        row_index  = 0
        numeric    = self.n_columns*[False]

        for row in self.rows:
            if row is None or isinstance(row,str):
                body.append(row)
                continue

            cells = []

            if self.row_headers is not None:
                if row_index < len(self.row_headers):
                    cells.append(str(self.row_headers[row_index]))
                else:
                    cells.append("")

            for i in range(0,len(row)):
                if self.formats is None or i >= len(self.formats):
                    column_format = None
                else:
                    column_format = self.formats[i]

                cells.append(self.__format_cell(row[i],column_format))

                # Columns holding anything else than strings are treated as
                # numeric columns and are aligned to the right by default.
                if self.row_headers is None:
                    column = i
                else:
                    column = i + 1
                if column < self.n_columns and \
                   not (row[i] is None or isinstance(row[i],str)):
                    numeric[column] = True

            body.append(cells)
            row_index += 1

        # The header lines, with the row header label prepended when the
        # row headers are used.
        header_lines = []
        for header_line in self.column_headers:
            header_lines.append([str(header) for header in header_line])

        if self.row_headers is not None and len(header_lines) > 0:
            for header_line in header_lines:
                header_line.insert(0,"")
            header_lines[-1][0] = self.row_header_label

        # The alignments of the columns.
        alignments = []
        for i in range(0,self.n_columns):
            if self.row_headers is None:
                given_index = i
            else:
                given_index = i - 1

            if self.alignments is not None and 0 <= given_index \
               and given_index < len(self.alignments):
                alignments.append(self.alignments[given_index])
            elif numeric[i]:
                alignments.append('r')
            else:
                alignments.append('l')

        # The field widths from the widest entry of each column.
        widths = self.n_columns*[0]
        for line in header_lines + body:
            if line is None or isinstance(line,str):
                continue
            for i in range(0,min(len(line),self.n_columns)):
                widths[i] = max(widths[i],len(line[i]))

        return header_lines, body, alignments, widths


    def string_table(self, plain=False):
        """Return the table as a plain-text string.

        Optional arguments
        ------------------
        plain : boolean
            Whether to return a bare, machine-readable form of the table
            instead of the standard human-readable one. The bare form
            contains no title, rules or explanatory texts and carries the
            column headers on a single comment line starting with '#'; it
            is meant for the data files read by plotting programs. Default
            is False.
        """
        header_lines, body, alignments, widths = self.__columns()

        def aligned(text,width,alignment):
            if alignment == 'l':
                return text.ljust(width)
            elif alignment == 'c':
                return text.center(width)
            else:
                return text.rjust(width)

        def line_string(cells):
            parts = []
            for i in range(0,self.n_columns):
                if i < len(cells):
                    parts.append(aligned(cells[i],widths[i],alignments[i]))
                else:
                    parts.append(widths[i]*" ")
            return self.column_separator.join(parts).rstrip()

        table_width = sum(widths) \
                      + max(0,self.n_columns - 1)*len(self.column_separator)

        if plain:
            tmp_str = ""
            for header_line in header_lines:
                tmp_str += "# " + line_string(header_line) + "\n"
            for row in body:
                if row is None:
                    continue
                elif isinstance(row,str):
                    tmp_str += "# " + row + "\n"
                else:
                    tmp_str += "  " + line_string(row) + "\n"
            return tmp_str

        indent  = self.indent*" "
        rule    = indent + table_width*self.rule_character + "\n"
        tmp_str = ""

        if self.title is not None:
            tmp_str += indent + self.title + "\n\n"

        # A note or a footnote may itself consist of several lines; the
        # renderer takes care of the indentation of all of them.
        def indented_text(text,first_prefix,continuation_prefix):
            line_list = text.split("\n")
            tmp_text  = indent + first_prefix + line_list[0] + "\n"
            for line in line_list[1:]:
                tmp_text += indent + continuation_prefix + line + "\n"
            return tmp_text

        for note in self.notes:
            tmp_str += indented_text(note,"","")
        if len(self.notes) > 0:
            tmp_str += "\n"

        for header_line in header_lines:
            tmp_str += indent + line_string(header_line) + "\n"

        tmp_str += rule

        for row in body:
            if row is None:
                tmp_str += rule
            elif isinstance(row,str):
                tmp_str += "\n" + indent + row + "\n"
            else:
                tmp_str += indent + line_string(row) + "\n"

        tmp_str += rule

        if len(self.summary) > 0:
            tmp_str += "\n"
            for summary_line in self.summary:
                tmp_str += indent + summary_line + "\n"

        if len(self.footnotes) > 0:
            tmp_str += "\n"
            for i in range(0,len(self.footnotes)):
                marker = self.footnote_marker(i) + " "
                tmp_str += indented_text(self.footnotes[i],marker,
                                         len(marker)*" ")

        return tmp_str + "\n"


    def data_file(self, filename, plain=True):
        """Write the table into a text file.

        Arguments
        ---------
        filename : str
            The name of the file to write.

        Optional arguments
        ------------------
        plain : boolean
            Whether to write the bare, machine-readable form of the table
            (see string_table). Default is True.
        """
        f = open(filename, 'w')
        f.write(self.string_table(plain=plain))
        f.close()


    def __repr__(self):
        """Return the plain-text rendering of the table."""
        return self.string_table()


    def __init__(self, rows,
                 column_headers=None,
                 row_headers=None,
                 row_header_label="",
                 title=None,
                 notes=None,
                 footnotes=None,
                 summary=None,
                 formats=None,
                 alignments=None,
                 table_type=None,
                 indent=None,
                 column_separator="  ",
                 rule_character=None):
        """Upon class initiation store the content of the table and fill in
        the layout options that were not given from the table type preset.
        """
        def text_list(text):
            if text is None:
                return []
            elif isinstance(text,str):
                return [text]
            else:
                return list(text)

        self.rows             = list(rows)
        self.title            = title
        self.notes            = text_list(notes)
        self.footnotes        = text_list(footnotes)
        self.summary          = text_list(summary)
        self.row_header_label = row_header_label
        self.column_separator = column_separator

        if row_headers is None:
            self.row_headers = None
        else:
            self.row_headers = list(row_headers)

        # A single list of column headers is a single header line.
        if column_headers is None:
            self.column_headers = []
        elif len(column_headers) > 0 and isinstance(column_headers[0],(list,tuple)):
            self.column_headers = [list(header_line) for header_line in column_headers]
        else:
            self.column_headers = [list(column_headers)]

        self.__resolve_options(table_type,indent,rule_character,formats,alignments)


    @classmethod
    def table_types(cls):
        """Return a sorted list of the names of the recognized table type
        presets.
        """
        return sorted(cls.__TABLE_TYPE_PRESETS.keys())


    @classmethod
    def footnote_marker(cls, index):
        """Return the marker of the footnote of the given index, i.e. 'a)'
        for the first footnote, 'b)' for the second one and so on. The
        marker is appended to the header of the column the footnote
        explains by the routine that constructs the table.
        """
        return chr(ord('a') + index) + ")"


    @classmethod
    def pseudospin_doublet_compound_table(cls, doublets, units,
                                          title=None,
                                          frame_label='input axis frame'):
        """Return a compound table summarizing the properties of several
        pseudospin doublets of a system, one doublet per line.

        The structure of the table depends on whether the doublets are
        Kramers doublets. For a Kramers system each line contains the index
        of the doublet, the indices of the two states spanning it, its
        energy, the three principal values of its g-tensor and the angle
        between its principal magnetic axis and that of the lowest-energy
        doublet. For a non-Kramers system each line contains the index of
        the quasi-doublet (or singlet), the indices of its states, the
        energies of the two states, the tunneling splitting, the z
        principal value of the g-tensor (the x and y principal values
        vanish by Griffith's theorem) and the same angle as in the Kramers
        case. The system is treated as a Kramers system when all the
        tabulated doublets are Kramers doublets.

        The g values and the principal magnetic axes are taken from the
        g-tensor of the doublet expressed in the input axis frame, i.e.
        they are consistent with the tabulation of the individual doublets.
        The principal magnetic axis is the principal axis of the largest
        principal g value, and since the sign of a principal axis is
        arbitrary, the reported angle is the acute angle between the two
        axes.

        Arguments
        ---------
        doublets : list
            The doublets to tabulate, in the order they are tabulated. Each
            item is either an instance of PseudoSpinDoublet or, for a
            singlet state of a non-Kramers system that is not part of a
            quasi-doublet, an item that is not an instance of
            PseudoSpinDoublet. An item given as a float (or as a list or
            tuple of floats) is interpreted as the energy (energies) of
            such a state and is tabulated in the energy column; any other
            item, e.g. None, leaves the line empty apart from the index.
        units : EnergyUnitSystem
            The unit system, used for the unit of the energy columns.

        Optional arguments
        ------------------
        title : str or None
            The title of the table. Default is None, in which case a
            standard title is used.
        frame_label : str
            The name of the coordinate frame the g-tensors and the
            principal magnetic axes refer to, stated in the note above the
            table. Default is 'input axis frame'.
        """
        from ouluspin import properties

        def is_doublet(item):
            return isinstance(item,properties.PseudoSpinDoublet)

        def principal_magnetic_axis(doublet):
            """The principal axis of the largest principal g value. The
            principal values of the g-tensor are non-negative and sorted in
            an ascending order, so this is the last eigenvector.
            """
            eigenvectors = np.asarray(doublet.input_frame_g_tensor.eigenvectors,
                                      dtype=np.float64)
            return eigenvectors[:,2]

        def state_energies(doublet):
            """The energies of the two states of a doublet, or None when
            the doublet was constructed without them.
            """
            return getattr(doublet,'state_energies',None)

        def singlet_energies(item):
            """The energies given in place of a doublet, as a list."""
            if isinstance(item,(int,float,np.integer,np.floating)):
                return [float(item)]
            elif isinstance(item,(list,tuple,np.ndarray)):
                return [float(energy) for energy in item]
            else:
                return []

        doublet_list = list(doublets)

        # The system is treated as a Kramers system only when every
        # tabulated doublet is a Kramers doublet.
        kramers_system = True
        for item in doublet_list:
            if not (is_doublet(item) and item.kramers is True):
                kramers_system = False
                break

        # The lowest-energy doublet provides the reference principal
        # magnetic axis. Doublets without energies cannot be ordered, in
        # which case the first doublet of the list is used.
        reference_axis = None
        lowest_energy  = None
        for item in doublet_list:
            if not is_doublet(item):
                continue

            energies = state_energies(item)
            if energies is None:
                if reference_axis is None:
                    reference_axis = principal_magnetic_axis(item)
                continue

            energy = min(energies)
            if lowest_energy is None or energy < lowest_energy:
                lowest_energy  = energy
                reference_axis = principal_magnetic_axis(item)

        def axis_angle(doublet):
            """The acute angle, in degrees, between the principal magnetic
            axis of the doublet and the reference axis.
            """
            if reference_axis is None:
                return None

            cosine = abs(float(np.dot(principal_magnetic_axis(doublet),
                                      reference_axis)))
            return np.degrees(np.arccos(min(1.0,cosine)))

        energy_unit = units.energy_unit_str
        angle_note  = "The angle between the principal magnetic axis of the doublet\n" \
                      "and that of the lowest-energy doublet, in degrees."
        angle_header = "Angle " + cls.footnote_marker(0)

        rows = []

        if kramers_system:
            column_headers = [["Doublet","States","E / " + energy_unit,
                               "g_x","g_y","g_z",angle_header]]
            formats        = [None,None,'.4f','.4f','.4f','.4f','.2f']

            for i in range(0,len(doublet_list)):
                doublet  = doublet_list[i]
                energies = state_energies(doublet)
                g_values = doublet.input_frame_g_tensor.eigenvalues

                if energies is None:
                    energy = None
                else:
                    energy = min(energies)

                rows.append([i + 1,
                             "{0},{1}".format(doublet.states[0],doublet.states[1]),
                             energy,
                             float(g_values[0]),float(g_values[1]),float(g_values[2]),
                             axis_angle(doublet)])
        else:
            column_headers = [["Doublet","States",
                               "E_1 / " + energy_unit,"E_2 / " + energy_unit,
                               "Gap / " + energy_unit,"g_z",angle_header]]
            formats        = [None,None,'.4f','.4f','.6e','.4f','.2f']

            for i in range(0,len(doublet_list)):
                item = doublet_list[i]

                if not is_doublet(item):
                    # A singlet state (or an unavailable doublet). Only the
                    # energies, when they were given, can be tabulated.
                    energies = singlet_energies(item)
                    rows.append([i + 1,"",
                                 energies[0] if len(energies) > 0 else None,
                                 energies[1] if len(energies) > 1 else None,
                                 None,None,None])
                    continue

                energies = state_energies(item)
                if energies is None:
                    first_energy  = None
                    second_energy = None
                else:
                    first_energy  = min(energies)
                    second_energy = max(energies)

                # The z principal value of the g-tensor, i.e. the largest
                # principal value; the transverse ones vanish by
                # Griffith's theorem.
                g_z = float(item.input_frame_g_tensor.eigenvalues[2])

                rows.append([i + 1,
                             "{0},{1}".format(item.states[0],item.states[1]),
                             first_energy,second_energy,
                             item.tunneling_gap,
                             g_z,
                             axis_angle(item)])

        if title is None:
            if kramers_system:
                table_title = "SUMMARY OF THE KRAMERS DOUBLETS"
            else:
                table_title = "SUMMARY OF THE PSEUDOSPIN DOUBLETS"
        else:
            table_title = title

        notes = ["The g-tensors and their principal magnetic axes are given in the",
                 frame_label + "."]
        if not kramers_system:
            notes.append("The transverse principal g values vanish by Griffith's theorem")
            notes.append("and are not tabulated.")

        return cls(rows,
                   column_headers=column_headers,
                   title=table_title,
                   notes=notes,
                   footnotes=[angle_note],
                   formats=formats,
                   table_type='pseudospin_doublets')


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
        from ouluspin import properties
        from ouluspin import units as unit_systems

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('ResultTable',test_name,
                                                       condition,print_output))

        # A simple table with a title, a note, a footnote and a summary
        # line.
        table = cls([[0,1.0,'a'],[1,2.5,'b']],
                    column_headers=['Index','Value','Label'],
                    title="TEST TABLE",
                    notes="A note.",
                    footnotes="A footnote.",
                    summary="A summary line.")

        table_str = table.string_table()

        check('the table renders', len(table_str) > 0)
        check('the string rendering is used by __repr__', repr(table) == table_str)
        check('the title is printed', "TEST TABLE" in table_str)
        check('the notes are printed', "A note." in table_str)
        check('the summary is printed', "A summary line." in table_str)
        check('the footnotes are labelled',
              "a) A footnote." in table_str)

        # A note or a footnote consisting of several lines is indented by
        # the renderer, the continuation lines of a footnote being aligned
        # with the text of its first line.
        multiline_table = cls([[1.0]],
                              column_headers=['Value'],
                              notes="First line.\nSecond line.",
                              footnotes="First line.\nSecond line.")
        multiline_str   = multiline_table.string_table()
        indent          = multiline_table.indent*" "

        check('a multi-line note is indented on every line',
              (indent + "First line." in multiline_str)
              and (indent + "Second line." in multiline_str))
        check('the continuation lines of a footnote are aligned',
              indent + "   Second line." in multiline_str)
        check('the column headers are printed',
              all(header in table_str for header in ['Index','Value','Label']))
        check('the default float format is used', "1.000000" in table_str)
        check('the horizontal rules are drawn',
              len([line for line in table_str.split("\n")
                   if line.strip() and set(line.strip()) == set('-')]) == 2)

        # The body lines of the table, i.e. the non-empty lines between the
        # first and the last horizontal rule of the rendering that are not
        # rules themselves.
        def body_lines(table_instance):
            lines = table_instance.string_table().split("\n")

            def is_rule(line):
                return bool(line.strip()) and set(line.strip()) <= set('-=')

            rule_indices = [i for i in range(0,len(lines)) if is_rule(lines[i])]

            return [line for line in lines[rule_indices[0]+1:rule_indices[-1]]
                    if line.strip() and (not is_rule(line))]

        check('all the rows are printed', len(body_lines(table)) == 2)

        # The explicit column formats and alignments.
        table = cls([[1.0,2.0]],
                    column_headers=['A','B'],
                    formats=['.2f','.4e'],
                    alignments='l')
        table_str = table.string_table()
        check('the explicit column formats are used',
              ("1.00" in table_str) and ("2.0000e+00" in table_str))
        check('the explicit alignment is used',
              any(line.startswith("1.00") for line in
                  [line.strip() for line in table_str.split("\n")]))

        # The row headers and the section and rule rows.
        table = cls([[1.0],"A section",[2.0],None,[3.0]],
                    column_headers=['Value'],
                    row_headers=['first','second','third'],
                    row_header_label='Row')
        table_str = table.string_table()
        check('the row headers are printed',
              all(header in table_str for header in ['first','second','third']))
        check('the row header label is printed', "Row" in table_str)
        check('the section headings are printed', "A section" in table_str)
        check('the rule rows are drawn',
              len([line for line in table_str.split("\n")
                   if line.strip() and set(line.strip()) == set('-')]) == 3)

        # The bare, machine-readable rendering and the data file.
        table = cls([[0,1.0],[1,2.0]],
                    column_headers=['Index','Value'],
                    title="TITLE",
                    notes="A note.")
        plain_str = table.string_table(plain=True)
        check('the bare rendering drops the title and the notes',
              (not "TITLE" in plain_str) and (not "A note." in plain_str))
        check('the bare rendering comments the header',
              plain_str.split("\n")[0].startswith("#"))
        check('the bare rendering has one line per row',
              len([line for line in plain_str.split("\n") if line.strip()]) == 3)

        filename = os.path.join(tempfile.gettempdir(),'ouluspin_result_table_test.dat')
        table.data_file(filename)
        f = open(filename)
        file_str = f.read()
        f.close()
        os.remove(filename)
        check('the data file contains the bare rendering', file_str == plain_str)

        # The table types and the footnote markers.
        check('the table types are listed',
              ('generic' in cls.table_types())
              and ('spherical_tensor' in cls.table_types()))
        check('the table type sets the indentation',
              cls([[1.0]],table_type='cartesian_tensor').indent == 4)
        check('the explicit indentation overrides the table type',
              cls([[1.0]],table_type='cartesian_tensor',indent=8).indent == 8)
        check('the table type sets the rule character',
              cls([[1.0]],table_type='transition_moments').rule_character == '=')
        check('the footnote markers are consecutive letters',
              (cls.footnote_marker(0) == 'a)') and (cls.footnote_marker(2) == 'c)'))

        # The compound table of pseudospin doublets. Two Kramers doublets
        # with mutually perpendicular Ising axes, at different energies.
        tmp_units        = unit_systems.EnergyUnitSystem('wavenumber')
        spin_matrix_list = debug_output.spin_matrices(1)

        axial_moment = [0.0*spin_matrix_list[0],
                        0.0*spin_matrix_list[1],
                        -2.0*tmp_units.mu_B*spin_matrix_list[2]]
        # The same doublet with the Ising axis rotated onto the x axis.
        rotated_moment = [-2.0*tmp_units.mu_B*spin_matrix_list[2],
                          0.0*spin_matrix_list[1],
                          0.0*spin_matrix_list[0]]

        ground_doublet = properties.PseudoSpinDoublet((0,1),axial_moment,tmp_units,
                                                      energies=[0.0,0.0],
                                                      kramers=True)
        excited_doublet = properties.PseudoSpinDoublet((0,1),rotated_moment,tmp_units,
                                                       energies=[100.0,100.0],
                                                       kramers=True)

        check('the doublet stores the energies of its states',
              np.allclose(ground_doublet.state_energies,[0.0,0.0]))

        table     = cls.pseudospin_doublet_compound_table([ground_doublet,
                                                           excited_doublet],
                                                          tmp_units)
        table_str = table.string_table()

        check('the compound table of Kramers doublets renders',
              len(table_str) > 0)
        check('the compound table of Kramers doublets has the g-tensor columns',
              all(header in table_str for header in ['g_x','g_y','g_z']))
        check('the compound table of Kramers doublets has one line per doublet',
              len(body_lines(table)) == 2)
        check('the compound table has a footnote on the angle',
              "a)" in table_str)
        check('the angle to the reference doublet vanishes',
              abs(table.rows[0][6]) < 1.0e-8)
        check('the angle between perpendicular magnetic axes is 90 degrees',
              abs(table.rows[1][6] - 90.0) < 1.0e-8)

        # A non-Kramers quasi-doublet together with a singlet state. The
        # quasi-doublet is split, so it is not a Kramers doublet and the
        # non-Kramers form of the table must be used. The moment is purely
        # off-diagonal within the doublet, as time-reversal symmetry
        # requires for two non-degenerate states.
        quasi_doublet_moment = [0.0*spin_matrix_list[0],
                                0.0*spin_matrix_list[1],
                                -2.0*tmp_units.mu_B*spin_matrix_list[0]]
        quasi_doublet = properties.PseudoSpinDoublet((0,1),quasi_doublet_moment,
                                                     tmp_units,
                                                     energies=[0.0,5.0])
        table     = cls.pseudospin_doublet_compound_table([quasi_doublet,200.0],
                                                          tmp_units)
        table_str = table.string_table()

        check('the compound table of non-Kramers doublets renders',
              len(table_str) > 0)
        check('the non-Kramers table has a tunneling gap column',
              "Gap" in table_str)
        check('the non-Kramers table tabulates only the axial g value',
              ("g_z" in table_str) and (not "g_x" in table_str))
        check('the non-Kramers table gives both state energies',
              (abs(table.rows[0][2] - 0.0) < 1.0e-8)
              and (abs(table.rows[0][3] - 5.0) < 1.0e-8))
        check('the tunneling gap is tabulated',
              abs(table.rows[0][4] - 5.0) < 1.0e-8)
        check('the singlet state is tabulated on its own line',
              (len(table.rows) == 2) and (abs(table.rows[1][2] - 200.0) < 1.0e-8))

        return debug_output.test_summary('ResultTable',result_list,print_output)
