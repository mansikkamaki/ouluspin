# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""Structures used to represent tables of calculated results.

The classes of this module separate the CONTENT of a result table (the
numbers, the headers and the explanatory texts) from its APPEARANCE (the
layout of the rendering that is asked for). The classes of the library that
report results construct an instance of ResultTable and return it; the
caller obtains the printable table simply by printing the instance or by
converting it to a str.

The structured representation is what makes the other output formats
possible: the same instance renders into the plain-text table of the
standard output of the library, into a LaTeX table and into an .odt or a
.docx document, without any of the classes that produce the results
knowing about the formats. The writers of the word-processor documents are
in the internal _documents module, which is imported only when they are
called.
"""

import sys

import numpy as np


class ResultTable:
    """A table of calculated results.

    An instance stores the table content as rows of cells together with the
    headers, the title and the explanatory texts belonging to the table. The
    appearance of the table is produced by the rendering methods: the
    plain-text renderer string_table(), which is also used by the __repr__
    dunder method, the LaTeX renderers latex_table() and
    latex_string_table(), and the word-processor renderers odt_table() and
    docx_table(). All of them use the same stored content.

    The word-processor renderers are written on the standard library only,
    so the library needs no external modules for them. The module writing
    the documents is imported when one of the renderers is called, which
    keeps it out of the ordinary use of the library.

    The renderers that are able to typeset their output, i.e. the LaTeX
    and the word-processor renderers, recognize the physical quantities
    stated by the headers of the columns and of the rows and set them the
    way they are meant to be read: the symbols are set in italics and the
    indices they carry as subscripts and superscripts, so that 'g_x' is
    set as a g with a subscript x and 'Re(X_k1q1)' with the two pairs of
    indices of the operator, whereas the unit of a header of the form
    'quantity / unit' is set upright. A Greek letter is written out by its
    name in the header, e.g. 'theta' or 'mu_B', and set as the letter. The
    markers of the footnotes are set as italic superscripts, and the
    negative numbers are written with the typographic minus sign in the
    word-processor documents and in the math mode of the LaTeX tables. The
    plain-text table prints all of these as they are given.

    A quantity whose symbol the plain-text table cannot print at all is
    named by a written-out word there and given as a dictionary stating
    both forms, e.g. {'text': 'Angle', 'typeset': 'theta'}; see the
    header_text class method.

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
        row_header_label. A header whose symbol the plain-text table
        cannot print is given as a dictionary stating both of its forms
        (see the header_text class method). Default is None, in which case
        no column headers are printed.
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
        lines, each of them indented by the renderer; the newlines are the
        manual wrapping of the plain-text table, and the renderings that
        wrap the text themselves set each note on a single line. Two
        separate notes are always set on separate lines. Default is None.
    footnotes : list of str or str or None
        Explanatory texts printed below the table. A text containing
        newlines is printed on several lines, the continuation lines being
        aligned with the text of the first one; as with the notes, the
        newlines are the manual wrapping of the plain-text table only. The
        footnotes are
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
    data_file(filename,plain=True,overwrite_file=False)
        Write the table into a text file, appending it to an existing
        file.
    odt_table(filename,overwrite_file=False)
        Write the table into an OpenDocument text (.odt) file, appending
        it to an existing document.
    docx_table(filename,overwrite_file=False)
        Write the table into an Office Open XML (.docx) file, appending
        it to an existing document.
    latex_table(filename,standalone=True,overwrite_file=False) : str or None
        Write the table into a LaTeX file, appending it to an existing
        document, or return it as a string when the filename is None.
    latex_string_table(standalone=True) : str
        Return the LaTeX rendering of the table as a string.

    Private methods
    ---------------
    __resolve_options(...)
        Fill in the options that were not given from the table type preset.
    __table_cells()
        Return the content of the table as the header lines and the rows
        of (text,is_number) pairs together with the alignments and the
        field widths of the columns.
    __columns()
        Return the content of the table in the column-wise form used by
        the plain-text renderer.
    __document_content()
        Return the content of the table in the renderer-independent form
        used by the word-processor and the LaTeX renderers.
    __format_cell(value,column_format)
        Return the printable string of a single cell.
    __is_number(value)
        Return whether the value of a cell is a number.
    __latex_number(text) : str
        Return the LaTeX rendering of a formatted number.
    __latex_text(text,markup=False) : str
        Return the LaTeX rendering of a text cell or of a header.
    __latex_cell(cell) : str
        Return the LaTeX rendering of one cell of the table.
    __latex_header(text) : str
        Return the LaTeX rendering of a column or a row header.
    __latex_quantity(atom) : str
        Return the LaTeX rendering of one physical quantity of a header.
    __latex_indices(group_list) : str
        Return the LaTeX rendering of the indices of a quantity.
    __latex_footnote_marker(letter) : str
        Return the LaTeX rendering of a footnote marker.

    Class methods
    -------------
    table_types() : list of str
        Return the names of the recognized table type presets.
    footnote_marker(index) : str
        Return the marker ('a)', 'b)', ...) of the footnote of the given
        index, to be appended to the header of the column the footnote
        explains.
    header_text(header,typeset=False) : str
        Return the text of a header, in the plain-text form or in the form
        used by the renderings that can typeset it.
    split_footnote_marker(text) : (str, str or None)
        Split a footnote marker off the end of a header and return the
        header and the letter of the marker.
    markup_atoms(text) : list of dict
        Split a header into its typesetting atoms, i.e. recognize the
        physical quantities it states.
    markup_segments(text) : list of dict
        Split a header into the formatted pieces of its rich-text
        rendering, used by the word-processor renderers.
    latex_escape(text) : str
        Return the text with the characters that are special in LaTeX
        escaped.
    pseudospin_doublet_compound_table(doublets,...) : ResultTable
        Return a compound table summarizing the properties of a list of
        pseudospin doublets of a system.
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if
        all tests passed.
    """

    # The typographic minus sign. The negative numbers of the renderings
    # that can typeset them are written with it instead of the hyphen the
    # plain-text table prints.
    MINUS_SIGN = "−"

    # The font of a table that is too wide for the page is made smaller, so
    # that the table still fits the width of the text. Each entry gives the
    # width of the table, in characters of the plain-text rendering, up to
    # which the font is scaled by the corresponding factor; a table wider
    # than the last entry uses the last factor. A table of some twenty
    # columns is still legible this way, and the reader of the document can
    # always turn the page instead.
    __FONT_SCALES = ((80,1.0),(100,0.9),(125,0.8),(150,0.7))

    # The Greek letters recognized in the headers of the tables. The
    # plain-text table cannot print them, so they are written out by their
    # names in the headers, and the renderings that are able to typeset
    # them set them as the letters. The names are also the names of the
    # corresponding LaTeX commands.
    GREEK_LETTERS = {
        'alpha':   "α", 'beta':    "β", 'gamma':   "γ", 'delta':   "δ",
        'epsilon': "ε", 'zeta':    "ζ", 'eta':     "η", 'theta':   "θ",
        'iota':    "ι", 'kappa':   "κ", 'lambda':  "λ", 'mu':      "μ",
        'nu':      "ν", 'xi':      "ξ", 'pi':      "π", 'rho':     "ρ",
        'sigma':   "σ", 'tau':     "τ", 'upsilon': "υ", 'phi':     "φ",
        'chi':     "χ", 'psi':     "ψ", 'omega':   "ω",
        'Gamma':   "Γ", 'Delta':   "Δ", 'Theta':   "Θ", 'Lambda':  "Λ",
        'Xi':      "Ξ", 'Pi':      "Π", 'Sigma':   "Σ", 'Phi':     "Φ",
        'Psi':     "Ψ", 'Omega':   "Ω",
    }

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
        'transition_moments':  {'float_format': '.6f', 'indent': 6, 'rule_character': '-'},
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


    def __is_number(self, value):
        """Return True when the value of a cell is a number rather than a
        string, an empty cell or a boolean. The renderers that typeset the
        numbers differently from the text use this to decide how a cell is
        printed.
        """
        if value is None or isinstance(value,str):
            return False

        if isinstance(value,(bool,np.bool_)):
            return False

        return isinstance(value,(int,float,complex,
                                 np.integer,np.floating,np.complexfloating))


    def __table_cells(self, typeset=False):
        """Return the content of the table in a column-wise form as a tuple

            (header_lines, body, alignments, widths)

        where header_lines is a list of lists of the formatted headers of
        the columns, body is the list of the rows with all cells converted
        into (text,is_number) pairs (the section headings and the rules
        being kept as they are), alignments is the list of the alignments
        of the columns and widths is the list of the field widths of the
        columns.

        The flag is_number of a cell tells whether the cell holds a number
        rather than a string. It is used by the renderers that typeset the
        numbers differently from the text, i.e. the LaTeX renderer, which
        sets the numbers in math mode.

        Optional arguments
        ------------------
        typeset : boolean
            Whether the headers are given in the form used by the renderers
            that are able to typeset them, i.e. whether a header that
            states a different form for them is taken in that form (see the
            header_text class method). Default is False, which gives the
            headers in the form used by the plain-text table.
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
                    cells.append((self.header_text(self.row_headers[row_index],
                                                   typeset),False))
                else:
                    cells.append(("",False))

            for i in range(0,len(row)):
                if self.formats is None or i >= len(self.formats):
                    column_format = None
                else:
                    column_format = self.formats[i]

                cells.append((self.__format_cell(row[i],column_format),
                              self.__is_number(row[i])))

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
            header_lines.append([self.header_text(header,typeset)
                                 for header in header_line])

        if self.row_headers is not None and len(header_lines) > 0:
            for header_line in header_lines:
                header_line.insert(0,"")
            header_lines[-1][0] = self.header_text(self.row_header_label,
                                                   typeset)

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

        # The field widths from the widest entry of each column. The
        # header lines hold plain strings and the body rows (text,is_number)
        # pairs.
        widths = self.n_columns*[0]

        for line in header_lines:
            for i in range(0,min(len(line),self.n_columns)):
                widths[i] = max(widths[i],len(line[i]))

        for line in body:
            if line is None or isinstance(line,str):
                continue
            for i in range(0,min(len(line),self.n_columns)):
                widths[i] = max(widths[i],len(line[i][0]))

        return header_lines, body, alignments, widths


    def __columns(self):
        """Return the content of the table in the column-wise form used by
        the plain-text renderer, i.e. as the tuple

            (header_lines, body, alignments, widths)

        returned by the __table_cells method with the cells of the body
        reduced from the (text,is_number) pairs to the plain strings.
        """
        header_lines, cell_body, alignments, widths = self.__table_cells()

        body = []
        for row in cell_body:
            if row is None or isinstance(row,str):
                body.append(row)
            else:
                body.append([cell[0] for cell in row])

        return header_lines, body, alignments, widths


    def __document_content(self):
        """Return the content of the table in the renderer-independent form
        used by the word-processor renderers of the _documents module and
        by the LaTeX renderer.

        The returned dictionary carries the texts belonging to the table
        together with its header lines, its rows and the layout information
        the renderers need; the items are documented in the docstring of
        the _documents module. The rows of the plain-text rendering that
        draw a horizontal rule are not rows of a typeset table, so they are
        converted into the flag rule_above of the row that follows them.
        """
        header_lines, cell_body, alignments, widths = \
            self.__table_cells(typeset=True)

        body       = []
        rule_above = False

        for row in cell_body:
            if row is None:
                rule_above = True
            elif isinstance(row,str):
                body.append({'type': 'section', 'text': row,
                             'rule_above': rule_above})
                rule_above = False
            else:
                body.append({'type': 'row', 'cells': list(row),
                             'rule_above': rule_above})
                rule_above = False

        return {'title':           self.title,
                'notes':           list(self.notes),
                'summary':         list(self.summary),
                'footnotes':       list(self.footnotes),
                'header_lines':    header_lines,
                'body':            body,
                'alignments':      alignments,
                'widths':          widths,
                'n_columns':       self.n_columns,
                'has_row_headers': self.row_headers is not None}


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

        # A section heading is separated from the rows above it by a blank
        # line, except when it directly follows a horizontal rule, i.e. the
        # rule below the column headers or a rule row of the table. There
        # the rule already separates the heading from what precedes it.
        after_rule = True

        for row in body:
            if row is None:
                tmp_str += rule
                after_rule = True
            elif isinstance(row,str):
                if not after_rule:
                    tmp_str += "\n"
                tmp_str += indent + row + "\n"
                after_rule = False
            else:
                tmp_str += indent + line_string(row) + "\n"
                after_rule = False

        tmp_str += rule

        if len(self.summary) > 0:
            tmp_str += "\n"
            for summary_line in self.summary:
                tmp_str += indent + summary_line + "\n"

        # The footnotes follow directly the line above them, i.e. the
        # closing rule of the table or the last summary line, without a
        # blank line in between.
        if len(self.footnotes) > 0:
            for i in range(0,len(self.footnotes)):
                marker = self.footnote_marker(i) + " "
                tmp_str += indented_text(self.footnotes[i],marker,
                                         len(marker)*" ")

        return tmp_str + "\n"


    def data_file(self, filename, plain=True, overwrite_file=False):
        """Write the table into a text file.

        When the file exists the table is appended to it, which makes it
        possible to collect several tables into one file by calling the
        method once per table.

        Arguments
        ---------
        filename : str
            The name of the file to write or to append to.

        Optional arguments
        ------------------
        plain : boolean
            Whether to write the bare, machine-readable form of the table
            (see string_table). Default is True.
        overwrite_file : boolean
            Whether an existing file is overwritten instead of the table
            being appended to it. Default is False.
        """
        if overwrite_file:
            f = open(filename, 'w')
        else:
            f = open(filename, 'a')

        f.write(self.string_table(plain=plain))
        f.close()


    def odt_table(self, filename, overwrite_file=False):
        """Write the table into an OpenDocument text (.odt) file.

        When the file exists the table is appended to the document,
        which makes it possible to collect several tables into one
        document by calling the method once per table. When the file does
        not exist a new document is created, laid out for an A4 page with
        margins of 2 cm and using the 11 pt font the tables are
        dimensioned for. The table always spans the full width of the text
        area, the width being divided between the columns in proportion to
        the widths they have in the plain-text rendering.

        The document is written using the standard library only, so no
        external modules are needed. The module writing it is imported
        when the method is called, not when ResultTable is imported.

        Arguments
        ---------
        filename : str
            The name of the file to write or to append to.

        Optional arguments
        ------------------
        overwrite_file : boolean
            Whether an existing file is overwritten by a new document
            instead of the table being appended to it. Default is False.
        """
        from ouluspin import _documents

        _documents.write_odt(self.__document_content(),filename,
                             overwrite_file=overwrite_file)


    def docx_table(self, filename, overwrite_file=False):
        """Write the table into an Office Open XML (.docx) file.

        The method is the .docx counterpart of odt_table and behaves in
        the same way: an existing file is appended to, a missing one is
        created with an A4 page layout and the 11 pt font, and the table
        spans the full width of the text area.

        Arguments
        ---------
        filename : str
            The name of the file to write or to append to.

        Optional arguments
        ------------------
        overwrite_file : boolean
            Whether an existing file is overwritten by a new document
            instead of the table being appended to it. Default is False.
        """
        from ouluspin import _documents

        _documents.write_docx(self.__document_content(),filename,
                              overwrite_file=overwrite_file)


    def __latex_number(self, text):
        """Return the LaTeX rendering of a formatted number, i.e. the
        number in math mode. A number in the exponent form is converted
        into a power of ten, e.g. '1.5000e-03' into '$1.5000 \\times
        10^{-3}$'.
        """
        exponent_position = text.lower().find('e')

        if exponent_position > 0:
            mantissa = text[:exponent_position]
            exponent = text[exponent_position+1:]

            # Drop the plus sign and the leading zeros of the exponent.
            sign = ""
            if exponent.startswith('-'):
                sign     = "-"
                exponent = exponent[1:]
            elif exponent.startswith('+'):
                exponent = exponent[1:]

            exponent = exponent.lstrip('0')
            if exponent == "":
                exponent = "0"
                sign     = ""

            return ("$" + mantissa + " \\times 10^{" + sign + exponent + "}$")

        return "$" + text + "$"


    def __latex_text(self, text, markup=False):
        """Return the LaTeX rendering of a text cell, of a header or of an
        explanatory line.

        The characters that are special in LaTeX are escaped. With markup
        set to True the physical quantities of the text are in addition
        recognized and typeset as such (see the markup_atoms class method),
        so that a header such as 'Re(X_k1q1)' is set as 'Re($X_{k_1,q_1}$)'
        and 'E / cm^-1' as '$E$ / $\\mathrm{cm}^{-1}$'.
        """
        if not markup:
            return self.latex_escape(text)

        tmp_str = ""

        for atom in self.markup_atoms(text):
            if atom['kind'] == 'text':
                tmp_str += self.latex_escape(atom['text'])
            elif atom['kind'] == 'number':
                # The math mode gives the number a proper minus sign.
                tmp_str += "$" + atom['text'] + "$"
            else:
                tmp_str += self.__latex_quantity(atom)

        return tmp_str


    def __latex_font_command(self, scale):
        """Return the LaTeX command that sets the font of a wide table, or
        an empty string when the table is set in the font of the running
        text.
        """
        if scale >= 1.0:
            return ""
        if scale >= 0.9:
            return "\\small"
        if scale >= 0.8:
            return "\\footnotesize"

        return "\\scriptsize"


    def __latex_footnote_marker(self, letter):
        """Return the LaTeX rendering of a footnote marker, i.e. the letter
        of the marker set as an italic superscript.
        """
        return "\\textsuperscript{\\textit{" + letter + "}}"


    def __latex_header(self, text):
        """Return the LaTeX rendering of a column header or of a row
        header. The physical quantities of the header are typeset as such
        and a footnote marker at its end is set as an italic superscript.
        """
        body, marker = self.split_footnote_marker(text)

        if marker is None:
            return self.__latex_text(text,markup=True)

        return (self.__latex_text(body,markup=True)
                + self.__latex_footnote_marker(marker))


    def __latex_quantity(self, atom):
        """Return the LaTeX rendering of one physical quantity of a header,
        i.e. its base symbol in math mode carrying its subscript and its
        superscript, e.g. '$X_{k_1,q_1}$' or '$\\mathrm{cm}^{-1}$'. The
        bars of a magnitude are drawn inside the math mode, where they are
        the vertical bars they are meant to be.
        """
        tmp_str = atom['base_latex']

        if atom['sub'] is not None:
            tmp_str += "_{" + self.__latex_indices(atom['sub']) + "}"

        # The bars of a magnitude enclose the symbol together with the
        # indices labelling it, but a power of the magnitude is taken of
        # the magnitude itself and stands outside the bars.
        if atom.get('bars',False):
            tmp_str = "|" + tmp_str + "|"

        if atom['sup'] is not None:
            tmp_str += "^{" + self.__latex_indices(atom['sup']) + "}"

        return "$" + tmp_str + "$"


    def __latex_indices(self, group_list):
        """Return the LaTeX rendering of the index groups of a subscript or
        of a superscript. An index that is itself a labelled quantity, such
        as the 'k1' of 'X_k1q1', carries its own subscript, so that the
        groups of 'k1q1' are rendered as 'k_{1},q_{1}'.
        """
        part_list = []

        for letters, digits in group_list:
            if letters == "":
                part_list.append(digits)
            elif digits == "":
                part_list.append(letters)
            else:
                part_list.append(letters + "_{" + digits + "}")

        return ",".join(part_list)


    def __latex_cell(self, cell):
        """Return the LaTeX rendering of one cell of the table, given as
        the (text,is_number) pair of the internal representation. The
        numbers are set in math mode and the text is escaped.
        """
        text, is_number = cell

        if text == "":
            return ""

        if is_number:
            return self.__latex_number(text)

        return self.__latex_text(text)


    def latex_table(self, filename, standalone=True, overwrite_file=False):
        """Write the table into a LaTeX file, or return it as a string.

        When the file exists the table is written at its end, before the
        \\end{document} line, which makes it possible to collect several
        tables into one document. When the file does not exist a new file
        is written, either a minimally compileable document containing the
        table or the table alone, depending on the standalone argument.

        The numbers of the table are set in math mode, as are the symbols
        carrying a subscript, such as the g_x of a column header. The
        horizontal rules are drawn with \\hline, so the table needs no
        additional LaTeX packages and can be pasted into any document.

        Arguments
        ---------
        filename : str or None
            The name of the file to write or to append to. When None, the
            table is not written but returned as a string, which is what
            the latex_string_table method does.

        Optional arguments
        ------------------
        standalone : boolean
            Whether a new file (or the returned string) is a minimally
            compileable LaTeX document instead of the table alone. The
            argument has no effect when the table is appended to an
            existing file. Default is True.
        overwrite_file : boolean
            Whether an existing file is overwritten by a new file instead
            of the table being written at the end of it. Default is False.
        """
        header_lines, cell_body, alignments, widths = \
            self.__table_cells(typeset=True)

        column_specification = ""
        for i in range(0,self.n_columns):
            if i < len(alignments) and alignments[i] in ('l','r','c'):
                column_specification += alignments[i]
            else:
                column_specification += 'r'

        def latex_row(cells, header=False):
            parts = []
            for i in range(0,self.n_columns):
                if i >= len(cells):
                    parts.append("")
                elif header:
                    parts.append(self.__latex_header(str(cells[i])))
                elif i == 0 and self.row_headers is not None:
                    # The leading cell of a row is the row header, which is
                    # typeset as the header it is.
                    parts.append(self.__latex_header(cells[i][0]))
                else:
                    parts.append(self.__latex_cell(cells[i]))
            return " & ".join(parts) + " \\\\\n"

        table_str = "\\begin{table}[htbp]\n\\centering\n"

        if self.title is not None:
            table_str += "\\caption{" + self.__latex_text(self.title) + "}\n"

        # The lines of a note, of a summary line or of a footnote are the
        # manual wrapping of the plain-text table. The typeset renderings
        # wrap the text themselves, so the lines of one entry are joined
        # into a single one and only the separate entries are set on
        # separate lines.
        def joined(text):
            return " ".join([line.strip() for line in text.split("\n")
                             if not line.strip() == ""])

        for note in self.notes:
            table_str += self.__latex_text(joined(note)) + " \\\\\n"

        # A table too wide for the page is set in a smaller font, inside a
        # group of its own so that the size of the caption and of the
        # explanatory texts is not changed.
        character_width = sum(widths) \
                          + max(0,self.n_columns - 1)*len(self.column_separator)
        size_command    = self.__latex_font_command(self.font_scale(character_width))

        if not size_command == "":
            # The space LaTeX leaves on both sides of every column is a
            # large part of the width of a table of many columns, so it is
            # narrowed along with the font. Both are set inside the group,
            # so the rest of the document keeps its own spacing.
            table_str += ("{" + size_command
                          + "\\setlength{\\tabcolsep}{2pt}\n")

        table_str += "\\begin{tabular}{" + column_specification + "}\n\\hline\n"

        for header_line in header_lines:
            table_str += latex_row(header_line,True)

        table_str += "\\hline\n"

        for row in cell_body:
            if row is None:
                table_str += "\\hline\n"
            elif isinstance(row,str):
                table_str += ("\\multicolumn{" + str(self.n_columns) + "}{l}{"
                              + self.__latex_text(row) + "} \\\\\n")
            else:
                table_str += latex_row(row)

        table_str += "\\hline\n\\end{tabular}\n"

        if not size_command == "":
            table_str += "}\n"

        for line in self.summary:
            table_str += "\\\\\n" + self.__latex_text(joined(line)) + "\n"

        # The footnotes are labelled with the marker letter set as an
        # italic superscript instead of the 'a)' of the plain-text table.
        for i in range(0,len(self.footnotes)):
            marker = self.__latex_footnote_marker(chr(ord('a') + i))
            table_str += ("\\\\\n" + marker + " "
                          + self.__latex_text(joined(self.footnotes[i])) + "\n")

        table_str += "\\end{table}\n"

        def standalone_document(body):
            return ("\\documentclass[11pt,a4paper]{article}\n"
                    "\\usepackage[a4paper,margin=2cm]{geometry}\n"
                    "\\begin{document}\n\n" + body + "\n\\end{document}\n")

        # An existing file is appended to, the table being written before
        # the end of the document.
        import os

        if filename is not None and os.path.exists(filename) \
           and not overwrite_file:
            f = open(filename)
            document = f.read()
            f.close()

            position = document.rfind("\\end{document}")
            if position < 0:
                document = document + "\n" + table_str
            else:
                document = document[:position] + table_str + document[position:]

            f = open(filename,'w')
            f.write(document)
            f.close()
            return None

        if standalone:
            document = standalone_document(table_str)
        else:
            document = table_str

        if filename is None:
            return document

        f = open(filename,'w')
        f.write(document)
        f.close()
        return None


    def latex_string_table(self, standalone=True):
        """Return the LaTeX rendering of the table as a string.

        Optional arguments
        ------------------
        standalone : boolean
            Whether the returned string is a minimally compileable LaTeX
            document instead of the table alone. Default is True.
        """
        return self.latex_table(None,standalone=standalone)


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

        The marker is written out in this form in the plain-text table. The
        renderers that can typeset it, i.e. the LaTeX and the
        word-processor renderers, set the letter alone as an italic
        superscript; they recognize the marker with the split_footnote_marker
        class method.
        """
        return chr(ord('a') + index) + ")"


    @classmethod
    def header_text(cls, header, typeset=False):
        """Return the text of a column or a row header.

        A header is ordinarily given as a str, which is used by all the
        renderings alike. A quantity whose symbol the plain-text table
        cannot print, such as an angle, is instead named by its written-out
        name there and by its symbol in the renderings that can typeset it.
        Such a header is given as the dictionary

            {'text': the plain-text name, 'typeset': the typeset name}

        e.g. {'text': 'Angle', 'typeset': 'theta'}, which the plain-text
        table prints as 'Angle' and the LaTeX and the word-processor
        renderings set as a theta. The Greek letters are written out by
        their names in the typeset form (see the markup_atoms class
        method).

        Optional arguments
        ------------------
        typeset : boolean
            Whether the typeset form of the header is returned instead of
            the plain-text one. Default is False.
        """
        if isinstance(header,dict):
            if typeset and 'typeset' in header:
                return str(header['typeset'])
            return str(header.get('text',""))

        return str(header)


    @classmethod
    def split_footnote_marker(cls, text):
        """Split a footnote marker off the end of a text and return the
        pair (body,marker), where the marker is the letter of the marker
        without its parenthesis, or None when the text carries no marker.

        The markers are written into the headers of the columns they
        explain by the routines that construct the tables, so that in the
        plain-text table they are a part of the header text. The renderers
        that set the markers as italic superscripts use this method to
        separate the marker from the header, e.g. 'Angle a)' into
        ('Angle','a').
        """
        import re

        match = re.match(r'^(.*?)[ ]*([a-z])\)$',str(text))

        if match is None:
            return str(text), None

        return match.group(1), match.group(2)


    @classmethod
    def markup_atoms(cls, text):
        """Split a header or a label into the list of its typesetting atoms
        and return the list.

        The tables of the library label their columns with the symbols of
        the physical quantities they hold, e.g. 'k1', 'S_0', 'g_x',
        'Re(X_k1q1)', '|C|^2' or 'E / cm^-1'. The plain-text table prints
        these as they are, but the LaTeX and the word-processor renderings
        are able to typeset them properly, and this method recognizes the
        quantities for them.

        Each atom of the returned list is a dictionary. A piece of ordinary
        text is the atom

            {'kind': 'text', 'text': the text}

        a header that states a plain number, such as the projection '-15/2'
        of a basis state or an energy, is the atom

            {'kind': 'number', 'text': the number}

        which the renderers set as a number, i.e. with the typographic
        minus sign of a negative value instead of the hyphen the plain-text
        table prints, and a physical quantity is the atom

            {'kind':       'quantity',
             'base':       the base symbol as (text,italic) pairs,
             'base_latex': the base symbol as LaTeX,
             'sub':        the subscript, or None,
             'sup':        the superscript, or None}

        where the subscript and the superscript are given as lists of index
        groups, each group being a (letters,digits) pair; the groups of the
        subscript of 'X_k1q1' are, for instance, [('k','1'),('q','1')].

        A base symbol of a single letter is a variable and is set in
        italics, whereas a base symbol of several letters is a unit, such
        as the 'cm' of 'cm^-1', and is set upright.
        """
        import re

        # A header that is a plain number, e.g. the projection '-15/2' or
        # '-1' of a basis state or an energy such as '105.1'. It is
        # recognized before anything else, since the solidus of a fraction
        # is not the solidus that separates a quantity from its unit.
        if re.match(r'^-?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?$',text.strip()) \
           and not text.strip() == "":
            return [{'kind': 'number', 'text': text.strip()}]

        def index_groups(index_text):
            """Split the text of a subscript or of a superscript into its
            index groups, e.g. 'k1q1' into [('k','1'),('q','1')] and '-1'
            into [('','-1')].
            """
            group_list = []
            for part in index_text.split(","):
                for match in re.finditer(r'([A-Za-z]*)(-?[0-9]*)',part):
                    if match.group(0) == "":
                        continue
                    group_list.append((match.group(1),match.group(2)))

            if len(group_list) == 0:
                return [(index_text,"")]

            return group_list

        def base_pieces(base_text):
            """Split a base symbol into (text,italic) pairs. The letters of
            a symbol of a single letter, and those inside the vertical bars
            of a magnitude such as '|C|', are variables and are set in
            italics; a symbol of several letters is a unit and is upright.
            """
            if base_text in cls.GREEK_LETTERS:
                return [(cls.GREEK_LETTERS[base_text],True)]

            if len(base_text) == 1 and base_text.isalpha():
                return [(base_text,True)]

            if base_text.startswith('|') and base_text.endswith('|'):
                inner = base_text[1:-1]
                if len(inner) == 1 and inner.isalpha():
                    return [("|",False),(inner,True),("|",False)]

            return [(base_text,False)]

        def base_latex(base_text):
            """Return the LaTeX rendering of a base symbol, the units being
            set upright with \\mathrm.
            """
            if base_text in cls.GREEK_LETTERS:
                return "\\" + base_text

            if len(base_text) == 1 and base_text.isalpha():
                return base_text

            if base_text.startswith('|') and base_text.endswith('|'):
                inner = base_text[1:-1]
                if len(inner) == 1 and inner.isalpha():
                    return "|" + inner + "|"

            if base_text.isalpha():
                return "\\mathrm{" + base_text + "}"

            return base_text

        def quantity(base_text, sub_text, sup_text, bars=False):
            if sub_text is None:
                sub = None
            else:
                sub = index_groups(sub_text)

            if sup_text is None:
                sup = None
            else:
                sup = index_groups(sup_text)

            return {'kind':       'quantity',
                    'base':       base_pieces(base_text),
                    'base_latex': base_latex(base_text),
                    'sub':        sub,
                    'sup':        sup,
                    'bars':       bars}

        # The recognized forms, in the order they are tried: a symbol or a
        # magnitude carrying a superscript ('cm^-1', '|C|^2'), a symbol
        # carrying a subscript ('S_0', 'X_k1q1'), a single letter followed
        # by an index ('k1'), and a single letter standing alone ('E').
        # A letter is taken to be a symbol only when it does not belong to
        # an ordinary word, which the surrounding letters tell, and when it
        # does not follow a number: a letter after a number is the unit of
        # that number, as the 'K' of the header '2.000 K', and is set
        # upright like the other units.
        # The Greek letters are written out by their names, which are
        # matched as whole words so that a name occurring inside an
        # ordinary word, such as the 'nu' of 'number', is not taken for a
        # symbol.
        # An index may itself consist of several parts separated by commas,
        # as in the 'mu_z,i' of the transition moment tables.
        index_str = r'[A-Za-z0-9]+(?:,[A-Za-z0-9]+)*'

        # A Greek letter may be followed directly by the symbol of another
        # quantity, as in the 'chiT' product of the susceptibility, which is
        # written without a space by the convention of the field. Such a
        # trailing symbol is a single capital letter, so it cannot be
        # confused with the continuation of an ordinary word: the 'nu' of
        # 'number' is still not taken for a symbol, since it is followed by
        # a lower-case letter.
        greek_str = ("(?<![A-Za-z])(?P<greek>"
                     + "|".join(sorted(cls.GREEK_LETTERS.keys(),
                                       key=len,reverse=True))
                     + r")(?:_(?P<greek_sub>" + index_str + r"))?"
                     r"(?P<greek_tail>[A-Z](?![A-Za-z0-9]))?(?![A-Za-z])")

        # The magnitude of a quantity, such as the '|X_k1q1|' of the tensor
        # tables or the '|mu_if|' of the transition moment tables, is
        # matched first, so that the bars end up around the whole quantity
        # together with the indices it carries.
        pattern = re.compile(greek_str +
                             r'|\|(?P<bar_base>[A-Za-z]+)'
                             r'(?:_(?P<bar_sub>' + index_str + r'))?\|'
                             r'(?:\^(?P<bar_sup>-?[A-Za-z0-9]+))?'
                             r'|(?P<sup_base>[A-Za-z]+|[0-9]+)'
                             r'\^(?P<sup>-?[A-Za-z0-9]+)'
                             r'|(?P<sub_base>[A-Za-z])_(?P<sub>' + index_str + r')'
                             r'|(?<![A-Za-z])(?P<index_base>[A-Za-z])'
                             r'(?P<index>[0-9]+)(?![A-Za-z0-9])'
                             r'|(?<![A-Za-z])(?<![0-9] )(?P<bare>[A-Za-z])'
                             r'(?![A-Za-z0-9])')

        # Within the unit of a header, i.e. after the solidus of a header
        # such as 'E / cm^-1' or 'B / T', the letters are the symbols of
        # the units and are set upright, so only the indices of the units
        # are recognized there. A unit named by a Greek letter, such as the
        # Bohr magneton 'mu_B', is recognized there as well.
        unit_pattern = re.compile(greek_str
                                  + r'|(?P<unit_base>[A-Za-z]+|[0-9]+)'
                                  + r'\^(?P<unit_sup>-?[A-Za-z0-9]+)')

        atom_list = []

        def add_text(piece):
            if piece == "":
                return
            if len(atom_list) > 0 and atom_list[-1]['kind'] == 'text':
                atom_list[-1]['text'] += piece
            else:
                atom_list.append({'kind': 'text', 'text': piece})

        def parse(part, unit):
            position = 0

            if unit:
                used_pattern = unit_pattern
            else:
                used_pattern = pattern

            for match in used_pattern.finditer(part):
                add_text(part[position:match.start()])

                def add_greek():
                    """Add the atom of a Greek letter and, when the letter
                    is followed directly by the symbol of another quantity
                    as in 'chiT', the atom of that symbol as well.
                    """
                    atom_list.append(quantity(match.group('greek'),
                                              match.group('greek_sub'),None))
                    if match.group('greek_tail') is not None:
                        atom_list.append(quantity(match.group('greek_tail'),
                                                  None,None))

                if unit:
                    # Only the units and the indices they carry are
                    # recognized within the unit of a header.
                    if match.group('greek') is not None:
                        add_greek()
                    else:
                        atom_list.append(quantity(match.group('unit_base'),None,
                                                  match.group('unit_sup')))
                elif match.group('greek') is not None:
                    add_greek()
                elif match.group('bar_base') is not None:
                    atom_list.append(quantity(match.group('bar_base'),
                                              match.group('bar_sub'),
                                              match.group('bar_sup'),
                                              bars=True))
                elif match.group('sup_base') is not None:
                    atom_list.append(quantity(match.group('sup_base'),None,
                                              match.group('sup')))
                elif match.group('sub_base') is not None:
                    atom_list.append(quantity(match.group('sub_base'),
                                              match.group('sub'),None))
                elif match.group('index_base') is not None:
                    atom_list.append(quantity(match.group('index_base'),
                                              match.group('index'),None))
                else:
                    atom_list.append(quantity(match.group('bare'),None,None))

                position = match.end()

            add_text(part[position:])

        # A header states the quantity and its unit as 'quantity / unit',
        # so everything after the first solidus belongs to the unit.
        solidus = text.find('/')

        if solidus < 0:
            parse(text,False)
        else:
            parse(text[:solidus],False)
            add_text(text[solidus:solidus+1])
            parse(text[solidus+1:],True)

        return atom_list


    @classmethod
    def markup_segments(cls, text):
        """Split a header or a label into the list of the typeset pieces of
        its rich-text rendering and return the list. The method is used by
        the word-processor renderers, which build the cells of their tables
        out of runs of differently formatted text.

        Each segment of the returned list is the dictionary

            {'text':     the text of the segment,
             'italic':   whether the text is set in italics,
             'position': 'normal', 'sub' or 'super'}

        The quantities recognized by the markup_atoms class method are
        split into their base symbols and the indices carried by them, e.g.
        'E / cm^-1' into the italic 'E', the upright ' / cm' and the
        superscript '-1'.
        """
        segment_list = []

        def add(text_piece, italic, position):
            if text_piece == "":
                return
            segment_list.append({'text':     text_piece,
                                 'italic':   italic,
                                 'position': position})

        def add_indices(group_list, position):
            for i in range(0,len(group_list)):
                letters, digits = group_list[i]
                if i > 0:
                    add(",",False,position)
                add(letters,True,position)
                add(digits,False,position)

        for atom in cls.markup_atoms(text):
            if atom['kind'] == 'text':
                add(atom['text'],False,'normal')
                continue

            if atom['kind'] == 'number':
                # A negative number is written with the typographic minus
                # sign rather than with the hyphen of the plain-text table.
                add(atom['text'].replace("-",cls.MINUS_SIGN),False,'normal')
                continue

            if atom.get('bars',False):
                add("|",False,'normal')

            for piece, italic in atom['base']:
                add(piece,italic,'normal')

            if atom['sub'] is not None:
                add_indices(atom['sub'],'sub')

            if atom.get('bars',False):
                add("|",False,'normal')

            if atom['sup'] is not None:
                add_indices(atom['sup'],'super')

        return segment_list


    @classmethod
    def font_scale(cls, character_width):
        """Return the factor the font of a table of the given width is
        scaled by in the renderings that typeset it.

        A table of many columns, such as the composition of the eigenstates
        of a J multiplet, does not fit the width of a page in the font of
        the running text. Rather than let it run over the margin, the
        renderings set a wide table in a smaller font, in which it still
        reads well; whether to keep it that way or to turn the table onto a
        page of its own is left to whoever writes the document.

        Arguments
        ---------
        character_width : int
            The width of the table in the characters of the plain-text
            rendering, i.e. the width of its widest line.
        """
        for limit, scale in cls.__FONT_SCALES:
            if character_width <= limit:
                return scale

        return cls.__FONT_SCALES[-1][1]


    @classmethod
    def latex_escape(cls, text):
        """Return the text with the characters that are special in LaTeX
        escaped.
        """
        tmp_text = str(text).replace("\\","\\textbackslash{}")
        for character in ['&','%','$','#','_','{','}']:
            tmp_text = tmp_text.replace(character,"\\" + character)
        tmp_text = tmp_text.replace("~","\\textasciitilde{}")
        tmp_text = tmp_text.replace("^","\\textasciicircum{}")

        # The vertical bar and the angle brackets are not themselves in the
        # text mode of LaTeX, where they come out as a dash and as inverted
        # punctuation marks, so they are written with the commands that
        # produce them.
        tmp_text = tmp_text.replace("|","\\textbar{}")
        tmp_text = tmp_text.replace("<","\\textless{}")
        tmp_text = tmp_text.replace(">","\\textgreater{}")

        return tmp_text


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

        # The plain-text table names the angle, which it cannot print as a
        # symbol, whereas the renderings that can typeset it set it as the
        # theta it is.
        angle_marker = cls.footnote_marker(0)
        angle_header = {'text':    "Angle " + angle_marker,
                        'typeset': "theta " + angle_marker}

        rows = []

        if kramers_system:
            column_headers = [["Doublet","States","E / " + energy_unit,
                               "g_x","g_y","g_z",angle_header]]
            # The energies are given with one decimal, which is the
            # accuracy of the quantum-chemical calculations they come
            # from. The tunneling gap of the non-Kramers table below is a
            # small difference of two energies and is a relative quantity
            # of a higher accuracy, so it keeps its own format.
            formats        = [None,None,'.1f','.4f','.4f','.4f','.2f']

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
            formats        = [None,None,'.1f','.1f','.6e','.4f','.2f']

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

        # A note is one entry even when it is wrapped by hand for the
        # plain-text table; the renderings that wrap the text themselves
        # set it on a single line.
        notes = ["The g-tensors and their principal magnetic axes are given in the\n"
                 + frame_label + "."]
        if not kramers_system:
            notes.append("The transverse principal g values vanish by Griffith's "
                         "theorem\nand are not tabulated.")

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
        if os.path.exists(filename):
            os.remove(filename)

        table.data_file(filename)
        f = open(filename)
        file_str = f.read()
        f.close()
        check('the data file contains the bare rendering', file_str == plain_str)

        # A second table is appended to the file, unless the file is
        # overwritten.
        table.data_file(filename)
        f = open(filename)
        file_str = f.read()
        f.close()
        check('a second table is appended to the data file',
              file_str == plain_str + plain_str)

        table.data_file(filename,overwrite_file=True)
        f = open(filename)
        file_str = f.read()
        f.close()
        os.remove(filename)
        check('the data file is overwritten with overwrite_file',
              file_str == plain_str)

        # The LaTeX rendering. The numbers are set in math mode, as are the
        # symbols carrying a subscript or a superscript.
        latex_table_instance = cls([[1,-3.56,1.5e-3,'text']],
                                   column_headers=['Re(X_k1q1)','g_x',
                                                   'E / cm^-1','Angle a)'],
                                   title="LATEX TEST",
                                   footnotes="A footnote.",
                                   formats=[None,'.2f','.4e',None])
        latex_str = latex_table_instance.latex_string_table(standalone=False)

        check('the LaTeX rendering builds a tabular environment',
              ("\\begin{tabular}" in latex_str)
              and ("\\end{tabular}" in latex_str))
        check('the LaTeX rendering sets the numbers in math mode',
              "$-3.56$" in latex_str)
        check('the LaTeX rendering writes the exponents as powers of ten',
              "$1.5000 \\times 10^{-3}$" in latex_str)
        check('the LaTeX rendering sets the subscripts in math mode',
              "$g_{x}$" in latex_str)
        check('the LaTeX rendering sets the units upright with a superscript',
              "$\\mathrm{cm}^{-1}$" in latex_str)
        check('the LaTeX rendering carries the title as the caption',
              "\\caption{LATEX TEST}" in latex_str)
        check('the LaTeX rendering labels the footnotes with a superscript',
              "\\textsuperscript{\\textit{a}} A footnote." in latex_str)
        check('the LaTeX rendering marks the footnoted header with a superscript',
              "Angle\\textsuperscript{\\textit{a}}" in latex_str)
        check('the LaTeX rendering sets a bare symbol in math mode',
              "$E$" in latex_str)
        check('the LaTeX rendering sets the indices of a symbol',
              "$X_{k_{1},q_{1}}$" in latex_str)
        check('the bare LaTeX rendering has no preamble',
              not "\\documentclass" in latex_str)
        check('the standalone LaTeX rendering is a complete document',
              ("\\documentclass" in latex_table_instance.latex_string_table())
              and ("\\end{document}" in latex_table_instance.latex_string_table()))

        # The recognition of the physical quantities of the headers. The
        # symbols are variables and are set in italics, whereas the units
        # of a header of the form 'quantity / unit' are set upright.
        def segment_of(text,piece):
            for segment in cls.markup_segments(text):
                if segment['text'] == piece:
                    return segment
            return None

        def latex_markup(text):
            """The LaTeX rendering of a header, as the renderer sets it."""
            return cls([[1.0]])._ResultTable__latex_text(text,markup=True)

        check('a symbol carrying an index is split into its parts',
              (segment_of('S_0','S')['italic'] is True)
              and (segment_of('S_0','0')['position'] == 'sub'))
        check('a symbol followed by an index is recognized',
              (segment_of('k1','k')['italic'] is True)
              and (segment_of('k1','1')['position'] == 'sub'))
        check('a bare symbol is set in italics',
              segment_of('E / cm^-1','E')['italic'] is True)
        check('the unit of a header is set upright',
              (segment_of('E / cm^-1','cm')['italic'] is False)
              and (segment_of('B / T',' / T')['italic'] is False))
        check('the index of a unit is a superscript',
              segment_of('E / cm^-1','-1')['position'] == 'super')
        check('the indices of a symbol are separated',
              [segment['text'] for segment in cls.markup_segments('X_k1q1')]
              == ['X','k','1',',','q','1'])
        check('an ordinary word is not taken for a symbol',
              cls.markup_segments('Doublet')[0]['italic'] is False)
        check('a footnote marker is split off a header',
              cls.split_footnote_marker('Angle a)') == ('Angle','a'))
        check('a header without a marker is left alone',
              cls.split_footnote_marker('Doublet') == ('Doublet',None))

        # A Greek letter is written out by its name in the header and set
        # as the letter by the renderings that can typeset it.
        check('a Greek letter is set as the letter',
              segment_of('theta','θ')['italic'] is True)
        check('a Greek letter name inside a word is not a symbol',
              cls.markup_segments('number')[0]['text'] == 'number')
        check('a Greek letter carries its indices',
              [segment['text'] for segment in cls.markup_segments('mu_B')]
              == ['μ','B'])
        check('an index of several parts is kept together',
              latex_markup('mu_z,i') == "$\\mu_{z,i}$")

        # A header that states a plain number, e.g. the projection of a
        # basis state, is set as a number, so that a negative value carries
        # the typographic minus sign instead of a hyphen. The solidus of a
        # fraction is not the solidus that separates a quantity from its
        # unit.
        check('a fraction is set as a number',
              latex_markup('-15/2') == "$-15/2$")
        check('an integer is set as a number',
              latex_markup('-1') == "$-1$")
        check('a decimal number is set as a number',
              latex_markup('105.1') == "$105.1$")
        check('a negative number carries the minus sign',
              cls.markup_segments('-15/2')[0]['text'] == cls.MINUS_SIGN + "15/2")
        check('a positive number is left alone',
              cls.markup_segments('15/2')[0]['text'] == "15/2")
        check('a unit is still told from a fraction',
              latex_markup('E / cm^-1') == "$E$ / $\\mathrm{cm}^{-1}$")

        # The characters that are not themselves in the text mode of LaTeX
        # are written with the commands that produce them; a bra-ket of a
        # note would otherwise come out as inverted punctuation marks.
        check('the angle brackets are escaped',
              cls.latex_escape("<J,M|i>")
              == "\\textless{}J,M\\textbar{}i\\textgreater{}")

        # A letter following a number is the unit of that number, not a
        # symbol, so it is set upright.
        check('a unit following a number is set upright',
              latex_markup('2.000 K') == "2.000 K")
        check('a symbol is still set in italics',
              latex_markup('E') == "$E$")
        check('a unit following a number is upright in the documents',
              all(segment['italic'] is False
                  for segment in cls.markup_segments('2.000 K')))

        # A table too wide for the page is set in a smaller font.
        check('a narrow table is set in the font of the text',
              cls.font_scale(60) == 1.0)
        check('a wide table is set in a smaller font',
              cls.font_scale(110) < 1.0)
        check('a very wide table is set in the smallest font',
              cls.font_scale(400) == cls.font_scale(150))

        narrow_latex = cls([[1.0]],column_headers=['Value']) \
                       .latex_string_table(standalone=False)
        wide_latex   = cls([20*[1.0]],column_headers=[str(i) for i
                                                      in range(0,20)]) \
                       .latex_string_table(standalone=False)

        check('the LaTeX rendering of a narrow table sets no font size',
              not "\\footnotesize" in narrow_latex)
        check('the LaTeX rendering of a wide table sets a smaller font',
              ("\\footnotesize" in wide_latex)
              or ("\\scriptsize" in wide_latex))
        check('the LaTeX rendering of a wide table narrows the columns',
              "\\tabcolsep" in wide_latex)
        check('the smaller font is set inside a group of its own',
              wide_latex.count("{") == wide_latex.count("}"))
        check('the bars of a magnitude enclose the whole quantity',
              latex_markup('|mu_if|') == "$|\\mu_{if}|$")
        check('a power of a magnitude stands outside the bars',
              latex_markup('|C|^2') == "$|C|^{2}$")

        # A header that the plain-text table cannot print as a symbol is
        # given in both forms, and each rendering takes the one it can set.
        two_form_header = {'text': 'Angle', 'typeset': 'theta'}
        check('the plain-text form of a header is the default',
              cls.header_text(two_form_header) == 'Angle')
        check('the typeset form of a header is used when asked for',
              cls.header_text(two_form_header,typeset=True) == 'theta')
        check('an ordinary header has one form only',
              (cls.header_text('Doublet') == 'Doublet')
              and (cls.header_text('Doublet',typeset=True) == 'Doublet'))

        two_form_table = cls([[1.0]],column_headers=[two_form_header])
        check('the plain-text table prints the written-out header',
              ("Angle" in two_form_table.string_table())
              and (not "theta" in two_form_table.string_table()))
        check('the LaTeX table sets the header as the symbol',
              "$\\theta$" in two_form_table.latex_string_table(standalone=False))

        # The manual wrapping of a note or of a footnote belongs to the
        # plain-text table; the renderings that wrap the text themselves
        # set each entry on a single line.
        wrapped_table = cls([[1.0]],
                            column_headers=['Value'],
                            notes="A note that is\nwrapped by hand.",
                            footnotes="A footnote that is\nwrapped by hand.")
        wrapped_latex = wrapped_table.latex_string_table(standalone=False)

        check('the plain-text table keeps the manual wrapping',
              "A note that is\n" in wrapped_table.string_table())
        check('the LaTeX table joins the lines of a note',
              "A note that is wrapped by hand." in wrapped_latex)
        check('the LaTeX table joins the lines of a footnote',
              "A footnote that is wrapped by hand." in wrapped_latex)
        check('the LaTeX table breaks the line only between the entries',
              wrapped_latex.count("wrapped by hand. \\\\")
              + wrapped_latex.count("wrapped by hand.\n") == 2)

        # The LaTeX file, and the appending of a second table to it.
        latex_name = os.path.join(tempfile.gettempdir(),'ouluspin_result_table_test.tex')
        if os.path.exists(latex_name):
            os.remove(latex_name)

        latex_table_instance.latex_table(latex_name)
        latex_table_instance.latex_table(latex_name)

        f = open(latex_name)
        latex_file_str = f.read()
        f.close()

        check('the LaTeX file is a compileable document',
              latex_file_str.count("\\documentclass") == 1)
        check('a second table is appended to the LaTeX file',
              latex_file_str.count("\\begin{tabular}") == 2)
        check('the appended table is written before the end of the document',
              latex_file_str.rfind("\\begin{tabular}")
              < latex_file_str.rfind("\\end{document}"))

        latex_table_instance.latex_table(latex_name,overwrite_file=True)

        f = open(latex_name)
        latex_file_str = f.read()
        f.close()
        os.remove(latex_name)

        check('the LaTeX file is overwritten with overwrite_file',
              latex_file_str.count("\\begin{tabular}") == 1
              and latex_file_str.count("\\documentclass") == 1)

        # The word-processor renderings. Both formats are ZIP archives of
        # XML documents, so the test checks that the archive holds the
        # expected entries, that they are well-formed XML and that a second
        # table is appended to an existing document.
        import xml.dom.minidom
        import zipfile

        for extension, entry_list, table_element in \
                (('.odt',['mimetype','content.xml','styles.xml',
                          'META-INF/manifest.xml'],'<table:table '),
                 ('.docx',['[Content_Types].xml','word/document.xml',
                           'word/styles.xml'],'<w:tbl>')):
            document_name = os.path.join(tempfile.gettempdir(),
                                         'ouluspin_result_table_test' + extension)
            if os.path.exists(document_name):
                os.remove(document_name)

            if extension == '.odt':
                latex_table_instance.odt_table(document_name)
                latex_table_instance.odt_table(document_name)
                content_entry = 'content.xml'
            else:
                latex_table_instance.docx_table(document_name)
                latex_table_instance.docx_table(document_name)
                content_entry = 'word/document.xml'

            archive   = zipfile.ZipFile(document_name,'r')
            name_list = archive.namelist()
            document  = archive.read(content_entry).decode('utf-8')
            archive.close()

            # The document is written anew when the file is overwritten,
            # so it holds the single table instead of the two of the
            # document written above.
            if extension == '.odt':
                latex_table_instance.odt_table(document_name,overwrite_file=True)
            else:
                latex_table_instance.docx_table(document_name,overwrite_file=True)

            archive             = zipfile.ZipFile(document_name,'r')
            overwritten_document = archive.read(content_entry).decode('utf-8')
            archive.close()
            os.remove(document_name)

            check('the ' + extension
                  + ' document is overwritten with overwrite_file',
                  overwritten_document.count(table_element) == 1)

            check('the ' + extension + ' document holds the expected entries',
                  all(entry in name_list for entry in entry_list))

            well_formed = True
            try:
                xml.dom.minidom.parseString(document)
            except Exception:
                well_formed = False

            check('the ' + extension + ' document is well-formed XML',well_formed)
            check('a second table is appended to the ' + extension + ' document',
                  document.count(table_element) == 2)
            check('the ' + extension + ' document carries the table content',
                  ("LATEX TEST" in document) and ("3.56" in document))
            check('the ' + extension
                  + ' document writes the negative numbers with a minus sign',
                  ("−3.56" in document) and (not "-3.56" in document))

            if extension == '.odt':
                subscript_markup   = 'style:text-position="sub'
                superscript_markup = 'style:text-position="super'
                italic_markup      = 'fo:font-style="italic"'
            else:
                subscript_markup   = '<w:vertAlign w:val="subscript"/>'
                superscript_markup = '<w:vertAlign w:val="superscript"/>'
                italic_markup      = '<w:i/>'

            check('the ' + extension + ' document sets the subscripts',
                  subscript_markup in document)
            check('the ' + extension + ' document sets the superscripts',
                  superscript_markup in document)
            check('the ' + extension + ' document sets the symbols in italics',
                  italic_markup in document)

            # The document wraps the text of a paragraph itself, so a note
            # or a footnote wrapped by hand is set on a single line.
            wrapped_name = os.path.join(tempfile.gettempdir(),
                                        'ouluspin_result_table_wrap' + extension)
            if os.path.exists(wrapped_name):
                os.remove(wrapped_name)

            if extension == '.odt':
                wrapped_table.odt_table(wrapped_name)
            else:
                wrapped_table.docx_table(wrapped_name)

            archive         = zipfile.ZipFile(wrapped_name,'r')
            wrapped_document = archive.read(content_entry).decode('utf-8')
            archive.close()
            os.remove(wrapped_name)

            check('the ' + extension + ' document joins the lines of a note',
                  "A note that is wrapped by hand." in wrapped_document)
            check('the ' + extension + ' document joins the lines of a footnote',
                  "A footnote that is wrapped by hand." in wrapped_document)

        # The table types and the footnote markers.
        check('the table types are listed',
              ('generic' in cls.table_types())
              and ('spherical_tensor' in cls.table_types()))
        check('the table type sets the indentation',
              cls([[1.0]],table_type='cartesian_tensor').indent == 4)
        check('the explicit indentation overrides the table type',
              cls([[1.0]],table_type='cartesian_tensor',indent=8).indent == 8)
        check('the table type sets the rule character',
              cls([[1.0]],table_type='cartesian_tensor').rule_character == '-')
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
        check('the compound table gives the energies with one decimal',
              "100.0" in table_str)
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
