# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""Internal writers of the images of a result plot.

The module draws the plots of the result_plot module and writes them on
disk as PNG, TIFF and PDF files. The drawing is done with Matplotlib,
which is the only part of the library needing a plotting library; the
module is imported by the writing methods of ResultPlot when they are
called, so that the ordinary use of the library never loads it and the
library can be used without Matplotlib installed.

The module is internal to the library and is not part of the public API.

The writers take the content of the plot as the dictionary returned by the
private __plot_content method of ResultPlot, which contains the following
items:

    plot_type   : the kind of the plot, i.e. 'standard_plot',
                  'effective_barrier' or 'energy_level_diagram'
    data_sets   : the data sets of a standard plot, each a dictionary with
                  the items 'x', 'y', 'label' and 'style'
    levels      : the states of an effective barrier, each a dictionary
                  with the items 'moment' and 'energy'
    transitions : the transitions of an effective barrier, each a
                  dictionary with the items 'from', 'to' and 'moment', the
                  indices referring to the levels
    columns     : the energy level structures of an energy level diagram,
                  each a dictionary with the items 'energies' and 'label'
    x_label     : the label of the horizontal axis
    y_label     : the label of the vertical axis
    title       : the title of the plot, or None
    legend      : whether the legends of the data sets are drawn
"""

import sys


# The message printed when Matplotlib is not installed. The plots are the
# only part of the library needing it, so the user is told how to get it
# rather than being left with an import error.
__MATPLOTLIB_MESSAGE = (
    "The writing of the plots needs the Matplotlib library, which was not\n"
    "found. Install it into the Python interpreter that runs the library,\n"
    "e.g. with\n\n"
    "    ouluspin-python -m pip install --user matplotlib\n\n"
    "The TIFF files need in addition the Pillow library, which Matplotlib\n"
    "ordinarily installs along with itself."
)

# The colours and the widths of the drawing. The effective barrier is drawn
# in the style established in the literature: black bars for the states and
# red arrows for the transitions.
LEVEL_COLOR      = 'black'
TRANSITION_COLOR = 'red'
LEVEL_WIDTH      = 2.5
LEVEL_HALF_WIDTH = 0.35

# The bars of an energy level diagram, in units of the spacing of the
# columns. The bars are narrow, as is usual in an energy level diagram,
# and the bars of a degenerate group are drawn beside each other with a
# gap between them, so that the degeneracy can be read off the diagram.
DIAGRAM_LEVEL_HALF_WIDTH = 0.11
DIAGRAM_LEVEL_GAP        = 0.05

# The width of the curves of a standard plot, in points, and the size of
# the markers drawn on them. The curves are drawn heavy enough to stay
# clearly visible when the figure is scaled down to the width of a column
# of a publication.
DATA_LINE_WIDTH = 2.0
MARKER_SIZE     = 4.5

# The width of the arrows of an effective barrier, in points, between the
# weakest and the strongest transition drawn.
MIN_ARROW_WIDTH = 0.4
MAX_ARROW_WIDTH = 4.0

# The transitions weaker than this fraction of the strongest one are not
# drawn at all, so that the plot does not fill with arrows that carry no
# information.
ARROW_THRESHOLD = 1.0e-3

# The appearance of the plots. The plots are meant for the figures of a
# publication, so a serif font is used and the frame and the ticks are
# drawn heavy enough to stay visible when the figure is scaled down to the
# width of a column.
PLOT_STYLE = {
    'font.family':         'serif',
    'font.serif':          ['STIXGeneral','DejaVu Serif'],
    'mathtext.fontset':    'stix',
    'font.size':           11.0,
    'axes.linewidth':      1.2,
    'axes.labelsize':      12.0,
    'xtick.labelsize':     11.0,
    'ytick.labelsize':     11.0,
    'xtick.direction':     'in',
    'ytick.direction':     'in',
    'xtick.top':           True,
    'ytick.right':         True,
    'xtick.major.width':   1.2,
    'ytick.major.width':   1.2,
    'xtick.major.size':    5.0,
    'ytick.major.size':    5.0,
    'legend.fontsize':     10.0,
    'savefig.bbox':        'tight',
    'savefig.pad_inches':  0.05,
}


def pyplot():
    """Return the pyplot module of Matplotlib, with the backend that needs
    no display selected. A RuntimeError carrying the installation
    instructions is raised when Matplotlib is not installed.
    """
    try:
        import matplotlib
    except ImportError:
        raise RuntimeError(__MATPLOTLIB_MESSAGE)

    # The plots are written into files, never shown on the screen, so the
    # backend that needs no display is used. It also lets the library be
    # used over a connection without graphics.
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    return plt


def axis_label(text):
    """Return the label of an axis as the mathematical text of Matplotlib.

    The labels state the physical quantities the way the headers of the
    result tables do, e.g. 'E / cm^-1' or 'chiT / cm^3 K mol^-1', and the
    markup of ResultTable recognizes the quantities they name. The pieces
    are turned into the mathematical text of Matplotlib, so that the
    symbols are set in italics, the indices as subscripts and superscripts
    and the Greek letters as the letters.
    """
    from ouluspin.result_table import ResultTable

    if text == "":
        return ""

    segment_list = [segment for segment in ResultTable.markup_segments(text)
                    if not segment['text'] == ""]

    part_list = []
    i         = 0

    while i < len(segment_list):
        base = segment_list[i]
        i   += 1

        # The indices belong to the symbol before them and have to be set
        # inside the same mathematical group as it, since a subscript
        # standing on its own is not a valid mathematical text.
        index_list = []
        while i < len(segment_list) \
              and segment_list[i]['position'] in ('sub','super'):
            index_list.append(segment_list[i])
            i += 1

        if len(index_list) == 0 and not base['italic']:
            # Ordinary text needs no mathematical text at all.
            part_list.append(base['text'])
            continue

        body = mathtext_symbol(base['text'])

        if not base['italic']:
            body = "\\mathrm{" + body + "}"

        for index in index_list:
            if index['position'] == 'sub':
                operator = "_"
            else:
                operator = "^"
            body += operator + "{" + mathtext_symbol(index['text']) + "}"

        part_list.append("$" + body + "$")

    return "".join(part_list)


def mathtext_symbol(text):
    """Return a symbol as the mathematical text of Matplotlib. The Greek
    letters are written out by the names of the corresponding commands, as
    in LaTeX.
    """
    from ouluspin.result_table import ResultTable

    for name, letter in ResultTable.GREEK_LETTERS.items():
        if text == letter:
            return "\\" + name

    # The minus sign of the indices is the one of the mathematical text.
    return text.replace("−","-")


def draw_standard_plot(axes, content):
    """Draw the data sets of a standard plot into the given axes.

    The labels of the data sets name physical quantities the same way the
    axis labels do, e.g. 'T = 1.800 K', so they are typeset by axis_label
    and the symbols end up in italics in the legend as well.
    """
    for data_set in content['data_sets']:
        style = data_set.get('style','line')

        if style == 'points':
            line_style   = 'none'
            marker_style = 'o'
        elif style == 'line_points':
            line_style   = '-'
            marker_style = 'o'
        else:
            line_style   = '-'
            marker_style = None

        label = data_set.get('label',"")
        if label == "":
            label = None
        else:
            label = axis_label(label)

        axes.plot(data_set['x'],data_set['y'],
                  linestyle=line_style,
                  marker=marker_style,
                  markersize=MARKER_SIZE,
                  linewidth=DATA_LINE_WIDTH,
                  label=label)

    # The susceptibility is reported as the chi*T product, which is drawn
    # from zero up by the convention of the field, so that the curves of
    # different compounds can be compared by eye.
    if content.get('y_from_zero',False):
        axes.set_ylim(bottom=0.0)


def degenerate_groups(energy_list, tolerance):
    """Return the energies grouped by degeneracy, as a list of lists of the
    energies of one group. The energies are sorted first, and a group is
    closed when the next energy lies further than the tolerance from the
    first energy of the group.
    """
    group_list = []

    for energy in sorted(energy_list):
        if len(group_list) > 0 and abs(energy - group_list[-1][0]) <= tolerance:
            group_list[-1].append(energy)
        else:
            group_list.append([energy])

    return group_list


def draw_energy_level_diagram(axes, content):
    """Draw the energy level structures of an energy level diagram into the
    given axes. Each structure is a column of horizontal bars and the
    horizontal axis is discrete.

    The states of a degenerate group are drawn beside each other instead of
    on top of each other, so that a doublet is seen to be a doublet rather
    than a single bar; the group stays centred on its column.
    """
    tick_positions = []
    tick_labels    = []

    tolerance = content.get('degeneracy_tolerance',0.0)

    for i in range(0,len(content['columns'])):
        column = content['columns'][i]

        for group in degenerate_groups(column['energies'],tolerance):
            # The bars of a group are laid side by side and the whole group
            # is centred on the column.
            bar_width   = 2.0*DIAGRAM_LEVEL_HALF_WIDTH
            group_width = len(group)*bar_width \
                          + (len(group) - 1)*DIAGRAM_LEVEL_GAP
            start       = i - 0.5*group_width

            for j in range(0,len(group)):
                left = start + j*(bar_width + DIAGRAM_LEVEL_GAP)

                axes.plot([left,left + bar_width],
                          [group[j],group[j]],
                          color=LEVEL_COLOR,
                          linewidth=LEVEL_WIDTH,
                          solid_capstyle='butt')

        tick_positions.append(i)
        tick_labels.append(column['label'])

    axes.set_xticks(tick_positions)
    axes.set_xticklabels([axis_label(label) for label in tick_labels])
    axes.set_xlim(-0.5,len(content['columns']) - 0.5)

    # The bars themselves mark the states, so the ticks of the discrete
    # axis are not drawn.
    axes.tick_params(axis='x',length=0.0)


def draw_effective_barrier(axes, content):
    """Draw an effective barrier plot into the given axes.

    The states are drawn as horizontal bars placed by their magnetic moment
    projection and their energy, and the transitions between them as arrows
    whose width follows the magnitude of the transition magnetic moment.
    The arrows are drawn first, so that the bars marking the states stay
    visible on top of them.
    """
    level_list = content['levels']

    if len(level_list) == 0:
        return

    moment_list = [level['moment'] for level in level_list]
    energy_list = [level['energy'] for level in level_list]

    # The width of an arrow follows the magnitude of the transition moment
    # relative to the strongest transition of the plot.
    transition_list = content['transitions']

    if len(transition_list) > 0:
        largest_moment = max([abs(transition['moment'])
                              for transition in transition_list])
    else:
        largest_moment = 0.0

    for transition in transition_list:
        if largest_moment <= 0.0:
            continue

        relative = abs(transition['moment'])/largest_moment

        if relative < ARROW_THRESHOLD:
            continue

        width = MIN_ARROW_WIDTH + relative*(MAX_ARROW_WIDTH - MIN_ARROW_WIDTH)

        first  = level_list[transition['from']]
        second = level_list[transition['to']]

        # The strongest transitions are drawn in full colour and the weaker
        # ones faded, which keeps the relaxation pathway visible among the
        # transitions that are drawn beside it.
        opacity = 0.15 + 0.85*relative

        axes.annotate("",
                      xy=(second['moment'],second['energy']),
                      xytext=(first['moment'],first['energy']),
                      arrowprops={'arrowstyle':      '-|>',
                                  'color':           TRANSITION_COLOR,
                                  'alpha':           opacity,
                                  'linewidth':       width,
                                  'mutation_scale':  8.0 + 4.0*width,
                                  'shrinkA':         0.0,
                                  'shrinkB':         0.0})

    for level in level_list:
        axes.plot([level['moment'] - LEVEL_HALF_WIDTH,
                   level['moment'] + LEVEL_HALF_WIDTH],
                  [level['energy'],level['energy']],
                  color=LEVEL_COLOR,
                  linewidth=LEVEL_WIDTH,
                  solid_capstyle='butt',
                  zorder=3)

    # A margin is left around the states so that the bars and the arrow
    # heads are not cut by the frame of the plot.
    moment_margin = 0.1*(max(moment_list) - min(moment_list) + 1.0)
    energy_margin = 0.1*(max(energy_list) - min(energy_list) + 1.0)

    axes.set_xlim(min(moment_list) - moment_margin,
                  max(moment_list) + moment_margin)
    axes.set_ylim(min(energy_list) - energy_margin,
                  max(energy_list) + energy_margin)


def write_plot(content, filename, file_format,
               width=6.0,
               size_ratio=4.0/3.0,
               resolution=600,
               compression=None):
    """Draw the plot and write it on disk.

    Arguments
    ---------
    content : dict
        The content of the plot (see the module docstring).
    filename : str
        The name of the file to write.
    file_format : str
        The format of the file, i.e. 'png', 'tiff' or 'pdf'.

    Optional arguments
    ------------------
    width : float
        The width of the image in inches. Default is 6.0.
    size_ratio : float
        The ratio of the width of the image to its height. Default is 4/3.
    resolution : int
        The resolution of the image in dots per inch. Default is 600.
    compression : int or str or None
        The compression of the raster formats, i.e. the compression level
        of a PNG file or the name of the compression of a TIFF file.
        Default is None, in which case the default of the format is used.
    """
    plt = pyplot()

    # The appearance is set for this figure alone, so that the settings of
    # a user who draws plots of their own beside the library are not
    # changed by writing a plot.
    with plt.rc_context(rc=PLOT_STYLE):
        return __write_figure(plt,content,filename,file_format,
                              width,size_ratio,resolution,compression)


def __write_figure(plt, content, filename, file_format,
                   width, size_ratio, resolution, compression):
    """Draw the figure and save it. Called by write_plot within the
    appearance settings of the library.
    """
    figure, axes = plt.subplots(figsize=(width,width/size_ratio))

    try:
        if content['plot_type'] == 'effective_barrier':
            draw_effective_barrier(axes,content)
        elif content['plot_type'] == 'energy_level_diagram':
            draw_energy_level_diagram(axes,content)
        else:
            draw_standard_plot(axes,content)

        if not content['x_label'] == "":
            axes.set_xlabel(axis_label(content['x_label']))
        if not content['y_label'] == "":
            axes.set_ylabel(axis_label(content['y_label']))

        if content['title'] is not None:
            axes.set_title(content['title'])

        if content['legend']:
            axes.legend(frameon=False)

        figure.tight_layout()

        save_arguments = {'dpi': resolution, 'format': format_name(file_format)}
        pil_arguments  = pil_options(file_format,compression)

        if len(pil_arguments) > 0:
            save_arguments['pil_kwargs'] = pil_arguments

        figure.savefig(filename,**save_arguments)
    finally:
        # The figures are held by Matplotlib until they are closed, so a
        # calculation writing many plots would otherwise fill the memory.
        plt.close(figure)


def format_name(file_format):
    """Return the name Matplotlib knows the given format by."""
    name = str(file_format).lower()

    if name in ('tif','tiff'):
        return 'tiff'
    if name == 'png':
        return 'png'
    if name == 'pdf':
        return 'pdf'

    raise RuntimeError("Unknown image format: " + str(file_format) + ". "
                       "The recognized formats are png, tiff and pdf.")


def pil_options(file_format, compression):
    """Return the options passed on to the writer of the raster formats,
    i.e. the compression of a PNG or a TIFF file. The vector format takes
    none.
    """
    name = format_name(file_format)

    if compression is None:
        return {}

    if name == 'png':
        return {'compress_level': int(compression)}

    if name == 'tiff':
        compression_name = str(compression).lower()

        if compression_name in ('none','raw',''):
            return {'compression': None}

        if compression_name == 'lzw':
            return {'compression': 'tiff_lzw'}

        return {'compression': compression_name}

    return {}
