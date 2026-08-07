#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""Perform on a Kuiva pseudospin file the analyses that aniso_analysis.py
performs on an ORCA aniso datafile, and report them as tables, as plots and
as a document.

A Kuiva pseudospin file (a .psd file) differs from an aniso datafile in
that it has already identified its matrices with a pseudospin basis: it
carries the pseudospin of every site, the ordered listing of the basis
states, the effective Hamiltonian and the Cartesian components of the
magnetic moment over that basis, and the unitary that maps the ab initio
states to it. The analysis is therefore the same as in aniso_analysis.py
but the setting up of the system is shorter, as three things that have to
be decided for an aniso datafile are read off the file instead:

  - the pseudospin of the multiplet, which is not given as an argument;
  - the ordering of the states, which needs no reorder_list, as the file
    lists its basis states in the order the library constructs them;
  - the quantization axis, which is the axis the states of the file are
    labelled by the projection M along, and not the principal magnetic axis
    of the ground doublet that the class determines for itself when it
    projects an aniso datafile. The two axes are the same one whenever the
    file was written with the states labelled along the magnetic axis of
    the ion, which is what Kuiva does by default; the script prints both,
    so that the choice can be seen.

This script treats a model of a single spin site, which is what a local
multiplet of one magnetic ion gives. A file of several sites is refused,
since the doublet grouping and the free-ion comparisons below are those of
one ion; the tensor decomposition of a multi-site model into the crystal
fields of the sites and the intersite exchange is the subject of
two_site_crystal_field.py.

The properties reported are those of aniso_analysis.py:

  - the pseudospin doublets of the multiplet, each with its g-tensor and
    its principal magnetic axes in the input axis frame, both in full and
    as a one-line-per-doublet summary;
  - the crystal-field parameters, i.e. the expansion of the Hamiltonian in
    the Iwahara--Chibotaru irreducible tensor operators;
  - the energies of the states of the model and their compositions in terms
    of the |S,M> basis states;
  - the transition magnetic moments between the states, from which the
    effective barrier of the reversal of the magnetization is read;
  - the static magnetic properties of a powder sample, i.e. the molar
    magnetic susceptibility as the chi*T product and the isothermal
    magnetization curves.

Everything printed on the standard output is also written into a
word-processor or a LaTeX document, the plots are written as image files,
and the quantization axis of the system is written into a small text file
of its own, so that it can be drawn into a picture of the structure.

Usage
-----
Run with the environment of the library set up (source setup.sh at the root
of the repository):

    ouluspin-python kuiva_analysis.py <file.psd> [<ion>]

The first argument is the Kuiva pseudospin file. The second argument, the
name of the ion, is optional and is used for the free-ion quantities the
plots are compared with and for the label of the axis file; IonData parses
the name case-insensitively and accepts the oxidation state as an Arabic or
as a Roman numeral ('Dy(III)', 'dy 3', 'Dy3+'). The pseudospin is NOT taken
from it, as the model of the file decides that.

The free-ion reference lines are those of the Hund's-rule ground multiplet
of the ion, so they are worth drawing when the model of the file is that
multiplet. A model of a single local doublet, e.g. the ground Kramers
doublet of a d(1) ion, is not the free-ion multiplet, and the ion is then
better left out: the plots are drawn without the reference lines.

The settings below the input decide what is calculated and in what form the
results are written; they are the part of the script meant to be edited.

Notes
-----
Both a Kramers model, i.e. one of a half-integer pseudospin, and a
non-Kramers model, i.e. one of an integer pseudospin, are handled. The
states of a Kramers multiplet are degenerate in pairs, whereas a
non-Kramers multiplet holds an odd number of states and one of them is left
over as a singlet; the grouping is done by the pseudospin_doublet_index_list
method of the system, which places the singlet where it leaves the smallest
splitting within the quasi-doublets.

A model of a single Kramers doublet, which is what the local multiplet of a
d(1) ion is, holds two states and no splitting at all: its crystal-field
table is empty, as a doublet carries no crystal field, and its barrier plot
holds a single transition. The analysis is worth reading for a model of a
larger multiplet; nothing in the script assumes one.

The phases of the states of a Kuiva file are arbitrary, as the file
canonicalizes none of them. They are corrected by the class method that
builds the system, which then checks that the operators behave properly
under time reversal and warns when they do not; a warning of that kind
means that the results below are not to be trusted.

The powder integration of the magnetic properties is what the calculation
costs: the Hamiltonian is diagonalized once per grid point and per field
strength. The grid is left at its default setting, which is the one
recommended for a calculation of both properties; see the class
documentation of ZCWGrid and of StaticMagneticProperties.
"""

import sys
import time

import numpy as np

import ouluspin


# The formats the results can be written in. One table format is used at a
# time, since all the tables go into one document; the images are written in
# every format that is listed.
TABLE_FORMATS = ('odt','docx','latex')
IMAGE_FORMATS = ('pdf','png','tiff')


if __name__ == '__main__':

    # ======================
    # === SETTINGS PHASE ===
    # ======================

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        sys.exit("usage: ouluspin-python kuiva_analysis.py "
                 "<file.psd> [<ion>]\n"
                 "e.g.:  ouluspin-python kuiva_analysis.py "
                 "ticl3_ground.psd 'Ti(III)'")

    filename = sys.argv[1]

    if len(sys.argv) == 3:
        ion = sys.argv[2]
    else:
        ion = None

    # The names of the files written. The suffix of each follows the format
    # it is written in.
    output_basename         = 'data'
    barrier_basename        = 'barrier'
    susceptibility_basename = 'susceptibility'
    magnetization_basename  = 'magnetization'
    axis_basename           = 'axis'

    # The format of the document holding the tables, and the formats of the
    # images. The document formats need nothing beyond the standard library;
    # the images need Matplotlib.
    table_output_format  = 'odt'
    image_output_formats = ['pdf']

    # The width of the plots and the font size of their text, chosen here
    # for a figure that goes into one column of a journal page.
    plot_width         = "7.5 cm"
    barrier_plot_width = "7.5 cm"
    plot_font_size     = 10

    # The temperatures and the fields the magnetic properties are
    # calculated at. Neither property is defined at zero temperature, but a
    # temperature list may begin at zero all the same: the classes raise a
    # vanishing temperature to MINIMUM_TEMPERATURE, which already gives the
    # zero-temperature limit.
    susceptibility_T_list = np.linspace(0,300,301)
    magnetization_T_list  = [1.8,2.0,2.5,3.0,4.0,5.0]
    magnetization_B_list  = np.linspace(0,10,21)

    # Rank combinations whose crystal-field parameters are all smaller than
    # this are left out of the crystal-field table.
    cf_rank_threshold = 1.0e-4

    units = ouluspin.EnergyUnitSystem('wavenumber')

    if not table_output_format in TABLE_FORMATS:
        sys.exit("unknown table format: " + str(table_output_format) + "\n"
                 "the recognized formats are "
                 + ", ".join(TABLE_FORMATS) + ".")

    for image_format in image_output_formats:
        if not image_format in IMAGE_FORMATS:
            sys.exit("unknown image format: " + str(image_format) + "\n"
                     "the recognized formats are "
                     + ", ".join(IMAGE_FORMATS) + ".")

    tik = time.time()


    # =========================
    # === CALCULATION PHASE ===
    # =========================

    # The file itself. Reading it here is not needed for the analysis, as
    # the class method below reads it again, but the summary of the file
    # states what the model is: how many sites it has, what their
    # pseudospins are and in which coordinate frame its operators are
    # written.
    pseudospin_file = ouluspin.KuivaPseudospinFile(filename,units)
    print(pseudospin_file)

    if not pseudospin_file.n_sites == 1:
        sys.exit("the file " + filename + " holds a model of "
                 + str(pseudospin_file.n_sites) + " spin sites,\n"
                 "whereas this script analyses a model of a single site.")

    # The data of the free ion, when an ion was given. They are used for
    # the reference lines of the plots and for the label of the axis file
    # only; the pseudospin of the analysis is the one of the file.
    if ion is None:
        ion_data = None
    else:
        ion_data = ouluspin.IonData(ion)
        print(ion_data)

    # Turn the Kuiva model into a pseudospin system. The basis, the
    # transformation into it and the quantization axis all come from the
    # file, so neither a pseudospin nor a reorder_list is given here.
    system = ouluspin.AbInitioElectronExchangeSystem\
                     .from_kuiva_data(filename,units)

    pseudospin = system.basis.pseudospin_list[0]

    # The pseudospin doublets, i.e. the states grouped in pairs, each with
    # its g-tensor and its principal magnetic axes in the input axis frame.
    doublet_index_list = system.pseudospin_doublet_index_list(pseudospin)
    doublets           = system.pseudospin_doublet_list(doublet_index_list)

    full_doublet_table    = system.pseudospin_doublet_table(doublets)
    summary_doublet_table = system.pseudospin_doublet_summary_table(doublets)

    # The crystal-field operator and the crystal-field states. The half
    # table leaves out the parameters that follow from the printed ones by a
    # phase.
    crystal_field       = system.hamiltonian_tensor()
    crystal_field_table = crystal_field.ITO_table(
        rank_threshold=cf_rank_threshold,half_table=True)

    hamiltonian = system.hamiltonian_operator()

    eigenvector_table         = hamiltonian.eigenvector_table()
    compact_eigenvector_table = hamiltonian.compact_eigenvector_table()

    # The transition magnetic moments, from which the effective barrier of
    # the reversal of the magnetization is read.
    transition_magnetic_moments = system.static_transition_magnetic_moments()
    transition_magnetic_moment_table = \
        transition_magnetic_moments.transition_magnetic_moment_table()

    # The static magnetic properties of a powder sample. The instances given
    # to StaticMagneticProperties state what is calculated and hold the
    # results afterwards, so the properties are read from them and not from
    # the return value.
    magnetization  = ouluspin.IsothermalStaticMagnetization(
        magnetization_T_list,magnetization_B_list)
    susceptibility = ouluspin.StaticMagneticSusceptibility(
        susceptibility_T_list)

    grid = ouluspin.ZCWGrid()

    ouluspin.StaticMagneticProperties\
            .from_ab_initio_system(system,grid,
                                   susceptibility=susceptibility,
                                   magnetization=magnetization)

    susceptibility_table = susceptibility.data_table()
    magnetization_table  = magnetization.data_table()


    # ====================
    # === OUTPUT PHASE ===
    # ====================

    # The tables in the order they are read in.
    table_list = [full_doublet_table,
                  summary_doublet_table,
                  crystal_field_table,
                  eigenvector_table,
                  compact_eigenvector_table,
                  transition_magnetic_moment_table,
                  susceptibility_table,
                  magnetization_table]

    for table in table_list:
        print(table)

    # The same tables into a document of their own. The tables that are
    # written are the ones that are read rather than searched through, so
    # the two tables that list every state one by one are left out. The
    # first table is written with overwrite_file, which starts a new
    # document; the rest are appended to it.
    document_table_list = [summary_doublet_table,
                           crystal_field_table,
                           compact_eigenvector_table,
                           transition_magnetic_moment_table,
                           susceptibility_table,
                           magnetization_table]

    if table_output_format == 'odt':
        output_filename = output_basename + '.odt'
        write_table     = lambda table, first: table.odt_table(
            output_filename,overwrite_file=first)
    elif table_output_format == 'docx':
        output_filename = output_basename + '.docx'
        write_table     = lambda table, first: table.docx_table(
            output_filename,overwrite_file=first)
    else:
        output_filename = output_basename + '.tex'
        write_table     = lambda table, first: table.latex_table(
            output_filename,overwrite_file=first)

    for i in range(0,len(document_table_list)):
        write_table(document_table_list[i],i == 0)

    print("Tables written into " + output_filename + ".")

    # The plots: the effective barrier of the reversal of the
    # magnetization, the chiT product and the magnetization curves. When an
    # ion was given, the susceptibility is drawn with the Curie value of the
    # free ion as a horizontal reference line, which the calculated curve
    # approaches from below once the whole ground multiplet is populated,
    # and the magnetization with the saturation value of an Ising-type
    # ground doublet, i.e. the powder value g_J*J/2, which is what the
    # curves saturate at when the ground doublet is axial and alone in the
    # low-energy region. Without an ion both lines are left out, which is
    # what a horizontal_line of None means.
    if ion_data is None:
        curie_line      = None
        saturation_line = None
    else:
        curie_line      = ion_data.curie_susceptibility()
        saturation_line = ion_data.ising_saturation_magnetization()

    barrier_plot        = ouluspin.ResultPlot(transition_magnetic_moments)
    susceptibility_plot = ouluspin.ResultPlot(susceptibility)
    magnetization_plot  = ouluspin.ResultPlot(magnetization)

    for image_format in image_output_formats:
        # The writing methods of the three formats take the same arguments,
        # so the one belonging to the format is picked by its name.
        barrier_writer        = getattr(barrier_plot,
                                        image_format + '_plot')
        susceptibility_writer = getattr(susceptibility_plot,
                                        image_format + '_plot')
        magnetization_writer  = getattr(magnetization_plot,
                                        image_format + '_plot')

        barrier_writer(barrier_basename + '.' + image_format,
                       overwrite=True,
                       width=barrier_plot_width,
                       font_size=plot_font_size)

        susceptibility_writer(
            susceptibility_basename + '.' + image_format,
            overwrite=True,
            width=plot_width,
            font_size=plot_font_size,
            horizontal_line=curie_line,
            horizontal_line_label="Curie chiT")

        magnetization_writer(
            magnetization_basename + '.' + image_format,
            overwrite=True,
            width=plot_width,
            font_size=plot_font_size,
            horizontal_line=saturation_line,
            horizontal_line_label="Ising M(sat)")

        print("Plots written in the " + image_format + " format.")

    # The two axes of the system, as unit vectors in the input axis frame of
    # the ab initio calculation. The quantization axis is the z axis of the
    # frame the pseudospin operators are written in, which for a model read
    # from a Kuiva file is the axis the states of the file are labelled
    # along; the magnetic axis of the ground doublet is evaluated from the
    # operators themselves. The two are the same axis when the file was
    # written with the states labelled along the magnetic axis of the ion,
    # and the angle between them says how far from that the file is.
    axis                  = system.quantization_axis()
    ground_doublet_axis   = system.ground_doublet_magnetic_axis()
    axis_angle            = np.degrees(np.arccos(
        min(1.0,abs(float(np.dot(axis,ground_doublet_axis))))))

    print()
    print("    Quantization axis (the labelling axis of the file):")
    print("      {0:12.8f} {1:12.8f} {2:12.8f}".format(*axis))
    print("    Principal magnetic axis of the ground doublet:")
    print("      {0:12.8f} {1:12.8f} {2:12.8f}".format(*ground_doublet_axis))
    print("    Angle between the two axes: {0:8.3f} degrees".format(axis_angle))
    print()

    # The quantization axis is written into a small file of its own, in the
    # form a molecular viewer reads, so that the easy axis can be drawn into
    # a picture of the structure. The label of the site is the symbol of the
    # ion when one was given.
    axis_filename = axis_basename + '.axis'

    if ion_data is None:
        axis_label = 'X'
    else:
        axis_label = ion_data.element

    with open(axis_filename,'w') as axis_file:
        axis_file.write(axis_label + "\n")
        for i in range(0,3):
            axis_file.write(" {0:14.8f}".format(axis[i]) + "\n")

    print("Quantization axis written into " + axis_filename + ".")

    tok = time.time()
    print()
    print("Total wall time: {0:18.3f}".format(tok - tik))
