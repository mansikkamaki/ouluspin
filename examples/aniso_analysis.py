#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""Perform on an ORCA aniso datafile the analyses that the SINGLE_ANISO
module performs, and report them as tables, as plots and as a document.

The datafile holds the matrices of the Hamiltonian and of the magnetic
moment operators in the basis of the spin-orbit states of an ab initio
calculation. The script projects them onto the pseudospin multiplet of the
ion, i.e. onto the ground J multiplet of a lanthanide(III) ion, and reports
the properties that are read off such a calculation:

  - the data of the free ion, i.e. its electron configuration, its
    Hund's-rule ground multiplet and the Lande g-factor of that multiplet;
  - the pseudospin doublets of the multiplet, each with its g-tensor and
    its principal magnetic axes in the input axis frame, both in full and
    as a one-line-per-doublet summary;
  - the crystal-field parameters, i.e. the expansion of the Hamiltonian in
    the Iwahara--Chibotaru irreducible tensor operators;
  - the energies of the crystal-field states and their compositions in
    terms of the |J,M> basis states;
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

    ouluspin-python aniso_analysis.py <file.anisofile> <ion>

The first argument is the ORCA aniso file and the second the name of the
ion, e.g. 'Dy(III)'. The pseudospin of the calculation and the free-ion
quantities the plots are compared with are read off the IonData class,
which parses the name of the ion case-insensitively and accepts the
oxidation state as an Arabic or as a Roman numeral ('Dy(III)', 'dy 3',
'Dy3+'). IonData gives the pseudospin in the doubled form used throughout
the library, i.e. 15 for the J = 15/2 of a Dy(III) ion.

The settings below the input decide what is calculated and in what form the
results are written; they are the part of the script meant to be edited.

Notes
-----
Both a Kramers ion, i.e. one of a half-integer J such as Dy(III), and a
non-Kramers ion, i.e. one of an integer J such as Ho(III), are handled. The
states of a Kramers multiplet are degenerate in pairs, whereas a non-Kramers
multiplet holds an odd number of states and one of them is left over as a
singlet; the grouping is done by the pseudospin_doublet_index_list method of
the system, which places the singlet where it leaves the smallest splitting
within the quasi-doublets. A non-Kramers ion whose ground state is that
singlet cannot be analysed this way, since the principal magnetic axes of
the system are those of the ground doublet, and the method stops with an
error.

The states of the datafile are reordered so that the pseudospin multiplet is
built of the lowest-energy states of the calculation.

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

    if len(sys.argv) != 3:
        sys.exit("usage: ouluspin-python aniso_analysis.py "
                 "<file.anisofile> <ion>\n"
                 "e.g.:  ouluspin-python aniso_analysis.py "
                 "Dy_complex.anisofile 'Dy(III)'")

    filename = sys.argv[1]
    ion      = sys.argv[2]

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

    # The data of the free ion. The pseudospin of the calculation is the
    # total angular momentum of the ground multiplet, which IonData stores
    # in the doubled form the library uses throughout, i.e. 15 for the
    # J = 15/2 of a Dy(III) ion.
    ion_data   = ouluspin.IonData(ion)
    pseudospin = ion_data.J

    # The states of the datafile are ordered by energy, whereas the
    # pseudospin basis is ordered by the projection M from -J up. The
    # reordering pairs the lowest ab initio state with the largest
    # projection, so that the multiplet is built of the lowest-energy states
    # of the calculation.
    reorder_list = list(range(pseudospin,-1,-1))

    # Turn the ab initio calculation into a pseudospin system.
    system = ouluspin.AbInitioElectronExchangeSystem\
                     .from_aniso_data(filename,
                                      pseudospin,
                                      units,
                                      reorder_list=reorder_list)

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

    print(ion_data)
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
    # magnetization, the chiT product and the magnetization curves. The
    # susceptibility is drawn with the Curie value of the free ion as a
    # horizontal reference line, which the calculated curve approaches from
    # below once the whole ground multiplet is populated, and the
    # magnetization with the saturation value of an Ising-type ground
    # doublet, i.e. the powder value g_J*J/2, which is what the curves
    # saturate at when the ground doublet is axial and alone in the
    # low-energy region.
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
            horizontal_line=ion_data.curie_susceptibility(),
            horizontal_line_label="Curie chiT")

        magnetization_writer(
            magnetization_basename + '.' + image_format,
            overwrite=True,
            width=plot_width,
            font_size=plot_font_size,
            horizontal_line=ion_data.ising_saturation_magnetization(),
            horizontal_line_label="Ising M(sat)")

        print("Plots written in the " + image_format + " format.")

    # The quantization axis of the system, i.e. the z axis of the frame the
    # pseudospin operators are written in, as a unit vector in the input
    # axis frame. It is written into a small file of its own, in the form a
    # molecular viewer reads, so that the easy axis can be drawn into a
    # picture of the structure.
    axis          = system.quantization_axis()
    axis_filename = axis_basename + '.axis'

    with open(axis_filename,'w') as axis_file:
        axis_file.write(ion_data.element + "\n")
        for i in range(0,3):
            axis_file.write(" {0:14.8f}".format(axis[i]) + "\n")

    print("Quantization axis written into " + axis_filename + ".")

    tok = time.time()
    print()
    print("Total wall time: {0:18.3f}".format(tok - tik))
