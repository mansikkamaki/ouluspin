# OuluSpin

**OuluSpin** is a Python 3 library for theoretical molecular magnetism. It
provides routines for the construction of pseudospin Hamiltonians and the
calculation of various magnetic properties using these Hamiltonians,
following the pseudospin formalism developed by Chibotaru and co-workers.
It also contains utilities for extracting ab initio quantum-chemical data
(mostly operator matrices) from the outputs of quantum-chemistry codes
(OpenMolcas, ORCA). The computationally intensive routines are written in
Fortran and interfaced to Python with f2py.

This README is the main documentation of the library. In addition, every
public class carries a detailed docstring documenting its constructor
arguments, attributes and methods; the docstrings are the authoritative
reference for the individual classes.

## Contents

- [Physical background](#physical-background)
- [Requirements](#requirements)
- [Installation and setup](#installation-and-setup)
- [Quick start](#quick-start)
- [Library overview](#library-overview)
- [Conventions](#conventions)
- [Examples](#examples)
- [Tests](#tests)
- [Repository layout](#repository-layout)
- [Citing](#citing)
- [References](#references)
- [License](#license)

## Physical background

The physical system is described by a pseudospin Hamiltonian constructed
in terms of products of equivalent operators,

$$
\tilde H
= \sum_{k_1,k_2,\cdots}\sum_{q_1,q_2,\cdots}
X_{k_1,q_1,k_2,q_2,\cdots}
\hat O_{k_1,q_1}(\mathbf{\tilde{S}}_1)
\otimes\hat O_{k_2,q_2}(\mathbf{\tilde{S}}_2)
\otimes\cdots
$$

where $k_i$ and $q_i$ are the ranks and components of the equivalent
operators $\hat O_{k_i,q_i}(\mathbf{\tilde{S}}_i)$ and
$\mathbf{\tilde S}_i$ are pseudospin operators acting on the individual
spin sites $i$. The individual equivalent operators are chosen as those
described by Iwahara and Chibotaru [2], whose matrix elements can be
evaluated as a ratio of two Clebsch–Gordan coefficients. Chibotaru–Ungur
equivalent operators [1] and Cartesian tensor forms are also supported
as input.

A typical workflow is:

1. Read operator matrices (Hamiltonian, magnetic moment) from the output
   of an ab initio calculation (`ouluspin.qc`).
2. Project the operators onto a pseudospin basis and decompose them into
   irreducible tensor operator (ITO) expansions
   (`ouluspin.pseudospin_operators`, `ouluspin.tensors`,
   `ouluspin.systems`).
3. Analyze the resulting parameters (crystal-field and exchange
   operators, g-tensors) and calculate measurable magnetic properties
   such as magnetization and susceptibility (`ouluspin.properties`,
   `ouluspin.integration`).

## Requirements

- Python ≥ 3.8 with `numpy` and `scipy`
- A Fortran compiler (`gfortran`, Intel `ifort`, or Intel `ifx` with
  numpy ≥ 1.26) for the compiled extension module
- A LAPACK/BLAS library (Intel MKL, OpenBLAS, or reference LAPACK/BLAS)

Optional, needed only by the plotting of the results (`ResultPlot`):

- `matplotlib` (and `pillow`, which it installs along with itself, for the
  TIFF files). Everything else, including the LaTeX, `.odt` and `.docx`
  renderings of the result tables, works without it. Install it into the
  interpreter that runs the library:

      ouluspin-python -m pip install --user matplotlib

  Take care that the installation does not replace the `numpy` of the
  interpreter: a `numpy` installed beside an Intel Python shadows the
  optimized one and can break the compiled Fortran extension, whose ABI is
  tied to the `numpy` it was built against. Should that happen, remove the
  shadowing copy with `ouluspin-python -m pip uninstall numpy`.

## Installation and setup

### 1. Build the Fortran extension module

```bash
cd src/fortran
./configure     # detects compiler, Python and LAPACK; writes make.inc
make            # builds and installs fortran_utils.so into the package
```

`configure` prefers the Intel toolchain (ifort + MKL + Intel Python) when
an oneAPI installation is found, and falls back to gfortran + system
LAPACK otherwise. Everything can be overridden, e.g.:

```bash
./configure --fc=gfortran --python=python3 --lapack-libs='-lopenblas'
```

or by editing the generated `make.inc` (see `make.inc.example`). Useful
Makefile targets: `make` (build + install into the package), `make test`
(build and run the Fortran test suite), `make clean`, `make distclean`.

**Important:** the extension module can only be imported by a Python
interpreter whose numpy is binary-compatible with the one used in the
build. Use the same interpreter for building and running (this is what
`setup.sh` and `configure` do by default).

The package itself can be imported without the compiled extension — for
example to browse docstrings — and an informative error is raised only
when a routine that needs the extension is actually called.

### 2. Set up the environment

Source the setup script (e.g. in your shell profile):

```bash
source /path/to/ouluspin/setup.sh
```

This puts the package on `PYTHONPATH`, selects the Python interpreter
(exported as `OULUSPIN_PYTHON`, with the `ouluspin-python` convenience
command) and, if an Intel oneAPI installation is present, adds the MKL and
Intel Fortran runtime libraries to `LD_LIBRARY_PATH`. Scripts are then run
with

```bash
ouluspin-python my_script.py
```

Alternatively, the pure-Python part can be installed with pip
(`pip install -e .` at the repository root); the Fortran extension is
still built with the Makefile as above.

## Quick start

The most important classes are re-exported at the package top level:

```python
import ouluspin

# Energy units used throughout a calculation.
units = ouluspin.EnergyUnitSystem('wavenumber')

# A pseudospin basis: pseudospins are given as multiples of two, so
# [15] is a single site with S = 15/2 (e.g. a Dy(III) J = 15/2 multiplet).
basis = ouluspin.PseudoSpinBasis([15])
```

A complete single-site crystal-field extraction from an ORCA aniso file:

```python
from ouluspin import pseudospin_operators, units as units_module
from ouluspin.qc import orca
from ouluspin.systems import electron_exchange_system

units = units_module.EnergyUnitSystem('wavenumber')

# Read the Hamiltonian and magnetic moment operator matrices.
calculation = orca.OrcaAnisoFile('Dy_complex.anisofile', units)

hamiltonian = pseudospin_operators.GeneralOperatorMatrix(
    calculation.hamiltonian(),
    diagonalize_operator_matrix=True, translate_eigenvalues=True)
magnetic_moment = pseudospin_operators.GeneralVectorOperatorMatrix(
    calculation.magnetic_moment(include_bohr_magneton=False),
    diagonalize_operator_matrix=True, translate_eigenvalues=False)

# Project onto a pseudospin S = 15/2 basis and decompose the Hamiltonian
# into an ITO expansion (the crystal-field parameters).
basis = pseudospin_operators.PseudoSpinBasis([15])
system = electron_exchange_system.AbInitioElectronExchangeSystem(
    hamiltonian, magnetic_moment, basis, units, print_output=True)

print(system.hamiltonian_tensor())
```

See [Examples](#examples) for full runnable scripts, including a coupled
two-site decomposition into one-site crystal fields and an intersite
exchange operator.

## Library overview

### `ouluspin.units`

- **`EnergyUnitSystem`** — defines the energy unit of a calculation,
  performs unit conversions and stores natural constants (Boltzmann
  constant, Planck constant, Bohr and nuclear magnetons, etc.) expressed
  in the chosen unit. Allowed energy units: `joule`, `wavenumber`,
  `kelvin`, `millielectronvolt`, `electronvolt`. An instance of this
  class is passed to essentially every other class in the library.

### `ouluspin.result_table`

- **`ResultTable`** — the structured representation of a table of results.
  The classes of the library return their result tables as instances of
  this class rather than as preformatted strings: the instance stores the
  content of the table (the rows, the headers, the title and the
  explanatory texts) and renders it, so that the same table can be
  produced in several output formats. Printing an instance, or converting
  it to a `str`, gives the plain-text table. The class method
  `pseudospin_doublet_compound_table` builds a compound one-line-per-doublet
  summary of a list of `PseudoSpinDoublet` instances.

  Besides the plain text, a table renders into the formats used when the
  results are written up:

  | Method | Output |
  | --- | --- |
  | `string_table(plain=False)` | the plain-text table; `plain=True` gives a bare, machine-readable form |
  | `data_file(filename)` | the bare form written into a text file, for plotting programs |
  | `latex_table(filename,standalone=True)` | a LaTeX table; a `None` filename returns the string |
  | `latex_string_table(standalone=True)` | the LaTeX table as a string |
  | `odt_table(filename)` | an OpenDocument text (`.odt`) document |
  | `docx_table(filename)` | an Office Open XML (`.docx`) document |

  The `.odt`, `.docx` and LaTeX writers append the table to the file when
  it already exists and create the document when it does not, so calling
  them once per table collects all the tables of a calculation into one
  document. The documents are laid out for an A4 page with 2 cm margins
  and an 11 pt font, and the tables span the full width of the text area.

  These renderings also typeset the physical quantities named by the
  headers: the symbols are set in italics with their indices as
  subscripts and superscripts, while the unit of a header of the form
  `quantity / unit` is set upright, so that `E / cm^-1` becomes an italic
  *E* over an upright cm⁻¹ and `Re(X_k1q1)` carries both pairs of
  operator indices. Greek letters are written out by name in the header
  (`theta`, `mu_B`) and set as the letters. Footnote markers become
  italic superscripts and negative numbers are written with a typographic
  minus sign. The plain-text table prints all of these as they are given,
  and a note or a footnote wrapped by hand for it is set on a single line
  where the text wraps by itself.

  The word-processor documents are written with the standard library
  only: both formats are ZIP archives of XML documents, so **no external
  modules such as `odfpy` or `python-docx` are needed**, and nothing
  beyond the standard library is imported unless one of the writers is
  actually called.

### `ouluspin.result_plot`

- **`ResultPlot`** — the structured representation of a plot of results,
  doing for the figures what `ResultTable` does for the tables. An
  instance is built from the object whose results are plotted, which
  decides the kind of the plot:

  | Source | Plot |
  | --- | --- |
  | `IsothermalStaticMagnetization` | magnetization against the field, one curve per temperature |
  | `StaticMagneticSusceptibility` | the χ*T* product against the temperature |
  | `StaticTransitionMagneticMoments` | the effective barrier of the reversal of the magnetization |
  | any operator carrying `eigenvalues` | an energy level diagram |

  `ResultPlot.from_data(x,y,...)` builds a plot from plain values. The
  axis labels, the legends and the drawing style are stored as attributes
  and can be changed afterwards, and the labels are typeset from the same
  markup as the table headers, so that `E / cm^-1` and `mu_B` come out as
  an italic *E* over cm⁻¹ and as μ<sub>B</sub>.

  The plot is written with `png_plot(filename)`, `tiff_plot(filename)` and
  `pdf_plot(filename)`. All of them take `size_ratio` and `resolution`
  (600 dpi by default, high enough for a publication), `draft_copy=True`
  to write a low-resolution copy beside the image for manuscript drafts,
  and `overwrite=False`, which makes writing over an existing file an
  error. PNG files are written with compression level 9 and TIFF files
  LZW-compressed by default; `compression` overrides both.

  Two compatible plots are combined with `+`, e.g. two χ*T* curves into
  one plot field or two level structures side by side. The criteria are
  strict: the kinds and both axis labels must agree, and effective
  barriers are never combined, since two barriers in one field cannot be
  told apart.

  Plotting is the only part of the library needing `matplotlib`; it is
  imported by the writing methods only when they are called, so the rest
  of the library works without it.

### `ouluspin.tensors`

Tensor structures used to parametrize the operators. Spherical tensor
objects support addition, scalar multiplication, rotation and conversion
between formats.

- **`IwaharaChibotaruSphericalTensor`** — an ITO expansion in the
  Iwahara–Chibotaru definition of the equivalent operators [2]; the
  central parameter format of the library.
- **`ChibotaruUngurSphericalTensor`** — a single-site ITO expansion in the
  Chibotaru–Ungur notation [1]; used as an input format (e.g. crystal-field
  parameters printed by SINGLE_ANISO) and convertible to the
  Iwahara–Chibotaru format.
- **`MixedCartesianIwaharaChibotaruSphericalTensor`** — an ITO expansion of
  a Cartesian vector operator (e.g. the magnetic moment): Cartesian in one
  index, Iwahara–Chibotaru spherical in the others.
- **`CartesianTensor`** — a Cartesian rank-two tensor (e.g. a g-tensor or
  ZFS tensor), convertible to Iwahara–Chibotaru parameters.
- **`Rotation`** — a rotation matrix and the corresponding Euler angles;
  used to rotate tensors between coordinate frames.

### `ouluspin.pseudospin_operators`

- **`PseudoSpinBasis`** — the basis states of a multi-site pseudospin
  system (pseudospins given as multiples of two).
- **`PseudoSpinOperator`** — constructs and diagonalizes a multi-site
  pseudospin operator from a list of spherical tensors.
- **`PseudoSpinVectorOperator`** — a vector (three-component) pseudospin
  operator; a convenience interface over three `PseudoSpinOperator`
  components.
- **`GeneralOperatorMatrix`** — stores a matrix representation of an
  operator in an arbitrary basis (e.g. an ab initio Hamiltonian matrix)
  and diagonalizes it; the standard container for ab initio input.
- **`GeneralVectorOperatorMatrix`** — the vector-operator counterpart of
  `GeneralOperatorMatrix` (e.g. the three Cartesian components of the
  magnetic moment operator).

### `ouluspin.properties`

- **`StaticMagneticProperties`** — evaluates static magnetic properties
  (isothermal magnetization, susceptibility) from the field-free
  Hamiltonian and magnetic moment operator matrices, with powder
  integration or fixed sample orientations.
- **`IsothermalStaticMagnetization`** — stores and prints isothermal
  magnetization values as a function of field and temperature.
- **`StaticMagneticSusceptibility`** — stores and prints the χT product
  as a function of temperature.
- **`StaticTransitionMagneticMoments`** — stores and prints tables of
  transition magnetic moment matrix elements between states.
- **`PseudoSpinDoublet`** — properties of a doublet described by an
  effective S = 1/2: the g-tensor and its principal axes, the tunneling
  gap, the energies of the two states, and the Kramers/non-Kramers
  classification.

### `ouluspin.integration`

Spherical grids for powder averaging:

- **`LebedevLaikovGrid`** — Lebedev–Laikov quadrature grids (built on the
  original Fortran 77 routines).
- **`ZCWGrid`** — Zaremba–Conroy–Wolfsberg grids.
- **`SimpleGrid`** — one or three Cartesian unit vectors, for single-axis
  or axis-resolved calculations.

### `ouluspin.qc`

Readers for quantum-chemistry outputs. All readers take an
`EnergyUnitSystem` and convert data to the chosen unit on reading.

- **`molcas.OpenMolcasCalculation`** — reads SINGLE_ANISO data from an
  OpenMolcas output file.
- **`orca.OrcaCalculation`** — reads data from an ORCA output file.
- **`orca.OrcaAnisoOutput`** — reads a standalone SINGLE_ANISO output file
  produced via ORCA.
- **`orca.OrcaAnisoFile`** — reads operator matrices from a `.anisofile`
  produced by an ORCA calculation.
- **`molcas.AnisoCalculation`** — common base class of the above; not
  instantiated directly.

### `ouluspin.systems`

- **`electron_exchange_system.ElectronExchangeSystem`** — a general
  multi-site pseudospin system defined by crystal-field/ZFS and exchange
  tensors in the Iwahara–Chibotaru form, used to evaluate magnetization
  and susceptibility.
- **`electron_exchange_system.AbInitioElectronExchangeSystem`** — builds a
  pseudospin system directly from ab initio operator matrices: projects
  the Hamiltonian and magnetic moment onto a pseudospin basis, and
  provides ITO decompositions (crystal-field and exchange parameters),
  transition magnetic moments and doublet g-tensor analyses.

### Fortran extension (`ouluspin._fortran`)

The computationally heavy routines live in the compiled extension module
`ouluspin._fortran.fortran_utils`, organized in Fortran modules:
`cg_utils` (Clebsch–Gordan coefficients and products), `angm_utils`
(Wigner d-matrices, 6j/9j symbols), `iwahara_utils` (Iwahara–Chibotaru
operator matrices and their diagonalization), `matrix_utils` (complex
matrix algebra, diagonalization, basis transformations), `cfp_utils`
(coefficients of fractional parentage for s/p/d/f shells),
`statmech_utils` (Boltzmann statistics), `powder_magnetization_utils`
(powder magnetization), `pseudospin_utils` and `grid_utils`
(Lebedev–Laikov grids). The leading underscore marks the package as
internal: the Python classes are the intended interface, and the routine
signatures may change without notice. The routines can nevertheless be
called directly when needed:

```python
from ouluspin._fortran import fortran_utils as fu
fu.cg_utils.cg(1, 1, 1, -1, 2, 0)   # angular momenta as doubled integers
```

## Conventions

- **Pseudospins and angular momenta are given as doubled integers**
  throughout the library and the Fortran routines: `15` means
  S = 15/2, `[15, 1]` means two sites with S = 15/2 and S = 1/2.
- **Basis ordering** is ascending in the pseudospin projection,
  |S,−S⟩, …, |S,S⟩ (for multi-site systems, the product basis ordered
  site by site).
- **Energy** is reported in the unit chosen in `EnergyUnitSystem`
  (customarily wavenumbers, cm⁻¹).
- **Magnetic susceptibility** is reported as molar susceptibility in cgs
  emu (χT in cm³ K mol⁻¹).
- **Magnetic field** is given either in cgs emu or in tesla depending on
  the situation (documented per method).
- **Magnetization** is reported as molar magnetization in units of the
  Bohr magneton.
- **Result tables** are returned as `ResultTable` instances, not as
  strings. Print them directly (`print(tensor.ITO_table())`) or convert
  them with `str()` when a string is needed. The same instance also
  writes itself into a LaTeX, `.odt` or `.docx` document; see
  [`ouluspin.result_table`](#ouluspinresult_table).

## Examples

Complete runnable scripts are in the [`examples/`](examples/) folder
(each documents the input files it needs; run them with
`ouluspin-python <script> <input>`):

- `one_site_crystal_field.py` — extract the crystal-field Hamiltonian of
  a single J = 15/2 multiplet (e.g. a Dy(III) complex) from an ORCA aniso
  file and decompose it into irreducible tensor operators.
- `two_site_crystal_field.py` — decompose the Hamiltonian of a coupled
  two-site system (S = 15/2 and S = 1/2) into the crystal fields of the
  individual sites and the intersite exchange operator.
- `average_crystal_field.py` — average the crystal fields of two
  SINGLE_ANISO calculations (one electron removed/added) after rotation
  into a common magnetic frame, and analyze the resulting system.

## Tests

After sourcing `setup.sh`:

```bash
ouluspin-python tests/class_tests.py     # tests of the Python classes
ouluspin-python tests/fortran_tests.py   # tests of the Fortran routines
```

`class_tests.py` runs the `run_tests` class method that every major
Python class implements (the tests check the class against analytic
reference results); `fortran_tests.py` exercises every Fortran module
against analytic values. `make test` in `src/fortran` builds the
extension and runs the Fortran suite in one step.

## Repository layout

    ouluspin/
    ├── pyproject.toml        Package metadata (pip install)
    ├── setup.sh              Environment setup script (source it)
    ├── LICENSE               GNU General Public License v3
    ├── src/
    │   ├── ouluspin/         The Python package
    │   │   ├── units.py                 Energy unit systems
    │   │   ├── result_table.py          Structured result tables
    │   │   ├── result_plot.py           Structured result plots
    │   │   ├── tensors.py               Spherical/Cartesian tensor classes
    │   │   ├── pseudospin_operators.py  Pseudospin bases and operators
    │   │   ├── properties.py            Magnetic property calculations
    │   │   ├── integration.py           Spherical integration grids
    │   │   ├── _documents.py            .odt and .docx writers (internal)
    │   │   ├── _images.py               Plot image writers (internal)
    │   │   ├── qc/                      Quantum-chemistry interfaces
    │   │   ├── systems/                 Higher-level physical systems
    │   │   └── _fortran/                Compiled Fortran extension (internal)
    │   └── fortran/          Fortran sources + build system (configure, make)
    ├── tests/                Test suites (Python classes and Fortran routines)
    └── examples/             Example scripts

## Citing

If you use OuluSpin in published work, please cite the papers describing
the physical formalism the library implements — reference [1] for the
pseudospin approach and reference [2] for the Iwahara–Chibotaru form of
the equivalent operators. If the Lebedev–Laikov integration grids are
used, please also cite reference [3], as requested by the authors of the
original quadrature routines.

## References

1. L. F. Chibotaru and L. Ungur, *Ab initio calculation of anisotropic
   magnetic properties of complexes. I. Unique definition of pseudospin
   Hamiltonians and their derivation*, J. Chem. Phys. **137**, 064112
   (2012). DOI: [10.1063/1.4739763](https://doi.org/10.1063/1.4739763)
2. N. Iwahara and L. F. Chibotaru, *Exchange interaction between J
   multiplets*, Phys. Rev. B **91**, 174438 (2015). DOI:
   [10.1103/PhysRevB.91.174438](https://doi.org/10.1103/PhysRevB.91.174438)
3. V. I. Lebedev and D. N. Laikov, *A quadrature formula for the sphere of
   the 131st algebraic order of accuracy*, Doklady Mathematics **59**, 477
   (1999).

## License

Copyright (C) 2020–2026 Akseli Mansikkamäki.

OuluSpin is free software: you can redistribute it and/or modify it under
the terms of the **GNU General Public License** as published by the Free
Software Foundation, either **version 3 of the License, or (at your option)
any later version**.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details. The full license text is in the [`LICENSE`](LICENSE) file.

Every source file carries an [SPDX](https://spdx.dev/) license identifier
(`SPDX-License-Identifier: GPL-3.0-or-later`).

The Lebedev–Laikov quadrature routines in
`src/fortran/lebedev_laikov.f` are a third-party contribution (C code by
Dmitri N. Laikov, translated to Fortran by Christoph van Wüllen); they
retain their original attribution header. Users of these routines are
asked to cite reference [3] in their publications.
