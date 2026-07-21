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
described by Iwahara and Chibotaru [3,4], whose matrix elements can be
evaluated as a ratio of two Clebsch–Gordan coefficients. Chibotaru–Ungur
equivalent operators [1] and Cartesian tensor forms are also supported
as input. The pseudospin formalism itself, i.e. the definition of the
pseudospin and the derivation of its Hamiltonian, is that of [1,2].

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

The powder integration of the static magnetic properties is parallelized
over the grid points with OpenMP, which `configure` enables by default and
`./configure --disable-openmp` turns off. The number of threads is set at
run time with `OMP_NUM_THREADS`.

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
  | `data_file(filename,plain=True)` | the bare form written into a text file, for plotting programs |
  | `latex_table(filename,standalone=True)` | a LaTeX table; a `None` filename returns the string |
  | `latex_string_table(standalone=True)` | the LaTeX table as a string |
  | `odt_table(filename)` | an OpenDocument text (`.odt`) document |
  | `docx_table(filename)` | an Office Open XML (`.docx`) document |

  The file writers append the table to the file when it already exists
  and create the file when it does not, so calling them once per table
  collects all the tables of a calculation into one file. The optional
  argument `overwrite_file`, accepted by `data_file`, `latex_table`,
  `odt_table` and `docx_table` and `False` by default, overwrites an
  existing file instead. The documents are laid out for an A4 page with
  2 cm margins and an 11 pt font, and the tables span the full width of
  the text area.

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
  `pdf_plot(filename)`. All of them take `resolution` (600 dpi by default,
  high enough for a publication), `draft_copy=True` to write a
  low-resolution copy beside the image for manuscript drafts, and
  `overwrite=False`, which makes writing over an existing file an error.
  PNG files are written with compression level 9 and TIFF files
  LZW-compressed by default; `compression` overrides both.

  The size of the image is given with `width` and `height`. A plain number
  is read as centimetres and a string carries its own unit, e.g.
  `width="8.5 cm"`, `width="3.35in"` or `width="1000px"`, with or without
  a space before the unit; the units are `cm`, `mm`, `in` and `px`, and a
  size in pixels is turned into a physical one through `resolution`. Given
  together the two fix the image and `size_ratio` is not used; one alone is
  completed by `size_ratio` (4/3 by default). The image comes out in
  exactly the size that was asked for, so a figure meant for the column of
  a journal can be written in the width it is to be printed in.

  `font_size` sets the main font size in points, i.e. the numbers along the
  axes, with the axis labels one point larger and the legends one point
  smaller. `typeface` sets the face the plot is set in: `'serif'` (the
  default, the usual choice of a publication in the field), `'sans-serif'`
  and `'monospace'`, the names `'sans'` and `'mono'` being accepted as
  well. The mathematical fonts follow the face, so the symbols match the
  text around them.

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
  Iwahara–Chibotaru definition of the equivalent operators [3,4]; the
  central parameter format of the library. Note that the definition was
  later revised in [5], which the library does not follow.
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
  pseudospin operator from a list of spherical tensors. Besides
  `eigenvalue_table()` and `eigenvector_table()` it offers
  `compact_eigenvector_table()`, which tabulates the composition of every
  eigenstate on a single row, one column per basis state labelled by its
  projection *M*; it is available for a basis of one spin site only, i.e.
  for the *J* multiplet of a lanthanide(III) ion and the like.
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
  Hamiltonian and magnetic moment operators of a system, with powder
  integration or fixed sample orientations. The class documentation
  includes a benchmark-based recommendation for the integration grid,
  which is what sets the cost of the calculation.
- **`IsothermalStaticMagnetization`** — stores and prints isothermal
  magnetization values as a function of field and temperature.
- **`StaticMagneticSusceptibility`** — stores and prints the χT product
  as a function of temperature.

  Both take the temperature range as it is given: neither property is
  defined at zero temperature, since the Boltzmann populations are not, so
  a temperature point at or below `MINIMUM_TEMPERATURE` (10⁻³ K) is raised
  to it. A range such as `numpy.linspace(0,300,301)` can therefore be used
  as it stands, and the raised point already gives the zero-temperature
  limit of the properties.
- **`StaticTransitionMagneticMoments`** — stores and prints tables of
  transition magnetic moment matrix elements between states. These are
  read as the qualitative effective barrier of the reversal of the
  magnetization of a single-molecule magnet [8].
- **`PseudoSpinDoublet`** — properties of a doublet described by an
  effective S = 1/2: the g-tensor and its principal axes, the tunneling
  gap, the energies of the two states, and the Kramers/non-Kramers
  classification. The g-tensor follows Section III.A of [1].

### `ouluspin.integration`

Spherical grids for powder averaging:

- **`LebedevLaikovGrid`** — Lebedev–Laikov quadrature grids [9] (built on
  the original Fortran 77 routines). Exact for the susceptibility already
  at a very small grid; converges slowly for the saturated magnetization.
- **`ZCWGrid`** — Zaremba–Conroy–Wolfsberg grids, in the form of Appendix 1
  of [10]. The better choice when the magnetization is calculated, and the
  better choice overall.
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

- **`electron_exchange_system.PseudoSpinSystem`** — the common base of the
  two system classes below. It carries the analysis that depends only on
  the pseudospin Hamiltonian and magnetic moment operator of a system and
  not on where those came from, so both systems are analysed through one
  interface: `static_transition_magnetic_moments`, `quantization_axis`,
  `ground_doublet_magnetic_axis`, `pseudospin_doublet`,
  `pseudospin_doublet_index_list`, `pseudospin_doublet_list`,
  `pseudospin_doublet_summary_table`, `pseudospin_doublet_table` and
  `static_magnetic_properties(grid,…)`, which returns the
  `StaticMagneticProperties` instance that evaluates the powder
  magnetization and susceptibility. It is not instantiated on its own.

  The static method
  `kramers_system_from_basis(basis)` decides whether a system is a Kramers
  system: the criterion is that the **sum** of the pseudospins is
  half-integer, i.e. that the system holds an odd number of electrons — not
  that the dimension of the basis is even, which is a different statement
  as soon as there is more than one site (two S = 1/2 sites span four
  states but hold two electrons).

  `pseudospin_doublet_index_list(pseudospin)` groups the states of a
  multiplet into the doublets the tabulation methods take. A Kramers
  multiplet splits evenly into doublets; a non-Kramers one holds an odd
  number of states and is left with one singlet, which is placed where it
  leaves the smallest splitting within the quasi-doublets and is reported
  by its energy alone. A non-Kramers system whose ground state is that
  singlet is an error, since the principal magnetic axes of the system are
  those of the ground doublet.

  Two axes are available as unit vectors in the input coordinate frame,
  i.e. the frame the operators of the system were given or calculated in.
  `quantization_axis()` returns the axis the pseudospin is quantized along,
  i.e. the *z* axis of the frame the pseudospin operators and the spherical
  tensors are written in. `ground_doublet_magnetic_axis()` returns the
  principal magnetic axis of the ground Kramers/Ising/pseudo doublet, i.e.
  the principal axis of its g-tensor belonging to the largest principal g
  value. When the quantization axis was chosen as that magnetic axis, which
  is the usual choice, the two agree; otherwise they differ. Both axes are
  directionless, so the sign is fixed by making the first component that is
  not numerically zero positive.

- **`electron_exchange_system.ElectronExchangeSystem`** — a general
  multi-site pseudospin system defined by crystal-field/ZFS and exchange
  tensors in the Iwahara–Chibotaru form, used to evaluate magnetization and
  susceptibility. The tensors are given as tuples of a tensor and the sites
  it acts on, all expressed in one common coordinate frame; note that the
  magnetic moment tensors imply the g-tensor, the multiplication by −μ_B
  being carried out by the class. Copies of the tensors are inflated to the
  sites of the whole system, so the tensors of the caller are left
  untouched and one tuple list can be used to build several systems.

  `hamiltonian_operator()` and `magnetic_moment_operator()` return the
  operators of the system, and `hamiltonian_tensor()` and
  `magnetic_moment_tensor()` return the tensor of the whole system (the sum
  of the given tensors, inflated). Everything listed under
  `PseudoSpinSystem` is available as well; since the tensors are used in
  the frame they were given in, the quantization axis is by construction
  the *z* axis of that frame, while `ground_doublet_magnetic_axis()` says
  where the easy axis actually lies in it.

- **`electron_exchange_system.AbInitioElectronExchangeSystem`** — builds a
  pseudospin system directly from ab initio operator matrices: projects
  the Hamiltonian and magnetic moment onto a pseudospin basis, and
  provides ITO decompositions (crystal-field and exchange parameters),
  transition magnetic moments and doublet g-tensor analyses. The
  projection and the crystal-field analysis follow [1,7]. Everything listed
  under `PseudoSpinSystem` is available; with no explicit `R` given to the
  constructor the quantization axis is the principal magnetic axis of the
  ground doublet, so the two axis methods return the same vector, and with
  an explicit `R` they differ.

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
(Lebedev–Laikov grids [9]). The coefficients of fractional parentage
follow the tabulation of [13] and the classification of [11,12], and are
built by the spectral method of [14]; the Clebsch–Gordan recursion is the
one of [15]. The leading underscore marks the package as
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
    ├── references.bib        BibTeX entries of the works cited below
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
the physical formalism the library implements — references [1,2] for the
pseudospin approach and its theory, reference [3] for the
Iwahara–Chibotaru form of the equivalent operators, and references [1,7]
for the ab initio methodology behind the crystal-field and g-tensor
analysis. If the effective barrier of the reversal of the magnetization
is reported, please cite reference [8]. If the Lebedev–Laikov integration
grids are used, please also cite reference [9], as requested by the
authors of the original quadrature routines.

The BibTeX entries of every reference below are in
[`references.bib`](references.bib) at the repository root.

## References

The BibTeX entries of all of the works below are collected in
[`references.bib`](references.bib) at the repository root, which is part of
the distribution. The citation key of each entry is given in brackets after
the reference.

### Pseudospin Hamiltonians and equivalent operators

1. L. F. Chibotaru and L. Ungur, *Ab initio calculation of anisotropic
   magnetic properties of complexes. I. Unique definition of pseudospin
   Hamiltonians and their derivation*, J. Chem. Phys. **137**, 064112
   (2012). DOI: [10.1063/1.4739763](https://doi.org/10.1063/1.4739763)
   [`Chibotaru2012`]
2. L. F. Chibotaru, *Ab Initio Methodology for Pseudospin Hamiltonians of
   Anisotropic Magnetic Complexes*, in S. A. Rice and A. R. Dinner (Eds.),
   Advances in Chemical Physics, Vol. **153**, pp. 397–519, Wiley (2013).
   DOI:
   [10.1002/9781118571767.ch6](https://doi.org/10.1002/9781118571767.ch6)
   [`Chibotaru2013`]
3. N. Iwahara and L. F. Chibotaru, *Exchange interaction between $J$
   multiplets*, Phys. Rev. B **91**, 174438 (2015). DOI:
   [10.1103/PhysRevB.91.174438](https://doi.org/10.1103/PhysRevB.91.174438)
   [`Iwahara2015`]
4. N. Iwahara, L. Ungur and L. F. Chibotaru, *$\tilde{J}$-pseudospin states
   and the crystal field of cubic systems*, Phys. Rev. B **98**, 054436
   (2018). DOI:
   [10.1103/PhysRevB.98.054436](https://doi.org/10.1103/PhysRevB.98.054436)
   [`Iwahara2018`]
5. N. Iwahara, Z. Huang, I. Neefjes and L. F. Chibotaru, *Multipolar
   exchange interaction and complex order in insulating lanthanides*,
   Phys. Rev. B **105**, 144401 (2022). DOI:
   [10.1103/PhysRevB.105.144401](https://doi.org/10.1103/PhysRevB.105.144401)
   [`Iwahara2022`]
6. A. Mansikkamäki, A. A. Popov, Q. Deng, N. Iwahara and L. F. Chibotaru,
   *Interplay of spin-dependent delocalization and magnetic anisotropy in
   the ground and excited states of $\mathrm{[Gd_2@C_{78}]^{-}}$ and
   $\mathrm{[Gd_2@C_{80}]^{-}}$*, J. Chem. Phys. **147**, 124305 (2017).
   DOI: [10.1063/1.5004183](https://doi.org/10.1063/1.5004183)
   [`Mansikkamaki2017`]

### Ab initio crystal field and magnetic properties

7. L. Ungur and L. F. Chibotaru, *Ab Initio Crystal Field for Lanthanides*,
   Chem. Eur. J. **23**, 3708–3718 (2017). DOI:
   [10.1002/chem.201605102](https://doi.org/10.1002/chem.201605102)
   [`Ungur2017`]
8. L. Ungur, M. Thewissen, J.-P. Costes, W. Wernsdorfer and L. F.
   Chibotaru, *Interplay of Strongly Anisotropic Metal Ions in Magnetic
   Blocking of Complexes*, Inorg. Chem. **52**, 6328–6337 (2013). DOI:
   [10.1021/ic302568x](https://doi.org/10.1021/ic302568x) [`Ungur2013`]

### Powder integration grids

9. V. I. Lebedev and D. N. Laikov, *A quadrature formula for the sphere of
   the 131st algebraic order of accuracy*, Dokl. Math. **59**, 477–481
   (1999). No DOI is available for this paper. [`Lebedev1999`]
10. M. Edén and M. H. Levitt, *Computation of Orientational Averages in
    Solid-State NMR by Gaussian Spherical Quadrature*, J. Magn. Reson.
    **132**, 220–239 (1998). DOI:
    [10.1006/jmre.1998.1427](https://doi.org/10.1006/jmre.1998.1427)
    [`Eden1998`]

### Angular momentum algebra and coefficients of fractional parentage

11. G. Racah, *Theory of Complex Spectra. III*, Phys. Rev. **63**, 367–382
    (1943). DOI:
    [10.1103/PhysRev.63.367](https://doi.org/10.1103/PhysRev.63.367)
    [`Racah1943`]
12. G. Racah, *Theory of Complex Spectra. IV*, Phys. Rev. **76**, 1352–1365
    (1949). DOI:
    [10.1103/PhysRev.76.1352](https://doi.org/10.1103/PhysRev.76.1352)
    [`Racah1949`]
13. C. W. Nielson and G. F. Koster, *Spectroscopic Coefficients for the
    $p^n$, $d^n$, and $f^n$ Configurations*, The M.I.T. Press, Cambridge MA
    (1963). [`Nielson1963`]
14. B. F. Bayman and A. Landé, *Tables of identical-particle fractional
    parentage coefficients*, Nucl. Phys. **77**, 1–80 (1966). DOI:
    [10.1016/0029-5582(66)90677-8](https://doi.org/10.1016/0029-5582(66)90677-8)
    [`Bayman1966`]
15. J. H. Luscombe and M. Luban, *Simplified recursive algorithm for Wigner
    $3j$ and $6j$ symbols*, Phys. Rev. E **57**, 7274–7277 (1998). DOI:
    [10.1103/PhysRevE.57.7274](https://doi.org/10.1103/PhysRevE.57.7274)
    [`Luscombe1998`]

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
asked to cite reference [9] in their publications.
