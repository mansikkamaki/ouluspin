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

The current version is **1.0.3**; see [Versioning](#versioning).

## Contents

- [Physical background](#physical-background)
- [Requirements](#requirements)
- [Installation and setup](#installation-and-setup)
- [Quick start](#quick-start)
- [Library overview](#library-overview)
- [Conventions](#conventions)
- [Examples](#examples)
- [Tests](#tests)
- [Versioning](#versioning)
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
  performs unit conversions and stores natural constants expressed in the
  chosen unit. Allowed energy units: `joule`, `wavenumber`, `kelvin`,
  `millielectronvolt`, `electronvolt`. An instance of this class is passed
  to essentially every other class in the library, and it is what fixes
  the unit every energy is read, stored and printed in; the readers of
  `ouluspin.qc` convert the data of a quantum-chemistry output into it on
  reading.

  The constants are taken from `scipy.constants`. Those that carry an
  energy — the Boltzmann constant `k_B`, the Planck constants `h` and
  `hbar`, and the Bohr and nuclear magnetons `mu_B` and `mu_N` — are
  stored in the chosen unit, which is what makes an expression such as
  `mu_B*B` come out in the unit of the calculation, and each of them is
  also available in SI as `k_B_si`, `h_si`, `mu_B_si` and so on. The
  constants that carry no unit system of their own, i.e. the free-electron
  g-factor `g_e` and the Avogadro constant `N_A`, are stored as they are.
  `convert_energy_unit(value, unit)` converts a value given in another of
  the allowed units into the unit of the system, and `energy_unit_str`
  gives the abbreviation the tables and the plots label their axes with.
  Printing the instance lists the unit, the constants and their values.

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

  `horizontal_line` draws a constant value across the plot field as a
  reference the data are compared with — the Curie χ*T* product of the free
  ion in a plot of the susceptibility, or the value the magnetization
  saturates at, both of which `IonData` gives:

  ```python
  ion = ouluspin.IonData('Dy(III)')

  chi_plot.pdf_plot('chiT.pdf',
                    horizontal_line=ion.curie_susceptibility(),
                    horizontal_line_label='Curie chiT')
  m_plot.pdf_plot('magnetization.pdf',
                  horizontal_line=ion.ising_saturation_magnetization(),
                  horizontal_line_label='Ising M(sat)')
  ```

  Any constant value is drawn the same way, and a list of values draws
  several lines. `horizontal_line_label` names the line in the legend and
  is typeset like the other labels; without it the line is drawn but does
  not enter the legend, and with it the legend appears even when the data
  sets alone would not bring one. `horizontal_line_style` chooses the style
  — `'dashed'` (the default, which separates the reference from the data at
  a glance), `'solid'`, `'dash-dot'` or `'dotted'` — and the labels and the
  styles are given either one per line or once for all of them. The
  vertical axis is widened to hold the line, so a value above the data is
  drawn as well.

  Two compatible plots are combined with `+`, e.g. two χ*T* curves into
  one plot field or two level structures side by side. The criteria are
  strict: the kinds and both axis labels must agree, and effective
  barriers are never combined, since two barriers in one field cannot be
  told apart.

  Plotting is the only part of the library needing `matplotlib`; it is
  imported by the writing methods only when they are called, so the rest
  of the library works without it.

### `ouluspin.tensors`

The tensor classes hold the **parameters** of an operator rather than its
matrix: every operator of the library is written as an expansion in
equivalent operators (an ITO expansion),

$$
\hat O = \sum_{k_1,q_1,k_2,q_2,\cdots} X_{k_1,q_1,k_2,q_2,\cdots}\,
\hat O_{k_1,q_1}(\mathbf{\tilde S}_1)\,
\hat O_{k_2,q_2}(\mathbf{\tilde S}_2)\cdots
$$

and a tensor instance stores the parameters *X* together with the ranks
*k* and components *q* that name them. One (*k*, *q*) pair belongs to each
spin site, so a one-site tensor carries rank sets of the form `[k, q]` and
a two-site tensor `[k1, q1, k2, q2]`; a term with *k* = 0 on a site
carries the identity there. The matrices themselves are built later, by
`pseudospin_operators`, from a tensor and a basis.

**Ranks and components are doubled**, exactly as the pseudospins are: a
rank-one (vector) operator is stored with `k = 2`, a rank-two operator
with `k = 4`, and `q` runs from `−k` to `k` in steps of two. The tables
print the true values, so a term stored as `[4, −4]` prints as *k* = 2,
*q* = −2.

Two tensors of the same number of sites are added with `+`, which adds
the parameters of equal rank sets, and multiplied by a scalar with `*`.
Every tensor also carries a free-form `frame` label (e.g.
`'input axis frame'`, `'principal magnetic axis frame'`) that the tables
print, that survives an addition only when both operands agree on it, and
that a rotation resets — after rotating, the caller relabels the frame it
has arrived in.

- **`IwaharaChibotaruSphericalTensor`** — an ITO expansion in the
  Iwahara–Chibotaru definition of the equivalent operators [3,4]; the
  central parameter format of the library, and the form in which
  crystal-field and exchange parameters are reported. Note that the
  definition was later revised in [5], which the library does not follow.

  An instance is built from the parameters directly,
  `IwaharaChibotaruSphericalTensor(rank_list, parameter_list)`, or from
  the operator it is meant to represent (pseudospins as multiples of two;
  the tensor is returned in the parameter form in every case):

  | Class method | Operator |
  | --- | --- |
  | `from_one_site_cartesian_operator(X, a, S)` | *X*·*S̃<sub>a</sub>*, a single Cartesian component, e.g. of a magnetic moment |
  | `from_two_site_isotropic_operator(X, S_A, S_B)` | *X*·**S̃**<sub>A</sub>·**S̃**<sub>B</sub>, a Heisenberg exchange term |
  | `from_two_site_ising_operator(X, S_A, S_B)` | *X*·*S̃*<sub>A,z</sub>·*S̃*<sub>B,z</sub>, an Ising exchange term |
  | `from_one_site_cartesian_tensor(M, S)` | **S̃**·**M**·**S̃**, e.g. a zero-field-splitting **D** tensor |
  | `from_two_site_cartesian_tensor(M, S_A, S_B)` | **S̃**<sub>A</sub>·**M**·**S̃**<sub>B</sub>, an anisotropic exchange **J** tensor |
  | `from_matrix_representation(matrix, basis)` | the expansion of an arbitrary operator matrix given in a `PseudoSpinBasis` |

  The parameters are reported by `ITO_table(symbol='X', title=None,
  order_of_magnitude=0, rank_threshold=0.0, half_table=False)`, one row
  per term with the ranks, the components and the real part, imaginary
  part and magnitude of the parameter. `rank_threshold` drops a whole rank
  combination whose parameters are all smaller than the threshold, which
  is what makes a crystal-field table readable, and `half_table=True`
  prints only the terms of *q*<sub>1</sub> ≥ 0, the rest following from
  them by a phase.

  The expansion is manipulated with `rotate(rotation)`,
  `purge_ranks(threshold=1.0e-6)` (drop negligible terms),
  `time_reversal_conjugate()` and `hermitian_conjugate()`. For
  multi-site tensors, `separate_tensors()` splits an expansion into the
  one-site tensors of each site and a remainder holding the terms that act
  on several sites at once — this is how a coupled Hamiltonian is
  separated into single-ion crystal fields and an intersite exchange
  operator (see `examples/two_site_crystal_field.py`);
  `couple_two_site_tensor()` performs the reverse coupling of a two-site
  tensor into a one-site one; `reorder_spin_sites(order)` permutes the
  sites; and `inflate_dimension(site_list)` embeds a tensor into a system
  of more sites by adding identity operators on the new ones. Note that
  `inflate_dimension` modifies the instance **in place** and returns
  nothing, so a tensor still needed in its original form is copied first.

  Back to Cartesian form: `cartesian_vector(S)` returns the rank-one part
  as a vector, `one_site_cartesian_tensor(S)` the rank-zero and rank-two
  parts as a `CartesianTensor`, and `two_site_cartesian_tensor(S_A, S_B)`
  the same for a two-site tensor.

- **`ChibotaruUngurSphericalTensor`** — a single-site ITO expansion in the
  Chibotaru–Ungur notation [1], which is the notation SINGLE_ANISO prints
  its crystal-field parameters in. The expansion is given as the real
  parameters (the coefficients of the operators *O<sub>n</sub><sup>m</sup>*)
  and the imaginary ones (the coefficients of
  *Ω<sub>n</sub><sup>m</sup>*) with their (*m*, *n*) rank–component pairs,
  together with the pseudospin the equivalent operators were built for.
  Its purpose is conversion: `iwahara_chibotaru_spherical_tensor()`
  returns the same operator in the Iwahara–Chibotaru form that the rest of
  the library uses, and `parameter_table(title)` tabulates the parameters
  as they were given.

- **`MixedCartesianIwaharaChibotaruSphericalTensor`** — an expansion of a
  Cartesian **vector** operator, carrying one Cartesian index *a* beside
  the spherical indices of the sites. The magnetic moment is such an
  operator, and the parameters
  *g<sup>a</sup><sub>k1,q1,…</sub>* generalize the usual g tensor: the
  Zeeman Hamiltonian is their contraction with the field,

  ```
  H_Zeeman = sum_{a,k1,q1,...} B_a * g^a_{k1,q1,...} * O_{k1,q1}(S_1) * ...
  ```

  The instance holds the three Cartesian components as
  `IwaharaChibotaruSphericalTensor` instances, reached with
  `component('x')`; `contract_with_vector(B)` performs the contraction
  above and returns the ordinary spherical tensor of one field direction,
  and `one_site_cartesian_tensor(S)` returns the rank-one part as the g
  tensor it corresponds to. It is built from an ab initio vector operator
  with `from_general_vector_operator_matrix(vector_matrix, basis)` or, for
  a free-ion-like moment, with
  `from_one_site_isotropic_operator(parameter, S)` — the value
  −*g<sub>J</sub>*·*μ*<sub>B</sub> gives the magnetic moment of a
  multiplet of Landé factor *g<sub>J</sub>*. The methods it shares with
  the scalar tensors (`rotate`, `purge_ranks`, `separate_tensors`,
  `inflate_dimension`, `reorder_spin_sites`, `ITO_table`, the two
  conjugates) act on the three components together.

- **`CartesianTensor`** — an ordinary 3×3 tensor, i.e. the familiar form of
  a g tensor, a zero-field-splitting **D** tensor or an exchange **J**
  tensor. It decomposes the tensor into the parts that have a physical
  meaning of their own — `trace()`, `isotropic_part()` (a third of the
  trace), `symmetric_part()` (traceless, the anisotropy) and
  `antisymmetric_part()` (the Dzyaloshinskii–Moriya-like part) — and
  converts to the spherical form with `rank_two_ito(S)`, which needs a
  symmetric tensor and returns the rank-zero and rank-two parameters, and
  `rank_one_ito(axis, S)`, which returns the rank-one parameters of one
  Cartesian row or column. A symmetric tensor is diagonalized upon
  construction, so `eigenvalues` and `eigenvectors` hold the principal
  values and principal axes; `principal_axis_table()` tabulates them (and
  returns `None` for a tensor that is not symmetric) and `tensor_table()`
  tabulates the tensor with its isotropic, symmetric and antisymmetric
  parts.

- **`Rotation`** — a rotation matrix together with the Euler angles of the
  Z–Y–Z convention, *R* = *R<sub>z</sub>*(γ)·*R<sub>y</sub>*(β)·*R<sub>z</sub>*(α).
  The matrix is checked on construction (real, orthogonal, determinant
  +1, and consistent with the angles derived from it), so an instance
  always represents a proper rotation. `wigner_D(J, M1, M2)` returns the
  Wigner *D* matrix element of the rotation (all three arguments doubled),
  which is what rotates a spherical tensor, and `inverse()` returns the
  reverse rotation. Rotations are what carry a calculation between
  coordinate frames: the qc readers return one with `read_rotation()`, and
  the system classes use them to report every quantity in the input frame
  of the ab initio calculation.

A tensor and a basis are all that is needed to build and diagonalize an
operator. Two coupled S = 1/2 sites, for instance:

```python
from ouluspin import tensors, pseudospin_operators, units as units_module

units = units_module.EnergyUnitSystem('wavenumber')

# Pseudospins are given as multiples of two, so [1,1] is two S = 1/2 sites.
basis    = pseudospin_operators.PseudoSpinBasis([1,1])
exchange = tensors.IwaharaChibotaruSphericalTensor\
                  .from_two_site_isotropic_operator(-5.0, 1, 1)

hamiltonian = pseudospin_operators.PseudoSpinOperator(basis, [exchange],
                                                      units)
print(hamiltonian.eigenvalue_table())
```

which gives the triplet at −1.25 cm⁻¹ and the singlet at 3.75 cm⁻¹, i.e.
a splitting of 5 cm⁻¹: with this sign convention a negative parameter is
a ferromagnetic coupling.

### `ouluspin.pseudospin_operators`

Where `tensors` holds the parameters of an operator, this module holds its
**matrix**: a basis plus a list of tensors gives a matrix representation,
which is built by the Fortran extension and diagonalized.

- **`PseudoSpinBasis`** — the basis states of a multi-site pseudospin
  system, |*S*<sub>0</sub>,*M*<sub>0</sub>⟩ ⊗ |*S*<sub>1</sub>,*M*<sub>1</sub>⟩ ⊗ …,
  constructed from the list of the pseudospins of the sites (as multiples
  of two, so `[15,1]` is a *J* = 15/2 ion coupled to an *S* = 1/2 radical).
  The states are ordered ascending in the projection and site by site, and
  `basis_table()` prints them with their labels, which is the quickest way
  to find the order a set of ab initio states has to be brought into.
  `conjugate_state_list()` and `conjugate_state_table()` pair the states
  that are time-reversal conjugates of each other, and
  `unitary_part_of_time_reversal_operator()` gives the operator itself;
  these are what the time-reversal checks of the system classes rest on.
  The number of states is `n_basis` and the number of sites `n_sites`.
- **`PseudoSpinOperator`** — constructs and diagonalizes a multi-site
  pseudospin operator from a basis, a list of
  `IwaharaChibotaruSphericalTensor` instances (which are summed) and an
  `EnergyUnitSystem`. The operator is assumed Hermitian, and the
  eigenvalues are ordered from the lowest up. `diagonalize_operator_matrix`
  and `store_operator_matrix` decide how much work is done and how much is
  kept, and `translate_eigenvalues=True` shifts the spectrum so that the
  ground state sits at zero, which is what one wants of a Hamiltonian and
  not of anything else. `construct_matrix()` returns the matrix itself.

  The spectrum is reported by `eigenvalue_table()`, by
  `eigenvector_table()`, which lists the composition of each eigenstate in
  the basis states, and by `compact_eigenvector_table()`, which puts every
  eigenstate on a single row with one column per basis state labelled by
  its projection *M* — the familiar table of the composition of a
  crystal-field spectrum, available for a basis of one spin site only,
  i.e. for the *J* multiplet of a lanthanide(III) ion and the like. All of
  them take a print limit, so that only the lowest states of a large basis
  are tabulated. `from_vector_operator(basis, vector, mixed_tensors, units)`
  builds the operator by contracting a Cartesian vector with mixed
  Cartesian–spherical tensors, i.e. it is how a Zeeman operator of a given
  field direction is constructed; the vector and the tensors must be
  expressed in the same coordinate frame.
- **`PseudoSpinVectorOperator`** — a vector (three-component) pseudospin
  operator, built from a basis and mixed Cartesian–spherical tensors. It
  is the container of the magnetic moment operator of a pseudospin system;
  `matrix_list()` returns the three Cartesian matrices, which is the form
  the property calculations take.
- **`GeneralOperatorMatrix`** — stores a matrix representation of an
  operator in an arbitrary basis (e.g. an ab initio Hamiltonian matrix)
  and diagonalizes it; the standard container for ab initio input, and the
  form in which the readers of `ouluspin.qc` are handed to the system
  classes. It takes the same `diagonalize_operator_matrix` and
  `translate_eigenvalues` arguments as `PseudoSpinOperator` and carries
  the resulting `eigenvalues` and `eigenvectors` as attributes.
- **`GeneralVectorOperatorMatrix`** — the vector-operator counterpart of
  `GeneralOperatorMatrix`, i.e. the three Cartesian components of a vector
  operator such as the magnetic moment, diagonalized and reached with
  `matrix_list()`.

### `ouluspin.properties`

- **`StaticMagneticProperties`** — evaluates static magnetic properties
  from the field-free Hamiltonian and the magnetic moment operator of a
  system. What is calculated is decided by what is handed to it: an
  `IsothermalStaticMagnetization` instance, a
  `StaticMagneticSusceptibility` instance, or both, each of which states
  the temperatures and fields to calculate at and holds the results
  afterwards. The operators may be either the pseudospin operators of a
  model or the ab initio matrices, and both must carry their matrix
  representations; `from_electron_exchange_system(system, grid, …)` and
  `from_ab_initio_system(system, grid, …)` take them from a system
  instance instead, which is the usual route.

  `sample_orientation` decides how the molecules of the sample are taken
  to be oriented: `'powder'` (the default) integrates over the grid,
  `'free'` lets the molecules rotate and weights the directions by
  Boltzmann statistics, and `'maximal'` searches the grid for the
  direction of the largest magnetization at each temperature. `n` is the
  number of magnetic subsystems per mole of sample and simply scales the
  result.

  The cost of the calculation is proportional to the number of grid
  points, since the Hamiltonian is diagonalized once per point and per
  field strength; choosing the grid is therefore the one decision that
  sets the cost. The class documentation carries a benchmark of the two
  grids on a strongly anisotropic Dy(III) system and the recommendation
  that follows from it (see [`ouluspin.integration`](#ouluspinintegration)).
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
  transition magnetic moment matrix elements between states, i.e.
  (|⟨I|μ<sub>x</sub>|J⟩| + |⟨I|μ<sub>y</sub>|J⟩| + |⟨I|μ<sub>z</sub>|J⟩|)/3.
  These are read as the qualitative effective barrier of the reversal of
  the magnetization of a single-molecule magnet [8], and `ResultPlot`
  draws exactly that picture from an instance. The elements are evaluated
  in the basis that diagonalizes the projection of the magnetic moment
  within each group of degenerate states, not in the arbitrary basis a
  diagonalization returns for a degenerate subspace — otherwise both the
  expectation values and the matrix elements between two doublets would
  depend on that arbitrary choice.
- **`PseudoSpinDoublet`** — properties of a doublet described by an
  effective S = 1/2: the g-tensor and its principal axes, the tunneling
  gap, the energies of the two states, and the Kramers/non-Kramers
  classification. The g-tensor follows Section III.A of [1].

### `ouluspin.integration`

Spherical grids for powder averaging. A grid point is an orientation, i.e.
a point on the unit sphere: every grid carries its points as Cartesian
unit vectors in `vectors` together with their `weights` (the ZCW grid
also keeps the Euler angles they were built from), and `n_grid_points`
counts them. A grid is handed as it is to `StaticMagneticProperties`, and
`data_table()` tabulates the points.

- **`ZCWGrid`** — Zaremba–Conroy–Wolfsberg grids, in the form of Appendix 1
  of [10]. The size is set by the parameter `M` (default 5, i.e. 233
  points), and `integration_range` covers the whole `'sphere'` (default),
  a `'hemisphere'` or an `'octant'`, the smaller ranges being usable when
  the symmetry of the system allows it. The points are distributed
  uniformly, which is what makes this the better choice when the
  magnetization is calculated, and the better choice overall.
- **`LebedevLaikovGrid`** — Lebedev–Laikov quadrature grids [9] (built on
  the original Fortran 77 routines). The size is set by `grid_quality`, an
  integer from 1 to 32 (default 17, i.e. 590 points). The grid is a
  quadrature exact for polynomials up to an order, so it integrates the
  susceptibility exactly already at `grid_quality = 7` (86 points); the
  saturated magnetization is not a smooth function of the direction, and
  there it converges slowly and erratically.
- **`SimpleGrid`** — one or three Cartesian unit vectors, for single-axis
  or axis-resolved calculations rather than powder averaging.
  `grid_type=0` (default) gives the three unit vectors and `grid_type=1` a
  single direction given by `grid_vector`.

### `ouluspin.data_tables`

- **`IonData`** — the data of a chemical ion, e.g. `IonData('Dy(III)')`. The
  name is parsed case-insensitively and accepts the oxidation state as an
  Arabic or a Roman numeral with the usual punctuation (`'Dy(III)'`,
  `'dy iii'`, `'Dy-III'`, `'Dy3'`, `'Dy 3+'`); a Roman numeral must be
  separated from the symbol, since `'Siii'` would otherwise be both Si(II)
  and S(III). Leaving the oxidation state out uses the state the element is
  ordinarily met in, so `'Dy'` is Dy(III) and `'Fe'` is Fe(III).

  The instance carries the electron configuration of the ion
  (`electron_configuration`, `valence_configuration`), the terms of its open
  shell (`terms`, `term_data`, `n_terms`, `n_states`, `n_spin_states`) and
  the quantum numbers of its
  Hund's-rule ground multiplet (`L`, `S`, `J`, `term_symbol`,
  `ground_multiplet_symbol`, `degeneracy`) together with its Landé
  g-factor in three forms: `lande_g_factor`, evaluated with the CODATA
  g-factor of the free electron (1.33410643 for Dy(III)) and meant for
  quantitative work; `lande_g_factor_simple`, the textbook approximation
  g_e = 2 (1.33333333); and `lande_g_factor_str`, the exact fraction the
  latter is (`'4/3'`), evaluated in exact arithmetic. The angular
  momenta are stored in the **doubled** form used throughout the library,
  i.e. `J = 15` for the J = 15/2 of Dy(III), and are printed as the true
  angular momenta in the tables. `J` is therefore the pseudospin to hand to
  `PseudoSpinBasis` or to `AbInitioElectronExchangeSystem.from_aniso_data`.

  `curie_susceptibility()` returns the χT product of the free ion given by
  the Curie law for its ground multiplet, in cm³ K mol⁻¹ — 14.19 for
  Dy(III) with the accurate g-factor, which is the default, and the 14.17
  of the textbook tables with `simple_g_factor=True`, those tables being
  computed in the same g_e = 2 approximation.

  `saturation_magnetization()` and `ising_saturation_magnetization()` return
  the magnetization the ground multiplet saturates at as B/T → ∞, in units
  of the Bohr magneton, and take the same `simple_g_factor` argument. The
  first is the isotropic value g_J·J of the free ion, whose whole ground
  multiplet is available to the field — 10 μ_B for Dy(III). The second is
  the **powder** value of an ion whose ground state is a maximally axial
  Ising-type doublet of M_J = ±J alone in the low-energy region: such a
  doublet has g_z = 2 g_J J and g_x = g_y = 0, so only the field along the
  easy axis acts on it, and averaging |cos θ| over the sphere gives one
  half. The Ising value is therefore half the isotropic one — 5 μ_B for
  Dy(III), the saturation ordinarily met in a Dy(III) single-molecule
  magnet — while along the easy axis of a single crystal the doublet
  saturates at the isotropic g_J·J.

  `coefficients_of_fractional_parentage()` returns the CFPs of the
  open shell; they are evaluated on the first call rather than upon
  construction. `states_by_multiplicity()` groups the terms and the states
  of the open shell by spin multiplicity. `data_table()`,
  `ground_multiplet_table()`, `term_table()` and `cfp_table()` return
  `ResultTable` instances, and `repr` of the instance is the plain-text
  rendering of `data_table()`. Both tables give the Landé g-factor, the χT
  product and the two saturation magnetizations in both conventions, side
  by side. The data table further
  lists the terms of the open
  shell and, beneath them, the number of terms, of spin states and of
  states of each spin multiplicity together with their totals — for the 4f9
  of Dy(III), 73 terms holding 735 spin states and 2002 states. The **spin
  states** of a multiplicity are the states of one spin component alone,
  i.e. the states divided by the multiplicity: a 6H term holds 6×11 = 66
  states and 11 spin states, so they are the states a spin-free calculation
  of that multiplicity carries.

  Only ions of a single open shell are treated, i.e. the d-block (Sc–Zn,
  Y–Cd, Hf–Hg) and f-block (La–Lu, Ac–Lr) ions whose electrons outside the
  noble-gas core all belong to one d or one f shell. An oxidation state too
  low for that (a neutral d-block atom, a mono-positive f-block ion) or too
  high (which would break into the core) is a fatal error stating which of
  the two is the case; a state that is possible but not chemically common is
  accepted with a warning, since the assignment of the whole valence to the
  open shell is then not certain — La(II) is taken here as 4f1 whereas it is
  in fact 5d1.

  The term structure is not tabulated but calculated: the terms come from
  `cfp_terms` of the Fortran extension and Hund's rules pick the ground term
  out of them. What is tabulated is only what does not follow from the
  physics, i.e. the chemical symbols, the blocks of the periodic table and
  the common and ordinary oxidation states of each element.

- **`MultipleIonData`** — the data of a whole set of ions gathered into
  tables of one ion per row, e.g. `MultipleIonData('lanthanides', 3)` for
  the trivalent lanthanides. Unlike `IonData`, which provides the
  quantitative values as well, this class is meant for the production of
  human-readable tables alone; every value it prints is taken from the
  `IonData` instance of the ion, which the `ion(name)` method returns.

  The scope is given by a set of elements and a set of oxidation states,
  both read case-insensitively. The elements are chemical symbols or one of
  the shorthands `'lanthanides'`/`'4f'`, `'actinides'`/`'5f'`, `'3d'`,
  `'4d'`, `'5d'` (Hf–Hg), `'d-block'`, `'f-block'` and `'all'`; the
  oxidation states are Arabic numbers or Roman numerals (`3`, `'III'`), or
  one of the shorthands `'common'` (the chemically common states of each
  element, the default), `'ordinary'` (the single state the element is
  ordinarily met in) and `'all'` (every state that leaves a single open
  shell). The rows are ordered first by atomic number and then by
  decreasing oxidation state.

  A combination of an element and an oxidation state that `IonData` refuses,
  i.e. one that leaves no single open shell, is kept as a row whose values
  are printed as dashes and named in a note of the table; a scope leaving no
  data at all, or an unrecognized element or oxidation state, is an error.

  The tables, all of them `ResultTable` instances, are
  `ground_multiplet_table()` (the valence configuration and S, L, J),
  `ground_magnetism_table()` (S, L, J, the Landé g-factor, the Curie χT
  product and the isotropic and Ising saturation magnetizations, all of
  them in both conventions for the free-electron g-factor), `number_of_spin_states_table()` (the number of spin states of
  each spin multiplicity, one column per multiplicity, the largest first,
  the multiplicities a configuration does not carry being dashes),
  `state_count_table()` (the number of terms, of spin states and of states)
  and `configuration_table()` (the atomic number, the electron count, the
  block, the core and the valence). `repr` of the instance is the
  plain-text rendering of `ground_multiplet_table()`.

### `ouluspin.qc`

Readers for quantum-chemistry outputs. All readers take an
`EnergyUnitSystem` and convert data to the chosen unit on reading. The
quantities they return are the input of everything else in the library,
and they come in two kinds: the **operator matrices** of a datafile, which
carry the whole ab initio result and are handed to
`GeneralOperatorMatrix` and `GeneralVectorOperatorMatrix`, and the
**parameters** already extracted by SINGLE_ANISO, which are returned
directly as tensors of `ouluspin.tensors`.

Everything a reader returns is expressed in the input axis frame, i.e. in
the coordinate frame of the ab initio calculation, which is what the
`frame` labels of the library refer to.

- **`orca.OrcaAnisoFile`** — reads operator matrices from a `.anisofile`
  produced by an ORCA calculation: `hamiltonian()` returns the matrix of
  the spin–orbit-coupled Hamiltonian, `magnetic_moment()` the three
  Cartesian components of the magnetic moment (with
  `include_bohr_magneton=False` when the moment is wanted in units of the
  Bohr magneton, which is what the pseudospin analysis takes), and
  `spin()` the three components of the spin. This is the reader the
  examples use.
- **`molcas.OpenMolcasCalculation`** — reads SINGLE_ANISO data from an
  OpenMolcas output file.
- **`orca.OrcaCalculation`** — reads data from an ORCA output file.
- **`orca.OrcaAnisoOutput`** — reads a standalone SINGLE_ANISO output file
  produced via ORCA.
- **`molcas.AnisoCalculation`** — the common base class of the three
  readers of SINGLE_ANISO output above, not instantiated directly. It
  defines what can be read from such an output: `read_pseudospin()`, the
  magnitude of the pseudospin of a multiplet; `read_magnetic_moment()`,
  the ITO expansion of the magnetic moment as a mixed
  Cartesian–spherical tensor; `read_zfs_tensor()`, the ITO expansion of
  the zero-field-splitting operator; `read_crystal_field()`, the ab initio
  crystal-field parameters of a lanthanide, available only when the
  crystal-field calculation was requested of SINGLE_ANISO; and
  `read_rotation()`, the rotation from the input frame to the principal
  magnetic axis frame, as a `Rotation`. A `multiplet_index` argument
  selects the multiplet, and `output_instance` selects which run to read
  from a file holding several SINGLE_ANISO outputs.

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
  S = 15/2, `[15, 1]` means two sites with S = 15/2 and S = 1/2. The
  **ranks and components of the tensor operators are doubled as well**, so
  a rank-one (vector) operator is stored with `k = 2` and a rank-two
  operator with `k = 4`, and `q` runs from `−k` to `k` in steps of two.
  Doubled values are what the attributes hold; the tables print the true
  values, i.e. J = 15/2 and k = 2.
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
- **Errors are fatal and are reported, not raised.** A mistake in the
  input of a class is printed as

      ERROR in IonData.
      ERROR: There is no element of the chemical symbol 'Xx'.
      Error termination.

  and stops the interpreter with exit status 1; a message of several lines
  is marked on every line. A condition that is suspicious but not fatal is
  printed the same way under `WARNING in <class>.` with `Warning:` on each
  line, and the calculation goes on. Every class reports through its own
  private `__error` and `__warning` methods, and the header names the class
  of the instance, so an error raised inside a base class points at the
  class that was actually built.
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
- `aniso_analysis.py` — the whole analysis of a single lanthanide(III) ion
  from an ORCA aniso file, i.e. what SINGLE_ANISO reports: the pseudospin
  doublets with their g-tensors and principal magnetic axes, the
  crystal-field parameters and states, the transition magnetic moments and
  the powder susceptibility and magnetization. The results go to the
  standard output, into an `.odt`, `.docx` or LaTeX document, and into
  plots, and the quantization axis is written into a file a molecular
  viewer reads. It takes the name of the ion as a second argument
  (`ouluspin-python aniso_analysis.py Dy_complex.anisofile 'Dy(III)'`),
  from which `IonData` gives the pseudospin and the free-ion values the
  plots are compared with.

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

## Versioning

The version of the library is read from the package:

```python
import ouluspin
ouluspin.__version__        # '1.0.3'
```

The numbering is [semantic versioning](https://semver.org), i.e.
MAJOR.MINOR.PATCH, and the number is raised according to the scope of the
change:

- **MAJOR** — a change that breaks the scripts written against the previous
  version: a class or a method removed or renamed, an argument removed or
  its meaning changed, or a change in the physics or in the conventions
  that changes the numbers a previous version produced.
- **MINOR** — new functionality that leaves the existing scripts working: a
  new class, a new method, a new optional argument, a new output format.
- **PATCH** — a fix to a wrong result or to a broken code path, a
  correction to the text of a message or of a table, added tests and
  documentation, and internal reorganization that leaves the behaviour
  alone.

Version 1.0.0 is the first numbered version of the library. The number in
use before it, 0.1.0, was a placeholder and does not describe any
particular state of the code.

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
    │   │   ├── data_tables.py           Data of the ions (IonData,
    │   │   │                            MultipleIonData)
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
3. N. Iwahara and L. F. Chibotaru, *Exchange interaction between J
   multiplets*, Phys. Rev. B **91**, 174438 (2015). DOI:
   [10.1103/PhysRevB.91.174438](https://doi.org/10.1103/PhysRevB.91.174438)
   [`Iwahara2015`]
4. N. Iwahara, L. Ungur and L. F. Chibotaru, *J̃-pseudospin states and the
   crystal field of cubic systems*, Phys. Rev. B **98**, 054436 (2018). DOI:
   [10.1103/PhysRevB.98.054436](https://doi.org/10.1103/PhysRevB.98.054436)
   [`Iwahara2018`]
5. N. Iwahara, Z. Huang, I. Neefjes and L. F. Chibotaru, *Multipolar
   exchange interaction and complex order in insulating lanthanides*,
   Phys. Rev. B **105**, 144401 (2022). DOI:
   [10.1103/PhysRevB.105.144401](https://doi.org/10.1103/PhysRevB.105.144401)
   [`Iwahara2022`]
6. A. Mansikkamäki, A. A. Popov, Q. Deng, N. Iwahara and L. F. Chibotaru,
   *Interplay of spin-dependent delocalization and magnetic anisotropy in
   the ground and excited states of [Gd₂@C₇₈]⁻ and [Gd₂@C₈₀]⁻*,
   J. Chem. Phys. **147**, 124305 (2017).
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
    pⁿ, dⁿ, and fⁿ Configurations*, The M.I.T. Press, Cambridge MA
    (1963). [`Nielson1963`]
14. B. F. Bayman and A. Landé, *Tables of identical-particle fractional
    parentage coefficients*, Nucl. Phys. **77**, 1–80 (1966). DOI:
    [10.1016/0029-5582(66)90677-8](https://doi.org/10.1016/0029-5582(66)90677-8)
    [`Bayman1966`]
15. J. H. Luscombe and M. Luban, *Simplified recursive algorithm for Wigner
    3j and 6j symbols*, Phys. Rev. E **57**, 7274–7277 (1998). DOI:
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
