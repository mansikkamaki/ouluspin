# OuluSpin examples

Worked examples of typical OuluSpin calculations. Each script needs an
input data file from an ab initio calculation (not included in the
repository because of their size); the required file type is documented at
the top of each script.

Run the scripts with the environment set up (see the README at the
repository root):

    source /path/to/ouluspin/setup.sh
    ouluspin-python one_site_crystal_field.py <input file>

Scripts:

- `one_site_crystal_field.py` — extract the crystal-field Hamiltonian of a
  single J = 15/2 multiplet (e.g. a Dy(III) complex) from an ORCA aniso
  file and decompose it into irreducible tensor operators.
- `two_site_crystal_field.py` — decompose the Hamiltonian of a coupled
  two-site system (S = 15/2 and S = 1/2) into the crystal fields of the
  individual sites and the intersite exchange operator.
- `average_crystal_field.py` — average the crystal fields of two
  OpenMolcas / SINGLE_ANISO calculations (one electron removed / added)
  after rotation into a common magnetic frame, and analyze the resulting
  system.
- `aniso_analysis.py` — the whole analysis of a single lanthanide(III)
  ion from an ORCA aniso file, i.e. what SINGLE_ANISO reports: the
  pseudospin doublets with their g-tensors and magnetic axes, the
  crystal-field parameters, the crystal-field states and their
  compositions, the transition magnetic moments, and the powder
  susceptibility and magnetization. Writes the tables into an `.odt`,
  `.docx` or LaTeX document, the plots as images, and the quantization
  axis into a file of its own. Takes the name of the ion as a second
  argument:

      ouluspin-python aniso_analysis.py Dy_complex.anisofile 'Dy(III)'
