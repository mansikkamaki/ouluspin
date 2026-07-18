# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

# Makefile for building the fortran_utils Python extension module of the
# OuluSpin library from the Fortran sources in this directory.
#
# Usage:
#
#     ./configure        # detect compiler, Python and LAPACK; writes make.inc
#     make               # build and install into ../ouluspin/_fortran
#
# Build configuration (compiler, Python interpreter, flags, LAPACK/BLAS
# libraries) lives in make.inc, which is created by ./configure and can be
# edited by hand afterwards; see make.inc.example for the documented
# variables.
#
# Targets:
#     all        (default) build the module and install it into the package
#     build      build fortran_utils.so in this directory only
#     install    copy fortran_utils.so into ../ouluspin/_fortran
#     test       build, install and run the Fortran test suite
#     clean      remove build artifacts (objects, modules, the local .so)
#     distclean  clean + remove make.inc and the generated signature file
#
# The build is done in three steps:
#   1. Compile the Fortran 77 Lebedev--Laikov routines to an object file.
#   2. Generate the f2py signature file (fortran_utils.pyf) from all the
#      Fortran 90 sources so that every routine is exposed to Python.
#   3. Compile and link everything into the fortran_utils extension module.

ifeq ($(wildcard make.inc),)
$(error make.inc not found -- run ./configure first, or copy and edit \
make.inc.example)
endif
include make.inc

MODULE  := fortran_utils
DESTDIR := ../ouluspin/_fortran

# The Fortran 77 Lebedev--Laikov grid routines. These are compiled
# separately into an object file which is linked into the module in the
# final step.
F77_SOURCE := lebedev_laikov.f
F77_OBJECT := lebedev_laikov.o

# All Fortran 90 module sources. The order matters: a module must appear
# after every module it uses. grid_utils.f90 provides the interface to the
# Lebedev--Laikov routines in $(F77_OBJECT).
F90_SOURCES := num_utils.f90 \
               math_utils.f90 \
               matrix_utils.f90 \
               angm_utils.f90 \
               cg_utils.f90 \
               iwahara_utils.f90 \
               cfp_utils.f90 \
               statmech_utils.f90 \
               powder_magnetization_utils.f90 \
               pseudospin_utils.f90 \
               grid_utils.f90

# Fortran-internal helper routines that must not get f2py wrappers. The
# functions log_factorial, cg_recursion_* and triangle_coefficient_log
# return real(real128) (quadruple precision) values, for which f2py cannot
# generate valid wrapper code. The remaining names are private helpers of
# cfp_utils that are not part of its exported interface. They are excluded
# from the signature file and therefore from the Python interface.
SKIPPED_ROUTINES := log_factorial cg_recursion_a cg_recursion_b \
                    cg_recursion_d triangle_coefficient_log wigner_6j_r128 \
                    validate_configuration validate_tensor_rank ensure_shell \
                    build_shell build_level \
                    find_term pair_overlap_factor phase_factor \
                    binomial_coefficient \
                    spin_sixj orbital_sixj u5_reduced_element \
                    canonicalize_subspace sort_records \
                    u_tensor_element_by_index v11_tensor_element_by_index

.PHONY: all build install test clean distclean

all: install

build: $(MODULE).so

$(F77_OBJECT): $(F77_SOURCE)
	$(FC) -c -fPIC $(F77FLAGS) $< -o $@

# The f2py signature file exposing the routines to Python; regenerated
# whenever a Fortran 90 source changes.
$(MODULE).pyf: $(F90_SOURCES)
	$(PYTHON) -m numpy.f2py -h $(MODULE).pyf -m $(MODULE) \
	    --overwrite-signature $(F90_SOURCES) \
	    skip: $(SKIPPED_ROUTINES) :

# f2py names the module after the running Python version, e.g.
# fortran_utils.cpython-39-x86_64-linux-gnu.so. It is renamed to plain
# fortran_utils.so, which the import machinery accepts regardless of the
# interpreter tag.
$(MODULE).so: $(MODULE).pyf $(F77_OBJECT) $(F90_SOURCES)
	$(PYTHON) -m numpy.f2py -c $(F2PY_FLAGS) $(LAPACK_LIBS) \
	    $(MODULE).pyf $(F77_OBJECT) $(F90_SOURCES)
	@set -e; for f in $(MODULE).cpython-*.so; do \
	    if [ -e "$$f" ]; then mv "$$f" $(MODULE).so; fi; \
	done
	@test -e $(MODULE).so

install: build
	cp $(MODULE).so $(DESTDIR)/$(MODULE).so
	@echo "Installed $(MODULE).so into $(DESTDIR)"

test: install
	PYTHONPATH="$(CURDIR)/..:$$PYTHONPATH" \
	    $(PYTHON) ../../tests/fortran_tests.py

clean:
	rm -f *.o *.mod $(MODULE).so

distclean: clean
	rm -f make.inc $(MODULE).pyf
