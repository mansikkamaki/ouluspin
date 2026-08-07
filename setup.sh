#!/bin/bash
# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

#
# Environment setup for the OuluSpin library. Source this file, do not
# execute it:
#
#     source /path/to/ouluspin/setup.sh
#
# After sourcing, OuluSpin scripts can be run with
#
#     ouluspin-python <script.py>
#
# or directly with "$OULUSPIN_PYTHON" <script.py>.
#
# The script
#
#   1. adds the src directory of the repository to PYTHONPATH so that
#      "import ouluspin" works from anywhere,
#   2. selects the Python interpreter used to run OuluSpin and exports it
#      as OULUSPIN_PYTHON, defining the convenience command ouluspin-python,
#   3. adds to LD_LIBRARY_PATH the runtime library directories the compiled
#      Fortran extension module needs (the Intel MKL and the Intel Fortran
#      runtime, when the module was built with the Intel toolchain).
#
# The runtime library directories of step 3 are taken from
# src/fortran/make.inc when it exists, i.e. from the directories the
# configure script of the extension module recorded when the module was
# built. This is the reliable source, since an Intel oneAPI installation
# holds several releases side by side and only the one the module was built
# against will do. If make.inc is absent, the directories of the newest
# release of the oneAPI installation are used instead.
#
# The interpreter is chosen as follows: if OULUSPIN_PYTHON is already set
# in the environment, it is respected; otherwise, if an Intel Distribution
# for Python is found under the oneAPI installation, it is used (a Fortran
# extension built with the Intel toolchain generally requires it);
# otherwise the system python3 is used.
#
# The Intel oneAPI installation is looked for at $ONEAPI_ROOT, defaulting
# to /opt/intel/oneapi. On systems without oneAPI everything Intel-related
# is silently skipped and the system python3 is used.

# Directory of this script (the repository root). Works in both bash and
# zsh when sourced.
if [ -n "${BASH_SOURCE[0]}" ]; then
    _ouluspin_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    _ouluspin_root="$(cd "$(dirname "$0")" && pwd)"
fi

# Root of an Intel oneAPI installation, if any.
_ouluspin_oneapi="${ONEAPI_ROOT:-/opt/intel/oneapi}"

# Prepend a directory to a path-like variable only if the directory exists
# and is not already present, to keep repeated sourcing idempotent.
_ouluspin_prepend_path() {
    # $1 = variable name, $2 = directory
    local _var="$1" _dir="$2" _current
    [ -d "${_dir}" ] || return 0
    eval "_current=\"\${${_var}}\""
    case ":${_current}:" in
        *":${_dir}:"*) ;;
        *) eval "export ${_var}=\"${_dir}\${${_var}:+:\${${_var}}}\"" ;;
    esac
}

# 1. Make "import ouluspin" work from anywhere.
_ouluspin_prepend_path PYTHONPATH "${_ouluspin_root}/src"

# 2. The Python interpreter: respect a pre-set OULUSPIN_PYTHON, then prefer
#    the Intel Python if present, then fall back to the system python3.
if [ -z "${OULUSPIN_PYTHON}" ]; then
    if [ -x "${_ouluspin_oneapi}/intelpython/latest/bin/python3" ]; then
        OULUSPIN_PYTHON="${_ouluspin_oneapi}/intelpython/latest/bin/python3"
    else
        OULUSPIN_PYTHON="$(command -v python3)"
    fi
fi
export OULUSPIN_PYTHON

ouluspin-python() {
    "${OULUSPIN_PYTHON}" "$@"
}

# 3. Runtime libraries for a fortran_utils.so built with the Intel
#    toolchain (MKL and the Intel Fortran runtime). These directories are
#    only added if they exist; on systems without oneAPI this does nothing.
#
#    The directories recorded in src/fortran/make.inc by the configure
#    script of the extension module are used when the file is there. The
#    line it writes has the form
#
#        export LD_LIBRARY_PATH := <dir>:<dir>:$(LD_LIBRARY_PATH)
#
_ouluspin_make_inc="${_ouluspin_root}/src/fortran/make.inc"
_ouluspin_libdirs=""
if [ -f "${_ouluspin_make_inc}" ]; then
    _ouluspin_libdirs="$(sed -n 's/^export LD_LIBRARY_PATH *:= *\(.*\):\$(LD_LIBRARY_PATH) *$/\1/p' \
                         "${_ouluspin_make_inc}")"
fi

if [ -n "${_ouluspin_libdirs}" ]; then
    # The directories of the build, in the order make.inc lists them. The
    # last one is prepended first, so that the resulting order is kept.
    _ouluspin_libdirs="$(echo "${_ouluspin_libdirs}" | tr ':' '\n' | tac)"
    while IFS= read -r _ouluspin_dir; do
        [ -n "${_ouluspin_dir}" ] && \
            _ouluspin_prepend_path LD_LIBRARY_PATH "${_ouluspin_dir}"
    done <<EOF
${_ouluspin_libdirs}
EOF
else
    # No make.inc: fall back to the newest release of the oneAPI
    # installation. Both the directory layout used since oneAPI 2024 and the
    # older one are covered, since neither is present in every release.
    _ouluspin_prepend_path LD_LIBRARY_PATH \
        "${_ouluspin_oneapi}/compiler/latest/linux/compiler/lib/intel64_lin"
    _ouluspin_prepend_path LD_LIBRARY_PATH "${_ouluspin_oneapi}/compiler/latest/lib"
    _ouluspin_prepend_path LD_LIBRARY_PATH "${_ouluspin_oneapi}/mkl/latest/lib/intel64"
    _ouluspin_prepend_path LD_LIBRARY_PATH "${_ouluspin_oneapi}/mkl/latest/lib"
fi

unset -f _ouluspin_prepend_path
unset _ouluspin_root _ouluspin_oneapi _ouluspin_make_inc _ouluspin_libdirs _ouluspin_dir
