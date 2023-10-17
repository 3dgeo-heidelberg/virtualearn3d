#!/bin/bash

# ############################################################################
#
# AUTHOR : Alberto M. Esmoris Pena
#
#
# BRIEF: Call this script to automatically build the sphinx-based HTML
#        documentation.
#
# WARNING:  This script MUST be called from the virtualearn3d/doc directory
#           where it is contained. Otherwise it might fail.
#
# ############################################################################


# ------------------------- #
# ---  BUILD FUNCTIONS  --- #
# ------------------------- #
function generate_modules {
    echo 'Generating modules ...'
    sphinx-apidoc -o source_src ../src
    echo 'Modules generated!'
}

function merge_sources {
    echo 'Merging sources ...'
    mkdir -p "source/_static"
    cp source_base/* "source"
    mv source_src/* "source"
    cp source_pages/* "source"
    echo 'Sources merged!'
}

function build_html {
    echo 'Building HTML documentation ...'
    make html
    local exit_status=$?
    echo 'HTML documentation built (see "build" directory)!'
    return ${exit_status}
}

function build_clean {
    echo 'Cleaning building files ...'
    rm -fr "source" "source_src"
    echo 'Building files cleaned!'
}


# ---------------- #
# ---   MAIN   --- #
# ---------------- #
# Procedurally generated rst files from python source code
generate_modules
echo -e "\n"
# Merge all rst files in the sources directory
merge_sources
echo -e "\n"
# Build the HTML documentation from the rst files in the sources directory
build_html
exit_status=$?
if [[ ${exit_status} != 0 ]]; then
    echo 'Build failed :('
    exit ${exit_status}
else
    echo 'Build succeeded :)'
fi
echo -e "\n"
# Clean files used during building
build_clean
