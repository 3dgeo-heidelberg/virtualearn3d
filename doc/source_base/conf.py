# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
# TODO Remove section ---
print(
    f'os.path.abspath: "{os.path.abspath(os.path.join("..", "..", "src"))}"'
)
# --- TODO Remove section
sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'VirtuaLearn3D'
copyright = '2023, Alberto Esmoris, Hannah Weiser, Bernhard Hoefle'
author = 'Alberto Esmoris, Hannah Weiser, Bernhard Hoefle'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']

# ---  HTML THEME  --- #
# -------------------- #
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 5
}


def setup(app):
    app.add_css_file('custom.css')

