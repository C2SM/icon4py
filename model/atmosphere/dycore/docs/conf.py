# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import sys, os
# Add the directory containing icon4py_sphinx to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "_ext")))
import icon4py_sphinx
import latex_sphinx

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'icon4py-atmosphere-dycore'
copyright = '2024, C2SM'
author = 'C2SM'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
    "sphinx_math_dollar",
    'sphinx_toolbox.collapse', # https://sphinx-toolbox.readthedocs.io/en/stable/extensions/collapse.html
]

# Make sure that the autosection target is unique
autosectionlabel_prefix_document = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = ['.rst', '.md']

# -- reST epilogue: macros / aliases -----------------------------------------
rst_epilog = """
.. |ICONtutorial| replace:: ICON Tutorial_
.. _Tutorial: https://www.dwd.de/EN/ourservices/nwp_icon_tutorial/nwp_icon_tutorial_en.html
.. |ICONdycorePaper| replace:: Zangl_etal_
.. _Zangl_etal: https://doi.org/10.1002/qj.2378
"""

# -- MathJax config ----------------------------------------------------------
mathjax3_config = {
    'chtml': {'displayAlign': 'left',
              'displayIndent': '1em'},
    'tex': {
        'inlineMath': [['\\(', '\\)']],
        'displayMath': [["\\[", "\\]"]],
        # Create mathjax macros from latex file
        'macros': latex_sphinx.tex_macros_to_mathjax('latex_macros.tex'),
    },
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for module names -------------------------------------------------
add_module_names = False

# -- More involved stuff ------------------------------------------------------

def setup(app):
    app.add_autodocumenter(icon4py_sphinx.ScidocMethodDocumenter)
