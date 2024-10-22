# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import re
import sys, os
# Add the directory containing icon4py_sphinx to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import icon4py_sphinx

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
    #"icon4py.model.atmosphere.dycore.sphinx" # disable while waiting for gt4py patch
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
"""

# -- MathJax config ----------------------------------------------------------
mathjax3_config = {
    'chtml': {'displayAlign': 'left',
              'displayIndent': '1em'},
    'tex': {
        'inlineMath': [['\\(', '\\)']],
        'displayMath': [["\\[", "\\]"]],
    },
}
# Import latex macros and extensions
mathjax3_config['tex']['macros'] = {}
with open('latex_macros.tex', 'r') as f:
    for line in f:
        macros = re.findall(r'\\(DeclareRobustCommand|newcommand|renewcommand){\\(.*?)}(\[(\d)\])?{(.+)}', line)
        for macro in macros:
            if len(macro[2]) == 0:
                mathjax3_config['tex']['macros'][macro[1]] = "{"+macro[4]+"}"
            else:
                mathjax3_config['tex']['macros'][macro[1]] = ["{"+macro[4]+"}", int(macro[3])]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for module names -------------------------------------------------
add_module_names = False

# -- More involved stuff ------------------------------------------------------

def setup(app):
    app.add_autodocumenter(icon4py_sphinx.FullMethodDocumenter)
