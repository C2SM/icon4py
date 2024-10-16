# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from sphinx.ext import autodoc
import re
import inspect

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

class FullMethodDocumenter(autodoc.MethodDocumenter):
    """Fully document a method."""

    objtype = 'full'

    priority = autodoc.MethodDocumenter.priority - 1

    def process_lines(self, docstr):
        def insert_before_first_non_space(s, char_to_insert):
            for i, char in enumerate(s):
                if char != ' ':
                    return s[:i] + char_to_insert + s[i:]
            return s
        # "Special" treatment of specific lines / blocks
        in_latex_block = False
        for iline, line in enumerate(docstr):
            # Make collapsible Inputs section
            if line.startswith('Inputs:'):
                docstr[iline] = '.. collapse:: Inputs:'
                docstr.insert(iline+1, '')
            # Identify LaTeX blocks and align to the left
            elif '$$' in line:
                if not in_latex_block and '\\\\' in docstr[iline+1]:
                    in_latex_block=True
                    continue
                else:
                    in_latex_block=False
            if in_latex_block:
                docstr[iline] = insert_before_first_non_space(line, '&')
        return docstr



    def get_doc(self):

        docstrings = re.findall(r'"""(.*?)"""', inspect.getsource(self.object), re.DOTALL)

        docstrings_list = []
        for docstring in docstrings:
            docstr = docstring.splitlines()
            # strip empty lines from the beginning
            while docstr[0] == '':
                docstr.pop(0)
            # strip leading and trailing whitespace of every line (maintain indentation)
            indent = len(docstr[0]) - len(docstr[0].lstrip(' '))
            docstr = [line[indent:].rstrip() for line in docstr]
            # strip empty lines from the end
            while docstr[-1] == '':
                docstr.pop(-1)
            # Aesthetics processing:
            # add a horizontal line at the end of the docstring
            docstr.append('')
            docstr.append('*' + '~' * 59 + '*')
            # and have one empty line at the end
            docstr.append('')
            docstr=self.process_lines(docstr)
            # add the processed docstring to the list
            docstrings_list.append(docstr)
        
        if docstrings_list:
            # remove the last aesthetic horizontal line
            docstrings_list[-1] = docstrings_list[-1][:-2]


        return docstrings_list
    
def setup(app):
    app.add_autodocumenter(FullMethodDocumenter)
