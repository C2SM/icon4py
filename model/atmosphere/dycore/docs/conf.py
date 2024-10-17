# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from sphinx.ext import autodoc
import os
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
    """
    'Fully' document a method, i.e. picking up and processing all docstrings in
    its source code.
    """

    objtype = 'full'
    priority = autodoc.MethodDocumenter.priority - 1

    def get_doc(self):
        # Override the default get_doc method to pick up all docstrings in the
        # source code
        source = inspect.getsource(self.object) # this is only the source of the method, not the whole file
        _, method_start_line = inspect.getsourcelines(self.object)

        docstrings = re.findall(r'"""(.*?)"""', source, re.DOTALL)
        docstrings_list = []
        for idocstr, docstring in enumerate(docstrings):
            formatted_docstr = None
            if idocstr==0:
                # "Normal" docstring at the beginning of the method
                formatted_docstr = docstring.splitlines()

            elif docstring.startswith('_scidoc_'):
                call_string = self.get_next_method_call(source, docstring)
                next_method_name = call_string[0].split('.')[-1].split('(')[0]
                formatted_docstr = docstring.splitlines()
                formatted_docstr = self.format_source_code(formatted_docstr)
                formatted_docstr = self.add_header(formatted_docstr, next_method_name+'()')
                formatted_docstr = self.process_scidocstrlines(formatted_docstr)
                formatted_docstr = self.add_next_method_call(formatted_docstr, call_string)

            if formatted_docstr is not None:
                if idocstr < len(docstrings)-1: # Add footer
                    formatted_docstr = self.add_footer(formatted_docstr)
                # add the processed docstring to the list
                docstrings_list.append(formatted_docstr)

        return docstrings_list

    def format_source_code(self, source_lines):
        # Clean up and format
        if source_lines[0].startswith('_scidoc_'):
            source_lines.pop(0) # remove the _scidoc_ prefix
        # strip empty lines from the beginning
        while source_lines[0] == '':
            source_lines.pop(0)
        # strip leading and trailing whitespace of every line (maintain indentation)
        indent = len(source_lines[0]) - len(source_lines[0].lstrip(' '))
        source_lines = [line[indent:].rstrip() for line in source_lines]
        # strip empty lines from the end
        while source_lines[-1] == '':
            source_lines.pop(-1)
        return source_lines

    def add_header(self, docstr_lines, title):
        # Add a title
        docstr_lines.insert(0, title)
        docstr_lines.insert(1, '='*len(title))
        docstr_lines.insert(2, '')
        return docstr_lines

    def add_footer(self, docstr_lines):
        # Add a horizontal line at the end of the docstring
        docstr_lines.append('')
        #docstr_lines.append('*' + '~' * 59 + '*')
        docstr_lines.append('.. raw:: html')
        docstr_lines.append('')
        docstr_lines.append('   <hr>')
        # and have one empty line at the end
        docstr_lines.append('')
        return docstr_lines

    def process_scidocstrlines(self, docstr_lines):
        # "Special" treatment of specific lines / blocks
        def insert_before_first_non_space(s, char_to_insert):
            for i, char in enumerate(s):
                if char != ' ':
                    return s[:i] + char_to_insert + s[i:]
            return s
        in_latex_block = False
        for iline, line in enumerate(docstr_lines):
            # Make collapsible Inputs section
            if line.startswith('Inputs:'):
                docstr_lines[iline] = '.. collapse:: Inputs'
                docstr_lines.insert(iline+1, '')
            # Identify LaTeX blocks and align to the left
            elif '$$' in line:
                if not in_latex_block and '\\\\' in docstr_lines[iline+1]:
                    in_latex_block=True
                    continue
                else:
                    in_latex_block=False
            if in_latex_block:
                docstr_lines[iline] = insert_before_first_non_space(line, '&')
        return docstr_lines

    def add_next_method_call(self, docstr_lines, formatted_call):
        docstr_lines.append('')
        docstr_lines.append('.. collapse:: Source code')
        docstr_lines.append('')
        docstr_lines.append('   .. code-block:: python')
        docstr_lines.append('')
        docstr_lines += ['      ' + line for line in formatted_call]
        return docstr_lines

    def get_next_method_call(self, source, docstring):
        index_start = source.find(docstring) + len(docstring)
        remaining_source = source[index_start:]
        stack = []
        start_index = None
        end_index = None
        for i, char in enumerate(remaining_source):
            if char == '(':
                if not stack:
                    start_index = i
                stack.append(char)
            elif char == ')':
                stack.pop()
                if not stack:
                    end_index = i
                    break
        if start_index is not None and end_index is not None:
            call_start_line = source[:index_start].count('\n')+1
            call_end_line = call_start_line + remaining_source[:end_index].count('\n')-1
            call_string = self.format_source_code(source.splitlines()[call_start_line:call_end_line+1])
            return call_string
        return None

def setup(app):
    app.add_autodocumenter(FullMethodDocumenter)
