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
    "myst_parser",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = ['.rst', '.md']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for module names -------------------------------------------------
#add_module_names = False

# -- More involved stuff ------------------------------------------------------

from sphinx.ext import autodoc
import re
import inspect

class SimpleMethodDocumenter(autodoc.MethodDocumenter):
    objtype = 'simple'
    #content_indent = '' # do not indent the content

    # do not display the method signature
    def format_args(self):
        return None

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
            # add the processed docstring to the list
            docstrings_list.append(docstr)

        #super_doc = super().get_doc()
        #import pdb; pdb.set_trace()
        
        if docstrings_list:
            # remove the last aesthetic horizontal line
            docstrings_list[-1] = docstrings_list[-1][:-2]


        return docstrings_list
    
def setup(app):
    app.add_autodocumenter(SimpleMethodDocumenter)
