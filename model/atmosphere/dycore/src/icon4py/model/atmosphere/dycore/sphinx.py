# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Sphinx documentation plugin used to document decorators.

Introduction
============

Usage
-----

The extension requires Sphinx 2.0 or later.

Add the extension to your :file:`docs/conf.py` configuration module:

.. code-block:: python

    extensions = (...,
                  'path.to.here.sphinx')

With the extension installed `autodoc` will automatically find gt4py decorated
objects with @field_operator, @scan_operator, @program (e.g. when using the
automodule directive) and generate the correct (as well as add a
``(gt4pydecor)`` prefix).
"""
import inspect

from docutils import nodes
from gt4py.next.ffront.decorator import FieldOperator, Program
from gt4py.next.ffront.field_operator_ast import ScanOperator
from sphinx.domains import python as sphinx_python
from sphinx.ext import autodoc


class Gt4pydecorDocumenter(autodoc.FunctionDocumenter):
    """Document gt4pydecor definitions."""

    objtype = "gt4pydecor"
    member_order = 11

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, (Program, FieldOperator, ScanOperator)) and member.__wrapped__

    def format_args(self):
        wrapped = getattr(self.object, "__wrapped__", None)
        if wrapped is not None:
            sig = inspect.signature(wrapped)
            if "self" in sig.parameters or "cls" in sig.parameters:
                sig = sig.replace(parameters=list(sig.parameters.values())[1:])
            return str(sig)
        return ""

    def document_members(self, all_members=False):
        pass

    def check_module(self):
        # Normally checks if *self.object* is really defined in the module
        # given by *self.modname*. But we have to check the wrapped function
        # instead.
        wrapped = getattr(self.object, "__wrapped__", None)
        if wrapped and wrapped.__module__ == self.modname:
            return True
        return super().check_module()


class Gt4pydecorDirective(sphinx_python.PyFunction):
    """Sphinx gt4pydecor directive."""

    def get_signature_prefix(self, sig):
        return [nodes.Text(self.env.config.gt4py_gt4pydecor_prefix)]


def autodoc_skip_member_handler(app, what, name, obj, skip, options):
    """Handler for autodoc-skip-member event."""
    # Some decorated methods have the property that *obj.__doc__* and
    # *obj.__class__.__doc__* are equal, which trips up the logic in
    # sphinx.ext.autodoc that is supposed to suppress repetition of class
    # documentation in an instance of the class. This overrides that behavior.
    if isinstance(obj, (Program, FieldOperator, ScanOperator)) and obj.__wrapped__:
        if skip:
            return False
    return None


def setup(app):
    """Setup Sphinx extension."""
    app.setup_extension("sphinx.ext.autodoc")
    app.add_autodocumenter(Gt4pydecorDocumenter)
    app.add_directive_to_domain("py", "gt4pydecor", Gt4pydecorDirective)
    app.add_config_value("gt4py_gt4pydecor_prefix", "(gt4pydecor)", True)
    app.connect("autodoc-skip-member", autodoc_skip_member_handler)

    return {"parallel_read_safe": True}
