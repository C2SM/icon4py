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
import types, typing
import ast

class FullMethodDocumenter(autodoc.MethodDocumenter):
    """
    'Fully' document a method, i.e. picking up and processing all 'tagged'
    comment blocks in its source code.
    """

    objtype = 'full'
    priority = autodoc.MethodDocumenter.priority - 1

    # Configuration options
    docstring_keyword = 'scidoc'
    var_type_in_inputs = True
    var_type_formatting = '``'
    print_variable_longnames = True

    def get_doc(self):
        # Override the default get_doc method to pick up all docstrings in the
        # source code
        source = inspect.getsource(self.object) # this is only the source of the method, not the whole file

        docstrings = self.get_docstring_blocks(source, self.docstring_keyword)

        docstrings_list = []
        for idocstr, docstring in enumerate(docstrings):
            formatted_docstr = None

            # Get some useful information on the following method call
            call_string = self.get_next_method_call(source, source.find(docstring) + len(docstring))
            next_method_info = self.get_method_info(call_string)
            # and process a scientific documentation docstring
            formatted_docstr = docstring.splitlines()
            formatted_docstr = self.format_docstring_block(formatted_docstr, self.docstring_keyword)
            formatted_docstr = self.add_header(formatted_docstr, next_method_info)
            formatted_docstr = self.process_scidocstrlines(formatted_docstr, next_method_info)
            formatted_docstr = self.add_next_method_call(formatted_docstr, call_string)

            if formatted_docstr is not None:
                if idocstr < len(docstrings)-1: # Add footer
                    formatted_docstr = self.add_footer(formatted_docstr)
                # add the processed docstring to the list
                docstrings_list.append(formatted_docstr)

        return docstrings_list
    
    def get_docstring_blocks(self, source, keyword):
        # which in this implementation are comment blocks
        comment_blocks = []
        source_lines = source.splitlines()
        in_block = False
        current_block = []
        for line in source_lines:
            stripped_line = line.strip()
            if stripped_line.startswith(f'# {keyword}:'):
                if in_block:
                    comment_blocks.append('\n'.join(current_block))
                    current_block = []
                in_block = True
                current_block.append(line)
            elif in_block:
                if stripped_line.startswith('#'):
                    current_block.append(line)
                else:
                    in_block = False
                    comment_blocks.append('\n'.join(current_block))
                    current_block = []
        if current_block:
            comment_blocks.append('\n'.join(current_block))
        return comment_blocks

    def format_docstring_block(self, docstr_lines, keyword):
        # Clean up and format
        if docstr_lines[0].strip().startswith(f'# {keyword}:'):
            docstr_lines.pop(0) # remove the {keyword} prefix
        while docstr_lines[0].strip() == '#':
            # strip empty lines from the beginning
            docstr_lines.pop(0)
        # strip leading and trailing whitespace of every line (maintain indentation)
        # as well as comment character and space (indent+2)
        indent = len(docstr_lines[0]) - len(docstr_lines[0].lstrip(' '))
        docstr_lines = [line[min(indent+2,len(line)):].rstrip() for line in docstr_lines]
        # strip empty lines from the end
        while docstr_lines[-1] == '':
            docstr_lines.pop(-1)
        return docstr_lines
    
    def add_header(self, docstr_lines, method_info):
        # Add a title with ReST formatting
        method_name = method_info['module_local_name'] + '.' + method_info['method_name']
        method_full_name = method_info['module_full_name'] + '.' + method_name
        title = f':meth:`{method_name}()<{method_full_name}>`'
        docstr_lines.insert(0, title)
        docstr_lines.insert(1, '='*len(title))
        docstr_lines.insert(2, '')
        return docstr_lines

    def add_footer(self, docstr_lines):
        # Add a horizontal line at the end of the docstring
        docstr_lines.append('')
        docstr_lines.append('.. raw:: html')
        docstr_lines.append('')
        docstr_lines.append('   <hr>')
        # and add one empty line at the end
        docstr_lines.append('')
        return docstr_lines

    def process_scidocstrlines(self, docstr_lines, next_method_info):
        # "Special" treatment of specific lines / blocks

        latex_multiline_block = False
        past_inputs = False

        def insert_before_first_non_space(s, char_to_insert):
            for i, char in enumerate(s):
                if char != ' ':
                    return s[:i] + char_to_insert + s[i:]
            return s

        for iline, line in enumerate(docstr_lines):

            # Make collapsible Inputs section
            if line.startswith('Inputs:'):
                past_inputs = True
                docstr_lines[iline] = '.. collapse:: Inputs'
                docstr_lines.insert(iline+1, '')

            # Identify LaTeX multiline blocks and align equations to the left
            elif '$$' in line:
                if not latex_multiline_block and '\\\\' in docstr_lines[iline+1]:
                    latex_multiline_block=True
                    continue
                else:
                    latex_multiline_block=False
            if latex_multiline_block:
                docstr_lines[iline] = insert_before_first_non_space(line, '&')

            # Add type information to variable names (only in bullet point lines)
            if (not latex_multiline_block) and ('-' in line) and (':' in line):
                if past_inputs and not self.var_type_in_inputs:
                    continue
                indent = len(line) - len(line.lstrip())
                split_line = line.split()
                for variable, var_type in next_method_info['var_types'].items():
                    if variable in split_line:
                        # Replace only exact matches
                        for j, part in enumerate(split_line):
                            if part == variable:
                                if self.print_variable_longnames:
                                    # long name version
                                    var_longname = next_method_info['var_longnames_map'][variable]
                                    prefix = '.'.join(var_longname.split('.')[:-1])
                                    suffix = var_longname.split('.')[-1]
                                    if prefix:
                                        vname = '*' + prefix + '*. **' + suffix + '**'
                                    else:
                                        vname = '**' + suffix + '**'
                                else:
                                    # short name version
                                    vname = variable
                                split_line[j] = f"{vname} {self.var_type_formatting}{var_type}{self.var_type_formatting}"
                        docstr_lines[iline] = ' '*indent + ' '.join(split_line)

        return docstr_lines

    def add_next_method_call(self, docstr_lines, formatted_call):
        # Add the source string of the next method call as a collapsible block
        docstr_lines.append('')
        docstr_lines.append('.. collapse:: Source code')
        docstr_lines.append('')
        docstr_lines.append('   .. code-block:: python')
        docstr_lines.append('')
        docstr_lines += ['      ' + line for line in formatted_call]
        return docstr_lines

    def get_next_method_call(self, source, index_start):
        # Find the next method call in the source code using a stack to find the
        # matching closing round brackets
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
            call_string = self.format_code_block(source.splitlines()[call_start_line:call_end_line+1])
            return call_string
        return None

    def format_code_block(self, code_lines):
        # Clean up and format
        while code_lines[0] == '':
            # strip empty lines from the beginning
            code_lines.pop(0)
        # strip leading and trailing whitespace of every line (maintain indentation)
        indent = len(code_lines[0]) - len(code_lines[0].lstrip(' '))
        code_lines = [line[indent:].rstrip() for line in code_lines]
        # strip empty lines from the end
        while code_lines[-1] == '':
            code_lines.pop(-1)
        return code_lines


    def get_method_info(self, call_string):
        method_info = {}
        local_method_name = call_string[0].split('(')[0] # get the method name from the call string (remove open bracket)
        # Get the module of this method from its name
        parent_name = local_method_name.split('.')[0]
        method_name = local_method_name.split('.')[-1]
        if parent_name in self.module.__dict__.keys():
            # we are importing directly from a module
            module_obj = getattr(self.module, parent_name)
            method_obj = getattr(module_obj, method_name)
            module_full_name = module_obj.__name__
        elif parent_name == 'self':
            # the method was renamed in the present class
            class_and_method_name = re.sub(self.modname, '', self.fullname)[1:]
            class_name = class_and_method_name.split('.')[0]
            class_obj = getattr(self.module, class_name)
            # Check if the method is defined in the class
            if hasattr(class_obj, method_name):
                method_obj = getattr(class_obj, method_name)
            else:
                # Handle the case where the method is imported and renamed in the class
                for attr_name in dir(class_obj):
                    attr = getattr(class_obj, attr_name)
                    if callable(attr) and hasattr(attr, '__name__') and attr.__name__ == method_name:
                        method_obj = attr
                        break
                else:
                    # Handle the case where the method is assigned to an instance variable using AST
                    source = inspect.getsource(class_obj)
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                            for stmt in node.body:
                                if isinstance(stmt, ast.Assign):
                                    for target in stmt.targets:
                                        if isinstance(target, ast.Attribute) and target.attr == method_name:
                                            if isinstance(stmt.value, ast.Attribute):
                                                original_method_name = stmt.value.attr
                                                if hasattr(class_obj, original_method_name):
                                                    method_obj = getattr(class_obj, original_method_name)
                                                    break
                                            elif isinstance(stmt.value, ast.Name):
                                                original_method_name = stmt.value.id
                                                if original_method_name in self.module.__dict__.keys():
                                                    module_obj = getattr(self.module, original_method_name)
                                                    method_obj = getattr(module_obj, method_name)
                                                    module_full_name = module_obj.__name__
                                                    break
                                            elif isinstance(stmt.value, ast.Call):
                                                # Traverse the call chain to get the original method and module
                                                call_chain = []
                                                current_node = stmt.value.func
                                                while isinstance(current_node, (ast.Attribute, ast.Name)):
                                                    if isinstance(current_node, ast.Attribute):
                                                        call_chain.append(current_node.attr)
                                                        current_node = current_node.value
                                                    elif isinstance(current_node, ast.Name):
                                                        call_chain.append(current_node.id)
                                                        break
                                                call_chain.reverse()
                                                if call_chain[-1] == 'with_backend':
                                                    # remove it from the call chain
                                                    call_chain.pop()
                                                if len(call_chain) == 1:
                                                    # the method is imported directly, e.g.
                                                    # from solve_nonhydro_program import stencil
                                                    # self._stencil = stencil.with_backend(...)
                                                    original_method_name = call_chain[0]
                                                    if original_method_name in self.module.__dict__.keys() and not isinstance(getattr(self.module, original_method_name), types.ModuleType):
                                                        # method_obj.definition_stage.definition.__module__
                                                        method_obj = getattr(self.module, original_method_name)
                                                        if type(method_obj).__module__.startswith('gt4py') and type(method_obj).__name__ == 'Program':
                                                            # it's a decorated gt4py program
                                                            module_full_name = method_obj.definition_stage.definition.__module__
                                                            break
                                                elif len(call_chain) >= 2:
                                                    # the method is called from a module import, e.g.
                                                    # import solve_nonhydro_program as nhsolve_prog
                                                    # self._stencil = nhsolve_prog.stencil.with_backend(...)
                                                    module_name = call_chain[0]
                                                    original_method_name = call_chain[1]
                                                    if module_name in self.module.__dict__.keys() and isinstance(getattr(self.module, module_name), types.ModuleType):
                                                        module_obj = getattr(self.module, module_name)
                                                        method_obj = getattr(module_obj, original_method_name)
                                                        module_full_name = module_obj.__name__
                                                        break
        #
        method_info['call_string'] = call_string
        method_info['method_name'] = method_name
        method_info['module_local_name'] = parent_name
        method_info['module_full_name'] = module_full_name
        method_info['annotations'] = method_obj.definition_stage.definition.__annotations__ if type(method_obj).__name__ == 'Program' else method_obj.__annotations__
        method_info['var_names_map'], method_info['var_longnames_map'] = self.map_variable_names(call_string)
        method_info['var_types'] = self.map_variable_types(method_info)
        return method_info

    def map_variable_names(self, function_call_str):
        # Extract argument names and their values using regex
        pattern = r'(\w+)\s*=\s*([^,]+)'
        matches = re.findall(pattern, ''.join(function_call_str))
        # Create a dictionary to map variable names to their full argument names
        variable_map = {}
        variable_longnames_map = {}
        for arg_name, arg_value in matches:
            # Extract the last part of the argument value after the last period
            short_name = arg_value.split('.')[-1]
            variable_map[arg_name] = short_name
            variable_longnames_map[short_name] = arg_value
        return variable_map, variable_longnames_map
    
    def map_variable_types(self, method_info):
        # Map variable short names (*not arg name*) to their types using the
        # annotations
        var_types = {}
        for arg_name, var_type in method_info['annotations'].items():
            if arg_name in method_info['var_names_map'].keys():
                var_types[method_info['var_names_map'][arg_name]] = self.format_type_string(var_type)
        return var_types
    
    # TODO: Implement a more sophisticated way to convert type strings to human-readable format
    def format_type_string(self, var_type):
        if isinstance(var_type, typing._GenericAlias):
            origin = var_type.__origin__
            assert origin.__name__ == 'Field' and origin.__module__.startswith('gt4py')
            args = var_type.__args__
            origin_str = origin.__name__ if hasattr(origin, '__name__') else str(origin).split('.')[-1]
            args_str = ', '.join(self.format_type_string(arg) for arg in args)
            return f"{origin_str}[{args_str}]"
        elif hasattr(var_type, '__name__') and var_type.__name__ != 'Dims':
            return var_type.__name__
        else:
            type_str = str(var_type)
            # Replace class types like <class 'numpy.int32'> with int32
            type_str = re.sub(r"<class '([\w\.]+)'>", lambda m: m.group(1).split('.')[-1], type_str)
            # Replace gt4py.next.common.Field[...] with Field[...]
            type_str = re.sub(r"gt4py\.next\.common\.", "", type_str)
            # Replace Dimension(value='K', kind=<DimensionKind.VERTICAL: 'vertical'>) with K
            type_str = re.sub(r"Dimension\(value='(\w+)', kind=<DimensionKind\.\w+: '\w+'>\)", r"\1", type_str)
            return type_str