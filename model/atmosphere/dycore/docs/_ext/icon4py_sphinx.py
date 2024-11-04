# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import ast
import inspect
import re
import textwrap
import types
import typing
from typing import Final

from sphinx.ext import autodoc

class ScidocMethodDocumenter(autodoc.MethodDocumenter):
    """
    'Fully' document a method, i.e. picking up and processing all 'tagged'
    comment blocks in its source code.
    """

    objtype = 'scidoc'
    priority = autodoc.MethodDocumenter.priority - 1

    # Configuration options
    docblock_keyword = 'scidoc'
    var_type_in_inputs = True
    var_type_formatting = '``'
    print_variable_longnames = True
    #: Footer lines with a horizontal line at the end  
    SCIDOC_FOOTER_LINES: Final[list[str]] = ["", ".. raw:: html", "",  "   <hr>"]  
    #: Code block lines for the source code of the next method call
    SCIDOC_CODE_BLOCK_LINES: Final[list[str]] = ["", ".. collapse:: Source code", "", " .. code-block:: python", ""]

    def get_doc(self) -> list[list[str]]:
        # Override the default get_doc method to pick up all tagged comment
        # blocks in the source code of the method
        source = inspect.getsource(self.object) # this is only the source of the method, not the whole file

        docblocks = self.get_documentation_blocks(source, self.docblock_keyword)

        docblocks_list = []
        for i, docblock in enumerate(docblocks):
            formatted_docblock = None

            # Get some useful information on the following method call
            call_string = self.get_next_method_call(source, source.find(docblock) + len(docblock))
            next_method_info = self.get_method_info(call_string)
            # and process the scidoc block
            formatted_docblock = docblock.splitlines()
            formatted_docblock = self.format_docblock(formatted_docblock, self.docblock_keyword)
            formatted_docblock = self.make_header(next_method_info) + formatted_docblock
            formatted_docblock = self.process_scidoc_lines(formatted_docblock, next_method_info)
            formatted_docblock += self.SCIDOC_CODE_BLOCK_LINES + [" "*6 + line for line in call_string]

            if formatted_docblock:
                if i < len(docblocks)-1: # Add footer
                    formatted_docblock += self.SCIDOC_FOOTER_LINES  
                # add the processed docblock to the list
                docblocks_list.append(formatted_docblock)

        return docblocks_list
    
    def get_documentation_blocks(self, source: str, keyword: str) -> list[str]:
        """
        Extract blocks of comments from the source code that start with a specific keyword.

        Args:
            source: The source code as a string.
            keyword: The keyword to look for in the comment blocks.

        Returns:
            A list of comment blocks that start with the specified keyword.
        """
        comment_blocks = []
        in_block = False
        for line in source.splitlines():
            stripped_line = line.strip()
            is_comment = stripped_line.startswith('#')
            if stripped_line.startswith(f'# {keyword}:'):
                if in_block:
                    raise RuntimeError("Nesting of documentation blocks is not supported ")
                comment_blocks.append(f'{line}')
                in_block = True
            elif in_block and is_comment:
                comment_blocks[-1] += f'\n{line}'
            in_block = is_comment and in_block # continue the block if the line is a comment (but don't start one if now with the keyword)
        return comment_blocks

    def format_docblock(self, docblock_lines: list[str], keyword: str) -> list[str]:
        """
        Format a block of documentation lines by cleaning up and removing keyword,
        whitespace and empty lines.

        Args:
            docblock_lines: The lines of the documentation block to be formatted.
            keyword: The keyword to look for and remove from the beginning of the docblock.

        Returns:
            The formatted documentation lines.
        """
        assert docblock_lines[0].strip().startswith(f'# {keyword}:')  
        docblock_lines.pop(0)  # remove the {keyword} prefix
        # Remove leading and trailing blank lines.
        # Blank lines in the middle are kept because they are used to separate
        # paragraphs and lists.
        while docblock_lines and docblock_lines[0].strip() == '#':
            docblock_lines.pop(0)
        while docblock_lines and docblock_lines[-1].strip() == '#':
            docblock_lines.pop(-1)
        # Dedent the comment block
        dedented_lines = textwrap.dedent('\n'.join(docblock_lines)).splitlines()
        # Remove leading '#'
        uncommented_lines = [line.lstrip('#') for line in dedented_lines]
        # Dedent again
        formatted_lines = textwrap.dedent('\n'.join(uncommented_lines)).splitlines()
        return formatted_lines

    def make_header(self, method_info: dict) -> list[str]:
        """
        Generate the header.

        Args:
            method_info: A dictionary information about the next method.

        Returns:
            A list containing the formatted title and its underline.
        """
        label = f"{method_info['local_parent_name']}.{method_info['local_shortname']}"  
        link_destination = f"{method_info['orig_parent_name']}.{method_info['orig_shortname']}"
        title = f":meth:`{label}()<{link_destination}>`" 
        return [title, '='*len(title), '']

    def process_scidoc_lines(self, docblock_lines: list[str], method_info: dict) -> list[str]:
        """
        Process a list of documentation string lines and apply specific formatting rules.
        - Drop the colons from the Outputs and Inputs sections.
        - Make the "Inputs" section collapsible.
        - Identify LaTeX multiline blocks and align equations to the left.
        - Add type information to variable names in bullet point lines.
        - Optionally print the long names of variables.

        Args:
            docblock_lines: The lines of the documentation string to process.
            method_info: A dictionary containing information about the next method.

        Returns:
            The processed documentation string lines with applied formatting rules.
        """
        
        latex_math = False
        latex_math_multiline = False
        past_inputs = False
        processed_lines = []

        for line_num, line in enumerate(docblock_lines):

            # Drop the colon from the Outputs section
            if line.startswith('Outputs:'):
                processed_lines.append('Outputs')
                continue

            # Drop the colon from the Inputs section and make it collapsible
            if line.startswith('Inputs:'):
                past_inputs = True
                processed_lines.append(".. collapse:: Inputs")
                processed_lines.append("")
                continue

            # Identify LaTeX math (multiline) blocks
            if line.strip().startswith('$$'):
                latex_math = not latex_math
                if docblock_lines[line_num+1].rstrip().endswith(r'\\'): # multiline math block :
                    latex_math_multiline = True
                else: # single line math block or end of block
                    latex_math_multiline = False
                processed_lines.append(line)
                continue

            # Process math lines
            if latex_math:
                processed_lines.append(self.process_math_line(line, latex_math_multiline))
                continue

            # Only in bullet point lines:
            # Add type information to variable names.
            # Optionally print the long names of variables.
            if ('-' in line) and (':' in line) and (not latex_math_multiline):
                if past_inputs and not self.var_type_in_inputs:
                    # Skip adding type information to variable names in the Inputs section
                    processed_lines.append(line)
                else:
                    # Add type information to variable names and optionally print the long names
                    indent = len(line) - len(line.lstrip())
                    split_line = line.split()
                    for variable, var_type in method_info['var_types'].items():
                        if variable in split_line:
                            if self.print_variable_longnames:
                                # long name version, with small font size for the prefix
                                var_longname = method_info['var_longnames_map'][variable]
                                prefix = '.'.join(var_longname.split('.')[:-1])
                                suffix = var_longname.split('.')[-1]
                                vname = (f"$\color{{grey}}{{\scriptstyle{{\\texttt{{{prefix}.}}}}}}$" if prefix else "") + f" **{suffix}**"
                            else:
                                # short name version
                                vname = variable
                            j = split_line.index(variable)
                            split_line[j] = f"{vname} {self.var_type_formatting}{var_type}{self.var_type_formatting}"
                            processed_lines.append(' '*indent + ' '.join(split_line))
                continue
            
            # Keep the line as is
            processed_lines.append(line)

        return processed_lines

    def process_math_line(self, line: str, multiline: bool) -> str:
        """
        Process a line of LaTeX math and apply specific formatting rules.

        Args:
            line: The line of the LaTeX math to process.
            multiline: A boolean flag indicating if the math block is multiline.

        Returns:
            The processed LaTeX math line with applied formatting rules.
        """
        symbols_needing_space = ['\Wrbf', '\Wlev', '\WtimeExner']

        if multiline:
            # Align multiline equations to the left
            # (single line equations are already left-aligned)
            start_idx = len(line) - len(line.lstrip()) 
            line = f"{line[:start_idx]}&{line[start_idx:]}"  

        # Add latex space character '\:' after symbols if followed by other symbols starting with '\' and a letter
        for symbol in symbols_needing_space:
            if symbol in line:
                line = re.sub(rf'(\{symbol})\s*(\\[a-zA-Z])', r'\1\\,\2', line)

        return line

    def get_next_method_call(self, source: str, index_start: int) -> list[str]:
        """
        Find the next method call in the source code starting from a given index.

        This method scans the source code starting from `index_start` to locate the
        next method call by identifying the matching closing round brackets using a stack.

        Args:
            source: The source code as a string.
            index_start: The index from which to start searching for the next method call.

        Returns:
            A list of strings representing the lines of code for the next method
            call. Returns None if no method call is found (hopefully never).
        """
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

    def format_code_block(self, code_lines: list[str]) -> list[str]:
        """
        Clean up and format a block of code by removing leading and trailing
        empty lines, and stripping leading and trailing whitespace from each
        line while maintaining indentation.

        Args:
            code_lines: A list of strings representing lines of code.

        Returns:
            A list of formatted code lines.
        """
        # Remove leading and trailing empty lines
        code_lines = [line for line in code_lines if line.strip()]
        # Dedent the code block
        return textwrap.dedent('\n'.join(code_lines)).splitlines()

    def get_method_info(self, call_string: list[str]) -> dict:
        """
        Extracts and returns detailed information about a method based on the provided call string.

        Args:
            call_string: A list containing the method call string.

        Returns:
            dict: A dictionary containing the following keys:
            - 'call_string': The call string.
            - 'local_shortname': The local shortname of the method.
            - 'local_parent_name': The local name of the parent/module where the method is defined.
            - 'orig_shortname': The original shortname of the method.
            - 'orig_parent_name': The original name of the parent/module where the method is defined.
            - 'annotations': The annotations of the method.
            - 'var_names_map': A dictionary mapping variable names as argument to the called method and variable short name (or argument value) in the present scope.
            - 'var_longnames_map': A dictionary mapping variable short and long names in the present scope.
            - 'var_types': A dictionary mapping variable short names and their types.
        """
        method_info = {}

        local_longname = call_string[0][:call_string[0].index("(")] # get the method name from the call string (remove open bracket)
        local_parent_name = local_longname.split('.')[0]
        local_shortname = local_longname.split('.')[-1]

        # Get the original method information
        orig = self.get_original_method_info(local_parent_name, local_shortname)

        method_info['call_string'] = call_string
        method_info['local_shortname'] = local_shortname
        method_info['local_parent_name'] = local_parent_name
        method_info['orig_shortname'] = orig['shortname']
        method_info['orig_parent_name'] = orig['parent_name']
        method_info['annotations'] = orig['method_obj'].definition_stage.definition.__annotations__ if type(orig['method_obj']).__name__ == 'Program' else orig['method_obj'].__annotations__
        method_info['var_names_map'], method_info['var_longnames_map'] = self.map_variable_names(call_string)
        method_info['var_types'] = self.map_variable_types(method_info)
        return method_info

    def get_original_method_info(self, local_parent_name: str, local_shortname: str) -> dict:
        """
        Retrieve information about a method based on the local parent name and short name.

        This method attempts to locate the original method object and its parent module or class name
        by examining the provided local parent name and short name. It handles various scenarios such as:
        - Direct import from a module
        - Method renamed in the current class
        - Method assigned to an instance variable using AST parsing

        Args:
            local_parent_name: The local name of the parent (module or class).
            local_shortname: The local short name of the method.

        Returns:
            dict: A dictionary containing the following keys:
                - 'method_obj': The original method object.
                - 'parent_name': The name of the parent module or class.
                - 'shortname': The short name of the method.
        """
        if local_parent_name in self.module.__dict__.keys():
            # Importing directly from a module
            module_obj = getattr(self.module, local_parent_name)
            method_obj = getattr(module_obj, local_shortname)
            parent_name = module_obj.__name__
        elif local_parent_name == 'self':
            # The method is defined or renamed in the present class
            class_and_method_name = re.sub(self.modname, '', self.fullname)[1:]
            class_name = class_and_method_name.split('.')[0]
            class_obj = getattr(self.module, class_name)
            # Check if the method is defined in the class
            if hasattr(class_obj, local_shortname):
                method_obj = getattr(class_obj, local_shortname)
            else:
                # Handle the case where the method is imported and renamed in the class
                for _, attr in vars(class_obj).items():
                    if callable(attr) and hasattr(attr, '__name__') and attr.__name__ == local_shortname:
                        method_obj = attr
                        break
                else:
                    # Handle the case where the method is assigned to an instance variable using AST
                    method_obj, parent_name = self.get_method_from_ast(class_obj, local_shortname)
        info = {
            'method_obj': method_obj,
            'parent_name': parent_name,
            'shortname': method_obj.__name__,
            }
        return info

    def get_method_from_ast(self, class_obj: object, local_shortname: str) -> tuple[object, str]:
        """
        Retrieve the original method object and its parent module or class name using AST parsing.

        Args:
            class_obj: The class object where the method is defined.
            local_shortname: The local short name of the method.

        Returns:
            A tuple containing:
                - The original method object.
                - The name of the parent module or class.
        """
        source = inspect.getsource(class_obj)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute) and target.attr == local_shortname:
                                if isinstance(stmt.value, ast.Attribute):
                                    original_method_name = stmt.value.attr
                                    if hasattr(class_obj, original_method_name):
                                        method_obj = getattr(class_obj, original_method_name)
                                        return method_obj, class_obj.__module__
                                elif isinstance(stmt.value, ast.Name):
                                    original_method_name = stmt.value.id
                                    if original_method_name in self.module.__dict__.keys():
                                        module_obj = getattr(self.module, original_method_name)
                                        method_obj = getattr(module_obj, local_shortname)
                                        return method_obj, module_obj.__name__
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
                                        # Remove it from the call chain
                                        call_chain.pop()
                                    if len(call_chain) == 1:
                                        # The method is imported directly, e.g.
                                        # from solve_nonhydro_program import stencil
                                        # self._stencil = stencil.with_backend(...)
                                        original_method_name = call_chain[0]
                                        if original_method_name in self.module.__dict__.keys() and not isinstance(getattr(self.module, original_method_name), types.ModuleType):
                                            method_obj = getattr(self.module, original_method_name)
                                            if type(method_obj).__module__.startswith('gt4py') and type(method_obj).__name__ == 'Program':
                                                # it's a decorated gt4py program
                                                return method_obj, method_obj.definition_stage.definition.__module__
                                    elif len(call_chain) >= 2:
                                        # The method is called from a module import, e.g.
                                        # import solve_nonhydro_program as nhsolve_prog
                                        # self._stencil = nhsolve_prog.stencil.with_backend(...)
                                        module_name = call_chain[0]
                                        original_method_name = call_chain[1]
                                        if module_name in self.module.__dict__.keys() and isinstance(getattr(self.module, module_name), types.ModuleType):
                                            module_obj = getattr(self.module, module_name)
                                            method_obj = getattr(module_obj, original_method_name)
                                            return method_obj, module_obj.__name__
        return None, None

    def map_variable_names(self, function_call_str: list[str]) -> tuple[dict[str,str], dict[str,str]]:
        """
        Map variable short names to their corresponding long names and argument names.

        This function takes a string representation of a function call, extracts the argument names
        and their values, and creates two dictionaries:
        - One mapping the argument names to their short names (the part after the last period).
        - Another mapping the short names to their full names.

        Args:
            function_call_str: A string representation of a function call with arguments.

        Returns:
            A tuple containing two dictionaries:
                - variable_map: A dictionary mapping variable names as argument to the called method and variable short name (or argument value) in the present scope.
                - variable_longnames_map: A dictionary mapping variable short and long names in the present scope.
        """
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
    
    def map_variable_types(self, method_info: dict) -> dict[str,str]:
        """
        Map variable short names to their types using the annotations provided in the method information.

        Args:
            method_info: A dictionary containing method information, including:
                - 'annotations': A dictionary where keys are argument names and values are their types.
                - 'var_names_map': A dictionary mapping argument names to their short names.

        Returns:
            A dictionary where keys are variable short names and values are their formatted types.
        """
        var_types = {}
        for arg_name, var_type in method_info['annotations'].items():
            if arg_name in method_info['var_names_map'].keys():
                var_types[method_info['var_names_map'][arg_name]] = self.format_type_string(var_type)
        return var_types
    
    def format_type_string(self, var_type: typing.Type) -> str:
        """
        Formats a type annotation into a string representation.

        Args:
            var_type: The type annotation to format.

        Returns:
            The formatted type string.

        Raises:
            AssertionError: If the origin of a generic alias is not 'Field' from the 'gt4py' module.

        Notes:
            - For generic types, the function recursively formats each argument.
            - For class types, it extracts and returns the class name.
            - It performs specific replacements for certain patterns:
                - Replaces class types like <class 'numpy.int32'> with 'int32'.
                - Replaces 'gt4py.next.common.Field[...]' with 'Field[...]'.
                - Replaces 'Dimension(value='K', kind=<DimensionKind.VERTICAL: 'vertical'>)' with 'K'.
        """
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