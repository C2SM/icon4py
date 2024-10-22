from sphinx.ext import autodoc
import re
import inspect
import typing

class FullMethodDocumenter(autodoc.MethodDocumenter):
    """
    'Fully' document a method, i.e. picking up and processing all docstrings in
    its source code.
    """

    objtype = 'full'
    priority = autodoc.MethodDocumenter.priority - 1

    # Configuration options
    var_type_in_inputs = True
    var_type_formatting = '``'

    def get_doc(self):
        # Override the default get_doc method to pick up all docstrings in the
        # source code
        source = inspect.getsource(self.object) # this is only the source of the method, not the whole file

        docstrings = re.findall(r'"""(.*?)"""', source, re.DOTALL)
        docstrings_list = []
        for idocstr, docstring in enumerate(docstrings):
            formatted_docstr = None
            if idocstr==0:
                # "Usual" docstring at the beginning of the method
                formatted_docstr = docstring.splitlines()

            elif docstring.startswith('_scidoc_'):
                # Get some useful information
                call_string = self.get_next_method_call(source, source.find(docstring) + len(docstring))
                next_method_info = self.get_method_info(call_string)
                # and process a scientific documentation docstring
                formatted_docstr = docstring.splitlines()
                formatted_docstr = self.format_source_code(formatted_docstr)
                formatted_docstr = self.add_header(formatted_docstr, next_method_info)
                formatted_docstr = self.process_scidocstrlines(formatted_docstr, next_method_info)
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
        while source_lines[0] == '':
            # strip empty lines from the beginning
            source_lines.pop(0)
        # strip leading and trailing whitespace of every line (maintain indentation)
        indent = len(source_lines[0]) - len(source_lines[0].lstrip(' '))
        source_lines = [line[indent:].rstrip() for line in source_lines]
        # strip empty lines from the end
        while source_lines[-1] == '':
            source_lines.pop(-1)
        return source_lines

    def add_header(self, docstr_lines, method_info):
        # Add a title with ReST formatting
        method_name = method_info['method_name']
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
                                split_line[j] = f"{part} {self.var_type_formatting}{var_type}{self.var_type_formatting}"
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
            call_string = self.format_source_code(source.splitlines()[call_start_line:call_end_line+1])
            return call_string
        return None

    def get_method_info(self, call_string):
        method_info = {}
        local_method_name = call_string[0].split('(')[0] # get the method name from the call string (remove open bracket)
        # Get the module of this method from its name
        module_name = local_method_name.split('.')[0] # this is not robust because it assumes we always import with module names
        method_name = local_method_name.split('.')[-1]
        module_obj = getattr(self.module, module_name)
        method_obj = getattr(module_obj, method_name)
        #
        method_info['call_string'] = call_string
        method_info['method_name'] = method_name
        method_info['module_local_name'] = module_name
        method_info['module_full_name'] = module_obj.__name__
        method_info['annotations'] = method_obj.definition.__annotations__
        method_info['var_names_map'] = self.map_variable_names(call_string)
        method_info['var_types'] = self.map_variable_types(method_info)
        return method_info

    def map_variable_names(self, function_call_str):
        # Extract argument names and their values using regex
        pattern = r'(\w+)\s*=\s*([^,]+)'
        matches = re.findall(pattern, ''.join(function_call_str))
        # Create a dictionary to map variable names to their full argument names
        variable_map = {}
        for arg_name, arg_value in matches:
            # Extract the last part of the argument value after the last period
            short_name = arg_value.split('.')[-1]
            variable_map[arg_name] = short_name
        return variable_map
    
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
            # Replace "Dims[Cell, K]" with "(Cell, K)"
            type_str = re.sub(r"Dims\[(\w+), (\w+)\]", r"(\1, \2)", type_str)
            # Replace "Dims[Kdim]" with "(Kdim)"
            type_str = re.sub(r"Dims\[(\w+)\]", r"(\1)", type_str)
            # Shorten "Cell" to "C"
            type_str = re.sub(r"Cell", "C", type_str)
            return type_str