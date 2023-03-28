#!/usr/bin/env python
# -*- coding: utf-8 -*-

#TODO: cleanup
import filecmp
import linecache
import os
import re
import shutil
import sys
import tempfile
from varname import nameof

def to_ascii(text):
    if sys.version_info[0] == 3:
        return bytes(text, 'ascii')
    else:
        return str(text)

"""
add_pp_ser_directives.py

Parser to add SerialBox directives in Fortran code, to be processed by the pp_ser.py
script.

The implementation uses two passes. The first pass collects all necessary calls
which is then used for a corresponding USE statement importing the necessary
methods from the Fortran serialization module. The second pass expands the directives.
"""

# information
__author__ = 'Christopher Bignamini'
__copyright__ = ''
__license__ = 'GPL'
__version__ = '0.1'
__date__ = ''
__email__ = 'bignamini@cscs.ch'

class AddPPSer:



    #module='m_serialize', ?
    def __init__(self, infile, outfile='', identical=True, verbose=False):

        # public variables
        self.verbose = verbose
        self.infile = infile          # input file
        self.outfile = outfile        # output file
        self.identical = identical    # write identical files (no preprocessing done)?

        # private variables
        self.__line = ''              # current line
        self.__linenum = 0            # current line number
        self.__outputBuffer = ''      # preprocessed file
        self.__skip_next_n_lines = 0  # Number of line to skip (use for lookahead)

        # parameter storage
        self.in_parameters = []
        self.final_in_parameters = []
        self.inout_parameters = []
        self.final_inout_parameters = []
        self.out_parameters = []
        self.final_out_parameters = []

    def addCartesianTypeDecomposition(self, var_name, parameter_list):
        print('cart varname')
        print(var_name)
        print('cart varname end')
        self.__outputBuffer += "real(wp) :: " + var_name + '_x(:,:,:), ' + var_name + '_y(:,:,:), ' + var_name + '_z(:,:,:)\n'
        parameter_list.append(('cart', 'real(c_double)', var_name))

    def addTangentVectorTypeDecomposition(self, var_name, parameter_list):
        print('tangent varname')
        print(var_name)
        print('tangent varname end')
        self.__outputBuffer += "real(wp) :: " + var_name + '_v1(:,:,:), ' + var_name + '_v2(:,:,:)\n'
        parameter_list.append(('tangent', 'real(c_double)', var_name))

    def addCartesianTypeSerializationDirectives(self, var_name, prefix):
        self.__outputBuffer += '!ser verbatim \"allocate(' + var_name + '_x(size(' + var_name + ', 1), size(' + var_name + ', 2), size(' + var_name + ', 3)))\"\n'
        self.__outputBuffer += '!ser verbatim \"allocate(' + var_name + '_y(size(' + var_name + ', 1), size(' + var_name + ', 2), size(' + var_name + ', 3)))\"\n'
        self.__outputBuffer += '!ser verbatim \"allocate(' + var_name + '_z(size(' + var_name + ', 1), size(' + var_name + ', 2), size(' + var_name + ', 3)))\"\n'
        # TODO: add decomposed variable to list?
        self.__outputBuffer += '!ser data ' + prefix + '_' + var_name + '_x(:,:,:) = ' + var_name + '(:,:,:)%x(1)\n'
        self.__outputBuffer += '!ser data ' + prefix + '_' + var_name + '_y(:,:,:) = ' + var_name + '(:,:,:)%x(2)\n'
        self.__outputBuffer += '!ser data ' + prefix + '_' + var_name + '_z(:,:,:) = ' + var_name + '(:,:,:)%x(3)\n'

    def addTangentVectorTypeSerializationDirectives(self, var_name, prefix):
        self.__outputBuffer += '!ser verbatim \"allocate(' + var_name + '_v1(size(' + var_name + ', 1), size(' + var_name + ', 2), size(' + var_name + ', 3)))\"\n'
        self.__outputBuffer += '!ser verbatim \"allocate(' + var_name + '_v2(size(' + var_name + ', 1), size(' + var_name + ', 2), size(' + var_name + ', 3)))\"\n'
        # TODO: add decomposed variable to list?
        self.__outputBuffer += '!ser data ' + prefix + '_' + var_name + '_v1(:,:,:) = ' + var_name + '(:,:,:)%v1\n'
        self.__outputBuffer += '!ser data ' + prefix + '_' + var_name + '_v2(:,:,:) = ' + var_name + '(:,:,:)%v2\n'

    def addCartesianTypeDecompositionCleanup(self, var_name):
        self.__outputBuffer += '!ser verbatim \"deallocate(' + var_name + '_x, ' + var_name + '_y, ' + var_name + '_z)\n'

    def addTangentTypeDecompositionCleanup(self, var_name):
        self.__outputBuffer += '!ser verbatim \"deallocate(' + var_name + '_v1, ' + var_name + '_v2)\n'

    # execute one parsing pass over file
    def parse(self, generate=False):
        # if generate == False we only analyse the file

        # reset flags (which define state of parser)
        self.__ser = False        # currently processing !$SER directives
        self.__line = ''          # current line
        self.__linenum = 0        # current line number
        self.__module = ''        # current module
        self.__outputBuffer = ''  # preprocessed file

                
        # open and parse file
        input_file = open(os.path.join(self.infile), 'r')
        try:
            # regex for start and end of functions/subroutines
            r_start = re.compile('^ *(subroutine|function).*', re.IGNORECASE)
            r_start_function = re.compile('^ *(function).*', re.IGNORECASE)
            r_end = re.compile('^ *(end subroutine|end function).*', re.IGNORECASE)
            
            # regex for parameter declaration
            r_parameter = re.compile('.*(::).*')
            r_intent_in = re.compile('.*INTENT\(.*IN.*\)', re.IGNORECASE)#\s* ::.*')
            r_intent_out = re.compile('.*INTENT\(.*OUT.*\)', re.IGNORECASE)#\s* ::.*')
            r_intent_inout = re.compile('.*INTENT\(.*INOUT.*\)', re.IGNORECASE)#\s* ::.*')
            r_par_split_pattern = r'::'
            r_par_list_split_pattern = r','

            # regex for execution path control
            r_return = re.compile('.*(return).*', re.IGNORECASE)
            
            # in/out/inout parameter storage
            self.in_parameters=[]
            self.final_in_parameters=[]
            self.out_parameters=[]
            self.final_out_parameters=[]
            self.inout_parameters=[]
            self.final_inout_parameters=[]
            nointent_parameters=[]

            # loop over file lines and identify functions/subroutines and parameter declaration
            is_fun_subroutine = False
            print_in_parameters = True
            print_inout_parameters = True
            print_out_parameters = True
            print_init_directives = True
            execute_variable_cleanup = True
            fun_subroutine_name = ""
            new_line = ''
            for line in input_file:
                # Skip line already handled
                if(self.__skip_next_n_lines > 0):
                    self.__skip_next_n_lines -= 1
                    self.__linenum += 1
                    continue
                self.__linenum += 1

                # Check if current line is empty, is a commented one or a preprocessor directive
                if ((line.strip() != "") and (re.match('\s*!', line)==None) and
                    (re.match('#if', line)==None) and (re.match('#endif', line)==None) and
                    (re.match('#else', line)==None) and (re.match('#undef', line)==None)): #TODO: more compact syntax

                    # Check if end of line comments are present and remove them
                    current_line = line
                    if(re.match('.*!', current_line)):
                        current_line = re.sub('!.*', '', current_line, re.IGNORECASE)

                    # Check if we are dealing with multiline
                    if(re.match('.* &', current_line, re.IGNORECASE)):
                        new_line += re.sub('&', '', current_line, re.IGNORECASE)
                        new_line = new_line.rstrip()
                        if(generate):
                            self.__outputBuffer += line
                        continue

                    new_line += current_line
                
                    # identify subroutine and function
                    # check if subroutine/function is started
                    m_start = r_start.search(new_line)
                    if(m_start):
                        # this is the start of a  subroutine/function
                        is_fun_subroutine = True
                        print_init_directives = False
                        m_start_function = r_start_function.search(m_start.group())
                        new_line_stripped = new_line.replace(" ","")
                        fun_subroutine_name = (re.split('\s*SUBROUTINE|\s*FUNCTION|\(',new_line_stripped))[1]
                        if(m_start_function):
                            # this is a function, look for return parameter
                            # TODO: we are assuming function name is the return parameter,
                            # include other cases
                            # TODO: use correct function return type
                            self.out_parameters.append(('', 'function_return_type', fun_subroutine_name))

                    # check if subroutine/function is finished                
                    m_end = r_end.search(new_line)
                    if(m_end):
                        # this is the end of a subroutine/function
                        is_fun_subroutine = False
                        print("Input parameters")
                        print(self.in_parameters)
                        print("Output parameters")
                        print(self.out_parameters)
                        print("Input/Output parameters")
                        print(self.inout_parameters)
                        print("No intent parameters")
                        print(nointent_parameters)
                        print_in_parameters = True
                        print_out_parameters= True
                        print_inout_parameters = True
                        print_init_directives = True
                        # print out/inout parameter serialization directives
                        for var in self.final_inout_parameters:
                            if(var[0] == ''):
                                self.__outputBuffer += "!$ser data end_inout_" + var[2] + "=" + var[2] + "\n"
                            elif(var[0] == 'cart'):
                                self.addCartesianTypeSerializationDirectives(var[2], 'end_inout')
                            elif(var[0] == 'tangent'):
                                self.addTangentVectorTypeSerializationDirectives(var[2], 'end_inout')
                        for var in self.final_out_parameters:
                            if(var[0] == ''):
                                self.__outputBuffer += "!$ser data end_out_" + var[2] + "=" + var[2] + "\n"
                            elif(var[0] == 'cart'):
                                self.addCartesianTypeSerializationDirectives(var[2], 'end_out')
                            elif(var[0] == 'tangent'):
                                self.addTangentVectorTypeSerializationDirectives(var[2], 'end_out')

                        # deallocate composed data types additional variables
                        for var in self.final_in_parameters:
                            if('t_cartesian_coordinates' in var[1]):
                                self.addCartesianTypeDecompositionCleanup(str(var[2]))
                            elif('t_tangent_vector' in var[1]):
                                self.addTangentTypeDecompositionCleanup(str(var[2]))
                        for var in self.final_inout_parameters:
                            if('t_cartesian_coordinates' in var[1]):
                                self.addCartesianTypeDecompositionCleanup(str(var[2]))
                            elif('t_tangent_vector' in var[1]):
                                self.addTangentTypeDecompositionCleanup(str(var[2]))
                        for var in self.final_out_parameters:
                            if('t_cartesian_coordinates' in var[1]):
                                self.addCartesianTypeDecompositionCleanup(str(var[2]))
                            elif('t_tangent_vector' in var[1]):
                                self.addTangentTypeDecompositionCleanup(str(var[2]))
                        self.in_parameters = []
                        self.out_parameters = []
                        self.inout_parameters = []
                        self.final_in_parameters = []
                        self.final_out_parameters = []
                        self.final_inout_parameters = []
                        nointent_parameters = []
                        fun_subroutine_name = ""


                    # if we are in subroutine/function look for parameters
                    if(is_fun_subroutine):
                        m_parameters = r_parameter.search(new_line)
                        if(m_parameters):
                            # get ready to print init directives when variable declaration block ends
                            print_init_directives = True
                            # get ready to cleanup variable list when declaration block ends
                            execute_variable_cleanup = True
                            # find parameters intent
                            if(r_intent_inout.search(m_parameters.group(0))):
                                # inout parameter
                                # TODO: code duplication in var split and cleanup
                                declaration_line = re.split(r_par_split_pattern, m_parameters.group())
                                cleaned_declaration_line=declaration_line[1]
                                cleaned_declaration_line=re.sub('\(.*\)', '', cleaned_declaration_line)
                                parameter_list = re.split(r_par_list_split_pattern, cleaned_declaration_line)
                                parameter_list = [var.strip(' ') for var in parameter_list]
                                parameter_list = [var.strip('&') for var in parameter_list]
                                var_type = re.split(r_par_list_split_pattern, declaration_line[0])
                                for var in parameter_list:
                                    self.inout_parameters.append(('', var_type[0], var))
                            elif(r_intent_in.search(m_parameters.group(0))):
                                # input parameter
                                # TODO: code duplication in var split and cleanup
                                declaration_line = re.split(r_par_split_pattern, m_parameters.group())
                                cleaned_declaration_line=declaration_line[1]
                                cleaned_declaration_line=re.sub('\(.*\)', '', cleaned_declaration_line)
                                parameter_list = re.split(r_par_list_split_pattern, cleaned_declaration_line)
                                parameter_list = [var.strip(' ') for var in parameter_list]
                                parameter_list = [var.strip('&') for var in parameter_list]
                                var_type = re.split(r_par_list_split_pattern, declaration_line[0])
                                for var in parameter_list:
                                    self.in_parameters.append(('', var_type[0], var))
                            elif(r_intent_out.search(m_parameters.group(0))):
                                # output parameter
                                # TODO: code duplication in var split and cleanup
                                declaration_line = re.split(r_par_split_pattern, m_parameters.group())
                                cleaned_declaration_line=declaration_line[1]
                                cleaned_declaration_line=re.sub('\(.*\)', '', cleaned_declaration_line)
                                parameter_list = re.split(r_par_list_split_pattern, cleaned_declaration_line)
                                parameter_list = [var.strip(' ') for var in parameter_list]
                                parameter_list = [var.strip('&') for var in parameter_list]
                                var_type = re.split(r_par_list_split_pattern, declaration_line[0])
                                for var in parameter_list:
                                    self.out_parameters.append(('', var_type[0], var))
                            else:
                                # nointent parameter
                                # TODO: code duplication in var split and cleanup
                                declaration_line = re.split(r_par_split_pattern, m_parameters.group())
                                cleaned_declaration_line=declaration_line[1]
                                cleaned_declaration_line=re.sub('\(.*\)', '', cleaned_declaration_line)
                                parameter_list = re.split(r_par_list_split_pattern, cleaned_declaration_line)
                                parameter_list = [var.strip(' ') for var in parameter_list]
                                parameter_list = [var.strip('&') for var in parameter_list]
                                var_type = re.split(r_par_list_split_pattern, declaration_line[0])
                                for var in parameter_list:
                                    nointent_parameters.append(('', var_type[0], var))
                                # if we are in a function, check if among nointent parameters we have
                                # the function output
                        else:

                            # here I'm assuming that all the declarations appear at the beginning
                            # of a function/subroutine, so their section should now be over
                            if(self.in_parameters or self.inout_parameters or
                               self.out_parameters or nointent_parameters):
                                # create final parameter lists with composed types
                                # check if we have composed types and add the decomposition code if needed
                                if(execute_variable_cleanup):
                                    for var in self.in_parameters:
                                        if('t_cartesian_coordinates' in var[1]):
                                            self.addCartesianTypeDecomposition(str(var[2]), self.final_in_parameters)
                                        elif('t_tangent_vector' in var[1]):
                                            self.addTangentVectorTypeDecomposition(str(var[2]), self.final_in_parameters)
                                        #else if (other composed types)
                                        else:
                                            self.final_in_parameters.append(var)
                                    for var in self.inout_parameters:
                                        if('t_cartesian_coordinates' in var[0]):
                                            self.addCartesianTypeDecomposition(str(var[1]), self.inout_parameters)
                                        elif('t_tangent_vector' in var[0]):
                                            self.addTangentVectorTypeDecomposition(str(var[1]), self.inout_parameters) 
        #                                #else if (other composed types)
                                        else:
                                            self.final_inout_parameters.append(var)
                                    for var in self.out_parameters:
                                        if('t_cartesian_coordinates' in var[0]):
                                            self.addCartesianTypeDecomposition(str(var[1]), self.out_parameters)
                                        elif('t_tangent_vector' in var[0]):
                                            self.addTangentVectorTypeDecomposition(str(var[1]), self.out_parameters) 
        #                                #else if (other composed types)
                                        else:
                                            self.final_out_parameters.append(var)
                                    execute_variable_cleanup = False

                                # add init serialization directives
                                if(print_init_directives):
                                    self.__outputBuffer += '!$ser init directory="./ser_data" prefix="' + fun_subroutine_name + '"\n'
                                    self.__outputBuffer += '!$ser mode write\n'
                                    self.__outputBuffer += '!$ser savepoint ' + fun_subroutine_name
                                    if('jg' in self.in_parameters):
                                        self.__outputBuffer += ' id=jg'
                                    self.__outputBuffer += '\n'
#$ser savepoint call-diffusion-init nproma=nproma date=TRIM(date) id=jg nshift_total=nshift nlev=nlev dtime=dtime linit=linit limited_area=l_limited_area num_cells=p_patch%n_patch_cells num_edges=p_patch%n_patch_edges num_vert=p_patch%n_patch_cells exit=.FALSE.
                                    print_init_directives = False

                                # add in and inout parameter serialization directives
                                if(print_in_parameters):
                                    print('self.final_in_parameters')
                                    print(self.final_in_parameters)
                                    print('self.final_in_parameters end') 
                                    for var in self.final_in_parameters:
                                        # check if composed types are present and if so add decomposition directives
                                        if(var[0] == ''):
                                            self.__outputBuffer += "!$ser data start_in_" + var[2] + "=" + var[2] + "\n"
                                        elif(var[0] == 'cart'):
                                            self.addCartesianTypeSerializationDirectives(var[2], 'start_in')
                                        elif(var[0] == 'tangent'):
                                            self.addTangentVectorTypeSerializationDirectives(var[2], 'start_in')
                                    print_in_parameters=False
                                if(print_inout_parameters):
                                    # check if composed types are present and if so add decomposition directives
                                    for var in self.final_inout_parameters:
                                        if(var[0] == ''):
                                            self.__outputBuffer += "!$ser data start_inout_" + var[2] + "=" + var[2] + "\n"
                                        elif(var[0] == 'cart'):
                                            self.addCartesianTypeSerializationDirectives(var[2], 'start_inout')
                                        elif(var[0] == 'tangent'):
                                            self.addTangentVectorTypeSerializationDirectives(var[2], 'start_inout')
                                    print_inout_parameters=False

                                # check if there is a return statement
                                m_return = r_return.search(new_line)
                                if(m_return):
                                    # TODO: there could be multiple return in the same subroutine/function
                                    # print out/inout parameter serialization directives
                                    for var in self.final_inout_parameters:
                                        if(var[0] == ''):
                                            self.__outputBuffer += "!$ser data ret_inout_" + var[2] + "=" + var[2] + "\n"
                                        elif(var[0] == 'cart'):
                                            self.addCartesianTypeSerializationDirectives(var[2], 'ret_inout')
                                        elif(var[0] == 'tangent'):
                                            self.addTangentVectorTypeSerializationDirectives(var[2], 'ret_inout')
                                    for var in self.final_out_parameters:
                                        if(var[0] == ''):
                                            self.__outputBuffer += "!$ser data ret_out_" + var[2] + "=" + var[2] + "\n"
                                        elif(var[0] == 'cart'):
                                            self.addCartesianTypeSerializationDirectives(var[2], 'ret_out')
                                        elif(var[0] == 'tangent'):
                                            self.addTangentVectorTypeSerializationDirectives(var[2], 'ret_out')
                                    for var in self.in_parameters:
                                        if('t_cartesian_coordinates' in var[1]):
                                            self.addCartesianTypeDecompositionCleanup(str(var[2]))
                                        elif('t_tangent_vector' in var[1]):
                                            self.addTangentTypeDecompositionCleanup(str(var[2]))


                    new_line = ''

                self.__outputBuffer += line

        finally:
            input_file.close()

    # main processing method
    def process(self):

        # parse file
#        self.parse()

        # generate output buffer
        self.parse(generate=True)
#        print(self.__outputBuffer)

        # write output
        if self.outfile != '':
            output_file = tempfile.NamedTemporaryFile(delete=False)
            # same permissions as infile
            os.chmod(output_file.name, os.stat(self.infile).st_mode)
            output_file.write(to_ascii(self.__outputBuffer))
            output_file.close()
            useit = True
            if os.path.isfile(self.outfile) and not self.identical:
                if filecmp.cmp(self.outfile, output_file.name):
                    useit = False
            if useit:
                try:
                    os.rename(output_file.name, self.outfile)
                except:
                    shutil.move(output_file.name, self.outfile)
            else:
                os.remove(output_file.name)
        else:
            print(self.__outputBuffer)


def parse_args():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-i', '--ignore-identical', help='Ignore files which are not modified by pre-processor',
                      default=False, action='store_true', dest='ignore_identical')
    parser.add_option('-d', '--output-dir', help='The target directory for writing pre-processed files',
                      default='', type=str, dest='output_dir')
    parser.add_option('-o', '--output', help='Output file name to preprocess single file',
                      default='', type=str, dest='output_file')
    parser.add_option('-r', '--recursive', help='Recursively process target directory and mirror tree',
                      default=False, action='store_true', dest='recursive')
    parser.add_option('-v', '--verbose', help='Enable verbose execution',
                      default=False, action='store_true', dest='verbose')
    parser.add_option('-p', '--no-prefix', help='Don\'t generate preprocessing macro definition for ACC_PREFIX',
                      default=True, action='store_false', dest='acc_prefix')
    parser.add_option('-a', '--acc-if', help='Add IF clause to OpenACC update statement',
                      default='', type=str, dest='acc_if')
    parser.add_option('-m', '--module', help='Extra MODULE to be add to the use statement',
                      default='', type=str, dest='modules')
    parser.add_option('-s', '--sp-as-var', help='Savepoint specified as variable instead of string',
                      default=False, action='store_true', dest='sp_as_var')
    (options, args) = parser.parse_args()
    if len(args) < 1:
        parser.error('Need at least one source file to process')
    if options.output_file and len(args) > 1:
        parser.error('Single source file required if output file is given')
    if options.recursive:
        if not options.output_dir:
            parser.error('Output directory is required with recursive option')
        for indir in args:
            if not os.path.isdir(indir):
                parser.error('Arguments need to be directories with recursive option')
    return options, args

if __name__ == "__main__":
    (options, args) = parse_args()
    if options.recursive:
        file_list = []
        for indir in args:
            build_tree(indir, options.output_dir, file_list, filter_fortran)
        args = file_list

    for infile in args:
        if options.output_dir:
            if options.recursive:
                outfile = os.path.join(options.output_dir,
                                       os.path.sep.join([p for p in os.path.dirname(infile).rsplit(os.path.sep) if p][1:]),
                                       os.path.basename(infile))
            else:
                outfile = os.path.join(options.output_dir, os.path.basename(infile))
        elif options.output_file:
            outfile = options.output_file
        else:
            outfile = ''

        # If output is to a file and the file is more updated than the input, skip
        if os.path.exists(outfile) and os.path.getctime(outfile) > os.path.getctime(infile):
            print('Skipping', infile)
        else:
            print('Processing file', infile)
            ser = AddPPSer(infile, outfile=outfile, identical=(not options.ignore_identical),
                           verbose=options.verbose)
            ser.process()
