'''
This script provides a wrapper for code to be run in parallel. Using this
wrapper provides the following advantages:

* Leverage git to provide commit identifier for code that run the simulation
* Tag the results with the associated git commit sha
* Saves the arguments and the parameters along the results
* Save the results as they are generated, not at the end
* Save the results in readable JSON format
* Decouple computation and presentation (aka simulation and plot generation)

The code is defined in a separate file and should
define the following:

    * A dictionary of simulation parameters that
        are global to all instances.

    * A list of args, tuples that will be provided
        to the function to be run in parallel.

    * A function named 'parallel_loop' that will
        be run in parallel with different parameters.

Dependencies:
* gitpython
* ipyparallel
'''
from __future__ import division, print_function

import argparse, datetime
import os, time, git, json, sys
import collections
import math

data_dir_format = '{date}_{name}{tag}/'
data_file_format = 'data_{pid}.json'
error_file_format = 'error_{pid}.json'

data_dir = None
data_file = 'data.json'
param_file = 'parameters.json'
args_file = 'arguments.json'

from .tools import get_git_hash, json_append, InvalidGitRepositoryError, DirtyGitRepositoryError

def run(func_parallel_loop, func_gen_args, func_init=None, base_dir=None, results_dir=None, description=None):
    '''
    Runs the simulation

    Parameters
    ----------
    func_parallel_loop: function
        The function that should be parallelized
    func_gen_args: function
        The function that will generate all the different inputs
        for func_parallel_loop
    func_init: function, optional
        A function that will be run before the simulation starts. This might
        generate some data or import some files for example
    base_dir: str, optional
        The location of the base directory for the simulation
    results_dir: str, optional
        The name of the directory where to save results
    description: str, optional
        A short description of the simulation for the help function
    '''
    import os, json

    if description is None:
        description = 'Generic simulation script'

    if base_dir is None:
        base_dir = './'
    base_dir = os.path.abspath(base_dir)

    if results_dir is None:
        results_dir = os.path.join(base_dir, 'data/')
    elif not os.path.isabs(results_dir):
        results_dir = os.path.join(base_dir, results_dir)

    # create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--dir', type=str, help='directory to store sim results')
    parser.add_argument('-p', '--profile', type=str, help='ipython profile of cluster')
    parser.add_argument('-t', '--test', action='store_true', help='test mode, runs a single loop of the simulation')
    parser.add_argument('-s', '--serial', action='store_true', help='run in a serial loop, ipyparallel not called')
    parser.add_argument('--dummy', action='store_true', help='tags the directory as dummy, can be used for running small batches')
    parser.add_argument('parameters', type=str, help='JSON file containing simulation parameters')

    cli_args = parser.parse_args()
    ipcluster_profile = cli_args.profile
    test_flag = cli_args.test
    serial_flag = cli_args.serial
    dummy_flag = cli_args.dummy
    data_dir_name = None
    parameter_file = cli_args.parameters

    # Check the state of the github repository
    if dummy_flag:
        tag = 'dummy'

    else:
        # Not a dummy run, try to get the git hash
        try:
            tag = get_git_hash(base_dir, length=10)

        except DirtyGitRepositoryError:
            if test_flag:
                import warnings
                warnings.warn('The git repo has uncommited modifications. Going ahead for test.')
                tag = 'test'
            else:
                raise ValueError('The git repo has uncommited modifications. Aborting simulation.')

        except InvalidGitRepositoryError:
            tag = ''

    # get all the parameters
    with open(parameter_file, 'r') as f:
        parameters = json.load(f)

    # if no name is given, use the parameters file name
    if 'name' not in parameters:
        name = os.path.splitext(os.path.basename(parameter_file))[0]
        parameters['name'] = name
    else:
        name = parameters['name']

    # record date and time
    date = time.strftime("%Y%m%d-%H%M%S")

    # for convenient access to parameters:
    p = collections.namedtuple('Struct', parameters.keys())(*parameters.values())

    # Save the result to a directory
    if data_dir_name is None:
        ttag = '_' + tag if tag != '' else tag
        data_dir = os.path.join(results_dir, data_dir_format.format(date=date, name=name, tag=ttag))
    else:
        data_dir = data_dir_name
    data_file_name = os.path.join(data_dir, data_file)

    # create directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # add a few practical things to the parameters
    parameters['_git_sha'] = tag
    parameters['_date'] = date
    parameters['_base_dir'] = base_dir
    parameters['_results_dir'] = data_dir
    parameters['_parallel'] = not serial_flag

    # Save the arguments in a json file
    param_file_name = os.path.join(data_dir, param_file)
    with open(param_file_name, "w") as f:
        json.dump(parameters, f, indent=2)
        f.close()

    # run the user provided init method
    if func_init is not None:
        func_init(parameters)

    # generate all the arguments to simulate
    arguments = func_gen_args(parameters)

    # Save the arguments in a json file
    args_file_name = os.path.join(data_dir, args_file)
    with open(args_file_name, "w") as f:
        json.dump(arguments, f, indent=0)
        f.close()

    # There is the option to only run one loop for test
    if test_flag:
        print('Running one test loop only.')
        arguments = arguments[:2]

    # Prepare a few things for the status line
    n_tasks = len(arguments)
    digits = int(math.log10(n_tasks) + 1)
    dformat = '{:' + str(digits) + 'd}'
    status_line = ('   ' + dformat + '/' 
            + dformat + (' tasks done. '
                        'Forecast end {:>20s}. '
                        'Ellapsed: {:>8s} Remaining: {:>8s}'))

    print('/!\\ the time estimate will only be correct '
          'when all tasks take about the same time to finish /!\\')

    forecast = 'NA'
    time_remaining = 'NA'


    # Main processing loop
    if serial_flag:
        # add parameters to builtins so that it is accessible in the namespace
        # of the calling script
        import builtins
        builtins.parameters = parameters

        print('Running everything in a serial loop.')

        # record start timestamp
        then = time.time()
        start_time = datetime.datetime.now()

        # Serial processing
        for i,ag in enumerate(arguments):
            result = func_parallel_loop(ag)

            # save the new result!
            json_append(data_file_name, result)

            # Now format some timing estimation
            n_remaining = n_tasks - (i+1)

            ellapsed = int(time.time() - then)
            ellapsed_fmt = '{:02}:{:02}:{:02}'.format(
                    ellapsed // 3600, ellapsed % 3600 // 60, ellapsed % 60)

            # estimate remaining time
            if ellapsed > 0:
                rate = (i+1) / ellapsed  # tasks per second
                delta_finish_min = int(rate * n_remaining / 60) + 1

                tdelta = datetime.timedelta(minutes=delta_finish_min)
                end_date = datetime.datetime.now() + tdelta

                # convert to strings
                forecast = end_date.strftime('%Y-%m-%d %H:%M:%S')
                s = int(tdelta.total_seconds())
                time_remaining = '{:02}:{:02}:{:02}'.format(s // 3600, s % 3600 // 60, s % 60)

            formatted_status_line = status_line.format(i+1, n_tasks, 
                    forecast, ellapsed_fmt, time_remaining)
            print(formatted_status_line, end='\r')

        # clean the output
        print(' ' * len(formatted_status_line))

        all_loops = int(time.time() - then)
        all_loops_format = '{:02}:{:02}:{:02}'.format(
                all_loops // 3600, all_loops % 3600 // 60, all_loops % 60)

        print('Total actual processing time: {} ({} s)'.format(all_loops_format, all_loops))


    else:
        # Parallel processing code
        import ipyparallel as ip

        print('Using ipyparallel processing.')

        # Start the parallel processing
        c = ip.Client(profile=ipcluster_profile)
        NC = len(c.ids)
        print(NC, 'workers on the job')

        # Clear the engines namespace
        c.clear(block=True)

        # Push the global config to the workers
        var_space = dict(
                parameters = parameters,
                )
        c[:].push(var_space, block=True)

        # record start timestamp
        then = time.time()
        start_time = datetime.datetime.now()

        # use a load balanced view
        lbv = c.load_balanced_view()

        # dispatch to workers
        ar = lbv.map_async(func_parallel_loop, arguments)

        # We use a try here so that if something happens,
        # we can catch it and abort the jobs on all engines
        try:
            for i, result in enumerate(ar):

                # save the new result!
                json_append(data_file_name, result)

                # Now format some timing estimation
                n_remaining = n_tasks - ar.progress

                ellapsed = int(time.time() - then)
                ellapsed_fmt = '{:02}:{:02}:{:02}'.format(
                        ellapsed // 3600, ellapsed % 3600 // 60, round(ellapsed % 60))

                if ar.progress > NC and n_remaining > NC:

                    # estimate remaining time
                    rate = ellapsed / ar.progress  # tasks per second
                    delta_finish_min = int(rate * n_remaining / 60) + 1

                    tdelta = datetime.timedelta(minutes=delta_finish_min)
                    end_date = datetime.datetime.now() + tdelta

                    # convert to strings
                    forecast = end_date.strftime('%Y-%m-%d %H:%M:%S')
                    s = int(tdelta.total_seconds())
                    time_remaining = '{:02}:{:02}:{:02}'.format(s // 3600, s % 3600 // 60, s % 60)

                formatted_status_line = status_line.format(ar.progress, n_tasks, 
                        forecast, ellapsed_fmt, time_remaining)
                print(formatted_status_line, end='\r')

            # clean the output
            print(' ' * len(formatted_status_line))

            print('Show all output from nodes, if any:')
            ar.display_outputs()

        except:
            # so here, things went south. Show the traceback
            # and abort all the jobs scheduled

            import traceback
            traceback.print_exc()

            print('Aborting all remaining jobs...')
            c.abort(block=True)

        all_loops = int(time.time() - then)
        all_loops_format = '{:02}:{:02}:{:02}'.format(
                all_loops // 3600, all_loops % 3600 // 60, all_loops % 60)

        print('Total actual processing time: {} ({} s)'.format(all_loops_format, all_loops))

    print('Saved data to folder: ' + data_dir)
