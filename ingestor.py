from pathlib import Path
from importlib import util as importutil
import warnings
import pandas as pd
import numpy as np

def check_file(file_name):
    """Checks to see if a file exists"""
    if not Path(file_name).exists():
        raise FileNotFoundError(file_name + " not found.")

def import_module_from_file(file_name):
    """helper function to import an arbitrary file as module"""
    check_file(file_name)
    module_name = Path(file_name).stem
    try:
        module_spec = importutil.spec_from_file_location(module_name, file_name)
        if module_spec is None:
            raise ImportError
        module_module = importutil.module_from_spec(module_spec)
        module_spec.loader.exec_module(module_module)
    except ImportError:
        error_string = file_name +  " not found"
        raise ImportError(error_string)

    return module_module

def fn_from_file(file, function_name):
    """Helper function to get a function from a py file"""
    return getattr(import_module_from_file(file), function_name)


class Context():
    """Class that defines algorithm parameters and functions"""
    def __init__(self, run_file_name=None):
        self.context_files = {
            'model_file': str(),
            'configuration_file': str(),
            'data_files': list(),
        }
        self.model = None
        self.initial_parameters = list()
        self.initial_values = list()
        self.time_span = list()
        self.datasets = list()
        self.data_parser = list()
        self.modelling_configuration = {
            'grid_size': 0,
            'basis_number': 0,
            'model_form': None,
            'knot_function': None,
        }
        self.fitting_configuration = {
            'regularisation_parameter': [],
            'weightings': None,
            'observation_vector': None,
        }
        self.visualisation_function = None

        if run_file_name:
            self.read_run_file(run_file_name)
            self.setup()

    def read_run_file(self, run_file_name):
        """Parses a run file"""
        check_file(run_file_name)
        with open(run_file_name, 'r') as run_file:
            run_configs = run_file.read().splitlines()
        for config in run_configs:
            config_values = config.split()
            config_type = config_values.pop(0)
            if config_type == "#":
                continue
            elif config_type in ['mf', 'model-file']:
                self.context_files['model_file'] = str(config_values[0])
            elif config_type in ['cf', 'configuration-file']:
                self.context_files['configuration_file'] = str(config_values[0])
            elif config_type in ['df', 'data-file']:
                self.context_files['data_files'].extend(config_values)
            else:
                error_string = "Unhandled config type: " + str(config_type)
                raise TypeError(error_string)

    def setup(self):
        """Reads in each of the context files"""
        self.parse_model_file()
        self.parse_config_file()
        self.parse_data_files()

    def parse_model_file(self):
        """Reads the model file to return the function that specifies the model"""
        if not self.context_files['model_file']:
            raise RuntimeError('No model file specified')
        else:
            check_file(self.context_files['model_file'])

        model_module = import_module_from_file(self.context_files['model_file'])
        self.model = model_module.model
        self.modelling_configuration['model_form'] = model_module.model_form()

    def parse_config_file(self):
        """Parses the configuration file"""
        if not self.context_files['configuration_file']:
            warnings.warn('No configuration file specified', RuntimeWarning)
            return

        check_file(self.context_files['configuration_file'])
        with open(self.context_files['configuration_file']) as config_file:
            configs = config_file.read().splitlines()

        for config in configs:
            config_values = config.split()
            config_type = config_values[0]
            config_values = config_values[1:]
            if config_type == '#':
                continue
            elif config_type in ['iv', 'initial-values']:
                self.initial_values = [float(val) for val in config_values]
            elif config_type in ['ip', 'initial-parameters']:
                self.initial_parameters = [float(val) for val in config_values]
            elif config_type in ['ts', 'time-span']:
                self.time_span = [float(val) for val in config_values]
            elif config_type in ['dp', 'data-parser']:
                self.data_parser = fn_from_file(config_values[0], config_values[1])
            elif config_type in ['gs', 'grid-size']:
                self.modelling_configuration['grid_size'] = int(config_values[0])
            elif config_type in ['bn', 'basis-number']:
                self.modelling_configuration['basis_number'] = int(config_values[0])
            elif config_type in ['kf', 'knot-function']:
                self.modelling_configuration['knot_function'] = fn_from_file(config_values[0], config_values[1])
            elif config_type in ['rg', 'regularisation']:
                self.fitting_configuration['regularisation_parameter'] = [float(val) for val in config_values]
            elif config_type in ['rv', 'regularisation-value']:
                self.fitting_configuration['regularisation_value'] = np.array([float(val) for val in config_values])
            elif config_type in ['vf', 'visualisation-function']:
                self.visualisation_function = fn_from_file(config_values[0], config_values[1])
            else:
                error_string = "Unhandled config type: " + str(config_type)
                raise TypeError(error_string)

    def parse_data_files(self):
        """Parses the data files"""
        if not self.context_files['data_files']:
            warnings.warn('Data files not specified', RuntimeWarning)
            return
        all_raw_data = []
        for data_file in self.context_files['data_files']:
            check_file(data_file)
            df_extension = data_file.split('.')[-1].lower()
            if df_extension == 'xlsx':
                all_raw_data.append(pd.read_excel(data_file))
            elif df_extension == "csv":
                all_raw_data.append(pd.read_csv(data_file))
            else:
                error_string = "Filetype not supported: " + str(df_extension)
                raise TypeError(error_string)
        clean_data, context_updates = self.data_parser(all_raw_data)
        self.datasets = clean_data
        self.batch_update(context_updates)

    def batch_update(self, update_dict):
        for key, value in update_dict.items():
            self.update(key, value)

    def update(self, attribute, value):
        if attribute not in dir(self):
            raise AttributeError(f"{attribute} not a valid attribute")
        this_attr = self.__getattribute__(attribute)
        if not isinstance(this_attr, dict):
            self.__setattr__(attribute, value)
        else:
            this_attr.update(value)