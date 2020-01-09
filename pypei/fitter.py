from . import modeller, objective

import casadi as ca
import numpy as np

class Solver(object):
    """ Wrapper for a CasADi NLP Solver """
    def __init__(self, objective=None, config=None):
        pass

    def run(self, config=None):
        pass

class Fitter(object):
    """ A simple workflow for fitting a model to data """

    def __init__(self, config=None):
        empty_dataset = dict(
                t = list(),
                y = list(),
            )
        self.model_config = dict(
            grid_size = int(),
            basis_number = int(),
            model_form = dict(
                state = int(),
                parameters = int(),
            ),
            time_span = list((int(), int())),
            knot_function = None, # function, default None
            model = None, # function, required
            dataset = empty_dataset,
        )
        self.objective_config = dict(
            dataset = empty_dataset,
            observation_vector = list(list()), # list of lists
            observation_model = None, # function, default None or missing
            weightings = list((list(), list())),
            regularisation_value = list(),
        )
        self.solver_config = dict(
            constraints = None, # function
            solve_opts = dict(), # options separate from the problem formulation
        )
        self.run_config = dict(
            p0 = list(),
            zero_init = True,
            parameters = list(),
            xlbounds = None,
            xubounds = None,
            glbounds = None,
            gubounds = None,
            callfrwd = None, # function, called pre-run
            callback = None, # function, called post-run
        )

        self.model = None
        self.objective = None
        self.solver = None

        if config:
            self.update_configs(config)
    
    def update_configs(self, config):
        self.update_dict(self.model_config, config)
        self.update_dict(self.objective_config, config)
        self.update_dict(self.solver_config, config)
        self.update_dict(self.run_config, config)

    @staticmethod
    def update_dict(template, updater):
        """ Update a dict's values from another dict, do not introduce new keys 
        
        Updates template's values from updater's values. 
        Keys that exist in updater, but not template, are not added to template.
        """
        for key in template:
            if key in updater:
                template[key] = updater[key]

    def build(self):
        self.model = modeller.Model(self.model_config)
        self.objective = objective.Objective()
        self.objective.make(self.objective_config, self.model)
        self.solver = Solver(objective, self.solver_config)
        self.init_run_config()

    def init_run_config(self):
        pass
        # will call update_run_config with specific parameters

    def update_run_config(self, config):
        pass

    def run(self, config=None):
        if config:
            self.update_run_config(config)
        return self.solver.run(self.run_config)