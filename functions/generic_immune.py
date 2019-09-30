import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from functions.clustering import find_paired_distances as fpd

# these parsing functions map the data selection functions
def parse_royal(raw_datasets):
    return parse(raw_datasets, select_data_royal)

def parse_torres(raw_datasets):
    return parse(raw_datasets, select_data_torres)

def parse_test(raw_datasets):
    return parse(raw_datasets, select_data_test)

def parse(raw_datasets, selection_function):
    """ Generic Data Parser for the Immune System Datasets

    Effects
    -------
    Removes data points with low values of x
    Shifts values of z, so they are negative
    Constructs a new field, y, that conatins the stacked values of x and z
    Creates the observation vector [1, 0, 1]
    """
    threshold_value = 1e-1
    clean_datasets = []
    updates = {'initial_values': [],
               'time_span': [],
               'fitting_configuration': {'weightings': [],
                                         'observation_vector': None,
                                        },
              }

    for data in raw_datasets:
        selected_data = selection_function(data)
        # threshold out low pathogen levels
        vals = selected_data['x']
        acceptable_vals = [v > threshold_value for v in vals]
        thresholded_data = selected_data.iloc[acceptable_vals.index(True):]
        clean_datasets.append(thresholded_data.reset_index())

    for dataset in clean_datasets:
        # shift initial conditions
        dataset['z'] = dataset['z'] - max(dataset['z'])

        # create the context updates
        updates['initial_values'].append([dataset['x'].iloc[0], 0, dataset['z'].iloc[0]])
        dataset['t'] = dataset['t'] - dataset['t'].iloc[0]
        updates['time_span'].append([0, dataset['t'].iloc[-1], dataset['t'].iloc[-1]])

        # smooshed_values = [v for v in dataset[['x', 'z']].values]
        smooshed_values = []
        for i, val in enumerate(dataset[['x', 'z']].values):
            if i == 0:
                smooshed_values.append(np.hstack([val, 0]))
            else:
                smooshed_values.append(np.hstack([val, np.nan]))

        dataset['y'] = pd.Series(smooshed_values, index=dataset.index)
        updates['fitting_configuration']['weightings'].append([1/max(dataset['x']),
                                                               -1/min(dataset['z']),
                                                               10]
                                                             )
        y_as_np = np.stack(dataset['y'].to_numpy())
        updates['fitting_configuration']['weightings'].append(fpd(y_as_np[:,0], y_as_np[:,1]))

    updates['fitting_configuration']['observation_vector'] = np.array([0, 2, 1])

    return clean_datasets, updates

def select_data_torres(data):
    """Map the data columns to model state variables"""
    data_cols = {
        'Day Post Infection': 't',
        'PD': 'x',
        'RBC': 'z',
        'Nkg7': 'w',
        # 'Status': 'status'
    }
    return data[data_cols.keys()].rename(columns=data_cols)

def select_data_royal(data):
    """Map the data columns to model state variables"""
    data_cols = {
        'day': 't',
        'parasite': 'x',
        'rbc': 'z',
        'weight': 'w',
    }
    return data[data_cols.keys()].rename(columns=data_cols)

def select_data_test(data):
    data_cols = {
        'TIME': 't',
        'X': 'x',
        'Z': 'z',
        'Y': 'w',
    }
    return data[data_cols.keys()].rename(columns=data_cols)

def visualise(ax, dataset):
    ax.plot(dataset['x'], dataset['w'], 'o-')

def knots_from_data(ts, n, dataset):
    """Selects the knots based on data weightings

    If n < number of data points, creates a sparse basis, where the knots are positioned at the
    points where the 2nd derivative is maximised

    If n > number of data points, creates a dense basis. Similar to above, selects points where the
    2nd derivative is maximised, the uniformly disperses knot locations over t
    """

    # calculate 2nd derivatives
    xdiffs = np.gradient(np.gradient(dataset['x'], dataset['t']), dataset['t'])
    zdiffs = np.gradient(np.gradient(dataset['z'], dataset['t']), dataset['t'])

    # rank the relative importance of each datapoint
    ntimes = len(dataset['t'])
    importance = sorted(range(ntimes), key=lambda i: np.abs(zdiffs * xdiffs)[i], reverse=True)

    if n <= ntimes:
        # ensure that 0 and -1 are in the knot vector
        temp_knots = importance[:n]
        if 0 in temp_knots:
            temp_knots.remove(0)
        if (ntimes-1) in temp_knots:
            temp_knots.remove(ntimes-1)
        knot_indices = [0] + sorted(temp_knots[:n-2]) + [-1]

        # match the times for knots
        corresponding_times = dataset['t'].iloc[knot_indices]
        return [min(ts, key=lambda t: np.abs(t-tk)) for tk in corresponding_times]
    else:
        corresponding_times = dataset['t'].iloc[importance[:n%ntimes]]
        add_knots = [min(ts, key=lambda t: np.abs(t-tk)) for tk in corresponding_times]
        candidates = np.linspace(0, dataset['t'].iloc[-1], n)
        # replace all closest candidates with important knots
        for k in add_knots:
            candidates[np.argmin(np.abs(candidates - k))] = k
        return list(candidates)

def behaviour_penalty(p):
    K = 100
    # fr/k -g, stability condition for type 2 equilibria
    x = p[7]*p[0]/p[1] - p[8]
    # prevent blowup of the exponential calculation
    if x <= 0:
        penalty = 1/(1+K**x)
        dpendx = -np.log(K)*(K**x)*penalty
    else:
        penalty = 1-1/(1+K**(-x))
        dpendx = -np.log(K)*K**(-x)/(K**(-x)+1)**2
    return penalty, np.array([p[7]/p[1]*dpendx, -p[7]*p[0]/(p[1]**2)*dpendx, 0, 0, 0, 0, 0, p[0]/p[1]*dpendx, -dpendx, 0, 0, 0,])
