import numpy as np

def knot_fn(ts, n, dataset):
    """ A curvature based knot location selection function 
    Inputs:
    ts - fine time grid
    n - number of knots
    dataset - ['y'] contains the data
              ['t'] contains the time gridding of data
    """
    y = dataset['y'].flatten()
    times = dataset['t'].flatten()
    diffs = np.gradient(np.gradient(y, times), times)
    ntimes = len(times)
    importance = sorted(range(ntimes), key=lambda i: np.abs(diffs[i]), reverse=True)
    if n <= ntimes:
        # ensure that 0 and -1 are in the knot vector
        temp_knots = importance[:n]
        if 0 in temp_knots:
            temp_knots.remove(0)
        if (ntimes-1) in temp_knots:
            temp_knots.remove(ntimes-1)
        knot_indices = [0] + sorted(temp_knots[:n-2]) + [-1]

        # match the times for knots
        corresponding_times = times[knot_indices]
        # align along fine grid (optional)
        return [min(ts, key=lambda t: np.abs(t-tk)) for tk in corresponding_times]
    else:
        # determine which time points to refine
        copies = (n//ntimes)*np.ones(ntimes)
        copies[importance[:(n%ntimes)]] += 1
        copies = [int(j) for j in copies]
        # compute the number of knot points in each gap
        kgn = [int(copies[0]-1)]
        for ci in copies[1:]:
            m = int(ci//2)
            kgn[-1]+=m
            kgn.append(m)
        kgn[-1]+=copies[-1]-1
        # select knots to keep, always keep end knots
        keep = [int(ci%2) for ci in copies]
        keep[-1] = 1
        knots = [times[0]]
        # construct knot locations
        for gapn, k, x0, x1, c in zip(kgn, keep[1:], times[:-1], times[1:], copies[1:]):
            lspc = np.linspace(x0, x1, gapn+2)
            frag = lspc[1:int(gapn+1+k)]
            knots.extend(frag)
        return knots