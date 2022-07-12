from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib import lines
from scipy import stats, interpolate
from numpy import linspace, exp

def form_xmonths(ax: plt.Axes, dspec=r'1 %b %Y', majors=1, minors=15, mindspec=r'15 %b'):
    """Formats an pyplot axis with ticks and labels at the first of each Month"""
    date_format = mdates.DateFormatter(dspec)
    min_date_format = mdates.DateFormatter(mindspec)
    major_format = mdates.DayLocator(majors)
    minor_format = mdates.DayLocator(minors)

    ax.xaxis.set_major_locator(major_format)
    ax.xaxis.set_minor_locator(minor_format)
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_minor_formatter(min_date_format)

def profile_1d_plotter(xvals, yvals, confidence=0.95, truth=None, ax=None, label=True):
    # create interp
    smooth = interpolate.interp1d(xvals, yvals, kind='cubic', bounds_error=False, fill_value=0.0)
    
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot()
    
    xts = linspace(xvals[0], xvals[-1], num=1001)
    lno, = ax.plot(xvals, yvals, 'ko')
    lns, = ax.plot(xts, smooth(xts), 'k')
    
    cval = exp(-0.5 * stats.chi2.ppf(confidence, df=1))
    lnc = ax.axhline(cval, color='blue', label='Profile Interval')
    
    ret_lns = [lno, lns, lnc]
    if truth is not None:
        lnt = ax.axvline(truth, color='r', label='Truth')
        ret_lns.append(lnt)
    
    if label:
        handles = ret_lns[2:]
        handles.insert(0, lines.Line2D([0], [0], color='k', marker='o', label='Profile Likelihood'))
        ax.legend(handles=handles, framealpha=1.0)

    return ax, ret_lns