from matplotlib import dates as mdates
from matplotlib import pyplot as plt

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


