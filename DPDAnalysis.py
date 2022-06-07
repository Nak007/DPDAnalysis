'''
Available methods are the followings:
[1] rollrate_table
[2] plot_rollrate
[3] vintage_table
[4] plot_vintage
[5] groupby_table
[6] plot_avgdpd
[7] matplotlib_cmap
[8] observed_rate
[9] plot_scores

Author: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 30-05-2022

'''
import time
from calendar import monthrange
from scipy.interpolate import interp1d
import pandas as pd, numpy as np
from matplotlib.colors import (ListedColormap, 
                               LinearSegmentedColormap)
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as transforms
from matplotlib.ticker import(FixedLocator, 
                              FixedFormatter, 
                              StrMethodFormatter,
                              FuncFormatter)

__all__ = ["rollrate_table", "plot_rollrate", 
           "vintage_table" , "plot_vintage", 
           "groupby_table"  , "plot_avgdpd", 
           "matplotlib_cmap", "create_mob",
           "observed_rate", "plot_scores"]

def matplotlib_cmap(name='viridis', n=10):

    '''
    Parameters
    ----------
    name : matplotlib Colormap str, default='viridis'
        Name of a colormap known to Matplotlib. 
    
    n : int, defualt=10
        Number of shades for defined color map.
    
    Returns
    -------
    colors : list of color-hex
        List of color-hex codes from defined Matplotlib Colormap. 
        Such list contains "n" shades.
        
    '''
    c_hex = '#%02x%02x%02x'
    c = cm.get_cmap(name)(np.linspace(0,1,n))
    c = (c*255).astype(int)[:,:3]
    colors = [c_hex % (c[i,0],c[i,1],c[i,2]) for i in range(n)]
    return colors

def delq_table(delq, values) -> pd.DataFrame:
    
    '''
    ** Private Function **
    Delinquency table
    '''
    columns = list(delq)
    bins = np.array([ 0., 29, 59., 89., np.inf])
    states = ["(1) Clean"     , "(2) 0<dpd<30" , 
              "(3) 30<=dpd<60", "(4) 60<=dpd<90", 
              "(5) dpd>90"]
    delq = np.digitize(delq.values, bins, right=True)
    delq = np.select([delq==n for n in range(len(bins))], states, -1)
    delq = pd.DataFrame(delq, columns=columns)
    if values is None: delq["value"] = 1
    else: delq["value"] = values

    dummy = dict([(c,states) for c in columns])
    dummy.update({"value": np.zeros(len(states))})
    return delq.append(pd.DataFrame(dummy), ignore_index=True)

def rollrate_table(X, mobs, values=None, percent=False):
    
    '''
    Roll rate is the proportion of customers who will be 'better', 
    'worse' or 'remain same' with time in terms of delinquency.
    
    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_mobs)
        Input data.
    
    mobs : list of variable names [start_mob, end_mob]
        Variables within data to use as starting and ending periods
        e.g. ["M4", "M12"] starts at 4 Months on book and ends at 12
        Months on book.
    
    values : np.ndarray, default=None
         An array of shape (n_samples,) to aggregate. If None, it
         defaults to an array of shape (n_samples,) filled with ones.
         
    percent : bool, default=False
        If True, it displays in percentage, otherwise actual number.
        
    References
    ----------
    [1] https://www.listendata.com/2019/09/roll-rate-analysis.html#:
        ~:text=Roll%20rate%20analysis%20is%20used,for%2090%20days%20
        or%20more.
         
    Returns
    -------
    df : pd.DataFrame
    
    '''
    start, end = mobs
    df = pd.pivot_table(delq_table(X[mobs].copy(), values), 
                        values='value', index=start, 
                        columns=end, aggfunc=np.sum).fillna(0)
    if percent==False: return df
    else: return (df/df.values.sum(1,keepdims=True)).fillna(0)

def plot_rollrate(delq, mobs, values=None, ax=None, colors=None, 
                  num_format=None, labels=None, tight_layout=True):
    '''
    Plot Roll-Rate
    
    Parameters
    ----------
    delq : pd.DataFrame of shape (n_samples, n_mobs)
        Input data.
    
    mobs : list of variable names [start_mob, end_mob]
        Variables within data to use as starting and ending periods
        e.g. ["M4", "M12"] starts at 4 Months on book and ends at 12
        Months on book.
    
    values : np.ndarray, default=None
         An array of shape (n_samples,) to aggregate. If None, it
         defaults to an array of shape (n_samples,) filled with ones.
         
    ax : matplotlib Axes, default=None
        Axes object to draw the plot onto, otherwise uses the default 
        axis with figsize = (8.3, 5.8).
        
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to 3 i.e. 
        ["better", "remain", "worst"]. If None, it uses default 
        colors.
  
    num_format : string formatter, default=None
        String formatters (function) for all numbers except 
        percentage. If None, it defaults to "{:,.0f}".format.
    
    labels : list of str, default=None
        List of labels for x, and y axes i.e. [xlabel, ylabel]. If 
        None, it defaults to ["Current Month", "Previous Month"].
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
         
    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot drawn onto it.
        
    '''
    # Default matplotlib figure, and axis.
    if ax is not None: fig = plt.gcf()
    else: fig, ax = plt.subplots(figsize=(8.3, 5.8))
    
    # Initialize parameters
    start, end = mobs
    if colors is None: colors = ["#1B9CFC", "#ffffff", "#FC427B"]
    if num_format is None: num_format = "{:,.0f}".format
    labels_ = ["Clean (0)", "1-29", "30-59", "60-89", "90+"]

    # Plot `matshow`
    a = np.array([np.arange(5)[::-1]-n for n in range(4,-1,-1)]).T
    cmap = LinearSegmentedColormap.from_list("customized", colors, N=9)
    cax = ax.matshow(a, cmap=cmap , alpha=0.9, interpolation='nearest')

    # Create colorbar
    notches = np.unique(a)
    ticks = np.linspace(min(notches)+0.5,
                        max(notches)-0.5, len(notches))
    cbar = fig.colorbar(cax, ax=ax, ticks=ticks, pad=0.12)
    cbar.ax.set_yticklabels(["{:d}".format(n) for n in notches])
    cbar.ax.set_ylabel("Delinquency Movement (Notch)", 
                       rotation=-90, va="bottom", fontsize=13)

    # Create annotation (N, %)
    coords = [(r,c) for r in range(5) for c in range(5)]
    args = (delq.fillna(0).copy(), mobs, values)
    delq_cnt = rollrate_table(*args, False).values.T.ravel()
    delq_pct = rollrate_table(*args, True ).values.T.ravel()
    for n in range(25):
        ax.annotate(num_format(delq_cnt[n]), coords[n], 
                    textcoords="offset points", xytext=(0,2), 
                    va="bottom", ha="center", fontsize=12)
        ax.annotate("({:.0%})".format(delq_pct[n]), coords[n], 
                    textcoords="offset points", xytext=(0,-2), 
                    va="top", ha="center",fontsize=12)

    # Grid lines
    for n in np.arange(0.5,4): 
        ax.axvline(n, lw=0.8, color="k")
        ax.axhline(n, lw=0.8, color="k")

    # Ticklabels on x-axis
    ax.set_xticks(range(5))
    tot_cnt = delq_cnt.reshape((5,5)).T.sum(0)
    ax.set_xticklabels([f"{s}" + "\n(" + num_format(v) +")" 
                        for s,v in zip(labels_, tot_cnt)], 
                       fontsize=13)
    
    # Ticklabels on y-axis
    ax.set_yticks(range(5))
    tot_cnt = delq_cnt.reshape((5,5)).T.sum(1)
    ax.set_yticklabels([f"{s}" + "\n(" + num_format(v) +")"
                        for s,v in zip(labels_,tot_cnt)], 
                       fontsize=13)

    # Labels 
    if labels is None: labels = ["Current Month", "Previous Month"]
    kwds = dict(fontsize=14, fontweight=600)
    ax.xaxis.set_label_position('top')
    ax.tick_params(which="both", bottom=False, left=False, top=False)
    ax.set_xlabel(labels[0], **kwds)
    ax.set_ylabel(labels[1], **kwds)
    
    # % of roll-forward and roll-backward
    trans = delq_pct.reshape(a.shape).T
    backward = np.where(np.sign(a)==-1, trans, 0).sum(1)
    forward  = np.where(np.sign(a)== 1, trans, 0).sum(1)
    for n in range(5):
        # Roll forward
        s = r"$\leftarrow$" + "{:.0%}".format(backward[n])
        ax.annotate(s, (4.5,n), textcoords="offset points", 
                    xytext=(2,2), va="bottom", ha="left", 
                    fontsize=12, color=colors[0])
        # Roll forward
        s = r"$\rightarrow$" + "{:.0%}".format(forward[n])
        ax.annotate(s, (4.5,n), textcoords="offset points", 
                    xytext=(2,-2), va="top", ha="left", 
                    fontsize=12, color=colors[-1])

    # Draw lines that separate unchanged group and the others.
    line = np.sort(np.r_[np.arange(-0.5,5),np.arange(-0.5,5)]) 
    ax.plot(line[:-3]+1, line[1:-2], lw=3, color="#2f3542")
    ax.plot(line[1:-2] , line[2:-1], lw=3, color="#2f3542")
    ax.grid(False, which="both")
    if tight_layout: plt.tight_layout()

    return ax

def vintage_table(X, groupby, dpd_cols, dpd_geq=30):
    '''Vintage analysis'''
    groupby, dpd_cols = list(groupby), list(dpd_cols)
    new_X = X[groupby + dpd_cols].copy()
    
    a = new_X[dpd_cols].values.copy()
    a = np.where(np.isnan(a), np.nan, a>=dpd_geq)
    a = np.cumsum(a, axis=1)
    new_X[dpd_cols] = np.where(np.isnan(a), np.nan, a>1)

    new_X["N"] = 1
    aggfnc = {**{"N":"count"},**dict([(c, np.nanmean) 
                                      for c in dpd_cols])}
    return new_X.groupby(groupby).agg(aggfnc)

def groupby_table(X, groupby, cols, agg=np.nanmean, factor=1):
    '''Groupby'''
    groupby, dpd_cols = list(groupby), list(cols)
    new_X = X[groupby + cols].copy()
    new_X["N"] = 1
    aggfnc = {**{"N":"count"},**dict([(c,agg) for c in dpd_cols])}
    new_X  = new_X.groupby(groupby).agg(aggfnc)
    new_X[cols] = new_X[cols]*factor
    return new_X

def color_map(colors, N):
    '''Return `N` shades of colors from `colors`'''
    cmap = LinearSegmentedColormap.from_list("customized", colors, N=N)
    c_hex = '#%02x%02x%02x'
    c = cmap(np.linspace(0,1,N))
    c = (c*255).astype(int)[:,:3]
    return [c_hex % (c[i,0],c[i,1],c[i,2]) for i in range(N)]

def plot_vintage(X, groupby, dpd_cols, dpd_geq=30, start_mth=0, 
                 ax=None, num_format=None, colors=None, 
                 plot_kwds=None, tight_layout=True, loc=None, 
                 focus=None, n_tops=None):
    '''
    Vintage plots.
    
    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_mobs)
        Input data.
    
    groupby : list of labels
        Used to determine the groups (columns) for the groupby.
    
    dpd_cols : list of labels
        List of delinquency or Day-Past-Due (DPD) columns.
    
    dpd_geq : int, default=30
        Any delinquency that is greater or equal to `dpd_geq` will
        be marked as ever-delinquent.
    
    start_mth : int, default=0
        The starting month(s) on book.
         
    ax : matplotlib Axes, default=None
        Axes object to draw the plot onto, otherwise uses the default 
        axis with figsize = max(6.3, n_mths*1.05).
        
    num_format : string formatter, default=None
        String formatters (function) for population count shown in 
        legend. If None, it defaults to "{:,.0f}".format.
        
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to 2. If 
        None, it uses default matplotlib colors.
  
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
    
    loc : str, default=None
        The location of the legend [1]. If None, legend will be 
        located on the right side of the plot.
        
    focus : str, default=None
        Column to focus from `dpd_cols`. If None, focusing is not
        implemented.
        
    n_tops : int, greater than 1, default=10
        Maximum number of top `n` groups to be displayed in colors,
        while the rest will be dimmed. If None, it displays all groups. 
        This is only relevant when `focus` is not None.
       
    References
    ----------
    [1] https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.axes.Axes.
        legend.html
         
    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot drawn onto it.
    
    '''
    vintage = vintage_table(X, groupby, dpd_cols, dpd_geq).copy()
    n_lines, n_mths = len(vintage), len(dpd_cols)
    x = np.arange(n_mths)

    # Default parameters
    args = (plot_kwds, num_format, ax, 
            colors, n_lines, n_mths, loc)
    kwds, num_format, ax, colors = default_params(*args)
        
    # Find focus
    dim, indices = find_focus(vintage, focus, n_tops, n_lines, dpd_cols)
    ax , samples = draw_lines(ax, vintage, x, dpd_cols, groupby, 
                              n_lines, dim, indices, num_format, 
                              kwds, colors)

    # Text (top left corner)
    ax = draw_text(ax, sum(samples), dpd_geq)
    
    # Set properties of x, and y axes.
    ax = set_axes(ax, x, start_mth, 
                  ylabel="Proportion", 
                  xlabel="Months on book", 
                  percent=True)

    # Draw focus line, and legend
    ax = draw_focus(ax, focus, dpd_cols)
    ax = draw_legend(ax, loc)
    ax.grid(False, which="both")
    if tight_layout: plt.tight_layout()
        
    return ax

def plot_avgdpd(X, groupby, dpd_cols, start_mth=0, ax=None, 
                num_format=None, colors=None, plot_kwds=None, 
                tight_layout=True, loc=None, focus=None, 
                n_tops=None):
    '''
    Averge delinquency or day-past-due plots.
    
    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_mobs)
        Input data.
    
    groupby : list of labels
        Used to determine the groups (columns) for the groupby.
    
    dpd_cols : list of labels
        List of delinquency or Day-Past-Due (DPD) columns.

    start_mth : int, default=0
        The starting month(s) on book.
         
    ax : matplotlib Axes, default=None
        Axes object to draw the plot onto, otherwise uses the default 
        axis with figsize = max(6.3, n_mths*1.05).
        
    num_format : string formatter, default=None
        String formatters (function) for population count shown in 
        legend. If None, it defaults to "{:,.0f}".format.
        
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to 2. If 
        None, it uses default matplotlib colors.
  
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
    
    loc : str, default=None
        The location of the legend [1]. If None, legend will be 
        located on the right side of the plot.
        
    focus : str, default=None
        Column to focus from `dpd_cols`. If None, focusing is not
        implemented.
        
    n_tops : int, greater than 1, default=10
        Maximum number of top `n` groups to be displayed in colors,
        while the rest will be dimmed. If None, it displays all groups. 
        This is only relevant when `focus` is not None.
       
    References
    ----------
    [1] https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.axes.Axes.
        legend.html
         
    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot drawn onto it.
    
    '''
    # Get data table
    avgdpd = groupby_table(X, groupby, dpd_cols)
    n_lines, n_mths = len(avgdpd), len(dpd_cols)
    x = np.arange(n_mths)

    # Default parameters
    args = (plot_kwds, num_format, ax, 
            colors, n_lines, n_mths, loc)
    kwds, num_format, ax, colors = default_params(*args)   
        
    # Plot lines.
    dim, indices = find_focus(avgdpd, focus, n_tops, n_lines, dpd_cols)
    ax , samples = draw_lines(ax, avgdpd, x, dpd_cols, groupby, 
                              n_lines, dim, indices, num_format, 
                              kwds, colors)

    # Text (top left corner)
    ax = draw_text(ax, sum(samples), None)
    
    # Set properties of x, and y axes.
    ax = set_axes(ax, x, start_mth,
                  ylabel="Average DPD (days)", 
                  xlabel="Months on book", 
                  percent=False)
    
    # Draw dpd-lines, focus-line, and legend
    ax = draw_dpd(ax)
    ax = draw_focus(ax, focus, dpd_cols)
    ax = draw_legend(ax, loc)
    ax.grid(False, which="both")
    if tight_layout: plt.tight_layout()
        
    return ax

def default_params(plot_kwds, num_format, ax, colors, n_lines, 
                   n_mths, loc):
    '''Private function: default parameters'''
    kwds = dict(linewidth=3, solid_capstyle='round', 
                solid_joinstyle="round", markersize=7,
                markeredgewidth=1)
    if plot_kwds is not None: kwds.update(plot_kwds)
    if num_format is None: num_format = "{:,.0f}".format  
    if ax is None: 
        width = max(6.3, n_mths*1.05) + (2 if loc is None else 0)
        ax = plt.subplots(figsize=(width, 4))[1]
    if colors is not None: colors = color_map(colors, n_lines)
    return kwds, num_format, ax, colors

def draw_lines(ax, data, x, dpd_cols, groupby, n_lines, 
               dim, indices, num_format, kwds, colors):
    '''Private function: plot lines'''
    data  = data.reset_index().copy()
    evers = data.loc[:, dpd_cols].values
    group = data.loc[:, groupby].astype(str).values
    samples = data["N"].values

    for n in range(n_lines):
        ever, N  = evers[n,:], num_format(samples[n])
        label = ", ".join(group[n,:]) + f" ({N})"
        if colors is not None: kwds.update({"color":colors[n]})
        if dim & (n not in indices):
            ax.plot(x, ever, label=label, 
                    **{**kwds, **{"color":"#dfe4ea", "zorder":-1}})
        else: ax.plot(x, ever, label=label, **kwds)
    return ax, samples

def draw_text(ax, n_samples, dpd_geq=None):
    '''Private function: draw text (top left corner)'''
    args = (ax.transAxes, ax.transAxes)
    trans= transforms.blended_transform_factory(*args)
    
    if dpd_geq is not None: 
        text = r"N(dpd $\geq$ {:,d}) = "\
        .format(int(dpd_geq))
    else: text = "N = "

    text += "{:,d}".format(n_samples)
    ax.text(0.01, 1, text, transform=trans, fontsize=13, va="top", 
            ha="left", color="k", bbox=dict(pad=2, facecolor="w", 
                                            edgecolor="none"))
    return ax

def draw_dpd(ax):
    '''Private function: draw dpd lines'''
    y_min, y_max = ax.get_ylim() 
    args = (ax.transAxes, ax.transData)
    trans = transforms.blended_transform_factory(*args)
    for d in [30, 60, 90, 120]:
        if d<=y_max:
            ax.axhline(d, lw=0.8, ls="--", color="#cccccc", zorder=-1)
            text = "{:,d} days".format(d)
            ax.text(0.01, d, text, transform=trans, fontsize=12, 
                    va="bottom", ha="left", color="#cccccc", zorder=-1)
    return ax

def set_axes(ax, x, start_mth=0, ylabel="", xlabel="Months on book", 
             percent=True):
    '''Private function: set properties of x, and y axes'''
    y_min, y_max = ax.get_ylim() 
    ax.set_ylim(y_min, y_max/0.85)
    for s in ["right","top"]:ax.spines[s].set_visible(False) 
    ax.set_xticks(x + start_mth)
    ax.set_xlabel(xlabel, fontsize=13, fontweight=600)
    ax.set_ylabel(ylabel, fontsize=13, fontweight=600)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    ax.tick_params(axis='both', labelsize=12)
    if percent:
        t = mpl.ticker.PercentFormatter(xmax=1)
        ax.yaxis.set_major_formatter(t)
    return ax

def find_focus(data, focus, n_tops, n_lines, dpd_cols):
    '''Private Function: find focus'''
    if focus is not None:
        if focus in dpd_cols:
            n_tops= max(min(n_tops, n_lines),1)
            indices = data.fillna(0).reset_index()\
            .sort_values(by=focus,ascending=False).copy()
            indices = np.array(indices.index)[:n_tops]
            return True, indices
        else: return False, []
    return False, []

def draw_focus(ax, focus, dpd_cols):
    '''Private function: draw focus line'''
    if focus is not None:
        index = np.isin(dpd_cols, [focus])
        if sum(index)>0:
            k = np.argmax(index)
            ax.axvline(k, lw=1, color="#cccccc", zorder=-1)
            args = (ax.transData, ax.transAxes)
            trans = transforms.blended_transform_factory(*args)
            ax.text(k, 1, focus, transform=trans, fontsize=13, 
                    va="bottom", ha="center", color="grey")
    return ax

def draw_legend(ax, loc):
    '''Private function: draw legend'''
    kwds = dict(fontsize=12, framealpha=0)
    if loc is None:
        legend = ax.legend(loc="upper left", **kwds) 
        legend.set_bbox_to_anchor([1,1], transform=ax.transAxes)
    else: ax.legend(loc=loc, **kwds)
    return ax

def observed_rate(X, scr_col, dpd_cols, val_cols, cr_lmt, dpd_geq=60, 
                  bins=None, factor=1):
    
    # Score bin edges
    bins ="fd" if bins is None else bins 
    scrs = X[scr_col].fillna(0).values.copy()
    bins = np.histogram_bin_edges(scrs, bins)
    bins = np.round(bins,0).astype(int)
    bins[[0,-1]] = 1, bins[-1] + 1
    cat = dict([(n,c) for n,c in enumerate(bins)])
    # right=False, bins[i-1] <= x < bins[i]
    indices = np.digitize(scrs, bins)

    # Month indices
    mths = X[dpd_cols].values.copy()
    mths[np.isnan(mths)==False] = 1
    
    # Observed rate
    ones = np.ones((len(scrs),1))
    delq = mths * (X[dpd_cols]>=dpd_geq).values
    data = np.hstack((indices.reshape(-1,1), ones, delq))
    data = pd.DataFrame(data, columns=["bin_lt","N"]+dpd_cols)
    data["bin_lt"] = data["bin_lt"].apply(lambda x: cat[x])
    aggfunc = dict([("N","sum")]+[(c,"sum") for c in dpd_cols])
    delq = data.groupby("bin_lt").agg(aggfunc).astype(int)
    
    # Exposure
    lmts = X[cr_lmt].values.reshape(-1,1)
    vals = mths * np.where((X[dpd_cols]>=dpd_geq),X[val_cols],0)
    data = np.hstack((indices.reshape(-1,1), lmts, vals))
    data = pd.DataFrame(data, columns=["bin_lt","cr_limit"] + val_cols)
    data["bin_lt"] = data["bin_lt"].apply(lambda x: cat[x])
    aggfunc = dict([("cr_limit","sum")]+[(c,"sum") for c in val_cols])
    vals = data.groupby("bin_lt").agg(aggfunc) * factor

    return delq, vals, bins

def plot_scores(data, ylabel, percent=False, ax=None, num_format=None, colors=None, 
                plot_kwds=None, tight_layout=True, loc=None):
    
    cols = data.columns[data.columns.str.contains("M")]
    base = data.columns[data.columns.str.contains("|".join(("N","cr_")))][0]
    if percent==False: base=None
    n_lines, n_pts = len(cols), len(data)
    args = (plot_kwds, num_format, ax, colors, n_lines, n_pts, loc)
    kwds, num_format, ax, colors = get_params(*args)
    ax = plot_lines(ax, data, cols, num_format, kwds, colors, base)
    ax = draw_legend(ax, loc)
    ax = set_xaxis(ax, data)
    ax = set_yaxis(ax, data, percent, ylabel)
    ax = set_params(ax)
    if tight_layout: plt.tight_layout()
    return ax

def get_params(plot_kwds, num_format, ax, colors, n_lines, n_pts, loc):
    '''Private function: default parameters'''
    kwds = dict(linewidth=2.5, solid_capstyle='round', 
                solid_joinstyle="round", markersize=7,
                markeredgewidth=1, marker="s")
    if plot_kwds is not None: kwds.update(plot_kwds)
    if num_format is None: num_format = "{:,.0f}".format  
    if ax is None: 
        width = max(5, n_pts*0.45) + (2 if loc is None else 0)
        ax = plt.subplots(figsize=(width, 4))[1]
    if colors is not None: colors = color_map(colors, n_lines)
    return kwds, num_format, ax, colors

def plot_lines(ax, data, cols, num_format, kwds, colors, base=None):
    '''Private function: plot lines'''
    x = np.arange(1,len(data)+1) - 0.5
    if base is None:
        base, total = np.ones(len(data)), 1
    else: base, total = data[base].values, sum(data[base])
    for n,col in enumerate(cols):
        y, N = data[col], sum(data[col])/total
        label = "{} ({})".format(col, num_format(N))
        if colors is not None: kwds.update({"color":colors[n]})
        ax.plot(x, y/base, label=label, **kwds)
    return ax

def set_xaxis(ax, data, label="FICO Scores"):
    '''Private function: set x-axis properties'''
    xticklabels = np.r_[0, np.array(data.index)]
    ax.set_xticks(np.arange(0, len(data)+1))
    ax.set_xticklabels(xticklabels, fontsize=12)
    ax.set_xlabel(label, fontsize=13, fontweight=600)
    ax.set_xlim(-0.2, len(xticklabels)-0.7)
    return ax

def set_yaxis(ax, data, percent=True, label="Proportion"):
    '''Private function: set y-axis properties'''
    y_min, y_max = ax.get_ylim() 
    ax.set_ylim(y_min, y_max/0.95)
    ax.set_ylabel(label, fontsize=13, fontweight=600)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.tick_params(axis='y', labelsize=12)
    if percent:
        t = mpl.ticker.PercentFormatter(xmax=1, decimals=1)
        ax.yaxis.set_major_formatter(t)
    return ax

def set_params(ax):
    for s in ["right","top"]:ax.spines[s].set_visible(False) 
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    return ax

def create_mob(X, dt_fmt="%d-%m-%y %H:%M", digit=2):
    
    start_time = time.time()    
    # Required fields
    fields = ["ip_id", "pd_lvl2", "cust_type", "apl_grp_no", 
              "apl_grp_type","fico_scor", "lnd_pos_dt", 
              "fnl_apl_dcsn_dt", "dlq_dys", "otsnd_bal_amt", 
              "pnp_amt", "fnl_cr_lmt"]
    
    # Groupby fields
    groupby= ["ip_id", "pd_lvl2", "cust_type", "apl_grp_no", 
              "apl_grp_type", "fnl_apl_dcsn_dt"]
    
    X = X[fields].copy()

    # Convert fields to `np.datetime64`
    dt0, dt1, dt2 = "fnl_apl_dcsn_dt", "lnd_pos_dt", "month"
    X[dt0] = pd.to_datetime(X[dt0], format=dt_fmt).dt.date
    X[dt0] = X[dt0].apply(lambda x: end_of_mth(x))
    X[dt1] = pd.to_datetime(X[dt1], format=dt_fmt).dt.date
    
    # Months on book
    X[dt2] =  np.round(((X[dt1] - X[dt0])/np.timedelta64(1,'M')),0)
    X = X.sort_values(by=["ip_id","apl_grp_no","lnd_pos_dt"])\
    .reset_index(drop=True)
    
    # Create fields for each month
    start, end = np.percentile(X[dt2].values, q=[0,100])
    X["dlq_day"] = [a for a in zip(X[dt2], X["dlq_dys"])]
    X["os_bals"] = [a for a in zip(X[dt2], X["otsnd_bal_amt"])]
    X["pnpamts"] = [a for a in zip(X[dt2], X["pnp_amt"])]
    
    # Aggregate functions
    aggfunc = {"dlq_day": create_cols(start, max(end,2)), 
               "os_bals": create_cols(start, max(end,2)),
               "pnpamts": create_cols(start, max(end,2), [0,1,2])}
    
    # Column fomats for `aggfnc`
    colfmts = {"dlq_day": "M{}".format, 
               "os_bals": "M{}_OS".format, 
               "pnpamts": "M{}_PNP".format}
    
    # Convert results to pd.DataFrame
    m_data, columns = [], []
    d = digit if isinstance(digit, int) else find_digit(end)
    group = X.groupby(["ip_id","apl_grp_no"])\
    .agg(aggfunc).reset_index()
    for key in aggfunc.keys():
        a = pd.DataFrame(group[key].values.tolist())
        columns += [colfmts[key](label_format(int(c), d)) 
                    for c in a.columns]
        m_data += [a]
        
    # Merge with groupby indices
    index  = group.drop(columns=aggfunc.keys())
    m_data = pd.DataFrame(np.hstack(m_data), 
                          columns=columns).astype(float)
    m_data = index.merge(m_data, right_index=True, left_index=True)
    del index
    
    aggfunc = {"fnl_cr_lmt": "max", "fico_scor" : "mean"}
    data = X.groupby(groupby).agg(aggfunc).reset_index()
    data = data.merge(m_data, how='inner', on=["ip_id","apl_grp_no"])
    r_time = time.gmtime(time.time() - start_time)
    r_time = time.strftime("%H:%M:%S", r_time)
    print('Total running time: {}'.format(r_time))
    
    return data

def create_cols(start, end, select=None):
    def create_cols_(x):
        default = dict([(n,np.nan) for n in np.arange(start, end+1)])
        default.update(dict(list(x)))
        if select is None: return default
        else: return dict([(n,default[n]) for n in select])
    return create_cols_

def end_of_mth(x):
    return x.replace(day=monthrange(x.year,x.month)[1])
                     
def label_format(a, d=2):
    if np.sign(a)>=0: return str(a).zfill(d)
    return "m" + str(abs(a)).zfill(d)
                         
def find_digit(a):
    if pow(10,np.log10(a))==a: return int(np.log10(a)+1)
    else:return int(np.ceil(np.log(a)/np.log(10)))