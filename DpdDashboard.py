'''
Available methods are the followings:
[1] dashboard_avgdpd
[2] dashboard_vintage
[3] dashboard_rollrate
[4] dashboard_dpdbins
[5] dashboard_pivottable

Author: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 30-06-2022

'''
from ipywidgets import (interact, fixed, IntSlider, Dropdown, 
                        FloatSlider, SelectionRangeSlider, Checkbox,
                        widgets, SelectMultiple, Label, HTMLMath,
                        interactive_output, HBox, VBox, Accordion, 
                        SelectionSlider)
from IPython.display import display
from calendar import month_abbr as abbr
import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import namedtuple
from itertools import product
from DpdAnalysis import *

__all__ = ["dashboard_avgdpd",
           "dashboard_vintage",
           "dashboard_rollrate",
           "dashboard_dpdbins",
           "dashboard_pivottable"]

def initial_parmas(X):
    columns  = np.r_[list(X)]
    osb_cols = columns[X.columns.str.contains(pat="_OS", case=True)].tolist()
    dpd_cols = columns[X.columns.str.contains(pat="M.[0-9]$")].tolist()
    cohorts  = dict([("{}-{}".format(abbr[int(d[-2:])], d[2:4]), d) 
                    for d in X["cohort"].unique()])
    return osb_cols, dpd_cols, cohorts

def widget_start_end(cohorts):
    kwds = dict(options=cohorts.keys(), 
                index=(0, len(cohorts)-1))
    return VBox([Label("Starting and ending cohorts"), 
                 SelectionRangeSlider(**kwds)])

def widget_products(x):
    products = x.unique().tolist() + ["All"]
    return VBox([Label("Type of products"), 
                 Dropdown(options=products, 
                          value=products[-1])])
def widget_customers(x):
    customers = x.unique().tolist() + ["All"]
    return VBox([Label("Type of customers"), 
                 Dropdown(options=customers, 
                          value=customers[-1])])

def widget_channels(x):
    channels = x.unique().tolist() + ["All"]
    return VBox([Label("Type of channels"), 
                 Dropdown(options=channels, 
                          value=channels[-1])])

def widget_cleanstatus():
    options = ["Yes","No (clean)", "All"]
    return VBox([Label("Ever delinquent"), 
                 Dropdown(options=options , 
                          value=options[-1])])

def widget_ficoscore(start=0, stop=1050, step=50):
    options = np.arange(start, stop, step)
    kwds = dict(options=options, index=(0, len(options)-1))
    return VBox([Label("FICO Scores"), 
                 SelectionRangeSlider(**kwds)])

def widget_util(start=0, stop=1.05, step=0.05):
    options = np.round(np.arange(start, stop, step), 2)
    kwds = dict(options=options, index=(0, len(options)-1))
    return VBox([Label("Max Utilization (M0, M1)"), 
                 SelectionRangeSlider(**kwds)])
    
def widget_mob(dpd_cols):
    rng = np.arange(1, len(dpd_cols))
    kwds = dict(value=rng[-1], min=1, max=rng[-1], step=1)
    return VBox([Label("Number of Months-On-Book (MOB)"), 
                 IntSlider(**kwds)])

def widget_focus(dpd_cols):
    kwds = dict(value=-1, min=-1, max=len(dpd_cols)-1, step=1)
    return VBox([Label("Focus on which MOB"), 
                 IntSlider(**kwds)])

def widget_top(value=5, min=1, max=100, step=1):
    kwds = dict(value=value, min=min, max=max, step=step)
    return VBox([Label(r"Highlight $n^{th}$ top cohort"), 
                 IntSlider(**kwds)])

def widget_groupby(options=None, text="criteria", start=0):
    if options is None:
        options = ['Cohort','Product', 'Customer', 
                   'Channel', 'Ever-delinquent', 
                   'FICO scores', 'Utilization']
    return VBox([Label("Group-by {} ({:,.0f})"
                       .format(text, len(options))), 
                 SelectMultiple(options=options, 
                                value=[options[start]], 
                                rows=len(options))])

def widget_legend(value=2, min=1, max=10, step=1):
    kwds = dict(value=value, min=min, max=max, step=step)
    return VBox([Label("Number of legend columns"), 
                 IntSlider(**kwds)])
    
def widget_width(value=10.5, min=1, max=20, step=0.1):
    kwds = dict(value=value, min=min, max=max, step=step)
    return VBox([Label("Plot width (inches) = {:,.2f}"
                       .format(value)), FloatSlider(**kwds)])

def widget_height(value= 6.0, min=1, max=20, step=0.1):
    kwds = dict(value=value, min=min, max=max, step=step)
    return VBox([Label("Plot height (inches) = {:,.2f}"
                       .format(value)), FloatSlider(**kwds)])

def widget_show(options):
    return VBox([Label("Display options ({:,.0f})"
                       .format(len(options))), 
                 SelectMultiple(options=options, 
                                value=[options[0]], 
                                rows=len(options))])

def widget_delq(value=0, min=0, max=150, step=1):
    kwds = dict(value=value, min=min, max=max, step=step)
    return VBox([Label(r"Ever-delinquent ($\geq$ x days)"), 
                 IntSlider(**kwds)])

def widget_cohorts(cohorts):
    return VBox([Label("Which cohort"), 
                 SelectionSlider(options=cohorts,
                                 value=cohorts[0],
                                 readout=True)])

def widget_current(n_cols):
    kwds = dict(value=1, min=1, max=len(n_cols)-1, step=1)
    return VBox([Label("Focus on which MOB"), 
                 IntSlider(**kwds)])

def widget_forward(n_cols):
    kwds = dict(value=1, min=1, max=len(n_cols)-1, step=1)
    return VBox([Label("Number of current months (Max)"), 
                 IntSlider(**kwds)])

def widget_backward(n_cols):
    kwds = dict(value=1, min=1, max=len(n_cols)-1, step=1)
    return VBox([Label("Number of previous months (Max)"), 
                 IntSlider(**kwds)])

def widget_orientation():
    options = ["Horizontal", "Vertical"]
    return VBox([Label("Orientation"), 
                 Dropdown(options=options , 
                          value=options[0])])

def widget_yscale(value=1, min=0.01, max=1, step=0.01):
    kwds = dict(value=value, min=min, max=max, step=step)
    return VBox([Label("Scale of y axis = {:,.2f}"
                       .format(value)), FloatSlider(**kwds)])

def widget_focus1(dpd_cols):
    kwds = dict(value=1, min=1, max=len(dpd_cols)-1, step=1)
    return VBox([Label("Focus on which MOB"), 
                 IntSlider(**kwds)])

def widget_states():
    states = ["x=0", "0<x<=30", "30<x<=60", 
              "60<x<=90", "x>90"]
    return VBox([Label("Type of states"), 
                 Dropdown(options=states, 
                          value=states[0])])

def get_UI01(X):
    
    # ======================= #
    # Create criteria widgets #
    # ======================= #
    osb_cols, dpd_cols, cohorts = initial_parmas(X)
    sta_end   = widget_start_end(cohorts) # Starting and ending cohorts
    products  = widget_products(X["pd_lvl2"]) # Products
    customers = widget_customers(X["cust_type"]) # Customers
    channels  = widget_channels(X["apl_grp_type"]) # Channels
    ever_dlq  = widget_cleanstatus() # Ever-delinquent
    fico_scor = widget_ficoscore(0, 1050, 50) # FICO scores
    util      = widget_util(0, 1.05, 0.05) # Utilization
    months    = widget_mob(dpd_cols) # Months on book
    focus     = widget_focus(dpd_cols) # Focus on which MOB
    n_tops    = widget_top(5, 1, 100, 1) # Highlight nth cohort 
    groupby   = widget_groupby(None) # Group-by criteria
    legend    = widget_legend(2, 1, 10, 1) # Number of legend columns
    width     = widget_width(10.5, 1, 20, 0.1) # figure ==> width
    height    = widget_height(6.0, 1, 20, 0.1) # figure ==> height
    show_opts = widget_show(['Average delinquency plot',
                             'Average delinquency table',
                             'O/S (millions)'])
 
    # Dictionary of inputs ==> dashboard
    values = {"sta_end"      : sta_end.children[1], 
              "pd_lvl2"      : products.children[1], 
              "cust_type"    : customers.children[1], 
              "apl_grp_type" : channels.children[1], 
              "clean"        : ever_dlq.children[1], 
              "fico_scor"    : fico_scor.children[1], 
              "util"         : util.children[1], 
              "groupby"      : groupby.children[1], 
              "show_options" : show_opts.children[1],
              "months"       : months.children[1],
              "focus"        : focus.children[1], 
              "n_tops"       : n_tops.children[1], 
              "ncol"         : legend.children[1], 
              "width"        : width.children[-1],
              "height"       : height.children[-1],
              "cohorts"      : fixed(cohorts), 
              "X"            : fixed(X.copy()), 
              "dpd_cols"     : fixed(dpd_cols), 
              "osb_cols"     : fixed(osb_cols)}
    
    # Layout
    children = [HBox([VBox([sta_end, fico_scor, util]), 
                     VBox([products, customers, channels]), 
                     VBox([ever_dlq])]),
                HBox([groupby, show_opts]), 
                HBox([VBox([months, focus, n_tops]),
                      VBox([legend, width, height])])]
    
    tab_nest = widgets.Tab()
    tab_nest.children = children
    tab_nest.selected_index = 0
    tab_nest.set_title(0, 'Data Filtering')
    tab_nest.set_title(1, 'Group-by & Display')
    tab_nest.set_title(2, 'Plot options')
    return tab_nest, values

def db_avgdpd_base(sta_end, pd_lvl2, cust_type, apl_grp_type, clean, 
                   fico_scor, util, groupby, show_options, months, 
                   focus, dpd_cols, osb_cols, n_tops, ncol, cohorts, 
                   X, width, height):
    
    # Determine starting and ending months
    sta_cohort, end_cohort = sta_end
    keys = np.array(list(cohorts.keys()))
    sta_mth = np.argmax(keys==sta_cohort)
    end_mth = np.argmax(keys==end_cohort) + 1
    values  = np.array(list(cohorts.values())) 
    cond = X["cohort"].isin(values[sta_mth:end_mth])
    figsize = (width, height)

    # Column mapping dictionary
    columns = dict([('Cohort', "cohort"),
                    ('Product', "pd_lvl2"), 
                    ('Customer', "cust_type"),  
                    ('Channel', "apl_grp_type"), 
                    ('Ever-delinquent', "clean"), 
                    ('FICO scores', "fico_bins"), 
                    ('Utilization', 'util_bins')])
    map_clean = {"Yes":False, "No (clean)":True}

    # FICO scores.
    min_scr, max_scr = fico_scor
    scores = X["fico_scor"].fillna(0)
    cond &= ((scores>=min_scr) & 
             (scores<=max_scr))

    # Utilization.
    min_util, max_util = util
    utilization = X["util"].fillna(0)
    cond &= ((utilization>=min_util) & 
             (utilization<=max_util))

    # Determine conditions.
    groupby = [columns[col] for col in groupby]
    if pd_lvl2!="All": cond &= (X["pd_lvl2"]==pd_lvl2)
    if cust_type!="All": cond &= (X["cust_type"]==cust_type)
    if apl_grp_type!="All": cond &= (X["apl_grp_type"]==apl_grp_type)
    if clean!="All": cond &= (X["clean"]==map_clean[clean])
        
    focus = None if focus==-1 else dpd_cols[focus]
    
    options = np.array(show_options)
    if 'Average delinquency plot' in options:
        ax = plot_avgdpd(X.loc[cond], groupby, dpd_cols[:months+1], 
                         start_mth=0, colors=None, focus=focus, 
                         n_tops=n_tops, ax=plt.subplots(figsize=figsize)[1], 
                         tight_layout=False)
        ax = relocate_legend(ax, ncol)
    
    if 'O/S (millions)' in options:
        df = groupby_table(X.loc[cond], groupby, osb_cols[:months+1], 
                           np.nansum, factor=1/10**6)
        display(df.round(2))

    if 'Average delinquency table' in options:
        df = groupby_table(X.loc[cond], groupby, dpd_cols[:months+1])
        display(df.round(2))

def relocate_legend(ax, ncol=2):
    args = ax.get_legend_handles_labels()
    legend = ax.legend(*args, ncol=ncol, loc="upper left",
                       fontsize=12, framealpha=0)
    legend.set_bbox_to_anchor([1,1], transform=ax.transAxes)
    return ax

def dashboard_avgdpd(X):
    ui, values = get_UI01(X)
    display(ui, interactive_output(db_avgdpd_base, values))

def get_UI02(X):
    
    # ======================= #
    # Create criteria widgets #
    # ======================= #
    osb_cols, dpd_cols, cohorts = initial_parmas(X)
    sta_end   = widget_start_end(cohorts) # Starting and ending cohorts
    products  = widget_products(X["pd_lvl2"]) # Products
    customers = widget_customers(X["cust_type"]) # Customers
    channels  = widget_channels(X["apl_grp_type"]) # Channels
    ever_dlq  = widget_cleanstatus() # Ever-delinquent
    fico_scor = widget_ficoscore(0, 1050, 50) # FICO scores
    util      = widget_util(0, 1.05, 0.05) # Utilization
    months    = widget_mob(dpd_cols) # Months on book
    focus     = widget_focus(dpd_cols) # Focus on which MOB
    n_tops    = widget_top(5, 1, 100, 1) # Highlight nth cohort 
    groupby   = widget_groupby(None) # Group-by criteria
    legend    = widget_legend(2, 1, 10, 1) # Number of legend columns
    width     = widget_width(10.5, 1, 20, 0.1) # figure ==> width
    height    = widget_height(6.0, 1, 20, 0.1) # figure ==> height
    yscale    = widget_yscale()
    show_opts = widget_show(['Vintage plot', 
                             'Number of samples (w/ dpd)', 
                             'Percentage of samples (w/ dpd)',
                             'Number of samples (w/o dpd)',
                             'Amount of O/S (w/ dpd)',
                             'Percentage of O/S (w/ dpd)',
                             'Amount of O/S (w/o dpd)'])
    delq = widget_delq(0, 0, 150, 1) # Delinquency (days)

    # Dictionary of inputs ==> dashboard
    values = {"sta_end"      : sta_end.children[1], 
              "pd_lvl2"      : products.children[1], 
              "cust_type"    : customers.children[1], 
              "apl_grp_type" : channels.children[1], 
              "clean"        : ever_dlq.children[1], 
              "fico_scor"    : fico_scor.children[1], 
              "util"         : util.children[1], 
              "groupby"      : groupby.children[1], 
              "show_options" : show_opts.children[1],
              "months"       : months.children[1],
              "focus"        : focus.children[1], 
              "n_tops"       : n_tops.children[1], 
              "ncol"         : legend.children[1], 
              "width"        : width.children[-1],
              "height"       : height.children[-1],
              "yscale"       : yscale.children[-1],
              "delq"         : delq.children[-1],
              "cohorts"      : fixed(cohorts), 
              "X"            : fixed(X.copy()), 
              "dpd_cols"     : fixed(dpd_cols), 
              "osb_cols"     : fixed(osb_cols)}
    
    # Layout
    children = [HBox([VBox([sta_end, fico_scor, util]), 
                     VBox([products, customers, channels]), 
                     VBox([ever_dlq, delq])]),
                HBox([groupby, show_opts]), 
                HBox([VBox([months, focus, n_tops]),
                      VBox([legend, width, height]),
                      VBox([yscale])])]
    
    tab_nest = widgets.Tab()
    tab_nest.children = children
    tab_nest.selected_index = 0
    tab_nest.set_title(0, 'Data Filtering')
    tab_nest.set_title(1, 'Group-by & Display')
    tab_nest.set_title(2, 'Plot options')
    return tab_nest, values

def db_vintage_base(**params):
    
    # Initialize parameters
    params = namedtuple("params",params.keys())(**params)
    width, height = params.width, params.height
    options  = np.array(params.show_options)
    dpd_cols = np.array(params.dpd_cols)
    osb_cols = np.array(params.osb_cols)
    focus, X = params.focus, params.X
    sta_cohort, end_cohort = params.sta_end
    cohorts = params.cohorts
    months, delq = params.months, params.delq
    n_tops, ncol = params.n_tops, params.ncol
    scale = params.yscale
    
    # Determine starting and ending months
    keys = np.array(list(cohorts.keys()))
    sta_mth = np.argmax(keys==sta_cohort)
    end_mth = np.argmax(keys==end_cohort) + 1
    values  = np.array(list(cohorts.values())) 
    cond = X["cohort"].isin(values[sta_mth:end_mth])
    figsize = (width, height)

    # Column mapping dictionary
    columns = dict([('Cohort', "cohort"),
                    ('Product', "pd_lvl2"), 
                    ('Customer', "cust_type"),  
                    ('Channel', "apl_grp_type"), 
                    ('Ever-delinquent', "clean"), 
                    ('FICO scores', "fico_bins"), 
                    ('Utilization', 'util_bins')])
    groupby = [columns[col] for col in params.groupby]

    # FICO scores.
    min_scr, max_scr = params.fico_scor
    scr = X["fico_scor"].fillna(0)
    cond &= ((scr>=min_scr) & (scr<=max_scr))

    # Utilization.
    min_util, max_util = params.util
    util = X["util"].fillna(0)
    cond &= ((util>=min_util) & (util<=max_util))
    
    # Other conditions.
    keys = ["pd_lvl2", "cust_type", "apl_grp_type", "clean"]
    map_clean = {"Yes":False, "No (clean)":True}
    for key in keys: 
        value = getattr(params, key)
        if value!="All":
            if key!="clean":cond &= (X[key]==value)
            else: cond &= (X[key]==map_clean[value])  
    focus = None if focus==-1 else dpd_cols[focus]
    
    if (len(options)>0) & (len(groupby)>0):

        if 'Vintage plot' in options:
            args = (X.loc[cond], groupby, dpd_cols[:months+1], delq)
            kwds = dict(colors=None, focus=focus, n_tops=n_tops, 
                        ax=plt.subplots(figsize=figsize)[1], 
                        tight_layout=False)
            ax = plot_vintage(*args, **kwds)
            ax = relocate_legend(ax, ncol)

            if scale<1:
                y_min, y_max = ax.get_ylim()
                y_max = (y_max-y_min)*scale + y_min
                ax.set_ylim(y_min, y_max)

        # Positional arguments
        cols = dpd_cols[:months+1]
        args = (X.loc[cond], groupby, cols, delq)

        if 'Number of samples (w/ dpd)' in options:

            df = vintage_table(*args, func=np.nansum)
            label = f"Vintage: Number of samples (dpd >= {delq})"
            columns = [("","N0")] + list(product([label], cols))
            df.columns = pd.MultiIndex.from_tuples(columns)
            display(df.astype(int))

        if 'Percentage of samples (w/ dpd)' in options:

            df = vintage_table(*args)
            df.loc[:,cols] = df[cols]*100
            label = f"Vintage: Percentage of samples (dpd >= {delq})"
            columns = [("","N0")] + list(product([label], cols))
            df.columns = pd.MultiIndex.from_tuples(columns)
            display(df.round(2))

        if 'Number of samples (w/o dpd)' in options:

            df = vintage_table(*tuple(list(args)[:-1] + 
                                      [0, np.nansum]))[cols]
            label = 'Number of samples (dpd = 0)'
            columns =  list(product([label], cols))
            df.columns = pd.MultiIndex.from_tuples(columns)
            display(df.astype(int))

        if 'Amount of O/S (w/ dpd)' in options:

            add = [np.nansum, osb_cols[:months+1], 1/10**6]
            df = vintage_table(*tuple(list(args) + add))
            label = f'Vintage: Amount of O/S MB (dpd >= {delq})'
            columns = [("","N0")] + list(product([label], cols))
            df.columns = pd.MultiIndex.from_tuples(columns)
            display(df.round(1))

        if 'Percentage of O/S (w/ dpd)' in options:

            add = [np.nansum, osb_cols[:months+1], 1/10**6]
            df0 = vintage_table(*tuple(list(args) + add))

            add = [0, np.nansum, osb_cols[:months+1], 1/10**6]
            df1 = vintage_table(*tuple(list(args)[:-1] + add))

            df0[cols] = (df0[cols] / df1[cols]) * 100
            label = f'Vintage: Percentage of O/S (dpd >= {delq})'
            columns = [("","N0")] + list(product([label], cols))
            df0.columns = pd.MultiIndex.from_tuples(columns)
            display(df0.round(2))

        if 'Amount of O/S (w/o dpd)' in options:

            add = [0, np.nansum, osb_cols[:months+1], 1/10**6]
            df = vintage_table(*tuple(list(args)[:-1] + add))
            label = 'Amount of O/S MB (dpd = 0)'
            columns = [("","N0")] + list(product([label], cols))
            df.columns = pd.MultiIndex.from_tuples(columns)
            display(df.round(1))
            
    else: print("No <Group-by criteria> or <Display options> selected")

def dashboard_vintage(X):
    ui, values = get_UI02(X)
    display(ui, interactive_output(db_vintage_base, values))

def get_UI03(X):
    
    # ======================= #
    # Create criteria widgets #
    # ======================= #
    osb_cols, dpd_cols, cohorts = initial_parmas(X)
    which = widget_cohorts(list(cohorts.keys()))
    forward   = widget_forward(dpd_cols)
    backward  = widget_backward(dpd_cols)
    products  = widget_products(X["pd_lvl2"]) # Products
    customers = widget_customers(X["cust_type"]) # Customers
    channels  = widget_channels(X["apl_grp_type"]) # Channels
    fico_scor = widget_ficoscore(0, 1050, 50) # FICO scores
    util      = widget_util(0, 1.05, 0.05) # Utilization
    focus     = widget_current(dpd_cols) # Focus on which MOB
    width     = widget_width(8.3, 1, 20, 0.1) # figure ==> width
    height    = widget_height(6.2, 1, 20, 0.1) # figure ==> height
    orient    = widget_orientation()
    show_opts = widget_show(['Roll-rate plot (Count)',
                             'Roll-rate plot (O/S, millions)'])
 
    # Dictionary of inputs ==> dashboard
    values = {"which"        : which.children[1], 
              "pd_lvl2"      : products.children[1], 
              "cust_type"    : customers.children[1], 
              "apl_grp_type" : channels.children[1], 
              "fico_scor"    : fico_scor.children[1], 
              "util"         : util.children[1], 
              "show_options" : show_opts.children[1],
              "focus"        : focus.children[1],
              "forward"      : forward.children[1],
              "backward"     : backward.children[1],
              "width"        : width.children[-1],
              "height"       : height.children[-1],
              "orient"       : orient.children[-1],
              "cohorts"      : fixed(cohorts), 
              "X"            : fixed(X), 
              "dpd_cols"     : fixed(dpd_cols), 
              "osb_cols"     : fixed(osb_cols)}
    
    # Layout
    children = [HBox([VBox([which, fico_scor, util]), 
                      VBox([products, customers, channels]),
                      VBox([focus, forward, backward])]),
                HBox([show_opts]), 
                HBox([VBox([width, height, orient])])]
    
    tab_nest = widgets.Tab()
    tab_nest.children = children
    tab_nest.selected_index = 0
    tab_nest.set_title(0, 'Data Filtering')
    tab_nest.set_title(1, 'Display')
    tab_nest.set_title(2, 'Plot options')
    return tab_nest, values

def db_rollrate_base(**params):
        
    # Initialize parameters
    params = namedtuple("params",params.keys())(**params)
    width, height = params.width, params.height
    options = np.array(params.show_options)
    dpd_cols = np.array(params.dpd_cols)
    osb_cols = np.array(params.osb_cols)
    focus, X = params.focus, params.X
    
    # Determine length of current and previous months
    n_curr = dpd_cols[focus: focus + params.forward]
    n_prev = dpd_cols[max(focus-params.backward,0):focus]
    X["n_curr"] = X[n_curr].fillna(0).max(axis=1)
    X["n_prev"] = X[n_prev].fillna(0).max(axis=1)

    # labels
    xlabel = "Current month ({})".\
    format("-".join(np.unique(n_curr[[0,-1]])))
    ylabel = "Previous month ({})".\
    format("-".join(np.unique(n_prev[[0,-1]])))
    
    # Cohort
    cohorts = params.cohorts
    keys = np.array(list(cohorts.keys()))
    index  = np.argmax(keys==params.which)
    values = np.array(list(cohorts.values())) 
    cond = X["cohort"].isin([values[index]])

    # FICO scores.
    min_scr, max_scr = params.fico_scor
    scr = X["fico_scor"].fillna(0)
    cond &= ((scr>=min_scr) & (scr<=max_scr))

    # Utilization.
    min_util, max_util = params.util
    util = X["util"].fillna(0)
    cond &= ((util>=min_util) & (util<=max_util))

    # Other conditions.
    keys = ["pd_lvl2", "cust_type", "apl_grp_type"]
    for key in keys: 
        if getattr(params,key)!="All":
            cond &= (X[key]==getattr(params,key))
            
    # Set figure, and gridspec
    if len(options)==2:
        if params.orient=="Horizontal": 
            args = (1,2); width *= 2
        else: args = (2,1); height *= 2
        fig  = plt.figure(figsize=(width,height))
        grid = gridspec.GridSpec(*args)
        axes = [fig.add_subplot(grid[i]) for i in range(2)]
    elif len(options)==1:
        fig  = plt.figure(figsize=(width,height))
        grid = gridspec.GridSpec(1,1)
        axes = [fig.add_subplot(grid[0])]
    else: pass 
   
    if len(options)>0:
        for opt,ax in zip(options, axes):    
            if opt=='Roll-rate plot (Count)':
                ax0 = plot_rollrate(X.loc[cond].copy(), ["n_prev","n_curr"], 
                                   labels=[xlabel, ylabel], ax=ax)
                title = ["Number of applications ({:,.0f})".format(sum(cond)), 
                         "cohort : {}".format(params.which)]
                ax0.set_title("\n".join(title), fontsize=20)
            else:
                a_curr = osb_cols[focus]
                values = X.loc[cond, a_curr].fillna(0)/10**6
                ax1 = plot_rollrate(X.loc[cond].copy(), ["n_prev", "n_curr"], 
                                    labels=[xlabel, ylabel], ax=ax, 
                                    values=values,num_format="{:,.1f}".format)
                title = ["O/S as of {} = {:,.1f}MB".format(n_curr[0], sum(values)), 
                         "cohort : {}".format(params.which)]
                ax1.set_title("\n".join(title), fontsize=20)
        grid.tight_layout(fig, h_pad=0.5, pad=0.1)
    else: print("No <Display options> selected")

def dashboard_rollrate(X):
    ui, values = get_UI03(X)
    display(ui, interactive_output(db_rollrate_base, values))

def get_UI04(X):
    
    # ======================= #
    # Create criteria widgets #
    # ======================= #
    osb_cols, dpd_cols, cohorts = initial_parmas(X)
    sta_end   = widget_start_end(cohorts) # Starting and ending cohorts
    products  = widget_products(X["pd_lvl2"]) # Products
    customers = widget_customers(X["cust_type"]) # Customers
    channels  = widget_channels(X["apl_grp_type"]) # Channels
    fico_scor = widget_ficoscore(0, 1050, 50) # FICO scores
    util      = widget_util(0, 1.05, 0.05) # Utilization
    focus     = widget_focus1(dpd_cols) # Focus on which MOB
    groupby   = widget_groupby(None) # Group-by criteria
    show_opts = widget_show(['Number of samples',
                             'Amount of O/S (MB)'])
 
    # Dictionary of inputs ==> dashboard
    values = {"sta_end"      : sta_end.children[1], 
              "pd_lvl2"      : products.children[1], 
              "cust_type"    : customers.children[1], 
              "apl_grp_type" : channels.children[1], 
              "fico_scor"    : fico_scor.children[1], 
              "util"         : util.children[1], 
              "groupby"      : groupby.children[1], 
              "show_options" : show_opts.children[1],
              "focus"        : focus.children[1], 
              "cohorts"      : fixed(cohorts), 
              "X"            : fixed(X.copy()), 
              "dpd_cols"     : fixed(dpd_cols)}
    
    # Layout
    children = [HBox([VBox([sta_end, fico_scor, util]), 
                     VBox([products, customers, channels]), 
                     VBox([focus])]),
                HBox([groupby, show_opts])]
    
    tab_nest = widgets.Tab()
    tab_nest.children = children
    tab_nest.selected_index = 0
    tab_nest.set_title(0, 'Data Filtering')
    tab_nest.set_title(1, 'Group-by & Display')
    return tab_nest, values

def db_dpdbins_base(**params):
        
    # Initialize parameters
    params = namedtuple("params",params.keys())(**params)
    options  = np.array(params.show_options)
    dpd_cols = np.array(params.dpd_cols)
    focus, X = params.focus, params.X
    sta_cohort, end_cohort = params.sta_end
    cohorts = params.cohorts
    
    # Determine starting and ending months
    keys = np.array(list(cohorts.keys()))
    sta_mth = np.argmax(keys==sta_cohort)
    end_mth = np.argmax(keys==end_cohort) + 1
    values  = np.array(list(cohorts.values())) 
    cond = X["cohort"].isin(values[sta_mth:end_mth])

    # Column mapping dictionary
    columns = dict([('Cohort', "cohort"),
                    ('Product', "pd_lvl2"), 
                    ('Customer', "cust_type"),  
                    ('Channel', "apl_grp_type"), 
                    ('Ever-delinquent', "clean"), 
                    ('FICO scores', "fico_bins"), 
                    ('Utilization', 'util_bins')])
    groupby = [columns[col] for col in params.groupby]

    # FICO scores.
    min_scr, max_scr = params.fico_scor
    scr = X["fico_scor"].fillna(0)
    cond &= ((scr>=min_scr) & (scr<=max_scr))

    # Utilization.
    min_util, max_util = params.util
    util = X["util"].fillna(0)
    cond &= ((util>=min_util) & (util<=max_util))
    
    # Other conditions.
    keys = ["pd_lvl2", "cust_type", "apl_grp_type"]
    for key in keys: 
        value = getattr(params, key)
        if value!="All": cond &= (X[key]==value)  
    focus = None if focus==-1 else dpd_cols[focus]
    
    if focus is not None:
        if (len(options)>0) & (len(groupby)>0):
            if 'Number of samples' in options:
                display(dpd_table(X.loc[cond], groupby, focus))
            if 'Amount of O/S (MB)' in options:
                display(dpd_table(X.loc[cond], groupby, focus, True))
        else: print("No <Group-by criteria> or <Display options> selected")
    else: print("No <MOB> selected.")

def dashboard_dpdbins(X):
    ui, values = get_UI04(X)
    display(ui, interactive_output(db_dpdbins_base, values))

def get_UI05(X):
    
    # ======================= #
    # Create criteria widgets #
    # ======================= #
    osb_cols, dpd_cols, cohorts = initial_parmas(X)
    sta_end   = widget_start_end(cohorts) # Starting and ending cohorts
    products  = widget_products(X["pd_lvl2"]) # Products
    customers = widget_customers(X["cust_type"]) # Customers
    channels  = widget_channels(X["apl_grp_type"]) # Channels
    fico_scor = widget_ficoscore(0, 1050, 50) # FICO scores
    util      = widget_util(0, 1.05, 0.05) # Utilization
    focus     = widget_focus1(dpd_cols) # Focus on which MOB
    groupby1  = widget_groupby(None, "index", 0) # Group-by criteria
    groupby2  = widget_groupby(None, "column", 1) # Group-by criteria
    states    = widget_states()
    show_opts = widget_show(['Number of samples',
                             'Amount of O/S (MB)'])
 
    # Dictionary of inputs ==> dashboard
    values = {"sta_end"      : sta_end.children[1], 
              "pd_lvl2"      : products.children[1], 
              "cust_type"    : customers.children[1], 
              "apl_grp_type" : channels.children[1], 
              "fico_scor"    : fico_scor.children[1], 
              "util"         : util.children[1], 
              "state"        : states.children[1],
              "groupby1"     : groupby1.children[1], 
              "groupby2"     : groupby2.children[1],
              "show_options" : show_opts.children[1],
              "focus"        : focus.children[1], 
              "cohorts"      : fixed(cohorts), 
              "X"            : fixed(X.copy()), 
              "dpd_cols"     : fixed(dpd_cols), 
              "osb_cols"     : fixed(osb_cols)}
    
    # Layout
    children = [HBox([VBox([sta_end, fico_scor, util]), 
                     VBox([products, customers, channels]), 
                     VBox([focus, states])]),
                HBox([groupby1, groupby2, show_opts])]
    
    tab_nest = widgets.Tab()
    tab_nest.children = children
    tab_nest.selected_index = 0
    tab_nest.set_title(0, 'Data Filtering')
    tab_nest.set_title(1, 'Group-by & Display')
    return tab_nest, values

def db_pivottable_base(**params):
        
    # Initialize parameters
    params = namedtuple("params",params.keys())(**params)
    options  = np.array(params.show_options)
    dpd_cols = np.array(params.dpd_cols)
    focus, X = params.focus, params.X
    sta_cohort, end_cohort = params.sta_end
    cohorts = params.cohorts
    
    # Determine starting and ending months
    keys = np.array(list(cohorts.keys()))
    sta_mth = np.argmax(keys==sta_cohort)
    end_mth = np.argmax(keys==end_cohort) + 1
    values  = np.array(list(cohorts.values())) 
    cond = X["cohort"].isin(values[sta_mth:end_mth])

    # Column mapping dictionary
    columns = dict([('Cohort', "cohort"),
                    ('Product', "pd_lvl2"), 
                    ('Customer', "cust_type"),  
                    ('Channel', "apl_grp_type"), 
                    ('Ever-delinquent', "clean"), 
                    ('FICO scores', "fico_bins"), 
                    ('Utilization', 'util_bins')])
    groupby1 = [columns[col] for col in params.groupby1]
    groupby2 = [columns[col] for col in params.groupby2]
    intersect = set(groupby1).intersection(groupby2)
    
    # State
    states = ["x=0", "0<x<=30", "30<x<=60", 
              "60<x<=90", "x>90", "All"]
    state = np.argmax(np.isin(states, params.state))

    # FICO scores.
    min_scr, max_scr = params.fico_scor
    scr = X["fico_scor"].fillna(0)
    cond &= ((scr>=min_scr) & (scr<=max_scr))

    # Utilization.
    min_util, max_util = params.util
    util = X["util"].fillna(0)
    cond &= ((util>=min_util) & (util<=max_util))
    
    # Other conditions.
    keys = ["pd_lvl2", "cust_type", "apl_grp_type"]
    for key in keys: 
        value = getattr(params, key)
        if value!="All": cond &= (X[key]==value)  
            
    if len(intersect)>0:
        print(f"Criterion must exist only either index or " 
              f"column. Found {intersect} in both axes.")
    elif (len(options)>0) & (len(groupby1)>0) & (len(groupby2)>0):
        if 'Number of samples' in options:
            df = pivot_table(X.loc[cond], groupby1, groupby2, 
                             mob=focus, state=state, 
                             return_counts=True, decimal=0)
            display(df)
        if 'Amount of O/S (MB)' in options:
            df = pivot_table(X.loc[cond], groupby1, groupby2, 
                             mob=focus, state=state, 
                             return_counts=False, decimal=2)
            display(df)
    else: print("No <Group-by criteria> or <Display options> selected")

def dashboard_pivottable(X):
    ui, values = get_UI05(X)
    display(ui, interactive_output(db_pivottable_base, values))