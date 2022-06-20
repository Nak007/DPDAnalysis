'''
Available methods are the followings:
[1] create_mob

Author: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 30-06-2022

'''
import time
from calendar import monthrange
from scipy.interpolate import interp1d
import pandas as pd, numpy as np

__all__ = ["create_mob"]

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