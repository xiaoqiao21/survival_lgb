#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import lightgbm as lgb
import numpy as np
from sksurv.metrics import concordance_index_censored
import os
import pandas as pd



os.getcwd()


# In[ ]:


def coxloss(preds, train_data):
    y_true = train_data.get_label()
    censor = (np.array(y_true) > 0).astype(int)
    labels = abs(y_true)
    orders = labels.argsort()
    ranks = orders.argsort()
    dd = censor[orders]
    haz = np.exp(preds[orders])
    rsk = np.flip(np.flip(haz, 0).cumsum(), 0)
    pp = np.tril(np.divide.outer(haz, rsk))
    grad = pp.dot(dd) - dd
    grad = grad[ranks]
    hh = np.tril(pp - np.divide.outer(np.square(haz), np.square(rsk)))
    hess = hh.dot(dd)
    hess = hess[ranks]
    return grad, hess


# In[ ]:


def evalc(preds, train_data):
    y_true = train_data.get_label()
    return 'concordance_index', concordance_index_censored(np.array(y_true > 0), abs(y_true), preds)[0], True


# In[ ]:


def coxdev(preds, train_data):
    y_true = train_data.get_label()
    censor = (y_true > 0).astype(int)
    y_true = abs(y_true)
    orders = y_true.argsort()
    dd = censor[orders]
    etas = preds[orders]
    rsk = np.flip(np.flip(np.exp(etas), 0).cumsum(), 0)
    return 'log_partial_likelihood', 2*np.mean(np.multiply(dd, etas - np.log(rsk))), True


# In[ ]:


xtr = pd.read_csv('xtr.csv')
xte = pd.read_csv('xte.csv')
ytr = pd.read_csv('ytr.csv')
yte = pd.read_csv('yte.csv')


# In[ ]:


params = {
    'verbose': 0
}


# In[ ]:


lgb_train = lgb.Dataset(xtr, ytr)
lgb_eval = lgb.Dataset(xte, yte, reference=lgb_train)


# In[ ]:


lgmmodel = lgb.train(params,
                     train_set = lgb_train,
                     valid_sets = lgb_eval,
                     num_boost_round = 100,
                     fobj = coxloss,
                     feval = coxdev)


lgmmodel2 = lgb.train(params,
                     train_set = lgb_train,
                     valid_sets = lgb_eval,
                     num_boost_round = 100,
                     fobj = coxloss,
                     feval = evalc)
