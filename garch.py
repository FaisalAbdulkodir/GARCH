import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from arch import arch_model
import plotly.express as px
import plotly.figure_factory as ff
import statsmodels.api as sm
from scipy.stats import t
from scipy.optimize import fmin, minimize

from ipywidgets import HBox, VBox, Dropdown, Output
from scipy.optimize import fmin, minimize
from scipy.stats import norm
from math import inf
from IPython.display import display

from sqlalchemy import create_engine
import matplotlib as mpl
import pyodbc
import pymysql

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
plt.rcParams["figure.dpi"] = 300
# %matplotlib inline

host="127.0.0.1"
port = 3306
database="garch"
user="root"

st.title('GARCH')

#side bar
# st.sidebar.header('Menu')


#load data bersih
# path = 'dataset/test.csv'
# df_garch = pd.read_csv(path, index_col='datetime')
try:
    conn = create_engine('mysql+pymysql://{}:@{}:{}/{}'.format(user,host,port,database))
except (Exception, NameError) as error:
    st.write("Error while connecting to mysql", error)
df_garch = pd.read_sql('SELECT * FROM garch', conn, index_col='datetime')
st.write(df_garch .head())

# st.write("""DATA""")


#mengubah matrix
def vecl(matrix):
    lower_matrix = np.tril(matrix,k=-1) 
    array_with_zero = np.matrix(lower_matrix).A1

    array_without_zero = array_with_zero[array_with_zero!=0]

    return array_without_zero

#modeling garch
def garch_t_to_u(df_garch, res):
    mu = res.params['mu']
    nu = res.params['nu']
    est_r = df_garch - mu
    h = res.conditional_volatility
    std_res = est_r / h
    # we could also just use:
    # std_res = res.std_resid
    # but it's useful to see what is going on
    udata = t.cdf(std_res, nu)
    return udata

#dcc
def loglike_norm_dcc_copula(theta, udata):
    N, T = np.shape(udata)
    llf = np.zeros((T,1))
    trdata = np.array(norm.ppf(udata).T, ndmin=2)
    
    
    Rt, veclRt =  dcceq(theta,trdata)

    for i in range(0,T):
        llf[i] = -0.5* np.log(np.linalg.det(Rt[:,:,i]))
        llf[i] = llf[i] - 0.5 *  np.matmul(np.matmul(trdata[i,:] , (np.linalg.inv(Rt[:,:,i]) - np.eye(N))) ,trdata[i,:].T)
    llf = np.sum(llf)

    return -llf

#dcc
def dcceq(theta,trdata):
    T, N = np.shape(trdata)

    a, b = theta
    
    if min(a,b)<0 or max(a,b)>1 or a+b > .999999:
        a = .9999 - b
        
    Qt = np.zeros((N, N ,T))

    Qt[:,:,0] = np.cov(trdata.T)

    Rt =  np.zeros((N, N ,T))
    veclRt =  np.zeros((T, int(N*(N-1)/2)))
    
    Rt[:,:,0] = np.corrcoef(trdata.T)
    
    for j in range(1,T):
        Qt[:,:,j] = Qt[:,:,0] * (1-a-b)
        Qt[:,:,j] = Qt[:,:,j] + a * np.matmul(trdata[[j-1]].T, trdata[[j-1]])
        Qt[:,:,j] = Qt[:,:,j] + b * Qt[:,:,j-1]
        Rt[:,:,j] = np.divide(Qt[:,:,j] , np.matmul(np.sqrt(np.array(np.diag(Qt[:,:,j]), ndmin=2)).T , np.sqrt(np.array(np.diag(Qt[:,:,j]), ndmin=2))))
    
    for j in range(0,T):
        veclRt[j, :] = vecl(Rt[:,:,j].T)
    return Rt, veclRt

model_parameters = {}
udata_list = []

#run garch ke semua kolom
def run_garch(df_garch, udata_list, model_parameters):
    for x in df_garch:
        am = arch_model(df_garch[x], dist = 't')
        short_name = x.split()[0]
        model_parameters[short_name] = am.fit(disp='off')
        udata = garch_t_to_u(df_garch[x], model_parameters[short_name])
        udata_list.append(udata)
    return udata_list, model_parameters

udata_list, model_parameters = run_garch(df_garch.dropna(), udata_list, model_parameters)

cons = ({'type': 'ineq', 'fun': lambda x:  -x[0]  -x[1] +1})
bnds = ((0, 0.5), (0, 0.9997))

opt_out = minimize(loglike_norm_dcc_copula, [0.01, 0.95], args = (udata_list,), bounds=bnds, constraints=cons)

llf  = loglike_norm_dcc_copula(opt_out.x, udata_list)

trdata = np.array(norm.ppf(udata_list).T, ndmin=2)
Rt, veclRt = dcceq(opt_out.x, trdata)

stock_names = [x.split()[0] for x in df_garch.columns]

corr_name_list = []
for i, name_a in enumerate(stock_names):
    if i == 0:
        pass
    else:
        for name_b in stock_names[:i]:
            corr_name_list.append(name_a + "-" + name_b)

dcc_corr = pd.DataFrame(veclRt, index = df_garch.dropna().index, columns= corr_name_list)
# dcc_corr
dcc_plot = px.line(dcc_corr, title = 'Dynamic Conditional Correlation plot', width=1000, height=500)
# dcc_plot.show()

# st.plotly_chart(dcc_plot)

garch_vol_df = pd.concat([pd.DataFrame(model_parameters[x].conditional_volatility/100)*1600 for x in model_parameters], axis=1)
garch_vol_df.columns = stock_names

garh_plot = px.line(garch_vol_df, title='GARCH Conditional Volatility', width=1000, height=500)
st.plotly_chart(garh_plot)