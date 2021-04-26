import math
import numpy as np
import matplotlib
import pandas_datareader as dr
import pandas as pd
import datetime as dt
from datetime import datetime
from scipy.stats import norm
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.objective_functions import negative_cvar

start = '2018-01-01'
#today = '2021-04-21'
today = datetime.today().strftime('%Y-%m-%d')

tickers = pd.read_excel('SigmaInput.xlsx')["Ticker"]
portfolioWeights = pd.read_excel('SigmaInput.xlsx')["Weight"]
#portfolioWeights = [0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616,0.047619047619047616]
price_data = pd.DataFrame()

for t in tickers:
    price_data[t] = dr.data.get_data_yahoo(t, start =start, end =today)['Adj Close']

return_data = (price_data/price_data.shift(1))-1

# function to obtain portfolio- return and volatility
def get_p_data(weights):
        weights = np.array(weights)
        p_return = np.sum(return_data.mean()*weights) * 252
        p_volatility = np.sqrt(np.dot(weights.T, np.dot(return_data.cov() * 252, weights)))
        return np.array([p_return, p_volatility])

theMean = get_p_data(portfolioWeights)[0]
theSD = get_p_data(portfolioWeights)[1]

#Function to compute VaR, given level, Mean, SD - assume Normality
def VaR(level, mean, standardDeviation):
    return norm.ppf(level, loc = mean, scale = standardDeviation)

#Function to compute cVaR, given level, Mean, SD - assume Normality
def cVaR(level, mean, standardDeviation):
    tail_loss = norm.expect(lambda x: x, loc = mean, scale = standardDeviation, lb = VaR(level,mean,standardDeviation))
    return (1 / (1 - level)) * tail_loss

VaR_95 = VaR(0.95,theMean,theSD)
cVaR_95 = cVaR(0.95,theMean,theSD)

#Function to compute the covariance matrix
def create_CovarianceMatrix (stock_data):
    return stock_data.cov()*252

e_cov = create_CovarianceMatrix(return_data)

#Function which finds the portfolio weights such that the cVaR is minimized
def minimize_cVaR(stock_data,cov_Matrix):
    ef = EfficientFrontier(None,cov_Matrix)
    return ef.custom_objective(negative_cvar,stock_data)

#minimize_cVaR outputs dictionary, these lines convert it to an array, this array contains ticker followed by corresponding weight
#we should still write short def to convert this array into array containing only the weights
def optimalWeights(stock_data,cov_Matrix):
    opt_Weights = minimize_cVaR(stock_data,cov_Matrix)
    data = list(opt_Weights.items())
    theOptimalWeights = np.array(data)
    return theOptimalWeights

print(optimalWeights(return_data,e_cov)
print(cVaR_95)
