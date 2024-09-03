# Portfolio Optimization for Cryptocurrencies

# Project Members
#- Simone Bonelli
#- Alessandro Ricchiuti
#- Bakak Feyzullayev
#- Alberto Yulian Hu

import yfinance as yf
import pandas as pd
import numpy as np
cryptos = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD', 'MANA-USD',
    'XRP-USD', 'DOGE-USD', 'BCH-USD', 'EOS-USD', 'XLM-USD', 'ATOM-USD', 'VET-USD', 'TRX-USD', 'XTZ-USD', 'THETA-USD',
    'AVAX-USD', 'FTM-USD', 'NEAR-USD', 'MATIC-USD', 'ALGO-USD', 'CRO-USD', 'FIL-USD', 'HBAR-USD', 'QNT-USD', 'ICP-USD',
    'AAVE-USD', 'GRT-USD', 'SUSHI-USD', 'YFI-USD', 'SNX-USD', 'RUNE-USD', '1INCH-USD', 'CRV-USD', 'BAL-USD',
    'KSM-USD', 'ZRX-USD', 'BAT-USD', 'ENJ-USD', 'ANKR-USD', 'CHZ-USD', 'STORJ-USD', 'DGB-USD', 'ZIL-USD', 'SC-USD',
    'ICX-USD', 'ONT-USD', 'QTUM-USD', 'OMG-USD', 'BNT-USD', 'KAVA-USD', 'LRC-USD', 'ZEN-USD', 'SKL-USD', 'OCEAN-USD',
    'REEF-USD', 'SRM-USD', 'AKT-USD', 'MLN-USD', 'STMX-USD', 'XEM-USD', 'FTT-USD', 'GALA-USD', 'HNT-USD', 'MKR-USD',
    'DAI-USD', 'BUSD-USD', 'USDT-USD', 'TUSD-USD', 'USDC-USD', 'UST-USD', 'LUNA-USD', 'AUDIO-USD', 'GNO-USD',
    'UMA-USD', 'POLS-USD', 'ORN-USD', 'DNT-USD', 'RLC-USD', 'SXP-USD', 'STPT-USD', 'WAVES-USD', 'CVC-USD', 'BLZ-USD',
    'MFT-USD', 'FUN-USD', 'RDN-USD', 'DOCK-USD', 'DATA-USD', 'PERP-USD', 'FORTH-USD', 'IDEX-USD', 'CEL-USD', 'QKC-USD'
]
data = yf.download(cryptos, start='2020-06-01', end='2023-01-01', interval='1d')['Adj Close']
returns = data.pct_change().dropna() # Find returns and eliminate Nan values

# Correlation Matrix
import seaborn as sns
corr_df = returns.corr()
sns.heatmap(corr_df)

shy = yf.download('SHY', start='2020-06-01', end='2023-01-01', interval='1d')
shy_close = shy['Close']
shy_returns = shy_close.pct_change().dropna()

def calculate_sharpe_ratios(df, risk_free_rate_df):
    sharpe_ratios = {}
    for column in df.columns:
        asset_returns = df[column]
        risk_free_rate = risk_free_rate_df.reindex(asset_returns.index).ffill().bfill()
        excess_returns = asset_returns - risk_free_rate
        mean_excess_return = excess_returns.mean()
        std_dev = excess_returns.std()

        sharpe_ratio = mean_excess_return / std_dev
        sharpe_ratios[column] = sharpe_ratio

    sharpe_ratios_df = pd.DataFrame(list(sharpe_ratios.items()), columns=['Asset', 'Sharpe Ratio'])
    return sharpe_ratios_df

SP = calculate_sharpe_ratios(returns, shy_returns) 
def select_top_n_sharpe_ratios(sharpe_ratios_df, n=10):
    top_n_sharpe_ratios_df = sharpe_ratios_df.sort_values(by='Sharpe Ratio', ascending=False).head(n)
    return top_n_sharpe_ratios_df

asset= select_top_n_sharpe_ratios(SP, n=10)
asset_names = asset['Asset'].to_numpy()
selected_df = returns[asset_names]

#%% Crypto Distribution
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from scipy.stats import norm


USDC = selected_df.iloc[:, 2].to_numpy()
mu, std = norm.fit(USDC)


plt.hist(USDC, bins=30, density=True, alpha=0.6, color='g', edgecolor='black', label='USDC')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Normal fit')
plt.legend()
plt.title('Histogram and Normal Distribution Fit for USDC')
plt.show()

kde = gaussian_kde(USDC)
x = np.linspace(min(USDC), max(USDC), 1000)
y = kde(x)


plt.hist(USDC, bins=30, density=True, alpha=0.6, color='g', edgecolor='black', label='USDC')
plt.plot(x, y, 'r', linewidth=2, label='KDE')
plt.legend()
plt.title('Histogram and KDE for USDC')

#%%
# MC simulation
def rejection_sampling(kde, min_val, max_val, num_samples):
    samples = []
    x = np.linspace(min_val, max_val, 1000)
    max_density = np.max(kde(x))  
    while len(samples) < num_samples:
        u = np.random.uniform(min_val, max_val)  
        v = np.random.uniform(0, max_density)  
        if v < kde(u):  
            samples.append(u)
    return samples

simulated_df = pd.DataFrame(columns=asset_names)
for col in selected_df.columns:
    prova = selected_df[col]
    kde = gaussian_kde(prova)
   
    # Sample values using Acceptance Rejection Method
    sampled_values = rejection_sampling(kde, min(prova), max(prova), num_samples=10000)
    simulated_df[col] = sampled_values

# Portfolio Optimization
import scipy.optimize as spopt
def port_ret(x, mu, annual):
    'x: weights of the portfolio'
    'mu: returns of the assets'
    return np.dot(x, mu) * annual

def port_variance(x, r, annual):
    'x: weights of the portfolio'
    'r: returns of the assets'
    S = r.cov()
    return np.dot(np.dot(x, S), x) * annual

def port_vola(x, r, annual):
    'x: weights of the portfolio'
    'r: returns of the assets'
    return np.sqrt(port_variance(x, r, annual))

def port_ret_eq(x, mu, annual, mu_0):
    'x: weights of the portfolio'
    'mu: returns of the assets'
    'mu_0: target return'
    return np.dot(x, mu) * annual - mu_0

# Real data
annual = 252
x0 = pd.Series(1/selected_df.shape[1], index=selected_df.columns)
mu = selected_df.mean()
r = selected_df.copy()

expected_return = port_ret(x0, mu, annual)
initial_variance = port_variance(x0, r, annual)
initial_volatility = port_vola(x0, r, annual)

mu_0 = 0.09  # target annual return
cons_MV = ({'type': 'eq', 'fun': lambda x: sum(x) - 1},
           {'type': 'eq', 'fun': port_ret_eq, 'args': (mu, annual, mu_0)})

res = spopt.minimize(port_variance, x0, method='SLSQP', args=(r, annual), 
                     constraints=cons_MV, options={'disp': True})

print("Optimization Result:", res)
print("Optimized Portfolio Weights:", res.x)
print("Optimized Portfolio Return:", port_ret(res.x, mu, annual))
print("Optimized Portfolio Variance:", port_variance(res.x, r, annual))
print("Optimized Portfolio Volatility:", port_vola(res.x, r, annual))

weights=res.x

# Simulated data
x0 = pd.Series(1/simulated_df.shape[1], index=simulated_df.columns)
mu = simulated_df.mean()
r = simulated_df.copy()

expected_return = port_ret(x0, mu, annual)
initial_variance = port_variance(x0, r, annual)
initial_volatility = port_vola(x0, r, annual)

mu_0 = 0.09  
cons_MV = ({'type': 'eq', 'fun': lambda x: sum(x) - 1},
           {'type': 'eq', 'fun': port_ret_eq, 'args': (mu, annual, mu_0)})

res = spopt.minimize(port_variance, x0, method='SLSQP', args=(r, annual), 
                     constraints=cons_MV, options={'disp': True})

print("Optimization Result:", res)
print("Optimized Portfolio Weights:", res.x)
print("Optimized Portfolio Return:", port_ret(res.x, mu, annual))
print("Optimized Portfolio Variance:", port_variance(res.x, r, annual))
print("Optimized Portfolio Volatility:", port_vola(res.x, r, annual))

weights2=res.x

weights = pd.Series(weights)
weights.name = 'Real data'
weights2 = pd.Series(weights2)
weights2.name = 'Simulated'
asset_names_df = pd.DataFrame(asset_names, columns=['Asset'])

w = pd.concat([asset_names_df, weights, weights2], axis=1)
column_sums = w.sum(axis=0) #the two portfolio weights sum to 1

fig, ax = plt.subplots(figsize=(10, 6))
w.set_index('Asset').plot(kind='bar', ax=ax)
plt.title('Asset Weights: Real vs Simulated')
plt.ylabel('Weights')
plt.xlabel('Assets')
plt.grid(True)
plt.xticks(rotation=0)

#%%check the result of MC simulation for just two columns of the dataframe
selected_df = selected_df.iloc[:, 1:3]

simulated_df = pd.DataFrame()

for col in selected_df.columns:
    prova = selected_df[col]
    kde = gaussian_kde(prova)
    x = np.linspace(min(prova), max(prova), 1000)
    y = kde(x)
    
    sampled_values = rejection_sampling(kde, min(prova), max(prova), num_samples=10000)
    simulated_df[col] = sampled_values[:len(prova)]

    # Plot histograms
    plt.figure(figsize=(12, 6))

    # Histogram for real data
    plt.subplot(2, 2, 1)
    plt.hist(prova, bins=30, alpha=0.7, color='blue', label='Real Data', density=True)
    plt.title(f'Histogram of Real Data for {col}')
    plt.xlabel(col)
    plt.ylabel('Probability Density')
    plt.legend()

    # Histogram for simulated data
    plt.subplot(2, 2, 2)
    plt.hist(simulated_df[col], bins=30, alpha=0.7, color='orange', label='Simulated Data', density=True)
    plt.title(f'Histogram of Simulated Data for {col}')
    plt.xlabel(col)
    plt.ylabel('Probability Density')
    plt.legend()
    
    # KDE plot for real data
    plt.subplot(2, 2, 3)
    plt.plot(x, y, color='blue', label='KDE Real Data')
    plt.title(f'KDE of Real Data for {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend()
    
    # KDE plot for simulated data
    kde_simulated = gaussian_kde(simulated_df[col])
    y_simulated = kde_simulated(x)
    plt.subplot(2, 2, 4)
    plt.plot(x, y_simulated, color='orange', label='KDE Simulated Data')
    plt.title(f'KDE of Simulated Data for {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plt.legend()