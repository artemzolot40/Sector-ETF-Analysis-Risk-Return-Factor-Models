import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

data_set = pd.read_csv("/Users/artemzolotovskii/Documents/Asset Pricing/dataset.csv")

#Question 1
data_set ["Date"] = pd.to_datetime(data_set["Date"], dayfirst=True)
data_set = data_set.sort_values ("Date")

print(data_set.head())

tickers = ["XLY", "XLV", "XLK", "^GSPC"]
print(tickers)

prices = data_set[["Date"] + tickers]
prices = prices.set_index("Date")

monthly_returns = prices.pct_change()
monthly_returns = monthly_returns.dropna()

rf_annual = 0.03
rf_monthly = rf_annual / 12

rows = []

for t in tickers:
    ret = monthly_returns[t]
    ret = ret.dropna()
    mean = ret.mean()
    std = ret.std()
    mean_eft = mean * 12
    mean_eft_percent = mean_eft * 100
    std_eft = std * np.sqrt(12)
    std_eft_percent = std_eft * 100
    excess_eft = mean_eft - rf_annual
    excess_eft_percent = excess_eft * 100
    sharpe_ratio = excess_eft / std_eft
    rows.append([t, mean_eft_percent, std_eft_percent, excess_eft_percent, sharpe_ratio])

table1 = pd.DataFrame(rows, columns=["Ticker", "Annualized Average Return (%)",
                                     "Annualized Standard Deviation (%)",
                                     "Annualized Mean Excess Return (%)",
                                     "Sharpe Ratio"])
table1 = table1.round(3)
print(table1.to_string(index=False, justify="center"))


std_prices = prices / prices.iloc[0]

plt.figure(figsize = (10,7))
plt.plot(std_prices.index, std_prices["XLY"], label = "XLY")
plt.plot(std_prices.index, std_prices["XLV"], label = "XLV")
plt.plot(std_prices.index, std_prices["XLK"], label = "XLK")
plt.plot(std_prices.index, std_prices["^GSPC"], label = "^GSPC")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Standardised Price")
plt.title("Standardised Price Series over the Sample Period")
plt.tight_layout()
plt.show()

data_set ["Date"] = pd.to_datetime(data_set["Date"], dayfirst=True)
data_set = data_set.sort_values ("Date")

tickers = ["XLY", "XLV", "XLK"]

prices = data_set[tickers].copy()
returns = prices.pct_change().dropna()

monthly_returns = prices.pct_change()
monthly_returns = monthly_returns.dropna()

mean_month = monthly_returns.mean()          # vector of means (monthly)
mean_ann = mean_month * 12           # annualised expected returns

cov_month = monthly_returns.cov()            # monthly covariance matrix
cov_ann = cov_month * 12

print("=== Annualised expected returns (mean_ann) ===")
print(mean_ann)
print()

print("=== Annualised covariance matrix (cov_ann) ===")
print(cov_ann)
print()

# --- Risk-free rate ---

rf = 0.03   # 3% per year as in the assignment

# --- Minimum-variance portfolio (MVP) ---

Sigma = cov_ann.values
inv_Sigma = np.linalg.inv(Sigma)
ones = np.ones(len(tickers))

w_mvp = inv_Sigma @ ones
w_mvp = w_mvp / (ones.T @ inv_Sigma @ ones)
w_mvp = w_mvp.round(decimals=3)

print("=== MVP weights ===")
for t, w in zip(tickers, w_mvp):
    print(t, ":", w)
print()

# --- Tangency (optimal risky) portfolio ---

mu = mean_ann.values
excess = mu - rf * ones              # (mu - r_f * 1)

w_tan_raw = inv_Sigma @ excess
# normalise so weights sum to 1
w_tan = w_tan_raw / np.sum(w_tan_raw)
w_tan = w_tan.round(decimals=3)

print("=== Tangency portfolio weights ===")
for t, w in zip(tickers, w_tan):
    print(t, ":", w)
print()

# --- Function to compute portfolio stats ---

def portfolio_stats(weights, mu_vec, cov_mat, rf_rate):
    r_p = float(weights @ mu_vec)
    var_p = float(weights @ cov_mat @ weights)
    sd_p = np.sqrt(var_p)
    sharpe_p = (r_p - rf_rate) / sd_p
    return r_p, sd_p, sharpe_p



mvp_r, mvp_s, mvp_sh = portfolio_stats(w_mvp, mu, Sigma, rf)
mvp_r = round(mvp_r, 4)
mvp_s = round(mvp_s, 4)
mvp_sh = round(mvp_sh, 4)
tan_r, tan_s, tan_sh = portfolio_stats(w_tan, mu, Sigma, rf)
tan_r = round(tan_r, 4)
tan_s = round(tan_s, 4)
tan_sh = round(tan_sh, 4)
print("=== MVP stats (annual) ===")
print("Return:", mvp_r)
print("Std dev:", mvp_s)
print("Sharpe:", mvp_sh)
print()

print("=== Tangency portfolio stats (annual) ===")
print("Return:", tan_r)
print("Std dev:", tan_s)
print("Sharpe:", tan_sh)
print()

# --- Efficient frontier (opportunity set) ---

frontier_rets = []
frontier_stds = []

# grid over weights w1, w2, w3 = 1 - w1 - w2, with w >= 0 (no short selling in the grid)
grid = np.linspace(0, 1, 101)
for w1 in grid:
    for w2 in grid:
        w3 = 1 - w1 - w2
        if w3 < 0:
            continue
        w = np.array([w1, w2, w3])
        r_p, s_p, _ = portfolio_stats(w, mu, Sigma, rf)
        frontier_rets.append(r_p)
        frontier_stds.append(s_p)

plt.figure(figsize=(9, 6))
plt.scatter(frontier_stds, frontier_rets, s=5, alpha=0.4, label="Portfolios (grid)")

# plot individual assets
asset_stds = np.sqrt(np.diag(Sigma))
asset_rets = mu
plt.scatter(asset_stds, asset_rets, color="black", marker="x", s=60, label="Individual assets")
for i, t in enumerate(tickers):
    plt.text(asset_stds[i] * 1.01, asset_rets[i] * 1.01, t)

# plot MVP and tangency
plt.scatter(mvp_s, mvp_r, color="red", label="MVP", zorder=5)
plt.scatter(tan_s, tan_r, color="blue", label="Tangency", zorder=5)

plt.xlabel("Annualised volatility")
plt.ylabel("Annualised expected return")
plt.title("Opportunity set and key portfolios (XLV, XLF, XOP)")
plt.legend()
plt.tight_layout()
plt.show()

mean_month = returns.mean()
mean_ann = mean_month * 12               # annual expected returns

cov_month = returns.cov()
cov_ann = cov_month * 12                 # annual covariance matrix

rf = 0.03                                # annual risk-free rate

# ===========================
# Equal-weight portfolio
# ===========================

N = len(tickers)
w_eq = np.array([1/N] * N)

def portfolio_stats(weights, mu_vec, cov_mat, rf_rate):
    r_p = float(weights @ mu_vec)
    sd_p = float(np.sqrt(weights @ cov_mat @ weights))
    sharpe_p = (r_p - rf_rate) / sd_p
    return r_p, sd_p, sharpe_p

eq_r, eq_s, eq_sh = portfolio_stats(
    w_eq,
    mean_ann.values,
    cov_ann.values,
    rf
)

print("=== Equal-weight portfolio (1/N) results ===")
print("Weights:", w_eq)
print("Annual return:", eq_r)
print("Annual st.dev:", eq_s)
print("Sharpe:", eq_sh)

prices = data_set[tickers + ["^GSPC"]]
returns = prices.pct_change().dropna()

SMB = data_set["SMB"].loc[returns.index]
HML = data_set["HML"].loc[returns.index]
RF =data_set["RF"].loc[returns.index]

market_excess = returns["^GSPC"] - RF
etf_excess = returns[tickers].sub(RF, axis=0)

single_rows = []

for t in tickers:
    y = etf_excess[t]              # dependent: Ri - Rf
    X = sm.add_constant(market_excess)  # independent: const + (Rm - Rf)

    model = sm.OLS(y, X).fit()

    alpha = model.params.iloc[0]
    beta_mkt = model.params.iloc[1]
    sigma2_e = np.var(model.resid, ddof=1)  # idiosyncratic variance
    R2 = model.rsquared

    single_rows.append([t, alpha, beta_mkt, sigma2_e, R2])

# Build the regression table DataFrame
reg_table = pd.DataFrame(
    single_rows,
    columns=["ETF", "alpha", "beta_mkt", "sigma2_e", "R2"]
)

print("\nRegression table (Q4a):")
print(reg_table.round(4).to_string(index=False))

ff_rows = []

for t in tickers:
    y = etf_excess[t]

    # Build X with market excess, SMB, HML
    X = pd.concat([market_excess, SMB, HML], axis=1)
    X.columns = ["MKT_RF", "SMB", "HML"]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    beta_M   = model.params["MKT_RF"]
    beta_SMB = model.params["SMB"]
    beta_HML = model.params["HML"]

    ff_rows.append([t, beta_M, beta_SMB, beta_HML])

# Build factor loadings table
ff_table = pd.DataFrame(
    ff_rows,
    columns=["ETF", "Beta_M", "Beta_SMB", "Beta_HML"]
)

print("\nTable 5: Factor loadings (Q4b):")
print(ff_table.round(4).to_string(index=False))

print("\n=== Q4(b) Fama-French 3-Factor Model ===")

for t in tickers:
    y = etf_excess[t]

    X = pd.concat([market_excess, SMB, HML], axis=1)
    X.columns = ["MKT_RF", "SMB", "HML"]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    print(f"\n--- {t} ---")
    print(model.summary())