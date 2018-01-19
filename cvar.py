import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats

start_time = time.time()

# Read CSV file
filepath = '/Users/matthewrosendin/Desktop/prices.csv'
df = pd.read_csv(filepath)

# Get the list of coin symbols
coins = df.drop('time', axis=1).columns

# Replace NAs with zeros
df.replace(0, np.nan, inplace=True)
df = df.reset_index(drop=True)

# Order rows ascending by time (oldest first, newest last)
df.sort_values(['time'], ascending=True, inplace=True)

# Truncate dataframe to prices on or after 2016-01-01
df = df[df['time'] >= '2016-01-01']
df = df.reset_index(drop=True)

# Create a dataframe of daily returns
objs = [df.time, df[coins].apply(lambda x: x / x.shift(1) - 1)]
df_daily_returns = pd.concat(objs=objs, axis=1)
df_daily_returns = df_daily_returns.reset_index(drop=True)

def get_summary_statistics(returns):
    """
    Returns summary statistics for the returns of the coins.

    Expected Return: Daily mean return of the coin
    Standard Deviation: Dollar amount dispersion from the daily expected return
    Kurtosis: Measure of whether the data are heavy-tailed or light-tailed
    Skewness: Measure of symmetry of the distribution of returns
    """
    coins = returns.columns
    metrics = pd.DataFrame(index=coins)
    metrics['Expected Return'] = returns[coins].apply(np.nanmean, axis=0)
    metrics['Standard Deviation'] = returns[coins].apply(np.nanstd, axis=0)
    metrics['Kurtosis'] = returns[coins].apply(
        func=lambda x: stats.kurtosis(x, nan_policy='omit'),
        axis=0
    )
    metrics['Skewness'] = returns[coins].apply(
        func=lambda x: stats.skew(x, nan_policy='omit'),
        axis=0
    )
    return round(metrics, 4)

# Print summary statistics
summ_stats = get_summary_statistics(df_daily_returns[coins])
print(summ_stats)

def get_cvar(portfolio_simulated, percentile):
    """
    Calculate conditional value at risk.
    """
    value = np.percentile(a=portfolio_simulated, q=percentile)

    if percentile < 50:
        return (portfolio_simulated[portfolio_simulated < value].mean())
    else:
        return (portfolio_simulated[portfolio_simulated > value].mean())

# Print conditional value at risk for NEO's returns at 5th percentile
cvar = round(get_cvar(df_daily_returns['NEO'].dropna(), 5) * 100, 2)
print('Conditional value at risk for NEO at 5th percentile: {}%'.format(cvar))

def best_fit_distribution(data, bins, ax=None):
    """
    Model data by finding best fit distribution to data.
    """
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        stats.alpha, stats.anglit, stats.arcsine, stats.beta, stats.betaprime,
        stats.bradford, stats.burr, stats.cauchy, stats.chi, stats.chi2,
        stats.cosine, stats.dgamma, stats.dweibull, stats.erlang, stats.expon,
        stats.exponnorm, stats.exponweib, stats.exponpow, stats.f,
        stats.fatiguelife, stats.fisk, stats.foldcauchy, stats.foldnorm,
        stats.frechet_r, stats.frechet_l, stats.genlogistic, stats.genpareto,
        stats.gennorm, stats.genexpon, stats.genextreme, stats.gausshyper,
        stats.gamma, stats.gengamma, stats.genhalflogistic, stats.gilbrat,
        stats.gompertz, stats.gumbel_r, stats.gumbel_l, stats.halfcauchy,
        stats.halflogistic, stats.halfnorm, stats.halfgennorm, stats.hypsecant,
        stats.invgamma, stats.invgauss, stats.invweibull, stats.johnsonsb,
        stats.johnsonsu, stats.ksone, stats.kstwobign, stats.laplace,
        stats.levy, stats.levy_l, stats.levy_stable, stats.logistic,
        stats.loggamma, stats.loglaplace, stats.lognorm, stats.lomax,
        stats.maxwell, stats.mielke, stats.nakagami, stats.ncx2, stats.ncf,
        stats.nct, stats.norm, stats.pareto, stats.pearson3, stats.powerlaw,
        stats.powerlognorm, stats.powernorm, stats.rdist, stats.reciprocal,
        stats.rayleigh, stats.rice, stats.recipinvgauss, stats.semicircular,
        stats.t, stats.triang, stats.truncexpon, stats.truncnorm,
        stats.tukeylambda, stats.uniform, stats.vonmises, stats.vonmises_line,
        stats.wald, stats.weibull_min, stats.weibull_max, stats.wrapcauchy
    ]

    # Keep track of the best fitting distribution, parameters, and SSE
    best_distribution = stats.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:
        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # Fit distribution to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # Determine if this distribution is a better fit
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            # TODO: May want to handle this exception.
            pass

    return (best_distribution.name, best_params)

def pdf(dist, params, size=10000):
    """
    Generate distributions's probability distribution function (PDF)
    """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get same start and end points of distribution
    start = (
        dist.ppf(0.01, *arg, loc=loc, scale=scale)
        if arg
        else dist.ppf(0.01, loc=loc, scale=scale)
    )
    end = (
        dist.ppf(0.99, *arg, loc=loc, scale=scale)
        if arg
        else dist.ppf(0.99, loc=loc, scale=scale)
    )

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    samples = dist.rvs(loc=loc, scale=scale, *arg, size=10)
    print('Samples from PDF: {}'.format(samples))
    pdf = pd.Series(y, x)
    return pdf

def generate_samples(dist, params, size):
    """
    Generate samples using the best fit distibution
    """

    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    samples = dist.rvs(loc=loc, scale=scale, *arg, size=size)
    return samples

def get_distributions(returns, coins, trials):
    """
    Find the best-fitting distributions for each coin
    """
    fitted_distributions = dict()
    results = pd.DataFrame(columns=coins)
    for i in coins:
        # Fit the data with distributions
        dist_name, dist_params = best_fit_distribution(df_daily_returns[i].dropna(inplace=False), bins=30)
        fitted_distributions[i] = [dist_name, dist_params]
        print('{0}: {1} with params of {2}'.format(i, dist_name, dist_params))

        # Generate samples
        dist_name = getattr(stats, dist_name)
        results[i] = generate_samples(dist_name, dist_params, trials)

    return results, fitted_distributions

samples, distributions = get_distributions(df_daily_returns, coins, 100000)

def get_simulated_portfolio(weights, simulated_returns):
    """
    Get portfolio returns based on returns and weights
    """
    return ((simulated_returns * weights).sum(1))

def get_minimum_risk(sim_returns, coins, percentile, bounds):
    weights = [1/len(coins)] * len(coins)
    def func(weights):
        portfolio_simulated = get_simulated_portfolio(weights, sim_returns)
        return (get_cvar(portfolio_simulated, percentile) * -1)

    def constraint1(weights):
        return(weights.sum() - 1)

    con1 = ({'type': 'eq', 'fun': constraint1})

    # Boundaries
    bnds = ((bounds[0], bounds[1]),) * len(coins)
    solution = minimize(
        func,
        weights,
        jac=False,
        bounds=bnds,
        constraints=con1,
        method='SLSQP',
        options={'disp': False})
    min_return = np.dot(solution.x, sim_returns.mean())
    return (min_return, solution.fun * -1)

def get_midpoint_return2(sim_returns, coins, percentile):
    values = samples.apply(lambda x: get_cvar(x, percentile))
    med_coin = values.ix[np.where(values == np.percentile(values, 50, interpolation='nearest'))[0]].index
    med_return =samples.mean()[med_coin]
    return (med_return.values)

def get_midpoint_return(sim_returns, coins, percentile, bounds):
    weights = [1/len(coins)] * len(coins)
    # For now, midpoint is the median 'risk' value among all coins
    midpoint = np.median(sim_returns.apply(lambda x: get_cvar(x, percentile))) * -1

    def func(weights):
        portfolio_simulated = get_simulated_portfolio(weights, sim_returns)
        return (get_cvar(portfolio_simulated, percentile)*-1)

    def constraint1(weights):
        return(weights.sum() - 1)

    def constraint2(weights):
        return(np.dot(weights, sim_returns.mean()) + midpoint)

    con1 = ({'type': 'eq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint2})

    bnds = ((bounds[0], bounds[1]),) * len(coins)
    solution = minimize(
        func,
        weights,
        jac=False,
        bounds=bnds,
        constraints=con1,
        method='SLSQP',
        options={'disp': False})
    min_return = np.dot(solution.x, sim_returns.mean())
    return (min_return)

def get_maximized_returns(sim_returns, coins, percentile, bounds):
    # Set sample weights
    weights = [1/len(coins)] * len(coins)

    def func(weights):
        # Objective function: maximize returns
        sim_portfolio = get_simulated_portfolio(weights, sim_returns)
        return (sim_portfolio.mean()*-1)

    # Constraints
    def constraint1(weights):
        return (weights.sum() - 1)

    con1 = ({'type': 'eq', 'fun': constraint1})

    # Boundaries
    bnds = ((bounds[0], bounds[1]),) * len(coins)

    # Optimize Solution
    solution = minimize(
        func,
        weights,
        jac=False,
        bounds=bnds,
        constraints=con1,
        method='SLSQP',
        options={'disp': False})
    max_return = solution.fun * -1
    opt_portfolio = get_simulated_portfolio(solution.x, sim_returns)
    max_risk = get_cvar(opt_portfolio, percentile)
    return (max_return, max_risk)

def efficient_frontier(sim_returns, coins, trials, percentile, bounds, points_between):
    # Get minimum risk
    min_return, min_risk = get_minimum_risk(sim_returns, coins, percentile, bounds)
    print('Min Return: {}'.format(min_return))
    # Get maximum return
    max_return, max_risk = get_maximized_returns(sim_returns, coins, percentile, bounds)
    print('Max Return: {}'.format(max_return))

    # Get midpoint return
    mid_return = get_midpoint_return2(sim_returns, coins, percentile)
    print('Mid Return: {}'.format(mid_return))

    # Set up optitmization
    points = np.linspace(min_return, max_return, points_between)
    values = pd.DataFrame(columns=np.hstack(('portfolio', coins, 'return', 'risk')))
    sim_portfolios = pd.DataFrame(columns=['portfolio_' + str(i) for i in range(points_between)])
    weights = [1 / len(coins)] * len(coins)

    for i, v in enumerate(points):
        def func(weights):
            portfolio_simulated = get_simulated_portfolio(weights, sim_returns)
            return (get_cvar(portfolio_simulated, percentile) * -1)

        def constraint1(weights):
            return (weights.sum() - 1)

        def constraint2(weights):
            # Portfolio return constraint
            return (np.dot(weights, sim_returns.mean()) - v)

        con1 = ({'type': 'eq', 'fun': constraint1},
                {'type': 'ineq', 'fun': constraint2})
        bnds = ((bounds[0], bounds[1]),) * len(coins)
        solution = minimize(
            func,
            weights,
            jac=False,
            bounds=bnds,
            constraints=con1,
            method='SLSQP',
            options={'disp': False})
        values = values.append({
            'BTC': round(solution.x[0],3),
            'ETH': round(solution.x[1],3),
            'BCH': round(solution.x[2],3),
            'XRP': round(solution.x[3],3),
            'LTC':round(solution.x[4], 3),
            'XMR': round(solution.x[5], 3),
            'ZEC': round(solution.x[6], 3),
            'DASH': round(solution.x[7], 3),
            'ETC': round(solution.x[8], 3),
            'NEO': round(solution.x[9],3),
            'Return': round(np.dot(solution.x, sim_returns.mean()),7),
            'Risk': round(solution.fun,7)}, ignore_index=True)
        sim_portfolios[sim_portfolios.columns[i]] = get_simulated_portfolio(solution.x, sim_returns)
        # print(get_simulated_portfolio(solution.x, returns_sim))
    values['portfolio'] = np.arange(0, points_between, 1)
    return (values, sim_portfolios)

print('Finished in {} s'.format(time.time() - start_time, 's'))
