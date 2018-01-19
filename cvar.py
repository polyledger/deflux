"""
This module creates optimal portfolio allocations, given a risk score.
"""

import math
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats


DEFAULT_COINS = [
    'BTC', 'ETH', 'BCH', 'XRP', 'LTC', 'XMR', 'ZEC', 'DASH', 'ETC', 'NEO'
]


class Allocator(object):
    def __init__(self, coins, percentile):
        self.coins = coins
        self.percentile = percentile

    def retrieve_data(self):
        """
        Retrives data as a DataFrame.
        """
        filepath = '/Users/matthewrosendin/Desktop/prices.csv'
        df = pd.read_csv(filepath)
        return df

    def solve_minimize(
        self,
        func,
        weights,
        constraints,
        lower_bound=0.0,
        upper_bound=1.0,
        func_deriv=False
    ):
        """
        Returns the solution to a minimization problem.
        """

        bounds = ((lower_bound, upper_bound), ) * len(self.coins)
        return minimize(
            fun=func, x0=weights, jac=func_deriv, bounds=bounds,
            constraints=constraints, method='SLSQP', options={'disp': False}
        )

    def get_min_risk(self, returns):
        """
        Minimizes the conditional value at risk of a portfolio.
        """
        weights = [1/len(self.coins)] * len(self.coins)  # Set sample weights

        def func(weights):
            """The objective function that minimizes risk."""
            simulated_portfolio = self.get_simulated_portfolio(weights, returns)
            return self.get_cvar(simulated_portfolio) * -1

        constraints = (
            {'type': 'eq', 'fun': lambda weights: (weights.sum() - 1)}
        )
        solution = self.solve_minimize(func, weights, constraints)
        min_return = np.dot(solution.x, returns.mean())
        return (min_return, solution.fun * -1)

    def get_max_return(self, returns):
        """
        Maximizes the returns of a portfolio.
        """
        weights = [1/len(self.coins)] * len(self.coins)  # Set sample weights

        def func(weights):
            """The objective function that maximizes returns."""
            sim_portfolio = self.get_simulated_portfolio(weights, returns)
            return sim_portfolio.mean() * -1

        constraints = (
            {'type': 'eq', 'fun': lambda weights: (weights.sum() - 1)}
        )

        solution = self.solve_minimize(func, weights, constraints)
        max_return = solution.fun * -1
        opt_portfolio = self.get_simulated_portfolio(solution.x, returns)
        max_risk = self.get_cvar(opt_portfolio)
        return (max_return, max_risk)

    def efficient_frontier(
        self,
        returns,
        trials,
        min_return,
        max_return,
        points
    ):
        """
        Returns a DataFrame of efficient portfolio allocations.
        """
        points = np.linspace(min_return, max_return, points)
        columns = [coin for coin in self.coins]
        values = pd.DataFrame(columns=columns)
        weights = [1 / len(self.coins)] * len(self.coins)

        for idx, val in enumerate(points):
            def func(weights):
                simulated_portfolio = self.get_simulated_portfolio(
                    weights, returns
                )
                return self.get_cvar(simulated_portfolio) * -1

            constraints = (
                {'type': 'eq', 'fun': lambda weights: (weights.sum() - 1)},
                {'type': 'ineq', 'fun': lambda weights: (
                    np.dot(weights, returns.mean()) - val
                )}
            )
            solution = self.solve_minimize(func, weights, constraints)

            columns = {}
            for index, coin in enumerate(self.coins):
                columns[coin] = math.floor(solution.x[index] * 100 * 100) / 100

            values = values.append(columns, ignore_index=True)
        return values

    def get_midpoint_return(self, returns, samples):
        values = samples.apply(lambda x: self.get_cvar(x))
        med_coin = values.ix[np.where(
            values == np.percentile(values, 50, interpolation='nearest')
        )[0]].index
        med_return = samples.mean()[med_coin]
        return med_return.values

    def get_cvar(self, simulated_portfolio):
        """
        Calculate conditional value at risk.
        """
        value = np.percentile(a=simulated_portfolio, q=self.percentile)

        if self.percentile < 50:
            result = simulated_portfolio[simulated_portfolio < value].mean()
        else:
            result = simulated_portfolio[simulated_portfolio > value].mean()
        return result

    def best_fit_distribution(self, data, bins):
        """
        Model data by finding best fit distribution to data.
        """
        # Get histogram of original data
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # Distributions to check
        dist_names = [
            'alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford',
            'burr', 'cauchy', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull',
            'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f',
            'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'frechet_r',
            'frechet_l', 'genlogistic', 'genpareto', 'gennorm', 'genexpon',
            'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic',
            'gilbrat', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy',
            'halflogistic', 'halfnorm', 'halfgennorm', 'hypsecant', 'invgamma',
            'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'ksone',
            'kstwobign', 'laplace', 'levy', 'levy_l', 'levy_stable',
            'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax',
            'maxwell', 'mielke', 'nakagami', 'ncx2', 'ncf', 'nct', 'norm',
            'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm',
            'rdist', 'reciprocal', 'rayleigh', 'rice', 'recipinvgauss',
            'semicircular', 't', 'triang', 'truncexpon', 'truncnorm',
            'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald',
            'weibull_min', 'weibull_max', 'wrapcauchy'
        ]

        # Keep track of the best fitting distribution, parameters, and SSE
        best_dist = stats.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf

        # Estimate distribution parameters from data
        for dist_name in dist_names:
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    dist = getattr(stats, dist_name)

                    # Fit distribution to data
                    params = dist.fit(data)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = dist.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    # Determine if this distribution is a better fit
                    if best_sse > sse > 0:
                        best_dist = dist
                        best_params = params
                        best_sse = sse

            except Exception:
                # TODO: May want to handle this exception.
                pass

        return (best_dist.name, best_params)

    def generate_samples(self, dist, params, size):
        """
        Generate samples using the best fit distibution
        """
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        samples = dist.rvs(loc=loc, scale=scale, *arg, size=size)
        return samples

    def get_distributions(self, returns, trials):
        """
        Find the best-fitting distributions for each coin
        """
        fitted_distributions = dict()
        results = pd.DataFrame(columns=self.coins)
        for i in self.coins:
            # Fit the data with distributions
            dist_name, dist_params = self.best_fit_distribution(
                data=returns[i].dropna(inplace=False),
                bins=30)
            fitted_distributions[i] = [dist_name, dist_params]
            print('{0}: {1} with params of {2}'.format(
                i, dist_name, dist_params)
            )

            # Generate samples
            dist = getattr(stats, dist_name)
            results[i] = self.generate_samples(dist, dist_params, trials)

        return results, fitted_distributions

    def get_simulated_portfolio(self, weights, returns):
        """
        Get portfolio returns based on returns and weights
        """
        return (returns * weights).sum(1)

    def allocate(self):
        """
        Returns an efficient portfolio allocation for the given risk index.
        """
        # Read CSV file
        df = self.retrieve_data()

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

        samples, distributions = self.get_distributions(
            returns=df_daily_returns,
            trials=100000
        )

        min_return, min_risk = self.get_min_risk(samples)
        max_return, max_risk = self.get_max_return(samples)

        frontier = self.efficient_frontier(
            returns=samples,
            trials=len(samples),
            min_return=min_return,
            max_return=max_return,
            points=6
        )

        return frontier
