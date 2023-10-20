# This file is part of the HapNe effective population size inference software.
# Copyright (C) 2023-present HapNe Developers.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from configparser import ConfigParser
from hapne.backend.DemographicHistory import PieceWiseExpDemographicHistory
from hapne.backend.StatsModel import StatsModel, IBDModel, LDModel, DemographicHistory2StatsModelAdapter
from hapne.backend.IO import DataHandler, IBDLoader, LDLoader
import numpy as np
from scipy.optimize import minimize
from numpy import log, exp
import scipy
import os
import pandas as pd
from hapne.utils import smoothing, LogBijection
from hapne.output import plot_results
import logging
from tqdm import tqdm
from pathlib import PurePath
import matplotlib.pyplot as plt
from scipy.stats import norm


class HapNe:
    def __init__(self, config: ConfigParser):
        """
        Class HapNe allows to fit a demographic history based on summary statistics
        :param config: see example config file
        """
        self.config = config
        self.method = self.config["CONFIG"]['method'].lower()
        # Logging
        verbose = self.config.get("CONFIG", "verbose", fallback=False)
        if verbose:
            logging.basicConfig(level=logging.INFO)
        # Method specific instances
        self.io = self.get_loader()
        self.stats_model = self.get_stats_model()

        # Parameter related to the model
        nb_points = 7 if self.method == "ld" else 8
        start = -5 if self.method == "ld" else -6
        if self.config.getfloat("CONFIG", "sigma2", fallback=None) is None:
            self.parameter_grid = np.array([10.**(start + ii) for ii in range(0, nb_points)])
        else:
            logging.info("Using user-provided sigma2")
            self.parameter_grid = np.array([self.config.getfloat("CONFIG", "sigma2")])
        self.u_min = self.get_default_u_min()  # Minimum value of u
        self.u_quantile = self.config.getfloat("CONFIG", "u_quantile", fallback=self.u_min)
        self.dt_min = max(0.25, self.config.getint("CONFIG", "dt_min", fallback=1))
        self.dt_max = self.config.getint("CONFIG", "dt_max", fallback=5000)
        self.t_max = self.config.getint("CONFIG", "t_max", fallback=125)
        self.buffer_time = self.t_max - 25  # the times averages the deep time effect
        # Parameters related to the fitting procedure
        self.mode = self.config.get("CONFIG", "mode", fallback="regularised")
        pseudo_diploid = self.config.getboolean("CONFIG", "pseudo_diploid", fallback=False)
        if self.mode == "regularised":
            default_params = 16 if pseudo_diploid else 21
        else:
            default_params = 50
        self.nb_parameters = self.config.getint("CONFIG", "nb_parameters", fallback=default_params)
        self.random_restarts = 3
        self.eps = 1e-4
        self.ftol = 1e-3
        self.n_min = 1e2
        self.n_max = 1e9
        self.signal_threshold = 6.635 if self.config['CONFIG']['METHOD'] == "ld" else 9.210
        self.nb_bootstraps = self.config.getint("CONFIG", "nb_bootstraps", fallback=100)

        # Initialisations
        self.times = scipy.stats.erlang.ppf(np.linspace(0, 1, self.nb_parameters + 1)[:-1], a=2,
                                            scale=1. / (2 * self.u_quantile))
        self.init_times()

    def get_loader(self) -> DataHandler:
        """
        Initialise the loader based on the method
        :return: subclass of DataHandler
        """
        if self.method == "ibd":
            dh = IBDLoader(self.config)
        elif self.method == "ld":
            dh = LDLoader(self.config)
        else:
            raise Exception("not implemented error", "method must either be LD or IBD")
        return dh

    def get_default_u_min(self):
        """
        Depending on the method, the default value is different.
        :return: minimum genetic distance in morgan
        """
        if self.method == "ibd":
            return 0.025
        elif self.method == "ld":
            return 0.005
        else:
            raise Exception("not implemented error", "method must either be LD or IBD")

    def get_stats_model(self) -> StatsModel:
        """
        Initialise the stats model to be used based on the method
        :return: subclass of StatsModel
        """
        if self.config["CONFIG"]['method'].lower() == "ibd":
            number_haploid_samples = 2 * self.config.getint("CONFIG", "nb_samples")
            nb_pairs = 0.5 * number_haploid_samples * (number_haploid_samples - 1)
            stats_model = IBDModel(nb_pairs, self.io.region_length().reshape([-1, 1]), self.io.phi2)

        elif self.config["CONFIG"]['method'].lower() == "ld":
            stats_model = LDModel(sigma=self.io.sigma)
        else:
            raise Exception("not implemented error", "method must either be LD or IBD")

        return stats_model

    def init_times(self):
        const_estimator = self.fit_constant()
        ne = np.ones(self.nb_parameters) * const_estimator
        self.update_times(ne)

    def fit_constant(self) -> float:
        def local_loss(ne):
            n = np.ones(self.nb_parameters) * ne
            return self.sse(n)
        res = minimize(lambda x: local_loss(self.x2n(x)), np.log(10_000))
        return self.x2n(res.x)[0]

    def fit(self):
        if self.mode == "regularised":
            ne_boot = self.fit_regularized()
        elif self.mode == "fixed":
            ne_boot = self.fit_fixed()
        elif self.mode == "mcmc":
            raise NotImplementedError("MCMC mode is not implemented yet")
            self.fit_mcmc()
        else:
            raise Exception("not implemented error", "mode must either be regularised, fixed or mcmc")
        self.save_results(ne_boot)

    def fit_fixed(self):
        self.initialize_fixed_prior_procedure()
        logging.info("Starting the fitting procedure")
        output_folder = self.get_output_folder()
        for _ in tqdm(range(self.nb_bootstraps)):
            mu_boot, indices = self.io.bootstrap(return_indices=True)
            self.stats_model.set_dataset(indices)
            theta_hat = self.optimize_theta(mu_boot, self.sig2)
            n_hat = self.x2n(theta_hat)
            demohist = self.n2demohist(n_hat)
            n_hat = demohist.n.ravel()
            # append n_hat to results file
            with open(output_folder / "hapne.boot", "a") as f:
                f.write(",".join([f"{x:.0f}" for x in n_hat]) + "\n")
        self.stats_model.set_dataset(np.arange(self.io.nb_regions()))
        logging.info("Fitting procedure finished")
        return pd.read_csv(output_folder / "hapne.boot", sep=",", header=None).values

    def initialize_fixed_prior_procedure(self):
        yearly_rate_growth = self.config.getfloat("CONFIG", "yearly_rate_growth", fallback=0.001)
        logging.info(f"Fitting with a fixed prior (std of growth rate assumed to be {yearly_rate_growth})")
        self.sig2 = ((1 + yearly_rate_growth) ** 29 - 1)**2
        self.times = np.linspace(0, self.t_max, self.nb_parameters)
        self.buffer_time = self.t_max - 25

    def optimize_theta(self, mu, sig2):
        self.times = np.linspace(0, self.t_max, self.nb_parameters)
        theta_0_mean = np.random.normal(np.log(10000), 1, self.nb_parameters)
        theta_0 = np.random.normal(theta_0_mean, 1, self.nb_parameters)
        res = scipy.optimize.minimize(lambda theta: -self.log_p(theta, mu, sig2),
                                      theta_0, method="L-BFGS-B", options={"ftol": 1e-4}
                                      )
        n_hat = self.x2n(res.x)
        demohist = self.n2demohist(n_hat)
        n_hat = demohist.n.ravel()
        return res.x

    def log_p(self, theta, mu, sig2):
        n = self.x2n(theta)
        pred = self.predict(n)
        log_likelihood = self.stats_model.log_likelihood(mu, pred)
        dt = np.diff(self.times)
        pop_ratio = (1 - np.exp(np.abs(np.diff(theta)) / dt))
        return log_likelihood - np.sum(dt * pop_ratio**2 / 2 / sig2)

    def fit_regularized(self):
        """
        Perform HapNe fitting procedure and save the results.
        """
        # Step 1 : calibrate times and get MAP Ne and best regularisation
        logging.info("Starting the calibration step...")
        ne_hat, sig2 = self.calibrate_method()
        # Step 2 : bootstrap results
        logging.info("Starting the bootstrapping procedure...")
        ne_bootstrap = self.bootstrap(ne_hat, sig2)
        # Step 3 : save results
        logging.info("Saving results...")
        return ne_bootstrap

    def calibrate_method(self):
        """
        Find the best regularisation parameter and time intervals
        """
        nb_iterations = 5
        # Save the results of each iterations
        times, nes = [np.zeros((nb_iterations, self.nb_parameters)) for _ in range(2)]
        sigmas, sses = [np.inf + np.zeros(nb_iterations) for _ in range(2)]
        # Run all iterations
        for ii in range(nb_iterations):
            times[ii, :] = self.times.copy()
            logging.info(f"Starting iteration {ii+1}")
            logging.info("*****")
            ne_hat, sig2, sse = self.run_relaxation()
            sigmas[ii], sses[ii] = sig2, sse
            logging.info(f"Iteration {ii+1}/{nb_iterations} : {sig2:.5f} - {sse:.2f} ")
            nes[ii, :] = ne_hat.ravel().copy()
            self.update_times(ne_hat)
            # Escape the iterations if time intervals have converged
            if self.time_intervals_have_converged(old_times=times[ii, :]):
                logging.info(f"Converged at iteration {ii}")
                return self.select_calibration(times[:ii + 1, :], nes[:ii + 1, :], sigmas[:ii + 1], sses[:ii + 1])

        if self.config.getfloat("CONFIG", "sigma2", fallback=None) is None:
            sig2message = f"Calibration step done, {sig2:.5f} selected!"
            logging.info(sig2message)
            self.io.summary_message += "\n" + sig2message + "\n"
            if sig2 < self.parameter_grid[2]:
                sig2message = "Evidence for fluctuation is weak, the results may be unreliable (looking too constant)"
                logging.warning(sig2message)
                self.io.summary_message += "\n" + sig2message + "\n"

        return self.select_calibration(times, nes, sigmas, sses)

    def run_relaxation(self):
        """
        Run a grid-search over different values of the hyperparameters
        Select the highest regularisation consistent with the signal.
        """
        nb_points = len(self.parameter_grid)
        # The loss is -log likelihood, hence does not depend on regularisation
        sses = np.inf + np.zeros(nb_points)
        nes = np.zeros((nb_points, self.nb_parameters))
        # Initialize the population size with the constant estimator
        ne_ini = np.ones(self.nb_parameters) + self.fit_constant()
        for ii, sigma2 in enumerate(self.parameter_grid):
            # minimize the penalized likelihood and get ne and loss (- log likelihood, no reg.)
            ne, sse = self.argmin_loss(ne_ini, sigma2)
            # Update the results if there is an improvement
            if sse < np.max(sses):
                nes[ii, :] = ne.copy()
                sses[ii] = sse
            # If not, the optimisation has not worked, do not save the results
            else:
                nes[ii, :] = nes[ii - 1, :]
                sses[ii] = sses[ii - 1]
            # Get the new inital paramter
            ne_ini = nes[ii, :]
            logging.info(f"{sigma2:.5f} - {sses[ii]:.2f}")
        # Select the best hyperparameter based on the grid search
        index = self.get_best_hyperparameter_index(sses)
        return nes[index, :], self.parameter_grid[index], self.sse(nes[index, :])

    def argmin_loss(self, ne_ini: np.ndarray, sigma2: float, data=None) -> tuple:
        best_loss = np.inf
        for _ in range(self.random_restarts):
            res = minimize(lambda x: self.loss(self.x2n(x), sigma2, data), self.n2x(ne_ini),
                           method='L-BFGS-B',
                           options={
                'ftol': self.ftol,
                'maxcor': 100,
                'eps': self.eps
            }
            )
            # Save the results if improved
            if res.fun < best_loss:
                ne = self.x2n(res.x)
                best_loss = res.fun
            # Add a perturbation to the initial guess
            ne_ini = ne_ini * np.random.uniform(0.75, 1.33, len(ne_ini))
        return ne, self.sse(ne)

    def loss(self, ne, sigma2, observations=None) -> float:
        return self.sse(ne, observations) - self.log_prior(ne, sigma2)

    def sse(self, ne, observations=None) -> float:
        """
        Compute the SSE of the Deviance residuals.
        :param : ne
        :param : observations : either none to use all data, or set of data
        """
        if observations is None:
            observations = self.io.mu
        predictions = self.predict(ne)
        return np.sum(self.stats_model.residuals(observations, predictions)**2)

    def log_prior(self, ne: np.ndarray, sigma2: float) -> float:
        """
        Returns the log prior, up to a constant
        """
        return -(np.sqrt(np.diff(np.log(ne))**2 + np.diff(self.times)**2).sum() - self.times[-1]) / sigma2

    def x2n(self, x: np.ndarray) -> np.ndarray:
        n = exp(x)
        return np.clip(n, self.n_min, self.n_max)

    def n2x(self, n: np.ndarray) -> np.ndarray:
        n = np.clip(n, self.n_min, self.n_max)
        return log(n)

    def get_best_hyperparameter_index(self, losses: np.ndarray) -> int:
        """
        Select the best hyperparameter consistent with the data
        """
        sig2_index = np.argmin(losses > np.min(losses) + self.signal_threshold)
        return sig2_index

    def time_intervals_have_converged(self, old_times: np.ndarray):
        """
        Determine whether the time intervals have converged.
        """
        # Return True if the maximum difference between two intervals is less than 2 generations
        return np.max(np.abs(old_times - self.times)) < 2

    def select_calibration(self, times: np.ndarray, nes: np.ndarray, sigmas: np.ndarray, losses: np.ndarray):
        """
        The calibration methods saves all intermediary results, this method returns the best one
        """
        best_calibration_index = np.argmin(losses)
        ne_hat = nes[best_calibration_index, :].ravel()
        sigma2 = sigmas[best_calibration_index]
        times = times[best_calibration_index, :].ravel()
        self.times = times
        return ne_hat, sigma2

    def update_times(self, ne):
        """
        Update the time intervals to match quantiles based on ne
        """
        demo = self.n2demohist(ne)
        self.times = demo.time_quantiles(self.u_quantile, self.nb_parameters,
                                         min_interval_length=self.dt_min,
                                         max_interval_length=self.dt_max, last_t=self.t_max)

    def n2demohist(self, n, return_coefs=False):
        """
        convert the parameters theta into a demographic history
        :param n: population size at the beginning of each interval
        :param return_coefs : if True also return the exponential coefficients
        :return : Demographic History 2 StatsModel Adapter
        """
        n = np.clip(n, 1e2, 1e8)
        demo_hist = PieceWiseExpDemographicHistory(n, u_min=self.u_quantile, delta_t_max=1,
                                                   time_thresholds=self.times)
        coefs = demo_hist.exp_coef
        demo_hist = DemographicHistory2StatsModelAdapter(demo_hist.time[:-1], demo_hist.n)
        if return_coefs:
            return demo_hist, coefs
        else:
            return demo_hist

    def predict(self, n):
        demo_hist = self.n2demohist(n, return_coefs=False)
        if self.io.time_heterogeneity is None:
            return self.stats_model.predict(self.io.bins, demo_hist)
        else:
            predictions = np.zeros((1, len(self.io.bins) - 1))
            # Define the points at which to evaluate the function for Simpson integration
            u_border = self.io.bins.reshape([1, -1])
            u_center = (u_border[:, 1:] + u_border[:, :-1]) / 2

            for ii, tau in enumerate(self.io.time_heterogeneity.taus.astype(int)):
                demo_hist_truncated = DemographicHistory2StatsModelAdapter(demo_hist.time[tau:-1] - demo_hist.time[tau],
                                                                           demo_hist.n[tau:])
                prediction_at_border = self.stats_model._predict_integrand(u_border, demo_hist_truncated) * \
                    self.io.time_heterogeneity.offset_correction_at_border[ii].reshape((1, -1))

                prediction_at_center = self.stats_model._predict_integrand(u_center, demo_hist_truncated) * \
                    self.io.time_heterogeneity.offset_correction_at_center[ii].reshape((1, -1))

                predictions_for_tau = 1. / 6 * (1 * prediction_at_border[:, 1:] +
                                                4 * prediction_at_center +
                                                1 * prediction_at_border[:, :-1]
                                                )
                predictions += self.io.time_heterogeneity.tau_density[ii] \
                    * predictions_for_tau.reshape((1, -1))
            return predictions.ravel()

    def bootstrap(self, ne_ini, sig2):
        """
        Sample chromosome arms with replacements to get an approximate CI for Ne
        """
        ne_boot = np.zeros((self.nb_bootstraps, self.t_max - 1), dtype=int)
        update_times = self.config.getboolean("CONFIG", "bootstrap_time_update", fallback=True)
        for ii in tqdm(range(self.nb_bootstraps)):
            mu_boot, indices = self.io.bootstrap(return_indices=True)
            self.stats_model.set_dataset(indices)
            sig2_boot = sig2 * 10.**np.random.uniform(0, 1)
            ne_hat, _ = self.argmin_loss(ne_ini, sig2_boot, mu_boot)
            ne_boot[ii, :] = self.n2demohist(ne_hat).n.ravel()[:self.t_max - 1]
            if update_times:
                self.update_times(ne_hat)

        self.stats_model.set_dataset(np.arange(self.io.nb_regions()))
        return ne_boot

    def save_results(self, n_boot: np.ndarray):
        output_folder = self.get_output_folder()
        self.save_hapne(n_boot, output_folder)
        self.plot_goodness_of_fit(np.median(n_boot, axis=0), output_folder)
        with open(output_folder / "config.ini", "w") as f:
            self.config.write(f)
        with open(output_folder / "summary_mesages.txt", "w") as f:
            f.write(self.io.summary_message)

    def save_hapne(self, n_boot, output_folder):
        n_quantiles = np.quantile(n_boot, [0.025, 0.25, 0.5, 0.75, 0.975], axis=0).T
        times = np.arange(1, 1 + n_boot.shape[1]).reshape(-1, 1)
        to_save = np.concatenate([times, n_quantiles], axis=1).astype(int)
        df = pd.DataFrame(data=to_save, columns=["TIME", "Q0.025", "Q0.25", "Q0.5", "Q0.75", "Q0.975"])
        transformer = LogBijection()
        w = 20
        for key in ['Q0.025', 'Q0.25', 'Q0.5', 'Q0.75', 'Q0.975']:
            df[key] = smoothing(df[key], transformer, w)
        df = df.astype(int)
        df.head(self.buffer_time).to_csv(output_folder / "hapne.csv", index=False)

        save_boot = self.config["CONFIG"].getboolean("save_bootstraps", fallback=False)
        if save_boot:
            np.savetxt(output_folder / "hapne.boot", n_boot)
        plot_results(self.config, save_results=True)

    def plot_goodness_of_fit(self, n, output_folder):
        self.times = np.arange(len(n))
        mu_hat = self.predict(n)
        mu = self.io.mu
        deviance_residuals = self.stats_model.residuals(mu, mu_hat)
        fig, axs = plt.subplots(1, 2, figsize=(7, 2.3))
        # plot the deviance residals against the quantiles of normal distribution
        quantiles = np.linspace(0.0, 1, deviance_residuals.shape[1] + 2)[1:-1]
        axs[0].plot(norm.ppf(quantiles), np.sort(deviance_residuals, axis=1).T, 'o--', markersize=1)
        axs[0].set_ylabel("Deviance residuals")
        axs[0].set_title("Normal Q-Q plot")
        axs[1].plot(self.io.bins[:-1], mu.T, "-", color="black", alpha=0.5)
        axs[1].plot(self.io.bins[:-1], mu_hat.T, "--", color="red", alpha=0.75)
        axs[1].set_title("Predicted vs Observed")
        axs[1].set_xlabel("Distance (cM)")
        axs[1].set_ylabel("Y (LD or Pibd)")
        for ax in axs:
            ax.spines.top.set_visible(False)
            ax.spines.right.set_visible(False)
        fig.tight_layout()
        fig.savefig(f"{output_folder}/residuals.png", dpi=300, bbox_inches="tight")

    def get_output_folder(self):
        output_folder = PurePath(self.config.get("CONFIG", "output_folder")) / PurePath("HapNe")
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        return output_folder
