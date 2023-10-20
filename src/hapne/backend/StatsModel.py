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

from hapne.backend.DemographicHistory2StatsModelAdapter import DemographicHistory2StatsModelAdapter
import numpy as np
from numpy import exp, sqrt
from scipy.stats import multivariate_normal as mvn
from scipy.special import xlogy


class StatsModel:
    def __init__(self):
        self.ql_correction = 1

    def predict(self, u: np.ndarray, demographic_model: DemographicHistory2StatsModelAdapter):
        """
        Make a prediction for the quantity of interest (LD or IBD).
        The method currently assumes that we are interested in all bins (we don't handle holes yet)
        :param u: (l+1,) Points delimiting the intervals in which the quantity was computed
        :param demographic_model:
        :return: y: (l,) predictions in each interval
        """
        return self._predict(StatsModel.format_u(u), demographic_model)

    def _predict(self, u, demographic_model: DemographicHistory2StatsModelAdapter):
        pass

    def log_likelihood(self, data, prediction):
        """
        Compute the (quasi)-log-likelihood for data and predictions using the appropriate model
        :param data: nb_chromosomes x nb_bins = l array
        :param prediction:array of length l
        :param variance: array of length l
        :return: scalar l(y, y_hat)
        """
        return self._log_likelihood(data, prediction)

    def residuals(self, data, prediction):
        return self._residuals(data, prediction)

    def _log_likelihood(self, data, prediction):
        pass

    def _residuals(self, data, prediction):
        pass

    def set_dataset(self, indices):
        """
        :param indices:
        :return:
        """
        pass

    @staticmethod
    def format_u(u):
        """
        Enforce the dimensionality of u
        :param u: limits of the bins
        :return: (1, l+1)
        """
        return np.reshape(u, (1, -1))

    def set_quasi_likelihood_correction(self, correction):
        self.ql_correction = correction


class LDModel(StatsModel):
    def __init__(self, sigma):
        """
        Initialise with the bin-wise variance
        :param sigma: (nb_bins, nb_bins) covariance matrix of each bin. See paper to avoid overfitting
        """
        super().__init__()
        self.phi2 = np.diag(sigma)
        self.sigma = sigma

    def predict_smc(self, u, demo_hist):
        f = (exp(-demo_hist.time[:-1] * (2 * u + demo_hist.coal_rate)) / (2 * u + demo_hist.coal_rate)
             - exp(-demo_hist.time[1:] * (2 * u + demo_hist.coal_rate)) / (2 * u + demo_hist.coal_rate))
        return (1 - u / 2. / demo_hist.coal_rate) * f

    def predict_erlang(self, u, demo_hist):
        f = u * (
            (1 + demo_hist.time[:-1] * (2 * u + demo_hist.coal_rate)) *
            exp(- demo_hist.time[:-1] * (2 * u + demo_hist.coal_rate))
            - np.nan_to_num(1 + demo_hist.time[1:] * (2 * u + demo_hist.coal_rate)) *
            exp(- demo_hist.time[1:] * (2 * u + demo_hist.coal_rate))) \
            / (2 * u + demo_hist.coal_rate) ** 2
        return f

    def predict_alpha_correction(self, u, demo_hist):
        f = (exp(-demo_hist.time[:-1] * (2 * u + 3 * demo_hist.coal_rate)) / (2 * u + 3 * demo_hist.coal_rate)
             - exp(-demo_hist.time[1:] * (2 * u + 3 * demo_hist.coal_rate)) / (2 * u + 3 * demo_hist.coal_rate))
        return u / 2. / demo_hist.coal_rate * demo_hist.alpha * f

    def _predict_integrand(self, u, demo_hist):
        return np.sum(exp(-demo_hist.acc_coal_rate + demo_hist.time[:-1] * demo_hist.coal_rate) * demo_hist.coal_rate *
                      (self.predict_smc(u, demo_hist)
                       + self.predict_erlang(u, demo_hist)
                       + self.predict_alpha_correction(u, demo_hist)
                       ), axis=0).reshape(-1)

    def _predict(self, u, dh: DemographicHistory2StatsModelAdapter):
        """
        Compute the averaged value of the probability of being IBD within bins delimited by u_{i}, u_{i+1}
        See parent method for the description of the parameters
        :return: (l, ) array
        """
        center_bins = ((u[0, :-1] + u[0, 1:]) / 2.).reshape([1, -1])
        y_border = self._predict_integrand(u, dh)
        y_center = self._predict_integrand(center_bins, dh)
        return 1 / 6. * (y_border[1:] + 4 * y_center + y_border[:-1])

    def _log_likelihood(self, data, prediction):
        """
        Compute the log likelihood of a normal distribution (up to a constant)
        :param data: (l,)
        :param prediction: (l,)
        :return:
        """
        return mvn.logpdf(data, prediction, self.sigma, allow_singular=True).sum()

    def _residuals(self, data, prediction):
        return (data - prediction) / sqrt(self.phi2)


class IBDModel(StatsModel):

    def __init__(self, nb_pairs, length, phi2):
        """
        :param nb_pairs: number of pairs considered
        :param length :(nb_chromosomes,) length of the different independent regions in Morgan
        """
        super().__init__()
        self.nb_pairs = nb_pairs
        self.length = length
        self.current_indices = np.arange(len(self.length))
        self.phi = np.maximum(1, np.sqrt(phi2))

    def set_dataset(self, new_indices):
        """
        When we are not using all data (for example splitting test train), we need to call this method first
        TODO: ask Fergus for a better design pattern
        :param new_indices: training, validation or bootstrapped indices
        :return:
        """
        self.current_indices = new_indices.copy()

    def _predict(self, u, dh: DemographicHistory2StatsModelAdapter):
        t1 = dh.time[:-1]
        t2 = dh.time[1:]

        # SMC
        def primitive(t):
            return exp(- t * (2 * u + dh.coal_rate)) * np.nan_to_num((1 + t * (2 * u + dh.coal_rate)))

        factor = (2 * dh.coal_rate * exp(-dh.acc_coal_rate + t1 * dh.coal_rate))

        interval_contributions = factor * (primitive(t1) - primitive(t2)) / (2 * u + dh.coal_rate) ** 2

        # Approximated SMC'
        def primitive_smcp(t):
            g = dh.coal_rate
            a = dh.alpha
            return exp(-2 * t * u - t * g) / (2 * g) * np.nan_to_num(
                exp(-2 * t * g) * a * np.nan_to_num(3 * g + 2 * u * (2 + 2 * t * u + 3 * t * g)) / (2 * u + 3 * g) ** 2
                +
                np.nan_to_num((8 * t * u ** 3 * (-a + 2 * t * g)
                               + g ** 2 * (2 - a + 2 * t * g)
                               + 2 * u * g * (3 + t * g) * (2 - a + 2 * t * g)
                               - 8 * u ** 2 * (a + t * (a * g - g * (3 + 2 * t * g)))) / (2 * u + g) ** 3)
            )

        interval_contributions_smcp = 0.5 * factor * (primitive_smcp(t1) - primitive_smcp(t2))
        segment_gt_u = np.sum(interval_contributions + interval_contributions_smcp, axis=0) * (
            self.length[self.current_indices])

        return -np.diff(segment_gt_u) * self.nb_pairs

    def _log_likelihood(self, data, prediction):
        """
        Compute the log likelihood of a poisson distribution
        :param data: (l,)
        :param prediction: (l,)
        :return:
        """
        return -(self._residuals(data, prediction) ** 2).sum()

    def _residuals(self, data, prediction):
        return np.sign(data - prediction) * \
            np.sqrt(2 * (xlogy(data, data / prediction) - (data - prediction))) / self.phi
