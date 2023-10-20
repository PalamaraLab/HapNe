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

import numpy as np
from numpy import exp, log
import scipy.stats
from numba import njit


DEMO_HIST_N_MIN = 100
DEMO_HIST_N_MAX = 1e8


class DemographicHistory:
    def __init__(self, t, n):
        """
        Initialise a piece-wise constant function representing a demography.

        Errors will be raised if t and n don't contain the same number of elements
                              if t don't start with 0.

        Values of n will be clipped between DEMO_HIST_N_MIN and DEMO_HIST_N_MAX

        :param t: list of positive number starting with 0
        :param n: list of elements
        """
        self.time = np.asarray(t).ravel()
        self.n = n

    @property
    def n(self):
        """
        :return: effective population size
        """
        return self.__n

    @n.setter
    def n(self, new_n):
        if self.is_n_valid(new_n):
            self.__n = np.asarray(np.clip(new_n, DEMO_HIST_N_MIN, DEMO_HIST_N_MAX)).ravel()
            self.__coal_rate = 1. / self.__n
            self.__acc_coal_rate = self.__compute_acc_coal_rate()
            self.alpha = self.__compute_alpha()

    @property
    def coal_rate(self):
        """
        :return: coalescent rate (corresponding to the inverse of the effective population size)
        """
        return self.__coal_rate

    @coal_rate.setter
    def coal_rate(self, new_coal_rate):
        new_coal_rate = np.asarray(new_coal_rate).ravel()
        new_n = 1. / new_coal_rate
        # set n to update all private elements
        self.n = new_n

    @property
    def acc_coal_rate(self):
        return self.__acc_coal_rate

    @property
    def dt(self):
        return self.__dt

    @property
    def time(self):
        return self.__time

    @time.setter
    def time(self, t):
        if self.is_t_valid(t):
            self.__time = np.append(np.asarray(t).ravel(), np.inf)
            self.__dt = np.diff(self.__time)

    def time_quantiles(self, u_min, nb_times, min_interval_length=1, max_interval_length=np.inf, last_t=None):
        """
        Given the current piece-wise constant function, this method computes the times corresponding to the quantiles
        of the distribution of the age of IBD segments greater than u_min.
        :param u_min: IBD length threshold, the length of the smallest IBD segments detected
        :param nb_times: number of times returned
        :param min_interval_length: min length of the intervals
        :param t_max: the last value to predict
        :return : (nb_times,)
        example:
        if nb_times = 4, then the times will correspond to the following quantiles (0, 0.25, 0.5, 0.75)
        """
        # Precompute useful quantities
        zi = self.get_age_ibd_segment_normalization_factor(u_min)
        z = np.sum(zi)
        zi /= z
        cum_prob_at_interval_end = np.cumsum(zi)
        # Define the quantiles and the number of parameters (depending on whether we impose tmax or not)
        if last_t is None:
            normalisation = 1
            offset = 0
        else:
            normalisation = self.age_ibd_segment_repartition_function(last_t, zi, u_min, z)
            offset = 1

        times = np.zeros([nb_times])
        global_missed_area = 0
        nb_it = 3
        for jj in range(nb_it):
            current_quantile = 0
            current_missed_area = 0
            for ii in range(nb_times - 1 - offset):
                next_quantile = min(current_quantile + (normalisation - global_missed_area) / (nb_times - offset),
                                    0.5 * (current_quantile + 1))
                t_quantile = self._find_t_quantile(next_quantile, cum_prob_at_interval_end, zi, z, u_min)
                t_quantile, missed_area = self._handle_next_time(t_quantile, times[ii],
                                                                 min_interval_length, zi, u_min, z)

                current_missed_area += missed_area
                current_quantile = next_quantile + missed_area
                times[ii + 1] = t_quantile
            if offset == 1:
                times[-1] = max(last_t, times[-2] + min_interval_length)

            if np.abs(current_missed_area - global_missed_area) < 1e-2:
                return times
            if jj < nb_it - 1:
                global_missed_area = current_missed_area
        if offset == 1:
            times[-1] = max(last_t, times[-2] + min_interval_length)
        return times

    def _find_t_quantile(self, next_quantile, cum_prob_at_interval_end, zi, z, u_min) -> float:
        """
        compute the time corresponding to next_quantile, with the other parameters corresponding
        to precomputed quantities
        """
        # Find which interval defined in the zi correspond to the current quantile
        interval = np.argmax(cum_prob_at_interval_end > next_quantile)
        # Next, iterate within this interval to find the best fitting value
        t_quantile = min(0.5 * (self.time[interval] + self.time[interval + 1]), 1000)
        t_max = min(self.time[interval + 1], 1000)
        t_min = self.time[interval]
        q_val = np.inf

        while (t_max - t_min > 0.25):
            q_val = self.age_ibd_segment_repartition_function(t_quantile, zi, u_min, z)
            if q_val < next_quantile:
                t_min = t_quantile
                t_quantile = 0.5 * (t_quantile + t_max)
            else:
                t_max = t_quantile
                t_quantile = 0.5 * (t_quantile + t_min)
        return t_quantile

    def _handle_next_time(self, t_quantile, previous_t, dt_min, zi, u_min, z):
        """
        Compute the next time according to dt_min and return the potentially surplus of the distribution captured within
        this interval
        """
        missed_area = 0
        if t_quantile < previous_t + dt_min:
            current_quantile = self.age_ibd_segment_repartition_function(t_quantile, zi, u_min, z)
            t_quantile = previous_t + dt_min
            missed_area = (self.age_ibd_segment_repartition_function(t_quantile, zi, u_min, z) - current_quantile)[0]
        return t_quantile, missed_area

    def get_age_ibd_segment_normalization_factor(self, u_min):
        """
        Compute P(l>u_min) for the current demographic model
        :param u_min : IBD segment threshold
        :return : (nb_intervals,) the integral of P(l>u_min|t)dt over the time intervals
        note : the normalization factor P(l>u_min) is the sum of all values
        """
        # Compute the contribution of each interval to the age distribution
        primitive_from = exp(-self.time[:-1] * (2 * u_min + self.coal_rate)) * (
            1 + self.time[:-1] * (2 * u_min + self.coal_rate))
        primitive_to = exp(-self.time[1:] * (2 * u_min + self.coal_rate)) * np.nan_to_num(
            (1 + self.time[1:] * (2 * u_min + self.coal_rate)))
        primitive_to[-1] = 0  # lim t-> inf exp(-t) * t is not handled in python

        zi = 2 * exp(-self.acc_coal_rate + self.time[:-1] * self.coal_rate) * self.coal_rate * (
            primitive_from - primitive_to) / (2 * u_min + self.coal_rate) ** 2
        return zi

    def age_ibd_segment_repartition_function(self, t, zi, u_min, z):
        """
        Compute the repartition function of the age of an IBD segment at time t
        :param t: time at which we evaluate the repartition function
        :param zi: probability of being in interval i
        :param u_min: minimum length of observed ibd
        :param z: normalisation factor
        """
        interval = np.argmax(self.time > t) - 1
        cum_prob_at_interval_end = np.cumsum(zi)

        cum_p = 0
        t_start = self.time[interval]
        gamma = self.coal_rate[interval]
        acc_gamma = self.acc_coal_rate[interval]
        if interval > 0:
            cum_p += cum_prob_at_interval_end[interval - 1]

        primitive_from = exp(-t_start * (2 * u_min + gamma)) * (
            1 + t_start * (2 * u_min + gamma))
        primitive_to = exp(-t * (2 * u_min + gamma)) * (
            1 + t * (2 * u_min + gamma))
        q_val = cum_p + 2 * gamma * exp(-acc_gamma + t_start * gamma) * (primitive_from - primitive_to) / (
            2 * u_min + gamma) ** 2 / z

        return q_val

    """
    Part for the probability of healing
    """

    def __compute_alpha(self):
        """
        the alpha factor is required to compute the probability of healing (see main text),
        this function computes it. This method is meant to be called each time we set a different n.
        """
        weighted_ndiff = np.append(self.n[0], np.diff(self.n, axis=0) * np.exp(2 * self.acc_coal_rate[1:]))
        return exp(-2 * self.acc_coal_rate + 2 * self.time[:-1] * self.coal_rate) * self.coal_rate * (
            np.cumsum(weighted_ndiff).reshape([-1, 1]))

    """
    check format
    """

    @staticmethod
    def is_t_valid(t):
        """
        check if the time t meets all criteria:
            * getitem is implemented
            * starts with 0
        """
        if t[0] != 0:
            raise Exception("The time provided to a DemographicHistory object must start with 0")
        elif not np.all(np.diff(t) >= 0):
            raise Exception("The time provided to a DemographicHistory object must be increasing")
        else:
            return True

    def is_n_valid(self, n):
        """
        check if the population size provided n metts all criteria:
            * has the same length as self.t
        """
        if len(n) != len(self.__dt):
            raise Exception("n and t must have the same length")

        else:
            return True

    def __compute_acc_coal_rate(self):
        """
        Compute the integral of coalescent rate, evaluated at self.__time
        :return: integral(0, t_{i})coal_rate(t) dt
        """
        acc_coal_rate = np.roll(self.__coal_rate * self.__dt, 1)
        acc_coal_rate[0] = 0
        return np.cumsum(acc_coal_rate)


class PieceWiseExpDemographicHistory(DemographicHistory):
    def __init__(self, n, u_min, time_thresholds=None, delta_t_max=3):
        """
        create a demographic history based on exponential coefficients
        :param n: population size at the beginning of each interval
        :param u_min: smallest detected segments
        :param time_thresholds: set up the times related to the given N. If None, then they are computed using a
        constant population model with Ne = infinity.
        :param delta_t_max: maximal size of a interval in the piece-wise constant representation
        """
        self.nb_intervals = len(n)
        if time_thresholds is None:
            self.quantile_times = self.get_quantile_times(u_min, self.nb_intervals)
        else:
            self.quantile_times = time_thresholds

        self.exp_coef = self.compute_exp_coef(n)
        self.n0 = n[0]
        self.delta_t_max = delta_t_max

        t, n = self.get_piecewise_constant_function()
        super().__init__(t, n)

    @staticmethod
    def get_quantile_times(u_min, nb_intervals):
        """
        Compute time intervals such that each interval contains the same expected number of coalescent events when
        looking at IBD segments of length bigger than u_min, assuming that 1/N(t) << u_min
        :param u_min : minimum length of IBD segments
        :param nb_intervals: number of intervals
        """
        quantiles = np.linspace(0, 1, nb_intervals + 1)
        ts = scipy.stats.erlang.ppf(quantiles, 2, scale=1. / (2 * u_min))
        return ts[:-1]

    def compute_exp_coef(self, n):
        """
        Compute the exponential coefficient linking ni with ni+1
        :param n: population size at the beginning of each interval
        :return : (nb_interval-1, )
        """
        coefs = log(n[:-1] / n[1:]) / np.diff(self.quantile_times)
        return coefs

    def get_piecewise_constant_function(self):
        """
        Convert the current representation (n(t=inf); exponential coefficients), into a piece-wise constant function
        """
        return exp2pwconst(self.quantile_times, self.n0, self.exp_coef, self.delta_t_max)


@njit
def exp2pwconst(time, n0, exp_params, delta_t_max):
    """
    Convert the current representation (n(t=inf); exponential coefficients), into a piece-wise constant function
    """
    t = np.linspace(0, time[-1] - 1, int(time[-1]))
    n = np.zeros_like(t)

    n_at_border = np.zeros(shape=(len(time)))
    n_at_border[0] = n0
    dt = np.diff(time)
    for i in range(1, len(time)):
        n_at_border[i] = n_at_border[i - 1] * exp(-dt[i - 1] * exp_params[i - 1])

    last_filled_t = 0
    for ii, next_border in enumerate(time[1:]):
        next_t = int(next_border)
        n[last_filled_t:next_t] = n_at_border[ii] * exp(-np.arange(next_t - last_filled_t) * exp_params[ii])
        last_filled_t = next_t

    return t[::delta_t_max], n[::delta_t_max]
