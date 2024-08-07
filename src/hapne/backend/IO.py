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
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import random
from scipy.stats import chi2, ttest_1samp
import logging
from hapne.ld import get_analysis_name, get_ld_output_folder
from hapne.utils import get_regions, get_region
from os.path import exists


class DataHandler:
    """
    This class handles the access and processing of the summary statistics
    """

    def __init__(self, config: ConfigParser, suffix="", extension=""):
        """
        :param config: see example.ini in the repository to know which information are available in the config file
        """
        self.output_folder = str(config["CONFIG"]["output_folder"]).strip("'")
        self.config = config

        self.genome_split = get_regions(config.get('CONFIG', 'genome_build', fallback='grch37'))

        self.nb_chromosomes = self.genome_split.shape[0]  # legacy

        self.u_min = config['CONFIG'].getfloat('u_min', fallback=self.get_default_u_min())
        self.u_max = config['CONFIG'].getfloat('u_max', fallback=self.get_default_u_max())

        self.suffix = suffix
        self.extension = extension

        self.sample_size = self.get_sample_size()

        message = f"Analysing {self.sample_size} "
        message += "haplotypes"
        logging.info(message)

        self._bins = self.read_bins()
        bin_from = self.apply_filter_u(self._bins[0, :].reshape(-1, 1))
        bin_to = self.apply_filter_u(self._bins[1, :].reshape(-1, 1))
        self.bins = np.append(bin_from.reshape(-1), bin_to[-1, 0]).reshape(-1)
        self._mu = None
        self.selected_regions = np.arange(self.genome_split.shape[0])

        default_loc = config["CONFIG"].get("output_folder")
        popname = config["CONFIG"].get("population_name")
        default_loc += f"/DATA/{popname}.age"
        self.time_heterogeneity_file = config['CONFIG'].get('age_samples', fallback=default_loc)
        if not exists(self.time_heterogeneity_file):
            self.time_heterogeneity_file = None
        else:
            self.time_heterogeneity_bin = config.getint("CONFIG", "age_samples_bin_width", fallback=1)

        self.apply_filter = config['CONFIG'].getboolean('filter_regions', fallback=True)
        self.summary_message = ""

    def get_default_u_min(self):
        raise Exception("Not Implemented Error", "Data Handler should not be used directly")

    @property
    def mu(self):
        return self._mu[self.selected_regions, :]

    def nb_regions(self):
        return len(self.selected_regions)

    def read_bins(self):
        raise Exception("Not Implemented Error", "Data Handler should not be used directly")

    def read_file(self, region_index, column):
        raise Exception("Not Implemented Error", "Data Handler should not be used directly")

    def apply_filter_u(self, data):
        """
        Remove the data that are not between self.u_min and self.u_max
        :param data: nd array nb_bins x any columns
        """
        bins_from = self._bins[0, :]
        bins_to = self._bins[1, :]

        # find the index that are in the desired range
        start_from = np.argmax(bins_from >= self.u_min)

        if np.max(bins_to) < self.u_max:
            end_at = len(bins_to)
        else:
            end_at = np.argmax(bins_to >= self.u_max) + 1

        return data[start_from:end_at, :]

    def load_data(self, column, apply_filter_u=True):
        """
        Load and concatenate data from all chromosomes files.
        :param col_name: name or number of the column in the data file
        :param apply_filter_u: bool True if we want to filter data outside the desired bin range
        """
        cumulated_data = []

        for region in range(1, self.nb_regions() + 1):
            data = self.read_file(region, column)
            if apply_filter_u:
                data = self.apply_filter_u(data.reshape(-1, 1))

            cumulated_data.append(data.flatten())

        return np.asarray(cumulated_data).T

    def k_fold_split(self, nb_folds=10, return_indices=False):
        """
        :return: generator (mu_train, mu_test)
        """
        kf = KFold(n_splits=nb_folds, shuffle=True)
        for train, test in kf.split(self.mu):
            if return_indices:
                yield self.mu[train, :], self.mu[test, :], train, test
            else:
                yield self.mu[train, :], self.mu[test, :]

    def bootstrap(self, return_indices=False):
        """
        return a bootstrap based on the current mu
        :return: self.mu.shape ndarray
        """
        selected_indices = random.choices(np.arange(self.nb_regions()), k=self.nb_regions())
        if return_indices:
            return self.mu[selected_indices, :], selected_indices
        else:
            return self.mu[selected_indices, :]

    def get_region_name(self, index):
        """
        Process the file describing the genomic regions
        :param index: 1-nb regions
        :return: name of the ith file
        """
        return str(self.genome_split.loc[index - 1, 'NAME']).strip()


class LDLoader(DataHandler):
    def __init__(self, config: ConfigParser):
        super().__init__(config, suffix="_", extension="r2")
        self.ld = self.load_data('R2', apply_filter_u=False)
        output_folder = get_ld_output_folder(config)
        bias_filename = output_folder / f"{get_analysis_name(config)}.ccld"
        try:
            data = pd.read_csv(bias_filename)
            region_1 = data.values[:, 0]
            region_2 = data.values[:, 1]
            # Select regions where the first 4 character do not match
            valid_rows = np.array([region_1[i][:5] != region_2[i][:5] for i in range(len(region_1))])
            self.ccld = data.values[valid_rows, 2].mean()
            self.bessel_cor = data.values[valid_rows, 4].mean()
            _, self.pval_no_admixture = ttest_1samp(data.values[valid_rows, 2].astype(float) -
                                                    data.values[valid_rows, 3].astype(float), 0)
            self.summary_message += f"CCLD: {self.ccld:.5f}. \n"
            self.summary_message += f"The p-value associated with H0 = no structure is {self.pval_no_admixture:.3f}.\n"
            self.summary_message += "If H0 is rejected, contractions in the recent past " \
                "might reflect structure instead of reduced population size."
            logging.warning(f"CCLD: {self.ccld:.5f}.")
            logging.warning(f"The p-value associated with H0 = no structure is {self.pval_no_admixture:.3f}.")
            logging.warning(
                           ("If H0 is rejected, contractions in the recent past might reflect "
                            "structure instead of reduced population size.")
            )

        except FileNotFoundError:
            logging.warning("Bias file not found, running HapNe-LD assuming no admixture LD and high coverage.")
            logging.warning(f"Expected to read ccld file in {bias_filename}")
            self.missingness = config.getfloat("CONFIG", "missingness", fallback=0)
            n_eff = self.sample_size * (1 - self.missingness)
            self.ccld = 4 / (n_eff - 1.) ** 2
            self.bessel_cor = (n_eff / (n_eff - 1.)) ** 2
            self.pval_no_admixture = np.nan

        self._mu = self.ld_to_p_ibd(self.apply_filter_u(self.ld)).T
        # Combine bins if we are dealing with a small genotype or high missingness
        # self.merge_bins_if_required()

        # Discard suspicious Regions
        self.filter_tolerance = config['CONFIG'].getfloat('filter_tol', fallback=5e-4)
        if self.apply_filter:
            self.apply_region_filter()
        logging.info(f"Analyzing {self.mu.shape[0]} regions ")

        self.phi2 = np.var(self.mu, axis=0).reshape(-1)
        self.sigma = np.cov(self.mu.T)

        if self.time_heterogeneity_file is not None:
            logging.info("Using time heterogeneity correction ")
            self.time_heterogeneity = LDTimeIO(self.time_heterogeneity_file, self.bins, self.time_heterogeneity_bin)
        else:
            logging.info("No age of samples found, assuming they originate from the same generation...")
            self.time_heterogeneity = None

    def get_sample_size(self):
        pseudo_diploid = self.config.getboolean("CONFIG", "pseudo_diploid", fallback=None)
        if pseudo_diploid is None:
            pseudo_diploid = False
            logging.warning("[CONFIG]pseudo_diploid not found in config file, assuming diploid. \n \
                            If you are analysing aDNA, set the flag to True.")
        try:
            region1 = get_region(1, self.config.get('CONFIG', 'genome_build', fallback='grch37'))
            default_loc = self.config.get("CONFIG", "output_folder") + "/DATA/GENOTYPES"
            genotypes_loc = self.config.get("CONFIG", "genotypes", fallback=default_loc)
            fam_file = pd.read_csv(f"{genotypes_loc}/{region1['NAME']}.fam", header=None, sep="\t")
            nb_individuals = fam_file.shape[0]
        except FileNotFoundError:
            nb_individuals = self.config.getint("CONFIG", "nb_individuals")
        return (2 - pseudo_diploid) * nb_individuals

    def apply_region_filter(self):
        """
        discard regions with high LD
        :return:
        """
        nb_regions = self._mu.shape[0]
        self.selected_regions = []
        for ii in range(nb_regions):
            jackknife = np.delete(self._mu, ii, axis=0)
            median = np.median(jackknife, axis=0)
            std = np.std(jackknife, axis=0, ddof=1)
            sse = np.sum((self._mu[ii, :] - median) ** 2 / std ** 2)
            ddof = self._mu.shape[1]
            pval = 1 - chi2.cdf(sse, ddof)
            if pval > self.filter_tolerance / nb_regions:
                self.selected_regions.append(ii)
            else:
                logging.warning(f"Discarding region {self.get_region_name(ii + 1)} with pval {pval:.5f}")
                self.summary_message += f"Discarding region {self.get_region_name(ii + 1)} with pval {pval:.5f}.\n"

        self.selected_regions = np.array(self.selected_regions)

    def read_bins(self):
        region = self.get_region_name(1)
        bins = pd.read_csv(f"{self.input_files_location()}/{region}.{self.extension}",
                           sep=",").values[:, 1:3].T
        return bins

    def ld_to_p_ibd(self, ld):
        """
        Convert ld + bias into probability of being IBD
        :param ld:
        """
        p_ibd = (ld - self.ccld) / (self.bessel_cor - 0.25 * self.ccld)
        # Approximated correction for the continuous-time approximation
        p_ibd *= np.exp(self.bins[:-1]).reshape((-1, 1))
        return p_ibd

    def read_file(self, region_index, column):
        region = self.get_region_name(region_index)
        filename = f"{self.input_files_location()}/{region}.{self.extension}"
        data = pd.read_csv(filename, sep=",")
        return data[column].values

    def is_admixture_significant(self):
        NotImplementedError

    def merge_bins_if_required(self):
        # Compute the average weight
        weights = self.load_data('WEIGHT', apply_filter_u=True).mean()
        # if the weight is less than 1e4, merge 2 bins into 1
        if weights < 2e4:
            bins_to_merge = 2
            nb_bins_first = self.bins.shape[0]
            nb_bins_end = (nb_bins_first - 1) // bins_to_merge + 1
            new_bins = np.zeros(nb_bins_end)
            new_mu = np.zeros([self.nb_regions(), nb_bins_end - 1])
            for ii in range(nb_bins_end - 1):
                new_bins[ii] = self.bins[bins_to_merge * ii]
                new_mu[:, ii] = 1. / bins_to_merge * (np.sum(self.mu[:, ii:(ii + bins_to_merge)], axis=1))
            new_bins[-1] = self.bins[14]

            self._mu = new_mu
            self.bins = new_bins

    def get_default_u_min(self):
        return 0.01

    def get_default_u_max(self):
        return 0.1

    def input_files_location(self):
        default_loc = self.output_folder + "/LD"
        return self.config["CONFIG"].get("ld_files", fallback=default_loc)


class IBDLoader(DataHandler):
    def __init__(self, config: ConfigParser):
        super().__init__(config, suffix=".", extension="ibd.hist")
        self.trim = config["CONFIG"].getfloat('segment_center_threshold_M', fallback=0.0)
        self._region_length = self.genome_split["LENGTH"].values / 100 - 2 * self.trim

        self._mu = self.load_data(2, apply_filter_u=True).T
        self.phi2 = np.ones(self.mu.shape[1])

        self.adjust_for_count()
        if self.apply_filter:
            self.apply_ibd_region_filter()
            self.adjust_for_count()
            logging.info(f"({self.genome_split.shape[0] - self.mu.shape[0]} were discarded)")
        logging.info(f"Last bin considered after filtering out small counts: {self.bins[-1]}")
        logging.info(f"Analyzing {self.mu.shape[0]} regions ")
        data = self._mu[self.selected_regions, :]
        mu = compute_poisson_mean(self.mu, self.region_length())
        mu = np.maximum(1. / self.region_length().sum(), mu)
        self.phi2 = np.var((data - mu) / np.sqrt(mu), axis=0, ddof=1)

        if self.time_heterogeneity_file is not None:
            logging.warning(
                "Samples with time offsets is not implemented for the IBD model yet,"
                + " falling back to no time offset")
            self.time_heterogeneity = None
        else:
            self.time_heterogeneity = None

    def adjust_for_count(self):
        mu_hat = self.mu.sum(axis=0) / self.region_length().sum()
        threshold = 1. / self.region_length().sum()
        has_hit = mu_hat >= threshold
        nb_valid_bins = np.max(np.nonzero(has_hit)) + 1
        self.bins = self.bins[:nb_valid_bins + 1]
        self._mu = self._mu[:, :nb_valid_bins]
        self.phi2 = self.phi2[:nb_valid_bins]

    def get_sample_size(self):
        return self.config["CONFIG"].getint("nb_samples")

    def read_bins(self):
        region = self.get_region_name(1)
        hist_files = self.input_files_location()
        return pd.read_csv(f"{hist_files}/{region}.{self.extension}",
                           sep="\t", header=None).values[0:, 0:2].T

    def input_files_location(self):
        default_loc = self.output_folder + "/HIST"
        return self.config["CONFIG"].get("hist_files", fallback=default_loc)

    def read_file(self, region_index, column):
        region = self.get_region_name(region_index)
        filename = f"{self.input_files_location()}/{region}.{self.extension}"
        data = pd.read_csv(filename, header=None, sep="\t").values[:, column]
        return data

    def get_default_u_min(self):
        return 0.02

    def get_default_u_max(self):
        return 0.5

    def region_length(self):
        return self._region_length[self.selected_regions]

    def apply_ibd_region_filter(self):
        """
        discard suspicious regions, assuming the data follow a negative binomial model
        :return:
        """

        def get_jackknife_residuals(data, indices, index):
            """
            Compute the residuals of genomic region index by computing the mean and
            the overdispersion parameter from all other regions.
            :param data:
            :param indices:
            :param index:
            :return:
            """
            train = np.delete(data, index, axis=0)
            validation = data[index, :]

            r_length = (self._region_length - 2 * self.trim)[indices]
            mu = compute_poisson_mean(data, r_length)
            mu_train = np.delete(mu, index, axis=0)
            phi2 = np.var((train - mu_train) / np.sqrt(mu_train), ddof=1, axis=0)

            return norm_residuals(validation, mu[index, :], phi2)

        def norm_residuals(y, mu, phi2):
            return (y - mu) / np.sqrt(mu) / np.sqrt(phi2)

        def get_jackknife_r2(data, indices, index):
            res = get_jackknife_residuals(data, indices, index)
            return np.sum(res ** 2)

        discarded = []
        n_bins = len(self.bins) - 1
        for _ in range(2):
            current_ibd = np.delete(self._mu, discarded, axis=0)
            indices = np.delete(np.arange(self.nb_regions()), discarded)

            for ii in range(len(indices)):
                sse = get_jackknife_r2(current_ibd, indices, ii)
                if min(chi2.cdf(sse, n_bins), 1 - chi2.cdf(sse, n_bins)) < 1e-12:
                    discarded.append(indices[ii])

        self.selected_regions = np.delete(self.selected_regions, discarded)


class LDTimeIO:
    def __init__(self, timefile, u=np.zeros(1), bin_width=1):
        """
        Precompute
        :param timefile: file containing the age of the samples.
        :param u : time at which we compute the time offsets
        :bin_width : Length of bins to compute the correction (the higher the faster)
        """
        u = u.reshape((1, -1))

        self.samples_age_in_gen = pd.read_csv(timefile, sep=",")
        self.samples_age_in_gen.columns = self.samples_age_in_gen.columns.str.replace(' ', '')

        self.most_recent = np.min(self.samples_age_in_gen["FROM"].values)
        self.oldest = np.max(self.samples_age_in_gen["TO"].values)
        logging.info("Reading the age of the samples")
        logging.info(f"Most recent sample: {self.most_recent} gen. bp")
        logging.info(f"Oldest sample: {self.oldest} gen. bp")

        try:
            assert self.oldest - self.most_recent > bin_width
        except AssertionError:
            logging.warning("The bin_width is too large for the data, using the smallest bin_width possible")
            bin_width = self.oldest - self.most_recent

        self.bins = np.arange(self.most_recent, self.oldest + bin_width + 1, bin_width)
        self.bin_width = bin_width
        self.middle_bins = self.bins[:-1] + bin_width / 2
        logging.info(f"HapNe will start predicting Ne from t={self.middle_bins[0]} gen bp")
        self.bins -= self.most_recent
        self.middle_bins -= self.most_recent

        self.age_density_in_bin = self._compute_age_density_in_bin()

        self.tau_density, self.dt_density = self._compute_pairwise_density()

        self.delta_ts = np.array([ii * bin_width for ii in range(len(self.middle_bins))])
        self.taus = self.middle_bins - self.middle_bins[0]

        self.offset_correction_at_border = self._compute_offset_correction(u)
        u_center = 0.5 * (u[0, 1:] + u[0, :-1]).reshape((1, -1))
        self.offset_correction_at_center = self._compute_offset_correction(u_center)

    def _compute_age_density_in_bin(self):
        """
        Convert a list of times at which individuals lived into the density of age of a ramdomly sampled individual
        :return:
        """
        # One row per samples. 1 when they are alive
        nb_samples = self.samples_age_in_gen.shape[0]
        age_density = self._compute_age_density()
        age_density_in_bin = np.zeros((nb_samples, len(self.middle_bins)))
        for ii in range(nb_samples):
            for jj in range(len(self.middle_bins)):
                age_density_in_bin[ii, jj] = np.sum(age_density[ii, jj * self.bin_width:(jj + 1) * self.bin_width])
        age_density_in_bin = age_density_in_bin / age_density_in_bin.sum(axis=1, keepdims=True)
        return age_density_in_bin

    def _compute_age_density(self):
        t_low = self.samples_age_in_gen["FROM"].values - self.most_recent
        t_high = self.samples_age_in_gen["TO"].values - self.most_recent
        nb_gen = self.oldest - self.most_recent + 1
        age_density = np.zeros((len(t_low) * 2, nb_gen))
        for ii in range(len(t_low)):
            age_density[2 * ii, t_low[ii]:t_high[ii] + 1] = 1
            age_density[2 * ii + 1, t_low[ii]:t_high[ii] + 1] = 1
        age_density = age_density / age_density.sum(axis=1, keepdims=True)
        return age_density

    def _compute_pairwise_density(self):
        nb_bins = len(self.middle_bins)
        s = self.samples_age_in_gen["FROM"].values.shape[0]
        delta_t_density_binned = np.zeros((nb_bins, nb_bins))
        n_terms = s * (s - 1)
        for tau in range(nb_bins):
            delta_t_density_binned[tau, 0] = 1 / n_terms * \
                cross_terms(self.age_density_in_bin[:, tau], self.age_density_in_bin[:, tau])
            for jj in range(0, tau):
                delta_t_density_binned[tau, tau - jj] = 1 / n_terms * \
                    2 * cross_terms(self.age_density_in_bin[:, tau], self.age_density_in_bin[:, jj])

        tau_density_binned = np.sum(delta_t_density_binned, axis=1)
        delta_t_density_binned = np.divide(delta_t_density_binned, tau_density_binned[:, None],
                                           out=np.zeros_like(delta_t_density_binned),
                                           where=(tau_density_binned[:, None] != 0))
        return tau_density_binned, delta_t_density_binned

    def _compute_offset_correction(self, u):
        delta_ts = self.delta_ts.reshape((-1, 1))
        u = u.reshape((1, -1))
        exponential_decays = np.exp(- (delta_ts - 0.75) * u)

        offset = np.zeros((len(self.taus), u.shape[1]))
        for ii in range(len(self.taus)):
            offset[ii, :] = (self.dt_density[ii].reshape((-1, 1)) * exponential_decays).sum(axis=0)
        return offset


def compute_poisson_mean(data, region_length):
    p = data.sum(axis=0) / region_length.sum()
    mu = p.reshape((1, -1)) * region_length.reshape(-1, 1)
    return mu


def cross_terms(f, g):
    return 0.5 * (np.sum(f + g, axis=0) ** 2 -
                  np.sum(f, axis=0) ** 2 - np.sum(g, axis=0) ** 2 - 2 * np.sum(f * g, axis=0))
