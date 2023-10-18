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

from hapne.backend.IO import LDTimeIO
import numpy as np
import pandas as pd
from configparser import ConfigParser
from hapne import hapne_ld


def test_age_samples():
    u = np.linspace(0.01, 0.1, 19)
    time_het = LDTimeIO("tests/files/samples.age", u)
    assert (time_het.age_density_in_bin[0, 5:16].sum() == 1)
    assert ((time_het.age_density_in_bin.sum(axis=1) == 1).all())


def test_binning():
    u = np.linspace(0.01, 0.1, 19)
    bin_width = 5
    time_het = LDTimeIO("tests/files/samples.age", u, bin_width=bin_width)

    times = pd.read_csv("tests/files/samples.age").values
    nb_gen = np.max(times[:, 1]) - np.min(times[:, 0]) + 1
    nb_samples = times.shape[0]
    densities = np.zeros((nb_samples, nb_gen))
    times = times - np.min(times[:, 0])
    for ii in range(nb_samples):
        densities[ii, times[ii, 0]:times[ii, 1]] = 1
        densities[ii, :] = densities[ii, :] / densities[ii, :].sum()

    nb_draws = 10000
    taus = np.zeros(nb_gen)
    delta_ts = np.zeros((nb_gen, nb_gen))
    for ii in range(nb_draws):
        ii, jj = np.random.randint(0, nb_samples, 2)
        ti = np.random.randint(times[ii, 0], times[ii, 1] + 1)
        tj = np.random.randint(times[jj, 0], times[jj, 1] + 1)
        tau = max(ti, tj)
        delta_t = abs(ti - tj)
        taus[max(ti, tj)] += 1
        delta_ts[tau, delta_t] += 1
    if taus[0] == 0:
        taus[0] += 1 / nb_draws
        delta_ts[0, 0] += 1

    taus = taus / taus.sum()
    delta_ts = delta_ts / delta_ts.sum(axis=1)[:, None]

    tau_error = np.zeros(len(time_het.tau_density))
    for ii in range(len(time_het.tau_density)):
        cumulative = taus[ii * bin_width: (ii + 1) * bin_width].sum()
        tau_error[ii] = (cumulative - time_het.tau_density[ii])
    assert (np.abs(tau_error.sum()) < 1e-5)


def test_prediction_long():
    config = ConfigParser()
    config["CONFIG"] = {
        "output_folder": "tests/files/const_ld_time_het",
        "population_name": "Const",
        "pseudo_diploid": "False",
        "age_samples": "tests/files/const_ld_time_het/offset_10.times",
        "nb_bootstraps": "10",
    }
    hapne_ld(config)
    inferred_ne = pd.read_csv("tests/files/const_ld_time_het/HapNe/hapne.csv")["Q0.5"].values
    true_ne = np.ones_like(inferred_ne) * 20_000
    assert np.mean(np.abs(inferred_ne - true_ne) / true_ne) < 0.025


def test_prediction_long_binned():
    config = ConfigParser()
    config["CONFIG"] = {
        "output_folder": "tests/files/const_ld_time_het",
        "population_name": "Const",
        "pseudo_diploid": "False",
        "age_samples": "tests/files/const_ld_time_het/offset_10.times",
        "nb_bootstraps": "10",
        "age_samples_bin_width": "4",
    }
    hapne_ld(config)
    inferred_ne = pd.read_csv("tests/files/const_ld_time_het/HapNe/hapne.csv")["Q0.5"].values
    true_ne = np.ones_like(inferred_ne) * 20_000
    # higher tolerance because of binning
    assert np.mean(np.abs(inferred_ne - true_ne) / true_ne) < 0.05


def test_prediction_short():
    config = ConfigParser()
    config["CONFIG"] = {
        "output_folder": "tests/files/const_ld_time_het_2",
        "population_name": "Const",
        "pseudo_diploid": "False",
        "age_samples": "tests/files/const_ld_time_het_2/offset_2.times",
        "nb_bootstraps": "10",
    }
    hapne_ld(config)
    inferred_ne = pd.read_csv("tests/files/const_ld_time_het_2/HapNe/hapne.csv")["Q0.5"].values
    true_ne = np.ones_like(inferred_ne) * 20_000
    assert np.mean(np.abs(inferred_ne - true_ne) / true_ne) < 0.025
