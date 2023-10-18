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

from hapne import hapne_ld, hapne_ibd
from configparser import ConfigParser
import pandas as pd
import numpy as np
import logging


def test_hapne_ld():
    config = ConfigParser()
    config.read("tests/files/const_hapne_ld.ini")
    config["CONFIG"]["nb_bootstraps"] = "10"
    hapne_ld(config)
    inferred_ne = pd.read_csv("tests/files/const_test_ld/HapNe/hapne.csv")["Q0.5"].values
    true_ne = np.ones_like(inferred_ne) * 20_000
    assert np.mean((inferred_ne - true_ne) / true_ne) < 0.01


def test_hapne_ld_sigma_hook():
    config = ConfigParser()
    config.read("tests/files/const_hapne_ld.ini")
    config["CONFIG"]["nb_bootstraps"] = "10"
    config["CONFIG"]["sigma2"] = "0.000001"
    hapne_ld(config)
    inferred_ne = pd.read_csv("tests/files/const_test_ld/HapNe/hapne.csv")["Q0.5"].values
    true_ne = np.ones_like(inferred_ne) * 20_000
    assert np.mean((inferred_ne - true_ne) / true_ne) < 0.01


def test_hapne_ld_fixed():
    config = ConfigParser()
    config.read("tests/files/const_hapne_ld_fixed.ini")
    config["CONFIG"]["nb_bootstraps"] = "10"
    hapne_ld(config)
    inferred_ne = pd.read_csv("tests/files/const_test_ld_fixed/HapNe/hapne.csv")["Q0.5"].values
    true_ne = np.ones_like(inferred_ne) * 20_000
    assert np.mean((inferred_ne - true_ne) / true_ne) < 0.01


def test_hapne_ibd_fixed():
    config = ConfigParser()
    config.read("tests/files/const_hapne_ibd_fixed.ini")
    config["CONFIG"]["nb_bootstraps"] = "10"
    hapne_ibd(config)
    inferred_ne = pd.read_csv("tests/files/const_test_ibd_fixed/HapNe/hapne.csv")["Q0.5"].values
    true_ne = np.ones_like(inferred_ne) * 20_000
    assert np.mean((inferred_ne - true_ne) / true_ne) < 0.01


def test_hapne_ibd():
    config = ConfigParser()
    config.read("tests/files/const_hapne_ibd.ini")
    config["CONFIG"]["nb_bootstraps"] = "10"
    hapne_ibd(config)
    inferred_ne = pd.read_csv("tests/files/const_test_ibd/HapNe/hapne.csv")["Q0.5"].values
    true_ne = np.ones_like(inferred_ne) * 20_000
    assert np.mean((inferred_ne - true_ne) / true_ne) < 0.01


if __name__ == "__main__":
    # set logging level to INFO
    logging.basicConfig(level=logging.INFO)
    test_hapne_ld_sigma_hook()
    # test_hapne_ibd()
