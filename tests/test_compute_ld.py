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

from hapne.ld import compute_ld_in_parallel, compute_cc_quantities_in_parallel, create_cc_file
from hapne.ld import compute_manhattan_ld_in_parallel
from configparser import ConfigParser
import pandas as pd


def test_ld():
    config = ConfigParser()
    config.read("tests/files/const.ini")
    compute_ld_in_parallel(0, config)
    ld = pd.read_csv("tests/files/const/LD/chr1.from752721.to121475791.r2")
    pibd = 1. / (1 + 2 * ld["BIN_TO[M]"] * 20_000)
    assert (ld["R2"] - pibd).mean() < 1e-4


def test_manhattan_ld():
    config = ConfigParser()
    config.read("tests/files/const.ini")
    compute_manhattan_ld_in_parallel(0, config)


def test_ccld():
    config = ConfigParser()
    config.read("tests/files/const.ini")
    create_cc_file(config)
    compute_cc_quantities_in_parallel(0, config)
