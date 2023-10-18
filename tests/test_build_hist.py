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
from hapne.ibd import build_hist_in_parallel
import pandas as pd
import numpy as np


def test_build_hist():
    config = ConfigParser()
    config.read("tests/files/const_hapne_ibd.ini")
    build_hist_in_parallel(0, config)
    converted_files = pd.read_csv(
        "tests/files/const_test_ibd/IBD/chr1.from752721.to121475791.ibd.hist",
        sep="\t", header=None)
    answer = pd.read_csv("tests/files/const_test_ibd/test.ibd.hist",
                         sep="\t", header=None)
    assert np.sum(np.abs(converted_files[2] - answer[2]).values) == 0


if __name__ == "__main__":
    test_build_hist()
