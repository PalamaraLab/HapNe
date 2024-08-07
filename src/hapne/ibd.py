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
import os
from hapne.utils import get_region, get_regions


def build_hist(config: ConfigParser):
    for ii in range(get_regions(config.get('CONFIG', 'genome_build', fallback='grch37')).shape[0]):
        build_hist_in_parallel(ii, config)


def build_hist_in_parallel(region_index: int, config: ConfigParser):
    region = get_region(region_index, config.get('CONFIG', 'genome_build', fallback='grch37'))
    name = region["NAME"]

    column_cm_length = config.get("CONFIG", "column_cm_length")
    ibd_folder = config.get("CONFIG", "ibd_files")
    hist_folder = get_hist_folder(config)
    # needs for on macos instead of zcat *
    command = f"for IBDFILE in `ls {ibd_folder}/{name}*.ibd.gz`" \
        + "; do " \
        + "gunzip -c $IBDFILE; " \
        + "done | " \
        + "awk -F\"\\t\" '{l=sprintf(\"%d\", 2*$" + f"{column_cm_length}" + "); c[l]++;} END{ for (i=1; i<=40; i++) " \
        + "print i/2/100 \"\\t\" (i+1)/2/100 \"\\t\" 0+c[i]; }'" \
        + f"> {hist_folder}/{name}.ibd.hist"
    os.system(command)


def get_hist_folder(config: ConfigParser):
    output_folder = config.get("CONFIG", "output_folder") + "/HIST"
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    return output_folder
