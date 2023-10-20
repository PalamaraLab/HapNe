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
from hapne.backend.HapNe import HapNe


def hapne_ld(config: ConfigParser):
    config["CONFIG"]["method"] = "ld"
    hapne = HapNe(config)
    hapne.fit()


def hapne_ibd(config: ConfigParser):
    config["CONFIG"]["method"] = "ibd"
    hapne = HapNe(config)
    hapne.fit()
