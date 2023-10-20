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

from hapne.backend.DemographicHistory import DemographicHistory
import numpy as np


class DemographicHistory2StatsModelAdapter(DemographicHistory):
    """
    Reformat the output of DemographicHistory so that it becomes a (n, 1) array default was (n,)
    """
    @staticmethod
    def shape_output(output):
        """
        :param output: property of the parent class
        :return: property in the correct format
        """
        return np.reshape(output, (-1, 1))

    @DemographicHistory.n.getter
    def n(self):
        return self.shape_output(super().n)

    @DemographicHistory.coal_rate.getter
    def coal_rate(self):
        return self.shape_output(super().coal_rate)

    @DemographicHistory.time.getter
    def time(self):
        return self.shape_output(super().time)

    @DemographicHistory.acc_coal_rate.getter
    def acc_coal_rate(self):
        return self.shape_output(super().acc_coal_rate)

    @DemographicHistory.dt.getter
    def dt(self):
        return self.shape_output(super().dt)
