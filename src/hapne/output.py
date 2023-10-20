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

import configparser
import os
import pandas as pd
import matplotlib.pyplot as plt
from hapne.utils import get_age_samples_in_bp


def plot_results(config: configparser.ConfigParser, color="tab:blue", label="", save_results=False):
    """
    Plot the results of the HapNe analysis
    """
    hapne_results = load_hapne_results(config)

    if config.get("CONFIG", "anno_file", fallback=None) is not None:
        years_per_gen = config.get("CONFIG", "years_per_gen", fallback=29.1)
        age_from, _ = get_age_samples_in_bp(config)
        time = min(age_from) + years_per_gen * hapne_results["TIME"]
        xlabel = "Time (years bp)"
    else:
        time = hapne_results["TIME"]
        xlabel = "Time (gen.)"
    fig, ax = plt.subplots(figsize=(5, 2.3))
    ax.plot(time, hapne_results["Q0.5"], color=color, label=label)
    ax.fill_between(time, hapne_results["Q0.25"], hapne_results["Q0.75"], color=color, alpha=0.5)
    ax.fill_between(time, hapne_results["Q0.025"], hapne_results["Q0.975"], color=color, alpha=0.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$N_e$")

    ax.set_yscale("log")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    if label != "":
        ax.legend()

    if save_results:
        output_folder = config.get("CONFIG", "output_folder")
        plt.savefig(os.path.join(output_folder, "HapNe/hapne_results.png"), bbox_inches="tight")
        plt.close()
    return fig, ax


def load_hapne_results(config: configparser.ConfigParser):
    """
    Load the results of the HapNe analysis
    """
    output_folder = config.get("CONFIG", "output_folder")
    hapne_results = pd.read_csv(os.path.join(output_folder, "HapNe/hapne.csv"))
    return hapne_results
