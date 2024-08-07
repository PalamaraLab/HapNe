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

import hapne.convert.mathii_scripts.py3Eigenstrat as pyEigenstrat
import os
import pandas as pd
from hapne.utils import get_region, get_regions


def eigenstrat2vcf(config: ConfigParser) -> None:
    """
    Convert Eigenstrat files to VCF format.
    :param config: ConfigParser object with the following sections:
    """
    # Get the config parameters
    eigen_root = config.get("CONFIG", "eigen_root")
    vcf_file = get_vcf_location(config) + ".vcf"
    keep = config.get("CONFIG", "keep", fallback=None)

    if keep is None:
        inds = None
    else:
        inds = pd.read_csv(keep, header=None)[0].tolist()

    conversion = {2: "0/0", 1: "0/1", 0: "1/1", 9: "./."}

    data = pyEigenstrat.load(eigen_root, inds=inds)
    header = "##fileformat=VCFv4.0\n" \
             "##source=eigenstrat2vcf.py\n" \
             "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n" \
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(data.ind["IND"])

    with open(vcf_file, "w") as f:
        f.write(header)
        for ii, d in enumerate(data):
            this_snp = data.snp[ii]
            line = "\t".join([this_snp["CHR"], str(this_snp["POS"]), this_snp["ID"],
                              this_snp["REF"], this_snp["ALT"], "100", "PASS", ".", "GT"])
            line = line + "\t" + "\t".join([conversion[x] for x in d])
            f.write("\n" + line)
    command = f"plink --vcf {vcf_file} \
        --export vcf \
        --const-fid \
        --out {get_vcf_location(config)}_tmp \
        --mind 0.8 \
        --geno 0.8 "
    os.system(command)
    # execute the gzip command to compress the file
    os.system(f"mv {get_vcf_location(config)}_tmp.vcf {vcf_file}")
    os.system("gzip " + vcf_file)

    # create the map from the .snp file
    save_maps_in = config.get("CONFIG", "output_folder") + "/DATA"
    for _, region in get_regions(config.get('CONFIG', 'genome_build', fallback='grch37')).iterrows():
        chr = region["CHR"]
        chr_from = region["FROM_BP"]
        chr_to = region["TO_BP"]

        bim = pd.read_csv(eigen_root + ".snp", header=None, delim_whitespace=True)
        bim = bim[bim[1] == chr]
        map = bim[[3, 2, 2]]
        map.columns = ["position", "COMBINED_rate(cM/Mb)", "Genetic_Map(cM)"]
        map.loc[:, "Genetic_Map(cM)"] = map.loc[:, "Genetic_Map(cM)"] * 100
        map.loc[:, "COMBINED_rate(cM/Mb)"] = map.loc[:, "COMBINED_rate(cM/Mb)"] * 100
        map.to_csv(save_maps_in + f"/chr{chr}.from{chr_from}.to{chr_to}.map", sep=" ", index=False)


def get_vcf_location(config: ConfigParser) -> str:
    """
    return the path where the vcf file must be saved/loaded
    Note that the extension .vcf(.gz) is not added
    """
    vcf_file = config.get("CONFIG", "vcf_file", fallback=None)
    if vcf_file is None:
        output_folder = config.get("CONFIG", "output_folder") + "/DATA"
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        population = config.get("CONFIG", "population_name", fallback="population")
        vcf_file = output_folder + f"/{population}"
    return vcf_file


def get_map_location(config: ConfigParser, fallback: str):
    location = config.get("CONFIG", "map", fallback=fallback)
    return location


def split_convert_vcf(config: ConfigParser) -> None:
    """
    Split the VCF file into multiple bed files.
    :param config: ConfigParser object with the following sections:
    """
    # Get the config parameters
    for ii in range(39):
        split_convert_vcf_in_parallel(ii, config)


def split_convert_vcf_in_parallel(index: int, config: ConfigParser):
    # check if ii in between 0 and 38
    if not (0 <= index < 39):
        raise ValueError(f"index must be between 0 and 38, but index={index}")

    save_in = config.get("CONFIG", "output_folder") + "/DATA/GENOTYPES"
    if not os.path.isdir(save_in):
        os.makedirs(save_in)

    region = get_region(index, config.get('CONFIG', 'genome_build', fallback='grch37'))
    chr = region["CHR"]
    chr_from = region["FROM_BP"]
    chr_to = region["TO_BP"]

    eigen_root = config.get("CONFIG", "eigen_root", fallback=None)
    keep = None
    keep_command = ""
    # The filtering option is carried out when converting eigenstrat files to vcf
    if eigen_root is None:
        keep = config.get("CONFIG", "keep", fallback=None)
    if keep is not None:
        keep_command = f"--keep {keep}"
    # run plink to convert
    vcf_loc = get_vcf_location(config)
    fallback = config.get("CONFIG", "output_folder") + f"/DATA/chr@.from{chr_from}.to{chr_to}.map"
    map_loc = get_map_location(config, fallback)
    command = f"plink --vcf {vcf_loc}.vcf.gz \
            --make-bed \
            {keep_command} \
            --out {save_in}/{region['NAME']} \
            --cm-map {map_loc} \
            --chr {chr} \
            --from-bp {chr_from} \
            --to-bp {chr_to} \
            --const-fid \
            --threads 1 \
            --memory 2048 \
            --maf 0.249 \
            --snps-only \
            --geno 0.8 "
    os.system(command)


def vcf2fastsmc(config: ConfigParser) -> None:
    """
    Split the VCF file into multiple haps files, ready to be analysed by FastSMC
    :param config: ConfigParser object
    """
    for ii in range(39):
        vcf2fastsmc_in_parallel(ii, config)


def vcf2fastsmc_in_parallel(index: int, config: ConfigParser) -> None:
    # check if ii in between 0 and 38
    if not (0 <= index < 39):
        raise ValueError(f"index must be between 0 and 38, but index={index}")

    save_in = config.get("CONFIG", "output_folder") + "/DATA/GENOTYPES"
    if not os.path.isdir(save_in):
        os.makedirs(save_in)

    region = get_region(index, config.get('CONFIG', 'genome_build', fallback='grch37'))
    chr = region["CHR"]
    chr_from = region["FROM_BP"]
    chr_to = region["TO_BP"]
    keep = config.get("CONFIG", "keep", fallback=None)
    if keep is not None:
        keep_command = f"--keep {keep}"

    # run plink to convert
    vcf_loc = get_vcf_location(config)
    command = f"plink2 --vcf {vcf_loc}.vcf.gz \
            --export haps bgz \
            {keep_command} \
            --out {save_in}/{region['NAME']} \
            --chr {chr} \
            --from-bp {chr_from} \
            --to-bp {chr_to} \
            --const-fid \
            --threads 1 \
            --memory 2048 \
            --maf 0.01 \
            --snps-only \
            --max-alleles 2"
    os.system(command)


def split_vcf(vcf_file: str, save_in: str, keep=None, genome_build='grch37',
              threads=1) -> None:
    """
    Split the VCF file into multiple vcf files corresponding to each chromosome arm.
    :param vcf_file: Path to the vcf fiel
    :param save_in: Path to the output folder
    :param keep: Optional path to the keep file (plink format)
    :genome_build : grch37 or grch38
    """
    if genome_build not in ['grch37', 'grch38']:
        raise ValueError("genome_build must be either grch37 or grch38")

    for ii in range(39):
        split_vcf_in_parallel(ii, vcf_file, save_in, keep, genome_build, threads)


def split_vcf_in_parallel(index: int, vcf_loc: str, save_in: str, keep=None, genome_build='grch37',
                          threads=1) -> None:
    # check if ii in between 0 and 38
    if not (0 <= index < 39):
        raise ValueError(f"index must be between 0 and 38, but index={index}")

    os.makedirs(save_in, exist_ok=True)

    region = get_region(index, genome_build)
    chr = region["CHR"]
    chr_from = region["FROM_BP"]
    chr_to = region["TO_BP"]
    if keep is not None:
        keep_command = f"--keep {keep}"
    else:
        keep_command = ""
    # run plink to convert
    command = f"plink2 --vcf {vcf_loc}.vcf.gz \
            --export vcf bgz \
            {keep_command} \
            --out {save_in}/{region['NAME']} \
            --chr {chr} \
            --from-bp {chr_from} \
            --to-bp {chr_to} \
            --const-fid \
            --threads {threads} \
            --memory 2048 \
            --maf 0.01 \
            --snps-only \
            --max-alleles 2"
    os.system(command)
