# HapNe
Haplotype-based inference of recent effective population size in modern and ancient DNA samples.

1. [System Requirements](#1-system-requirements)
2. [Installation Guide](#2-installation-guide)
3. [Demo](#3-demo)
4. [Instructions for use](#4-instructions-for-use)
    - [4.1 HapNe-LD](#41-hapne-ld)
    - [4.2 HapNe-IBD](#42-hapne-ibd)
    - [4.3 Analyses of ancient samples](#43-analyses-of-ancient-samples)
5. [Understanding the output](#5-understanding-the-output)
6. [FAQ](#6-faq)
7. [Citation](#7-citation)
8. [Acknowledgements](#8-acknowledgements)

## 1. System Requirements
The software dependencies are listed in the conda_environment.yml and setup.cfg files. 
The software works on Unix systems (Linux and macOS).
The software has been tested on macOS (Ventura 13.3.1, M1 and Intel) and Linux (Ubuntu 22.04.3 LTS). 
It does not require non-standard hardware.

## 2. Installation Guide
If `plink` and `plink2` are already installed on your system, you can install HapNe using pip:

`pip install hapne`

However, we strongly recommend to install HapNe within a conda environment by running:

`conda env create --file conda_environment.yml`

using the following conda_environment.yml file (~ 5 minutes):

```yml
name: HapNe
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - python
  - pytest
  - numpy
  - pandas
  - plink
  - plink2
  - flake8
  - numba
```
You can then install HapNe using pip (1 minute):
`pip install hapne`

If you wish to modify the code, you can install HapNe in editable mode by running `pip install .` within this repository. 

## 3. Demo
The `tests` folder contains a demo for all functionalities of HapNe, and the expected output of each functionality.

Running `pytest tests` is typically done in less than 20 minutes. 

## 4. Instructions for use
### 4.1 HapNe-LD
HapNe-LD can be run by adapting the following config file:
```
[CONFIG]
vcf_file=data
keep=data.keep
map=genetic_map_chr@_combined_b37.txt
pseudo_diploid=False
output_folder=HapNe/data
population_name=POP
genome_build=grch37
```
* vcf_file: path to the vcf file (without the .vcf.gz extension)
* keep (optional): samples to keep, useful to filter out relatives 
* map: path to SHAPEIT-format recombination map files. The first column is the physical position (in bp), the second one is the rate (in cM/Mb) and the third one is the genetic position (in cM).
* pseudo_diploid: False for modern, true for ancient data
* output_folder: folder where the results will be saved
* population_name: name to be used for the analysis
* genome_build: genome build used (grch37 (default) or grch38)

HapNe can be run using the following Pyton script:
```python
from configparser import ConfigParser
import argparse

from hapne.convert.tools import split_convert_vcf
from hapne.ld import compute_ld, compute_ccld
from hapne import hapne_ld


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HapNe-LD pipeline')
    parser.add_argument('--config_file',
                    help='configfile')
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config_file)
    print("Starting stage 1")
    split_convert_vcf(config)
    print("Starting stage 2")
    compute_ld(config)
    compute_ccld(config)
    print("Starting stage 3")
    hapne_ld(config)
```

### 4.2 HapNe-IBD
Running HapNe-IBD requires the use of an IBD detection software as a first step. Experiments in the HapNe paper used [HapIBD](https://doi.org/10.1016/j.ajhg.2020.02.010) with the postprocessing tool provided [here](https://faculty.washington.edu/browning/refined-ibd.html#gaps), following the instructions and recommendations from the publications and website. 

As input, HapNe-IBD considers IBD length histograms that are split by chromosome arm. We provide a naive pipeline that can make it easier to get the desired outcome.
Starting from a vcf file with phased samples containing all chromosomes, the pipeline allows splitting the vcf file by chromosome arm:

```python
from hapne.convert.tools import split_vcf
split_vcf(vcf_file: str, save_in: str, keep=None, genome_build='grch37')
```
where:
* vcf_file: path to the vcf file (without the .vcf.gz extension)
* save_in: folder where the results will be saved
* keep (optional): samples to keep in plink format, useful to filter out relatives
* genome_build: genome build used (grch37 (default) or grch38)

The method will create 39 new vcfs files in the save_in folder. You can then run HapIBD on each of these files separately, and create *.ibd.gz files following the same naming convention.

For example, you can use a script like:
```
for file in ./data/*.vcf.gz
do
    CHR=$(echo $file | awk -F'chr' '{print $2}' | awk -F'.' '{print $1}')
    PREFIX=$(echo $file | awk -F'.vcf.gz' '{print $1}' | awk -F'/' '{print $NF}')
    java -jar hap-ibd.jar gt=$file map=plink.chr$CHR.GRCh38.map  out=IBD/$PREFIX
    echo "Hap-IBD done for $file"
done
```
Do not forget to merge ibd and hbd files if you use this script. It is also recommended to merge adjacent segments.

HapNe-IBD requires a config file to set the following options:
```
[CONFIG]
output_folder=output_folder
nb_samples=nb_diploid_samples_in_analysis
population_name=pop_name
ibd_files=output_folder_of_ibd_files
column_cm_length=8
genome_build=grch38
```
Where:
* column_cm_length is the index of the column containing the length (in centimorgans) for each IBD segment in the ibd.gz file.
* nb_samples is the number of diploid samples, aka the number of individuals, used in the analysis.

Using this config file, HapNe-IBD can be run using the following script:
```python
from configparser import ConfigParser
from hapne.ibd import build_hist
import argparse
from hapne import hapne_ibd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HapNe-IBD preprocessing pipeline')
    parser.add_argument('--config_file',
                    help='configfile')
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config_file)
    build_hist(config)
    hapne_ibd(config)
```

### 4.3 Analyses of ancient samples
The HapNe package contains a pipeline to easily analyze samples from the ["Allen Ancient DNA Resource" data set](https://reich.hms.harvard.edu/allen-ancient-dna-resource-aadr-downloadable-genotypes-present-day-and-ancient-dna-data).
After downloading the data, HapNe can take a file with the indices of samples to be analyzed as input (Caribbean_Ceramic_recent.keep in the following example).

Note that it is assumed that samples present in the keep file are unrelated. Kinship information is usually present in the anno file and can be used to filter out related individuals. 

To perform the analysis, create the following configuration file:

```
[CONFIG]
eigen_root=DATA/v50.0_1240k_public
anno_file=DATA/v50.0_1240k_public.anno
keep=CONFIG/Caribbean_Ceramic_recent.keep
pseudo_diploid=True
output_folder=RESULTS/Caribbean_Ceramic_recent
population_name=Caribbean_Ceramic_recent
```
eigen_root describes the location to the main data set, anno_file points to the annotation file, keep refers to a file containing the indices of the individuals to be analyzed (one index per row).

The output will be written in a new output_folder folder. pseudo_diploid must be set to true when studying ancient data. Finally, population_name will be used to name the output files. 

Next, the following pipeline.py script can be run using
python pipeline.py --config_file config.ini 
 
```python
from configparser import ConfigParser
import pandas as pd
import argparse

from hapne.convert.tools import eigenstrat2vcf
from hapne.convert.tools import split_convert_vcf
from hapne.ld import compute_ld, compute_ccld, create_cc_file
from hapne.utils import get_age_from_anno
from hapne import hapne_ld


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HapNe-LD pipeline')
    parser.add_argument('--config_file',
                    help='configfile')
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config_file)
    print("Starting stage 1")
    eigenstrat2vcf(config)
    print("Starting stage 2")
    split_convert_vcf(config)
    print("Starting stage 3")
    compute_ld(config)
    compute_ccld(config)
    print("Starting stage 4")
    get_age_from_anno(config)
    hapne_ld(config)
```

## 5. Understanding the output
HapNe creates different folders in the output folder provided in the config file. The main output is written in the HapNe folder. It contains the following files:
* `config.ini` : copy of the config file used
* `summary.txt` : summary of the analysis, with warnings cautioning about potential biases. 
* `hapne.csv` : The haploid effective population size at each generation. The csv files contains the Maximum-Likelihood estimate (MLE), as well as the estimated quantiles (0.025, 0.25, 0.5, 0.75, 0.975) obtained from the bootstrap procedure.
* `assessment.png` : visualization of the Pearson residuals of each chromosome arm. The residuals should be normally distributed, and centered around 0.
* `hapne_results.png` : A visual representation of the results. The blue line represents the MLE, and the light-shaded area represents the 95% confidence interval.

**Note: the population size produced in output by HapNe is haploid (divide by 2 to obtain diploid size estimates).**

## 6. FAQ
### General Questions
1. **Can HapNe be used to analyze non-human data?**

Our testing has been limited to human-like evolutionary models. HapNe may also work in non-human data, but we recommend using simulations to test the accuracy under other evolutionary settings. We are also updating the software to allow using non-human genome builds and welcome suggestions on how to facilitate these analyses.

2. **Can HapNe be used to infer population sizes beyond 100 generations ago?**

Depending on the sample size and the demography, HapNe might be able to infer fluctuations at times older than 100 generations. We have not extensively tested this, so we recommend running simulations to verify the accuracy under specific evolutionary parameters. The software can be used to infer Ne at times deeper than the default 100 generations by setting the `t_max` in parameter in the config file. It may also be useful to modify the number of inferred parameters, `nb_parameters`, and the maximum length of a time interval in the output, `dt_max`.

### HapNe-LD
1. **The summary message contains a warning about CCLD**

   CCLD (cross-chromosome LD) arises when there is population structure among the samples or when the population has encountered a recent admixture event. In such scenarios, HapNe-LD might be biased and output spurious collapses that should not be trusted. In modern data, consider running HapNe-IBD.

2. **I observe a collapse in the population's recent past**

   Check that there is no warning about CCLD in the summary message.
   Check that your samples are unrelated.

3. **I get a flat output with small confidence intervals**

   Check the summary message in the output folder. This situation is generally encountered when there is no signal in the dataset, and HapNe relies on its prior (flat demographic history). This scenario is flagged by a warning in the summary message. Please refer to our manuscript for additional details.

### HapNe-IBD
1. **I am observing wild oscillations in HapNe’s output**

   Genotyping or switch errors can cause IBD segments to be split into multiple smaller segments, which can bias the output of HapNe. Consider running HapNe-LD on the data if you do not trust the phasing or have reasons to believe IBD detection could be noisy.

2. **I get a flat output with small confidence intervals**

   Check the summary message in the output folder. This situation is generally encountered when there is no signal in the dataset, and HapNe relies on its prior (flat demographic history). This scenario is flagged by a warning in the summary message. Please refer to our manuscript for additional details.


## 7. Citation
If you use this software, please cite:

R. Fournier, Z. Tsangalidou, D. Reich, P. Palamara. Haplotype-based inference of recent effective population size in modern and ancient DNA samples. Nature Communications, 2023.

## 8. Acknowledgements
Two scripts of the `convert` module were downloaded from the following repositories and edited to fit into this package:
- https://github.com/mathii/pyEigenstrat
- https://github.com/mathii/gdc
