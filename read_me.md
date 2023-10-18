# HapNe
Haplotype-based inference of recent effective population size in modern and ancient DNA samples

## Summary 
1. Prerequisites 
2. HapNe-LD
3. HapNe-IBD
4. Analyses of ancient samples 
5. How to cite

## 1. Prerequisites
Some pre-processing features require plink1.9 and plink2 to be installed. 
HapNe assumes that the commands `plink` and `plink2` work in the terminal.

All functionalities have been tested on macOS and Linux within the following conda environment: 

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
We strongly encourage to install HapNe within this environment by running: 
`conda env create --file conda_environment.yml`

## 2. HapNe-LD
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
* keep (facultative): samples to keep, useful to filter out relatives 
* map: path to the genetic maps
* pseudo_diploid: False for modern data, true for ancient ones
* output_folder: folder where the results will be saved
* population_name: name of the analysis
* genome_build: genome build used (grch37 (default) or grch38)

The analysis can be run using a script like this one:
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

# 3. Running HapNe-IBD
Running HapNe-IBD requires the use of an IBD detection software as a first step. We recommend using [HapIBD](https://doi.org/10.1016/j.ajhg.2020.02.010) with the postprocessing tool provided [here](https://faculty.washington.edu/browning/refined-ibd.html#gaps), following the instructions and recommendations from the publications and website. 

HapNe requires the IBD software to be run on each chromosome arm separately. It is possible to split a single vcf file into multiple files corresponding to each chromosome arm by using the following script:

```python
from hapne.convert.tools import split_vcf
split_vcf(vcf_file: str, save_in: str, keep=None, genome_build='grch37')
```
where:
* vcf_file: path to the vcf file (without the .vcf.gz extension)
* save_in: folder where the results will be saved
* keep (facultative): samples to keep in plink format, useful to filter out relatives
* genome_build: genome build used (grch37 (default) or grch38)

The method will create 39 new vcfs files in the save_in folder. You can then run HapIBD on each of these files separately, and create *.ibd.gz files following the same naming convention.

The next step consists of writing a config file following this example:
```
[CONFIG]
vcf_file=data
keep=data.keep
map=genetic_map_chr@_combined_b37.txt
pseudo_diploid=False
output_folder=HapNe/data
population_name=POP
ibd_files=OUTPUT_OF_HAPIBD
column_cm_length=9
genome_build=grch37 # or grch38
```
Where:
* column_cm_length is the index of the column containing the length information in the ibd.gz file.

Using this config file, HapNe-IBD can be run using the following script:
```python
from configparser import ConfigParser
from hapne.ibd import build_hist

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

## 4. aDNA analyses
HapNe provides a pipeline to easily study samples from the ["Allen Ancient DNA Resource" data set](https://reich.hms.harvard.edu/allen-ancient-dna-resource-aadr-downloadable-genotypes-present-day-and-ancient-dna-data).
After downloading the data, HapNe can take a file with the indices of samples to study as input (Caribbean_Ceramic_recent.keep in the following example).

Note that it is assumed that samples present in the keep file are unrelated. Kinship information is usually present in the anno file. 

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
eigen_root describes the location to the main data set, anno_file points to the annotation file, keep refers to as a file containing the indices of the individuals to study (one index per row).

The output will be written in a new output_folder folder. pseudo_diploid must be set to true when studying ancient data. Finally, population_name will be used to name the output files. 

Next, the following pipeline.py script can be run using
python pipeline.py --config_file config.ini 
 
```python
from configparser import ConfigParser
import pandas as pd
import argparse

from hapne.convert.eigenstrat2vcf import eigenstrat2vcf
from hapne.convert.eigenstrat2vcf import split_convert_vcf
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

# 5. How to cite? 

If you use this software, please cite:

R. Fournier, D. Reich, P. Palamara. Haplotype-based inference of recent effective population size in modern and ancient DNA samples. (preprint) bioRxiv, 2022.

# 6. FAQ

### HapNe-IBD
1. **I am observing wild oscillations in HapNe output**

   Genotyping or switch errors can cause IBD segments to be split into multiple smaller segments, which can bias the output of HapNe. Consider running HapNe-LD on the data if you do not trust the phasing.

2. **I get a flat output with small confidence intervals**

   Check the summary message in the output folder. This situation is generally encountered when there is no signal in the dataset, and HapNe relies on its prior (flat demographic history).

### HapNe-LD
1. **The summary message contains a warning about CCLD**

   CCLD (cross-chromosome LD) arises when there is population structure among the samples or when the population has encountered a recent admixture event. In such scenarios, HapNe-LD might be biased and output spurious collapses that should not be trusted. In modern data, consider running HapNe-IBD.

2. **I observe a collapse in the population's recent past**

   Check that there is no warning about CCLD in the summary message.
   Check that your samples are unrelated.

3. **I get a flat output with small confidence intervals**

   Check the summary message in the output folder. This situation is generally encountered when there is no signal in the dataset, and HapNe relies on its prior (flat demographic history).

# Acknowledgments  
Two scripts of the `convert` module were downloaded from the following repositories and edited to fit into this package:
- "https://github.com/mathii/pyEigenstrat" 
- "https://github.com/mathii/gdc"
