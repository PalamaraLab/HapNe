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

# Copyright 2015 Iain Mathieson
# See src/hapne/convert/mathii_scripts/LICENSE for license information

# Minor modifications of pyEigenstrat.py by Romain Fournier to make the script work with python3
###############################################################################3

################################################################################

# Class for reading packed and unpacked Eigenstrat/Ancestrymap format files.
# packedancestrymap format description by Nick Patterson below:
#
################################################################################
#
# usage:
#
# files named root.{ind,snp,geno} either packed or unpacked
#
# > data = pyEigenstrat.load("root", [pops = [], inds = [], snps = []])
# to load the data - with optionally including only certain populations
# individuals or snps
#
# > genotypes = data.geno()
# to load all the data or iterate line by line (snp by snp) without loading
# the whole file into memory:
# > for snp in data: print(snp)
#
################################################################################
# packedancestrymap format
#
#
# nind # individuals (samples)
# nsnp # snps
#
# 1.
# record len (rlen)
#
# Here is a C-fragment
#  y  =  (double) (nind * 2) / (8 * (double) sizeof (char)) ;
#   rlen  =  lround(ceil(y)) ;
#   rlen  =  MAX(rlen, 48)  ;
#
# The genotype file will contain 1 header record of rlen bytes and then
# nsnp records of genotype data.
#
# a) Header record
#
# sprintf(hdr, "GENO %7d %7d %x %x", nind, nsnp, ihash, shash)
#  wwhere ihash and shash are hash values whose calculation we don't describe hhere.
#
# b) data records
# genotype values are packed left to right across the record.
# Order
# byte 1:  (first sample, second sample, ...
# byte 2:  (fourth sample ...
#
# Values   00  =  0
#          01  =  1
#          10  =  2
#          11  =  3
# And the last byte is padded with 11 if necessary
#
# Nick 7/23
################################################################################
# imports


import numpy as np

################################################################################

# datatype definitions
dt_snp1 = np.dtype([("ID", np.str_, 16), ("CHR", np.str_, 2), ("POS", np.int32)])
dt_snp2 = np.dtype([("ID", np.str_, 16), ("CHR", np.str_, 2), ("POS", np.int32),
                    ("REF", np.str_, 1), ("ALT", np.str_, 1)])
dt_ind = np.dtype([("IND", np.str_, 32), ("POP", np.str_, 32)])

###########################################################################


def load(file_root, pops=None, inds=None, exclude_inds=None, snps=None):
    """
    Investigate the geno file, and return either a packed
    or unpacked eigenstrat object as appropriate
    """
    geno_file = open(file_root + ".geno", "rb")
    head = geno_file.read(4)
    geno_file.close()
    if head == b"GENO":
        return packed_data(file_root, pops, inds, exclude_inds, snps)
    else:
        return unpacked_data(file_root, pops, inds, exclude_inds, snps)

###########################################################################


class data():
    """
    Base class.
    """

    def __init__(self, file_root, pops=None, inds=None, exclude_inds=None, snps=None):
        """
        We expect to see files file_root.{snp,ind,geno}. the .geno
        file might be either packed or unpacked.
        """

        snp, snp_include = load_snp_file(file_root, snps)
        ind, ind_include = load_ind_file(file_root, pops, inds, exclude_inds)

        # Snp and ind data
        self.snp = snp
        self.ind = ind
        self._file_root = file_root
        self._snp_include = snp_include
        self._ind_include = ind_include

        # Genotypes might be set later, geno file used for iterator.
        self._geno = None
        self._geno_file = self.open_geno_file(file_root)
        # Which snp are we on.
        self._isnp = 0

    def __iter__(self):
        return self

    # Interface follows:

    def open_geno_file(self, file_root):
        """
        Open the genotype file.
        """
        raise NotImplementedError("Don't call the base class")

    def geno(self):
        """
        If this is called, load the whole genotype matrix, and return it
        buffer it in case we want to load it again.
        """
        raise NotImplementedError("Don't call the base class")

    def __next__(self):
        raise NotImplementedError("Don't call the base class")


###########################################################################
# END CLASS

class unpacked_data(data):
    """
    Read unpacked data
    """

    def open_geno_file(self, file_root):
        """
        Open the genotype file.
        """
        return open(file_root + ".geno", "rb")

    def geno(self):
        """
        If this is called, load the whole genotype matrix, and return it
        buffer it in case we want to load it again.
        """
        if self._geno is not None:
            return self._geno

        geno = np.genfromtxt(self._file_root + ".geno", dtype='i1', delimiter=1,
                             usecols=np.where(self._ind_include)[0])

        # If we only loaded one individual, don't drop the second dimension.
        if len(geno.shape) == 1:
            geno.shape = (geno.shape[0], 1)

        geno = geno[self._snp_include, :]
        self._geno = geno
        return geno

    def __next__(self):
        while True:
            line = next(self._geno_file)
            print(line)
            self._isnp += 1
            if self._snp_include[self._isnp - 1]:
                break

        gt = np.array(list(line[:-1]), dtype='i1')
        gt = gt[self._ind_include]
        return gt

###########################################################################
# END CLASS


class packed_data(data):
    """
    Read packed data
    """

    def open_geno_file(self, file_root):
        """
        Open the genotype file (in binary mode). Read the header.
        """
        geno_file = open(file_root + ".geno", "rb")
        header = geno_file.read(20)         # Ignoring hashes
        if header.split()[0] != b"GENO":
            raise Exception("This does not look like a packedancestrymap file")
        nind, nsnp = [int(x) for x in header.split()[1:3]]

        self._nind = nind
        self._nsnp = nsnp
        self._rlen = max(48, int(np.ceil(nind * 2 / 8)))    # assuming sizeof(char) = 1 here
        geno_file.seek(self._rlen)         # set pointer to start of genotypes
        return geno_file

    def geno(self):
        """
        If this is called, load the whole genotype matrix, and return it
        buffer it in case we want to load it again.
        """
        if self._geno is not None:
            return self._geno

        geno = np.fromfile(self._file_root + ".geno", dtype='uint8')[self._rlen:]  # without header
        geno.shape = (self._nsnp, self._rlen)
        geno = np.unpackbits(geno, axis=1)[:, :(2 * self._nind)]
        geno = 2 * geno[:, ::2] + geno[:, 1::2]
        geno = geno[:, self._ind_include]
        geno[geno == 3] = 9  # set missing values

        # If we only loaded one individual, don't drop the second dimension.
        if len(geno.shape) == 1:
            geno.shape = (geno.shape[0], 1)

        geno = geno[self._snp_include, :]
        self._geno = geno
        return geno

    def __next__(self):

        while True:
            if self._isnp >= self._nsnp:
                raise StopIteration()
            record = self._geno_file.read(self._rlen)
            self._isnp += 1
            if self._snp_include[self._isnp - 1]:
                break

        gt_bits = np.unpackbits(np.fromstring(record, dtype='uint8'))
        gt = 2 * gt_bits[::2] + gt_bits[1::2]
        gt = gt[:self._nind][self._ind_include]
        gt[gt == 3] = 9  # set missing values

        return gt

###########################################################################
# END CLASS


def load_snp_file(file_root, snps=None):
    """
    Load a .snp file into the right format.
    """
    snp_file = open(file_root + ".snp", "r")
    line = snp_file.readline()
    bits = line.split()
    snpdt = dt_snp1                     # does the snp file have the alleles in?
    snpcol = (0, 1, 3)
    if len(bits) not in [4, 6]:
        raise Exception("SNP file should have either 4 or 6 columns")
    elif len(bits) == 6:
        snpdt = dt_snp2
        snpcol = (0, 1, 3, 4, 5)

    snp_file.seek(0)
    snp = np.genfromtxt(snp_file, dtype=snpdt, usecols=snpcol)
    snp_file.close()

    include = np.ones(len(np.atleast_1d(snp)), dtype=bool)
    if snps is not None:
        include = np.in1d(snp["ID"], snps)
        snp = snp[include]

    return snp, include

###########################################################################


def load_ind_file(file_root, pops=None, inds=None, exclude_inds=None):
    """
    Load a .ind file, restricting to the union of specified
    individuals and individuals in the specified populations.
    """
    ind = np.genfromtxt(file_root + ".ind", dtype=dt_ind, usecols=(0, 2))   # ignore sex

    include = np.ones(len(ind), dtype=bool)
    if pops or inds or exclude_inds:
        include = np.zeros(len(ind), dtype=bool)
        if pops:
            include = np.in1d(ind["POP"], pops)
        if inds:
            include = np.logical_or(include, np.in1d(ind["IND"], inds))
        if exclude_inds:
            include = np.logical_and(include, ~np.in1d(ind["IND"], exclude_inds))

    ind = ind[include]
    return ind, include

###########################################################################
