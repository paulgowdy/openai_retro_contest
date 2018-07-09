from Bio.Blast import NCBIWWW, NCBIXML
from Bio import SeqIO

import pandas as pd
import numpy as np
import csv

from gemstone_utils import *

import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET

data_filename = 'data/inqteldata_082217.csv'

iqt_seq_082217 = pd.read_csv(data_filename, delimiter = ',', header = 0)

sequences = iqt_seq_082217[['Sequence.data']]

seq = sequences.iat[1, 0]

print "searching"
result_handle = NCBIWWW.qblast("blastn", "nt", seq)

#print "saving"

#with open("my_blast.xml", 'w') as out_handle:
#    out_handle.write(result_handle.read())

#result_handle.close()

E_VAL_THRESH = 0.001

print "reading"
blast_record = NCBIXML.read(result_handle)

seq_hits = {}

for alignment in blast_record.alignments:

    for hsp in alignment.hsps:

        if hsp.expect < E_VAL_THRESH:

            if alignment.title not in seq_hits:

                seq_hits[alignment.title] = 1

            else:

                seq_hits[alignment.title] += 1

for j in seq_hits:

    print j, seq_hits[j]
