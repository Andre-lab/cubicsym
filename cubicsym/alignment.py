#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Mads Jeppesen
@Date: 12/5/22
"""
import random
import re
from Bio.Align.Applications import MafftCommandline


def write_fasta_on_chain_set(in_file, pose1, pose2, chain_set1, chain_set2):
    with open(in_file, "w") as f:
        # first write pose 1 / chain_set2
        for chain1 in chain_set1:
            f.write(f">1_{chain1}\n")
            f.write(pose1.chain_sequence(chain1) + "\n")
        for chain2 in chain_set2:
            f.write(f">2_{chain2}\n")
            f.write(pose2.chain_sequence(chain2) + "\n")


def sequence_alignment_on_chain_set(pose1, pose2, chain_set1, chain_set2):
    # 1. get the sequence for all chains and create a fasta file:
    in_file = f"/tmp/{''.join([str(random.randint(0, 9)) for i in range(10)])}.fasta"
    write_fasta_on_chain_set(in_file, pose1, pose2, chain_set1, chain_set2)
    mafft_cline = MafftCommandline(input=in_file, clustalout=True)  # , thread=1)
    stdout, stderr = mafft_cline()
    new = stdout.split("\n")
    new.pop(0)  # remove header
    new = [i for i in new if i != ""]  # remove all empty elements
    mafft_alignment = {}
    stars = ""
    for i in new:
        if i[0] != " ":
            i = re.split('\s+', i)
            idn, string = i[0], i[1]
            if idn in mafft_alignment:
                mafft_alignment[idn] += string
            else:
                mafft_alignment[idn] = string
        else:
            # so there's 16 letters from start to sequence, including the id number:
            # 1               IVPFIRSLLMPTTGPASIPDDTLEKHTLRSETSTYNLTVGDTGSGLIVFFPGFPGSIVGA
            # 60              HYTLQSNGNYKFDQMLLTAQNLPASYNYCRLVSRSLTVRSSTLPGGVYALNGTINAVTFQ
            #                 ***********************************************************
            stars += i[16:]
    ids = ["1_" + str(chain) for chain in chain_set1] + ["2_" + str(chain) for chain in chain_set2]
    counter = {i: 0 for i in ids}
    alignment_map = {i: [] for i in ids}

    for star in stars:
        if star == " ":  # no match
            # increase counters for which there are letters, and if '-' dont.
            for i in mafft_alignment.keys():
                letter = mafft_alignment[i][counter[i]]
                if letter != "-":
                    counter[i] += 1
        # if "*" that means that that particular sequence residue matches across all chains
        # and we therefore want to put that into the alignment map
        elif star == "*":
            for i in alignment_map.keys():
                alignment_map[i].append(counter[i])
                # Now that we have processed that letter increase all counters by 1
                counter[i] += 1

    # since the sequences were from different chains, to get the correct residue number we have to multiply them by pose.chain_begin(chain)
    # this will automatically also push this to the correct 1-indexing for pose
    correct_alignment_map = {}
    for chain in chain_set1:
        idx = "1_" + str(chain)
        correct_alignment_map[idx] = [i + pose1.chain_begin(chain) for i in alignment_map[idx]]
    for chain in chain_set2:
        idx = "2_" + str(chain)
        correct_alignment_map[idx] = [i + pose2.chain_begin(chain) for i in alignment_map[idx]]

    # assert all lengths are the same
    assert all([len(correct_alignment_map[ids[0]]) == len(i) for i in correct_alignment_map.values()])
    # assert all residues are the same
    unique_sequences = set()
    for chain in chain_set1:
        idx = "1_" + str(chain)
        seq = "".join([pose1.residue(i).name1() for i in correct_alignment_map[idx]])
        unique_sequences.add(seq)
    for chain in chain_set2:
        idx = "2_" + str(chain)
        seq = "".join([pose2.residue(i).name1() for i in correct_alignment_map[idx]])
        unique_sequences.add(seq)
    assert len(unique_sequences) == 1, f"The sequence alignment does not consists of of identical chains (pose name = {pose1.pdb_info().name()})"

    return correct_alignment_map