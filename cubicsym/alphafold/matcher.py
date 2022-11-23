#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for Matcher
@Author: Mads Jeppesen
@Date: 11/18/22
"""
import numpy as np
from pyrosetta.rosetta.core.scoring import CA_rmsd
from shapedesign.src.utilities.alignment import tmalign
from pyrosetta.rosetta.std import map_unsigned_long_unsigned_long # core::Size, core::Size


class Matcher:
    """After some experimentation it seems like both of the functions CA_rmsd and tmalign aligns correctly but bases their score
    (RMSD and TMscore) on matching chains. Therefor the correct mapping between chains should be applied. This class does this in
    its apply function."""

    def get_resis_from_chain(self, pose, chain: int):
        return list(range(pose.chain_begin(chain), pose.chain_end(chain) + 1))

    def get_resi_combos_from_pose(self, pose, int_combo):
        return [i for ii in [list(range(pose.chain_begin(j), pose.chain_end(j) + 1)) for j in int_combo] for i in ii]

    def construct_map_from_resis(self, resis, resis_ref):
        m = map_unsigned_long_unsigned_long()
        for resi, resi_ref in zip(resis, resis_ref):
            m[resi] = resi_ref
        return m

    def CA_rmsd_no_super(self, pose1, pose2, resis1, resis2):
        """Calcuates the CA rmsd between pose1 and pose2 using resis1 and resis2 without doing alignment."""
        dist, n = 0, 0
        for resi1, resi2 in zip(resis1, resis2):
            if pose1.residue(resi1).is_protein() and pose2.residue(resi2).is_protein():
                ca1 = pose1.residue(resi1).xyz("CA")
                ca2 = pose2.residue(resi2).xyz("CA")
                dist += ca1.distance(ca2) ** 2 # Alternatively np.linalg.norm(ca2 - ca1)
                n += 1
        return np.sqrt(dist / n)

    def get_chain_matches(self, pose1, pose2, chain_set1=None, chain_set2=None):
        """Returns a list of tuple of chain pairs that are closest to each between pose1 and pose2. If not all chains are to be considered,
        chain_set1 and chain_set2 can be specified. These must contain a list of rosetta chain numbers to be considered for either poses.
        """
        if chain_set1 is None:
            chain_set1 = list(range(1, pose1.num_chains() + 1))
        if chain_set2 is None:
            chain_set2 = list(range(1, pose2.num_chains() + 1))
        rmsd_map = {} #{k1: {k2: None for k2 in chain_set2} for k1 in chain_set1}
        for chain1 in chain_set1:
            chain_1_resis = self.get_resis_from_chain(pose1, chain1)
            for chain2 in chain_set2:
                chain_2_resis = self.get_resis_from_chain(pose2, chain2)
                rmsd_map[(chain1, chain2)] = self.CA_rmsd_no_super(pose1, pose2, chain_1_resis, chain_2_resis)
        best_rmsd_map = [k for k,v in sorted(rmsd_map.items(), key = lambda k: k[1])] #[(a, min(b, key=b.get)) for a, b in rmsd_map.items()]
        # now we have to pick from the best until
        chain_matches = []
        for match in best_rmsd_map:
            if match[0] not in [i[0] for i in chain_matches]:
                if match[1] not in [i[1] for i in chain_matches]:
                    chain_matches.append(match)
        return chain_matches

    def get_resis_from_chain_matches(self, pose, pose_ref, chain_match):
        resis1, resis2 = [], []
        for (chain1, chain2) in chain_match:
            resis1 += self.get_resis_from_chain(pose, chain1)
            resis2 += self.get_resis_from_chain(pose_ref, chain2)
        return resis1, resis2

    def apply(self, pose, pose_ref, pose_chains=None, pose_ref_chains=None, move_poses=False):
        """Aligns pose on pose_ref with TMalign and calculates a chain independent TMscore and RMSD (which Rosetta does not do).
        Can optionally specify chains to move. By default all chains will Not move. If the chains are a subset of the full chains the
        specified chains will move independenly of the others and the geometrical orientation between chains of the original pose is lost.
        If you want to move the pose set move_poses=True. Under the hoood the poses will be deepcopied.
        """

        # initial alignment without specifying chains. Tmscore will not be good since it is based on the matches to the same chains IDs
        if move_poses is False:
            pose = pose.clone()
            pose_ref = pose_ref.clone()
        if pose_chains is None and pose_ref_chains is None:
            _ = tmalign(pose, pose_ref)
        else:
            assert pose_chains is not None and pose_ref_chains is not None
            assert len(pose_chains) == len(pose_ref_chains), "chain lengths must match" # this is important to have correct matching chains
            initial_pose_resi = self.get_resi_combos_from_pose(pose, pose_chains)
            initial_pose_ref_resi = self.get_resi_combos_from_pose(pose_ref, pose_ref_chains)
            _ = tmalign(pose, pose_ref, initial_pose_resi, initial_pose_ref_resi)
        # now we align again, now specifying the correct chain mapping. If chains are already specifed as arguments these are used.
        chain_matches = self.get_chain_matches(pose, pose_ref, chain_set1=pose_chains, chain_set2=pose_ref_chains)
        pose_resi, pose_ref_resi = self.get_resis_from_chain_matches(pose, pose_ref, chain_matches)
        tmscore = tmalign(pose, pose_ref, pose_resi, pose_ref_resi)
        rmsd = CA_rmsd(pose_ref, pose, self.construct_map_from_resis(pose_ref_resi, pose_resi))
        return chain_matches, tmscore, rmsd
