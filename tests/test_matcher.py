#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the Matcher class
@Author: Mads Jeppesen
@Date: 11/21/22
"""

def test_matcher():
    from cubicsym.alphafold.matcher import Matcher
    from pyrosetta import pose_from_file, init
    import math
    init()
    matcher = Matcher()

    # The results were checked with pymol (see more here: https://pymolwiki.org/index.php/Rms)
    # commands to run:
    #   sele onto, pose and n. CA and (chain A or chain B)
    #   sele from, ref and n. CA and (chain A or chain E)
    #   alter all,segi=""
    #   alter all,chain=""
    #   rms_cur from, onto
    #   rms from, onto
    # rms does fitting and rms_cur does not.
    # rms_cur was matched with results from matcher.CA_rmsd_no_super
    # rms was matched with results from rosettas CA_rmsd function

    ##############################
    # Test w. sequence alignment #
    ##############################

    # 4ZOR
    pose = pose_from_file("/home/mads/production/afoncubic/final/I/031122/output/4ZOR_2_1/relaxed_model_1_multimer_v2_pred_0.pdb")
    pose_ref = pose_from_file("/home/shared/databases/SYMMETRICAL/I/idealized/crystal_repr/native/4ZOR_crystal.pdb")
    chain_matches, alignment, tmscore, rmsd = matcher.apply(pose, pose_ref, pose_chains=(1,2), pose_ref_chains=(2,6), move_poses=True)
    rmsd_no_align = matcher.CA_rmsd_no_super(pose, pose_ref, *matcher.get_resis_from_chain_matches(chain_matches, alignment))
    # i cant do alignment in pymol beacuase chain dont match so I just eye balled it and it looks correctish
    assert math.isclose(rmsd_no_align,  1.001, abs_tol=1e-3)
    assert math.isclose(rmsd, 0.986, abs_tol=1e-3)


    ############################################################################################
    # chain 1, 2 are closest to 4, but this tests that unique chains are matched to each other #
    ############################################################################################

    # 1VEI
    pose = pose_from_file("/home/mads/production/afoncubic/final/T/031122/output/1VEI_3_1/relaxed_model_2_multimer_v2_pred_0.pdb")
    pose_ref = pose_from_file("/home/shared/databases/SYMMETRICAL/T/idealized/crystal_repr/native/1VEI_crystal.pdb")
    chain_matches, alignment, tmscore, rmsd = matcher.apply(pose, pose_ref, pose_chains=(1,2), pose_ref_chains=(1,4), move_poses=True)
    rmsd_no_align = matcher.CA_rmsd_no_super(pose, pose_ref, *matcher.get_resis_from_chain_matches(chain_matches, alignment))
    assert math.isclose(rmsd_no_align,  30.647, abs_tol=1e-3)
    assert math.isclose(rmsd, 23.846, abs_tol=1e-3)

    ##############
    # GOOD RMSDs #
    ##############

    # 1VEI
    pose = pose_from_file("/home/mads/production/afoncubic/final/T/031122/output/1VEI_3_1/relaxed_model_2_multimer_v2_pred_0.pdb")
    pose_ref = pose_from_file("/home/shared/databases/SYMMETRICAL/T/idealized/crystal_repr/native/1VEI_crystal.pdb")
    chain_matches, alignment, tmscore, rmsd = matcher.apply(pose, pose_ref, pose_chains=(1,2), pose_ref_chains=(2,4), move_poses=True)
    rmsd_no_align = matcher.CA_rmsd_no_super(pose, pose_ref, *matcher.get_resis_from_chain_matches(chain_matches, alignment))
    assert math.isclose(rmsd_no_align, 2.279, abs_tol=1e-3)
    assert math.isclose(rmsd,  1.628, abs_tol=1e-3)

    # 2CC9
    pose = pose_from_file("/home/mads/production/afoncubic/final/T/031122/output/2CC9_3_1/relaxed_model_4_multimer_v2_pred_0.pdb")
    pose_ref = pose_from_file("/home/shared/databases/SYMMETRICAL/T/idealized/crystal_repr/native/2CC9_crystal.pdb")
    chain_matches, alignment, tmscore, rmsd = matcher.apply(pose, pose_ref, pose_chains=(1, 2, 3), pose_ref_chains=(1, 2, 3), move_poses=True)
    rmsd_no_align = matcher.CA_rmsd_no_super(pose, pose_ref, *matcher.get_resis_from_chain_matches(chain_matches, alignment))
    assert math.isclose(rmsd_no_align,   0.410, abs_tol=1e-3)
    assert math.isclose(rmsd, 0.400 , abs_tol=1e-3)

    #############
    # BAD RMSDs #
    #############

    # 2CC9
    pose = pose_from_file("/home/mads/production/afoncubic/final/T/031122/output/2CC9_3_1/relaxed_model_4_multimer_v2_pred_0.pdb")
    pose_ref = pose_from_file("/home/shared/databases/SYMMETRICAL/T/idealized/crystal_repr/native/2CC9_crystal.pdb")
    chain_matches, alignment, tmscore, rmsd = matcher.apply(pose, pose_ref, pose_chains=(1, 2, 3), pose_ref_chains=(1, 4, 7), move_poses=True)
    rmsd_no_align = matcher.CA_rmsd_no_super(pose, pose_ref, *matcher.get_resis_from_chain_matches(chain_matches, alignment))
    assert math.isclose(rmsd_no_align, 23.404, abs_tol=1e-3)
    assert math.isclose(rmsd, 23.277, abs_tol=1e-3)

    # 2CC9
    pose = pose_from_file("/home/mads/production/afoncubic/final/T/031122/output/2CC9_3_1/relaxed_model_4_multimer_v2_pred_0.pdb")
    pose_ref = pose_from_file("/home/shared/databases/SYMMETRICAL/T/idealized/crystal_repr/native/2CC9_crystal.pdb")
    chain_matches, alignment, tmscore, rmsd = matcher.apply(pose, pose_ref, pose_chains=(2,3), pose_ref_chains=(1,5), move_poses=True)
    rmsd_no_align = matcher.CA_rmsd_no_super(pose, pose_ref, *matcher.get_resis_from_chain_matches(chain_matches, alignment))
    assert math.isclose(rmsd_no_align, 17.956, abs_tol=1e-3)
    assert math.isclose(rmsd, 16.025, abs_tol=1e-3)

    # 1VEI
    pose = pose_from_file("/home/mads/production/afoncubic/final/T/031122/output/2CC9_3_1/relaxed_model_4_multimer_v2_pred_0.pdb")
    pose_ref = pose_from_file("/home/shared/databases/SYMMETRICAL/T/idealized/crystal_repr/native/2CC9_crystal.pdb")
    chain_matches, alignment, tmscore, rmsd = matcher.apply(pose, pose_ref, pose_chains=(1, 2), pose_ref_chains=(1, 5), move_poses=True)
    rmsd_no_align = matcher.CA_rmsd_no_super(pose, pose_ref, *matcher.get_resis_from_chain_matches(chain_matches, alignment))
    assert math.isclose(rmsd_no_align,  17.953, abs_tol=1e-3)
    assert math.isclose(rmsd, 16.023, abs_tol=1e-3)

    ###################################################################
    # Test that if we dont move the chains the rmsds are as we expect #
    ###################################################################

    # 1VEI
    pose = pose_from_file("/home/mads/production/afoncubic/final/T/031122/output/1VEI_3_1/relaxed_model_2_multimer_v2_pred_0.pdb")
    pose_ref = pose_from_file("/home/shared/databases/SYMMETRICAL/T/idealized/crystal_repr/native/1VEI_crystal.pdb")
    chain_matches, alignment, tmscore, rmsd = matcher.apply(pose, pose_ref, pose_chains=(1,2), pose_ref_chains=(1,5), move_poses=False)
    rmsd_no_align = matcher.CA_rmsd_no_super(pose, pose_ref, *matcher.get_resis_from_chain_matches(chain_matches, alignment))
    assert math.isclose(rmsd_no_align, 35.975, abs_tol=1e-3)
    assert math.isclose(rmsd, 22.488, abs_tol=1e-3) # Ca_RMSD is the same as it does optimization






