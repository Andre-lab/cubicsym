

def test_model_extra_chains():
    from simpletestlib.setup import setup_test
    from cubicsym.cubicsetup import CubicSetup
    from cubicsym.actors.symdefswapper import SymDefSwapper
    from pyrosetta import get_fa_scorefxn
    sym = "I"
    for pdbid, hand in {'1STM': True, '1NQW': False, '1B5S': True, '6S44': False}.items():
        # setup
        pose_in, pmm, cmd, symdef = setup_test(name=sym, file=pdbid, mute=True, return_symmetry_file=True)
        pose_asym = setup_test(name=sym, file=pdbid, mute=True, return_symmetry_file=False, symmetrize=False, pymol=False)
        sfxn = get_fa_scorefxn()
        # cs_in = CubicSetup(symdef)
        sds = SymDefSwapper(pose_in, symdef)
        # sds.foldHF_setup._set_init_vrts()
        # pose_asym = cs_in.make_asymmetric_pose(pose_in)
        # pose_asym = sds.foldHF_setup.make_asymmetric_pose(pose_in)

        # HFfold
        pose_HF = pose_asym.clone()
        cs_HF = sds.foldHF_setup.add_extra_chains()
        cs_HF.make_symmetric_pose(pose_HF)
        sfxn.score(pose_HF)
        # cs_HF.visualize(ip="10.8.0.26")#, apply_dofs=False)
        pose_HF.pdb_info().name("HF_pose")
        pmm.apply(pose_HF)


        # 3fold
        pose_3F = pose_asym.clone()
        cs_3F = sds.fold3F_setup.add_extra_chains()
        cs_3F.make_symmetric_pose(pose_3F)
        sfxn.score(pose_3F)
        pose_3F.pdb_info().name("3F_pose")
        # pmm.apply(pose_3F)
        # cs_3F.visualize(ip="10.8.0.26")

        # 2fold
        pose_2F = pose_asym.clone()
        cs_2F = sds.fold2F_setup.add_extra_chains()
        cs_2F.make_symmetric_pose(pose_2F)
        sfxn.score(pose_2F)
        pose_2F.pdb_info().name("2F_pose")
        pmm.apply(pose_2F)
        cs_2F.visualize(ip="10.8.0.26")
