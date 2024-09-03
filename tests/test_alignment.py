def test_alignment():
    from simpletestlib.setup import setup_test
    from cubicsym.alignment import sequence_alignment_on_chain_set
    pose, _, _ = setup_test("I", "1STM")
    pose2 = pose.clone()
    sequence_alignment_on_chain_set(pose, pose2, [1, 2], [3, 4])