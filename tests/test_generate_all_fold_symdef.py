#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test for the script generate_all_fold_symdef
@Author: Mads Jeppesen
@Date: 1/18/23
"""

def test_generate_all_fold_symdef():
    from scripts.generate_all_fold_symdef import main, create_other_symmetries
    input = "inputs/2CC9.cif"
    symdef = "inputs/2CC9.symm"
    outdir = "outputs"
    create_other_symmetries(input, symdef, outdir)


def test_generate_for_test_purposes_for_evodock():
    from scripts.generate_all_fold_symdef import main, create_other_symmetries
    input = "inputs/2CC9.cif"
    from cubicsym.paths import T
    p = T.parent.parent.joinpath("idealized")
    file = "6M8V"
    input = str(p.joinpath(f"input/native/{file}.cif"))
    symdef = str(p.joinpath(f"symdef/native/{file}.symm"))
    sym_outdir = "/home/mads/projects/evodock/inputs/symmetry_files"
    create_other_symmetries(input, symdef, sym_outdir)

