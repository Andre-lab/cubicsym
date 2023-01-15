#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test for asssembly.py
@Author: Mads Jeppesen
@Date: 7/28/22
"""

def test_show():
    from shapedesign.settings import SYMMETRICAL
    from cubicsym.assembly.assembly import Assembly
    assembly = Assembly(mmcif=SYMMETRICAL.joinpath(f"I/unrelaxed/native/6QCM.cif"), assembly_id="1",
                        rosetta_units="EB FB GB HB C D F H L O KA LA MA NA OA PA QA RA SA TA UA VA WA XA YA ZA AB BB CB DB A B E G I J K M N P Q R S T U V W X Y Z AA BA CA DA EA FA GA HA IA JA")
    assembly.set_server_proxy("http://10.8.0.10:9123")
    assembly.set_server_root_path("/Users/mads/mounts/mailer")
    assembly.show("csa_full", map_subunit_ids_to_chains=True)