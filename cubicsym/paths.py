#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paths to different databases + IP addres to server
@Author: Mads Jeppesen
@Date: 4/7/22
"""
from pathlib import Path
DATA = Path(__file__).parent.joinpath("data")
assert DATA.exists(), "data folder does not exist!"
