# -*- coding: utf-8 -*-
"""兼容旧入口：复用唯一的阈值回放与按教材留一实现。"""
import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(HERE, "calib_vecgate.py"), run_name="__main__")
