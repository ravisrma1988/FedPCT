"""Run an unmodified (frozen) script on FMoW-WILDS: patch data locations, then execute it as __main__.
Usage: python fmow_run.py <script.py> [args ...]      (as cct_run.py for CCT)
"""
import runpy
import sys

import fmow_env

fmow_env.use_fmow()
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
