"""Run an unmodified (frozen) script on CCT: patch data locations, then execute it as __main__.
Usage: python cct_run.py <script.py> [args ...]
"""
import runpy
import sys

import cct_env

cct_env.use_cct()
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
