#!/usr/bin/env python3
"""Run from exp0_2: python run.py. Or from ECML-PKDD: python -m test_method.exp0_2.main"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))
from test_method.exp0_2.main import main
main()
