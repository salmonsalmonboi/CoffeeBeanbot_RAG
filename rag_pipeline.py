# rag_pipeline.py
# Thin re-export shim so 03_app.py can do:
#   from rag_pipeline import initialize_pipeline, ask_beanbot
#
# All actual logic lives in 02_rag_pipeline.py

from importlib import import_module
import sys, pathlib

# Add project root to sys.path so the import resolves
sys.path.insert(0, str(pathlib.Path(__file__).parent))

_mod = import_module("02_rag_pipeline")

initialize_pipeline = _mod.initialize_pipeline
ask_beanbot = _mod.ask_beanbot
