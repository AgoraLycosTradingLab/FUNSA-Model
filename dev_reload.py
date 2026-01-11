import importlib
import sys

def reload_funsa_v2():
    mods = [m for m in sys.modules if m.startswith("funsa_v2")]
    for m in mods:
        importlib.reload(sys.modules[m])
