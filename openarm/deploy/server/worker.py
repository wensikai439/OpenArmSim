"""Set both cuRobo runtime paths before starting the unchanged planning worker."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import curobo.runtime as public
import curobo._src.runtime as internal
public.cache_dir = internal.cache_dir = str(ROOT / '.runtime/cache/curobo')
runpy.run_module('tools.scene_cache', run_name='__main__')
