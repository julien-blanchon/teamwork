import sys
import traceback
import os
import time
import pdb
from pdb import Pdb
import atexit
from pathlib import Path

pdb_file: Path | None = None
last_check_time: float | None


def pdb_check():
    global pdb_file
    global last_check_time

    if pdb_file is None:
        pid = os.getpid()
        pdb_file = Path(f"/tmp/{pid}.pdb_watch")
        last_check_time = None
        print(f"`touch {pdb_file}` to enter pdb")

    try:
        file_mtime = pdb_file.stat().st_mtime
        if last_check_time is None:
            last_check_time = file_mtime
            return
    except FileNotFoundError:
        pdb_file.touch()
        last_check_time = time.time()
        return

    if file_mtime > last_check_time:
        last_check_time = file_mtime
        pdb = Pdb()
        pdb.message(f"entering pdb ({pdb_file} was modified)")
        pdb.set_trace(sys._getframe().f_back)


def pdb_exception(e: Exception):
    traceback.print_exception(e)
    print("\n>>> Uncaught exception detected. Starting PDB...\n")
    # ...then start the debugger in post-mortem mode.
    pdb.post_mortem(e.__traceback__)


def cleanup():
    if pdb_file is not None:
        pdb_file.unlink(missing_ok=True)


atexit.register(cleanup)
