"""Nonblocking byte-range lock on Windows; flock on Unix."""

import os


def acquire(file):
    if os.name == "nt":
        import msvcrt

        # Windows permits locking beyond EOF. Do not read or initialize the
        # byte: another lease's mandatory lock would deny that I/O before we
        # can report contention as BlockingIOError.
        file.seek(0)
        try:
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError() from exc
    else:
        import fcntl

        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
