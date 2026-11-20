"""Keep a guarded command in its owned process group if its parent disappears."""

import argparse
import os
import signal
import subprocess
import threading
from contextlib import AbstractContextManager


class WindowsProcess:
    """Retain the kernel process object, whose lifetime survives PID reuse."""

    def __init__(self, pid):
        import ctypes
        from ctypes import wintypes

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.api.OpenProcess.restype = wintypes.HANDLE
        self.api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.api.WaitForSingleObject.restype = wintypes.DWORD
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
            ctypes.POINTER(wintypes.FILETIME)
        ] * 4
        # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION.
        self.handle = self.api.OpenProcess(0x00100000 | 0x1000, False, pid)
        if not self.handle:
            raise ProcessLookupError(pid)
        times = [wintypes.FILETIME() for _ in range(4)]
        if not self.api.GetProcessTimes(self.handle, *(ctypes.byref(t) for t in times)):
            self.close()
            raise ProcessLookupError(pid)
        self.birth = str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)

    def alive(self):
        # WAIT_TIMEOUT is the only result proving the process is still alive.
        return self.api.WaitForSingleObject(self.handle, 0) == 258

    def close(self):
        self.api.CloseHandle(self.handle)


class PosixProcess:
    def __init__(self, pid):
        import psutil

        self.process = psutil.Process(pid)
        self.birth = str(self.process.create_time())

    def alive(self):
        import psutil

        try:
            return self.process.is_running() and self.process.status() != "zombie"
        except psutil.Error:
            return False

    def close(self):
        pass


def open_parent(pid):
    return (WindowsProcess if os.name == "nt" else PosixProcess)(pid)


class ParentProcess(AbstractContextManager):
    def __init__(self, pid, birth=None):
        self.handle = open_parent(pid)
        if birth is not None and self.handle.birth != birth:
            self.handle.close()
            raise ProcessLookupError("Guard parent PID was reused")

    def alive(self):
        try:
            return self.handle.alive()
        except (OSError, RuntimeError):
            return False

    def __exit__(self, *_):
        self.handle.close()


def parent_birth():
    with ParentProcess(os.getpid()) as process:
        return process.handle.birth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--parent-start")
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    # Group interrupts reach the actual worker too. Let it checkpoint while
    # this lightweight group leader stays alive until the parent escalates.
    signal.signal(getattr(signal, "SIGUSR1", signal.SIGINT), lambda *_: None)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, lambda *_: None)
    finished = threading.Event()
    parent = ParentProcess(args.parent_pid, args.parent_start)

    def watch_parent():
        from .memory import kill

        while not finished.wait(0.1):
            if not parent.alive():
                kill(type("OwnedGroup", (), {"pid": os.getpid()})())
                return

    watcher = threading.Thread(target=watch_parent, daemon=True)
    watcher.start()
    try:
        if not parent.alive():
            return 1
        # Inherit our group; killing it covers the command and its descendants.
        child = subprocess.Popen(
            argv, env=os.environ | {"SONGE_GUARD_PID": str(os.getpid())}
        )
        return child.wait()
    finally:
        finished.set()
        # Join before releasing a Windows handle used by the watcher.
        watcher.join()
        parent.__exit__()


if __name__ == "__main__":
    raise SystemExit(main())
