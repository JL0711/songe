"""Measure host process trees and the selected device across whole phases."""

import os
import threading
from contextlib import AbstractContextManager

from ..memory import process_tree_bytes, snapshot


class Meter(AbstractContextManager):
    def __init__(self, platform, baseline, backend=None, interval=0.1):
        self.platform, self.baseline, self.backend = platform, baseline, backend
        self.interval = interval
        self.done = threading.Event()
        self.host = self.device = 0
        self.error = None
        # Calibration must measure the same tree as the outer guard, including
        # its Python group leader. Otherwise a precise peak always looks exceeded.
        leader = os.environ.get("SONGE_GUARD_PID", "")
        self.pid = (
            int(leader)
            if leader.isdigit() and int(leader) == os.getppid()
            else os.getpid()
        )

    def sample(self):
        sample = snapshot(self.platform)
        rss = process_tree_bytes(self.platform, self.pid)
        self.host = max(self.host, rss, self.baseline.available - sample.available)
        if self.backend and not sample.vram:
            self.host = max(self.host, self.backend.peak_memory())
        if sample.vram:
            self.device = max(
                self.device,
                *(
                    max(0, a.memory_free - b.memory_free)
                    for a, b in zip(self.baseline.vram, sample.vram, strict=True)
                ),
            )
            if self.backend:
                self.device = max(self.device, self.backend.peak_memory())

    def __enter__(self):
        self.sample()
        self.entry_host, self.entry_device = self.host, self.device

        def watch():
            while not self.done.wait(self.interval):
                try:
                    self.sample()
                except Exception as exc:
                    self.error = exc
                    return

        self.watcher = threading.Thread(target=watch, daemon=True)
        self.watcher.start()
        return self

    def __exit__(self, *exc):
        self.done.set()
        self.watcher.join()
        self.sample()
        if self.error and not exc[0]:
            raise self.error

    def result(self):
        return {
            "host_peak_bytes": self.host,
            "device_peak_bytes": self.device,
            "extra_host_bytes": max(0, self.host - self.entry_host),
            "extra_device_bytes": max(0, self.device - self.entry_device),
        }
