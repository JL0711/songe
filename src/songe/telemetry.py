import json
import logging
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path


class DiagnosticHandler(RotatingFileHandler):
    def handleError(self, record):
        # A full/unavailable log disk must not print logging's own traceback.
        pass


class EventFormatter(logging.Formatter):
    def format(self, record):
        # Ordinary events retain safe locations only. Explicit child diagnostics
        # include the full exception chain in this private file, never the UI.
        data = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        if record.exc_info:
            data["exception"] = record.exc_info[0].__name__
            data["stack"] = [
                {"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                for f in traceback.extract_tb(record.exc_info[2])
            ]
        if getattr(record, "diagnostic", False) and record.exc_info:
            data["traceback"] = "".join(traceback.format_exception(*record.exc_info))
        if getattr(record, "child_stderr", None):
            data["traceback"] = record.child_stderr
        return json.dumps(data)


def configure(root):
    from .storage import private_dir

    private_dir(root)
    os.environ["SONGE_DIAGNOSTICS_DIR"] = str(root)
    logger = logging.getLogger("songe")
    for old in list(logger.handlers):
        if isinstance(old, RotatingFileHandler):
            logger.removeHandler(old)
            old.close()
    handler = DiagnosticHandler(root / "songe.log", maxBytes=2_000_000, backupCount=3)
    (root / "songe.log").chmod(0o600)
    handler.setFormatter(EventFormatter())
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False


def failure_hint(phase, root=None):
    # `songe doctor` shows the log path; status lines stay short.
    return f"{phase.lower()} failed · songe doctor"


def log_child_stderr(stderr):
    if stderr and any(
        isinstance(h, RotatingFileHandler) for h in logging.getLogger("songe").handlers
    ):
        logging.getLogger("songe").error(
            "heavy_child_stderr",
            extra={
                "child_stderr": stderr.decode(errors="replace")
                if isinstance(stderr, bytes)
                else stderr
            },
        )


def diagnostic_main(operation, root=None):
    root = root or os.environ.get("SONGE_DIAGNOSTICS_DIR")
    if root is None:
        from .config import data_dir

        root = data_dir()
        if "--data-dir" in sys.argv and sys.argv.index("--data-dir") + 1 < len(
            sys.argv
        ):
            root = Path(sys.argv[sys.argv.index("--data-dir") + 1])
    from .memory import MEMORY_EXIT, MemoryPressure, memory_stop_line

    try:
        configure(Path(root))
        return operation()
    except MemoryPressure as exc:
        # Short memory is a wait, not a failure; the parent reads this line.
        logging.getLogger("songe").info("heavy_child_memory_stop %s", exc)
        print(memory_stop_line(exc), file=sys.stderr)
        raise SystemExit(MEMORY_EXIT) from None
    except Exception as exc:
        # Children exit quietly; the parent displays a single actionable line.
        # The private file retains the complete exception chain for diagnosis.
        if "[metal::malloc]" in str(exc):
            # A device allocation limit (no private data) the parent acts on.
            print(str(exc).splitlines()[0], file=sys.stderr)
        if any(
            isinstance(h, RotatingFileHandler)
            for h in logging.getLogger("songe").handlers
        ):
            logging.getLogger("songe").exception(
                "heavy_child_failed", extra={"diagnostic": True}
            )
        print(failure_hint("Background work", root), file=sys.stderr)
        raise SystemExit(1) from None
