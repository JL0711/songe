"""Presentation of recorded status text without changing worker records."""

import re


def terse(value):
    text = str(value)
    known = {
        "Waiting: battery low": "battery low",
        "Will sleep when idle": "idle gap pending",
        "Stopping: battery low; saving checkpoint": "battery low · checkpoint pending",
        "Sleep could not complete; current version preserved": "Sleep failed · serving retained",
        "Run songe init": "uninitialized · songe init",
        "Service stopped. Run songe serve.": "service stopped · songe serve",
        "Service stopped. Run songe doctor.": "service stopped · songe doctor",
        "not supported yet": "unsupported",
        "Waiting for measured trainability": "not measured yet",
        "calibration required": "not measured yet",
    }
    if text in known:
        return known[text]
    architecture = re.fullmatch(
        r"Architecture (\S+) (?:not supported yet by \S+|unsupported for GGUF training)",
        text,
    )
    if architecture:
        return f"{architecture[1]} not supported yet"
    if text.startswith("Waiting for "):
        return text.removeprefix("Waiting for ").removesuffix(".") + " pending"
    if text.startswith("Waiting before "):
        return text.removeprefix("Waiting before ") + " pending"
    return text.replace(". Run songe doctor", " · songe doctor").replace(
        "Work failed; served version preserved. Diagnostics:",
        "Work failed · serving retained · log:",
    )
