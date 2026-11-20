"""Power policy regressions using fake native APIs, sysfs, and workers only."""

import asyncio
from types import SimpleNamespace

import pytest

from songe import schedule
from songe.api import snapshot
from songe.cli import print_status
from songe.platform import Memory, Power, power_status
from songe.platform.darwin import Darwin
from songe.platform.linux import Linux
from songe.platform.windows import Windows
from songe.sleep import worker


class NativeFunction:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


@pytest.mark.parametrize(
    "warning,mode,low",
    [(1, False, False), (2, False, True), (3, False, True), (1, True, True)],
)
@pytest.mark.parametrize("remaining", [600, -1, -2])
def test_macos_os_warnings_low_power_and_remaining(
    monkeypatch, warning, mode, low, remaining
):
    from songe.platform import darwin

    iokit = SimpleNamespace(
        IOPSGetTimeRemainingEstimate=NativeFunction(remaining),
        IOPSGetBatteryWarningLevel=NativeFunction(warning),
    )
    monkeypatch.setattr(darwin.ctypes, "CDLL", lambda _: iokit)
    monkeypatch.setattr(darwin, "process_info_value", lambda *args: mode)
    power = Darwin().power()
    assert power == Power(remaining == -2, low, 600 if remaining == 600 else None)
    assert Darwin().on_ac_power() == (remaining == -2)
    assert power.constrained() == (low and remaining != -2)


def test_macos_missing_apis_are_unconstrained(monkeypatch):
    from songe.platform import darwin

    def missing(*args):
        raise OSError("unavailable")

    monkeypatch.setattr(darwin.ctypes, "CDLL", missing)
    assert Darwin().power() == Power()


@pytest.mark.parametrize(
    "ac,flags,saver,life,low,seconds",
    [
        (0, 0, 0, 600, False, 600),
        (0, 2, 0, 600, True, 600),
        (0, 4, 0, 600, True, 600),
        (0, 2 | 8, 0, 600, True, 600),
        (0, 0, 1, 600, True, 600),
        (0, 255, 0, 0xFFFFFFFF, False, None),
        (0, 128, 1, 0, False, None),
        (1, 4, 1, 0, True, 0),
        (255, 4, 1, 0xFFFFFFFF, True, None),
    ],
)
def test_windows_native_flags_and_sentinels(ac, flags, saver, life, low, seconds):
    def status(pointer):
        info = pointer._obj
        info.ac, info.flags, info.saver, info.life = ac, flags, saver, life
        info.percent = 0  # Policy must never invent a percentage threshold.
        return 1

    platform = Windows(
        kernel=SimpleNamespace(GetSystemPowerStatus=status), user=object()
    )
    power = platform.power()
    assert power == Power(None if ac == 255 else bool(ac), low, seconds)
    assert power.constrained(300) == (ac == 0 and (low or seconds == 0))


def test_windows_failed_telemetry():
    platform = Windows(
        kernel=SimpleNamespace(GetSystemPowerStatus=lambda _: 0), user=object()
    )
    assert platform.power() == Power()


def supply(root, name, **values):
    path = root / "class/power_supply" / name
    path.mkdir(parents=True)
    for key, value in values.items():
        (path / key).write_text(str(value), encoding="utf-8")
    return path


@pytest.fixture
def linux(tmp_path, monkeypatch):
    monkeypatch.setattr("songe.platform.linux.shutil.which", lambda _: None)
    return Linux(sysfs=tmp_path)


@pytest.mark.parametrize(
    "level,low",
    [
        ("Normal", False),
        ("Low", True),
        ("Critical", True),
        ("Unknown", False),
        ("", False),
    ],
)
def test_linux_capacity_level_and_remaining(linux, tmp_path, level, low):
    battery = supply(
        tmp_path,
        "BAT0",
        type="Battery",
        status="Discharging",
        capacity_level=level,
        capacity=0,
        time_to_empty_now=123,
    )
    fraction = None if level in {"Normal", "Low", "Critical"} else 0.0
    assert linux.power() == Power(False, low, 123, fraction)
    (battery / "time_to_empty_now").write_text("invalid", encoding="utf-8")
    (battery / "time_to_empty_avg").write_text("321", encoding="utf-8")
    assert linux.power() == Power(False, low, 321, fraction)
    supply(tmp_path, "AC", type="Mains", online=1)
    assert not linux.power().constrained(1000)


def test_linux_no_system_battery_or_telemetry(linux, tmp_path):
    assert linux.power() == Power()
    supply(
        tmp_path,
        "mouse",
        type="Battery",
        scope="Device",
        status="Discharging",
        capacity_level="Critical",
        time_to_empty_now=0,
    )
    supply(
        tmp_path,
        "BAT0",
        type="Battery",
        present=0,
        status="Discharging",
        capacity_level="Low",
        time_to_empty_now=0,
    )
    assert linux.power() == Power()
    supply(
        tmp_path,
        "BAT1",
        type="Battery",
        status="Unknown",
        capacity_level="Low",
        time_to_empty_now=-1,
    )
    assert not linux.power().constrained(300)


@pytest.mark.parametrize("warning", ["low", "critical", "action"])
def test_linux_optional_upower(linux, monkeypatch, warning):
    from songe.platform import linux as module

    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/upower")
    monkeypatch.setattr(
        module,
        "read",
        lambda *args: (
            f"  present: yes\n  state: discharging\n  warning-level: {warning}\n"
            "  time to empty: 2.5 minutes\n"
        ),
    )
    assert linux.power() == Power(False, True, 150)


@pytest.mark.parametrize("value", [None, OSError("missing"), AttributeError("missing")])
def test_missing_power_does_not_invalidate_hardware_probes(value):
    def power():
        if isinstance(value, Exception):
            raise value
        return value

    platform = fake_platform(power)
    assert power_status(platform) == Power()
    assert schedule.probe(platform).available


def fake_platform(power):
    return SimpleNamespace(
        name="test",
        power=power,
        idle_seconds=lambda: 1800,
        thermal=lambda: 0,
        memory=lambda: Memory(24 * 10**9, 20 * 10**9),
        accelerators=lambda: [],
        keep_awake=lambda _: None,
    )


@pytest.mark.parametrize(
    "power,blocked",
    [
        (Power(False, True), True),
        (Power(False, False, 299), True),
        (Power(False, False, 300), False),
        (Power(False, False, 301), False),
        (Power(False), False),
        (Power(True, True, 0), False),
        (Power(), False),
    ],
)
async def test_scheduler_slice_admission(config, store, monkeypatch, power, blocked):
    config.calibration_available = True
    config.max_sequence, config.response_tokens = 256, 32
    config.token_chunk = 1  # A complete geometry; this test admits training.
    store.set(
        "calibration",
        {
            "rows": [
                dict(
                    sequence=256,
                    response_tokens=32,
                    token_chunk=1,
                    accumulation=1,
                    host_peak_bytes=100,
                    device_peak_bytes=0,
                    step_seconds=1,
                    generation_tps=50,
                )
            ]
        },
    )
    platform = fake_platform(lambda: power)
    scheduler = schedule.Scheduler(config, store, None, platform)
    scheduler.last_traffic -= 2000
    monkeypatch.setattr(scheduler, "enough", lambda: asyncio.sleep(0, result=True))
    launched = []

    async def spawn(*args, **kwargs):
        launched.append(args)
        return SimpleNamespace(pid=1, returncode=0)

    monkeypatch.setattr(schedule.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(schedule, "guard", lambda *args, **kwargs: None)
    await scheduler.tick()
    if scheduler.guard_task:
        await scheduler.guard_task
    assert bool(launched) is not blocked
    if blocked:
        assert "battery low" in snapshot(config, store, scheduler, live=True)["reason"]
    else:
        args = launched[0]
        assert float(args[args.index("--slice-seconds") + 1]) == 300


async def test_scheduler_battery_stop_uses_traffic_interrupt_and_records_reason(
    config, store, monkeypatch, capsys
):
    signals = []
    child = SimpleNamespace(returncode=None)

    async def wait():
        child.returncode = 0

    child.wait = wait
    monkeypatch.setattr(schedule, "interrupt_group", signals.append)
    platform = fake_platform(lambda: Power(False, True))
    scheduler = schedule.Scheduler(config, store, None, platform)
    scheduler.process = child
    night = store.night(None)
    slice_id = store.slice()
    await scheduler.tick()
    await scheduler.tick()
    assert signals == [child]
    assert "battery low" in scheduler.reason
    await scheduler.finish()
    assert store.slices()[0]["id"] == slice_id
    assert store.slices()[0]["interrupted_by"] == "battery low"
    assert store.nights()[0]["id"] == night
    assert store.nights()[0]["record"]["reason"] == "battery low"
    print_status(snapshot(config, store, scheduler, live=True))
    output = capsys.readouterr().out
    assert "Waiting for power" in output and "plug in to continue" in output


@pytest.mark.parametrize(
    "power", [Power(False), Power(False, False, 1), Power(True, True, 0), Power()]
)
async def test_running_slice_allows_healthy_battery_ac_and_unknown(
    config, store, monkeypatch, power
):
    monkeypatch.setattr(
        schedule, "interrupt_group", lambda _: pytest.fail("interrupted")
    )
    scheduler = schedule.Scheduler(config, store, None, fake_platform(lambda: power))
    scheduler.process = SimpleNamespace(returncode=None)
    await scheduler.tick()
    assert scheduler.cancelled_at is None


@pytest.mark.parametrize("power", [Power(False, True), Power(False, False, 59)])
def test_direct_worker_admission(config, store, monkeypatch, power):
    monkeypatch.setattr(worker, "current", lambda: fake_platform(lambda: power))
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.setattr(worker, "preview", lambda *args: pytest.fail("started work"))
    assert worker.run(config, store, slice_seconds=60) == {"interrupted": True}
    assert store.slices()[0]["interrupted_by"] == "battery low"


@pytest.mark.parametrize("capacity,constrained", [(9, True), (10, True), (11, False)])
def test_linux_percentage_only_uses_default_warning_point(
    linux, tmp_path, capacity, constrained
):
    supply(tmp_path, "BAT0", type="Battery", status="Discharging", capacity=capacity)
    assert linux.power().constrained() is constrained
    supply(tmp_path, "AC", type="Mains", online=1)
    assert not linux.power().constrained()


def test_os_low_battery_level_overrides_percentage(linux, tmp_path):
    supply(
        tmp_path,
        "BAT0",
        type="Battery",
        status="Discharging",
        capacity=5,
        capacity_level="Normal",
    )
    assert not linux.power().constrained()
