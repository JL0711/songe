"""One local screen, with the same recorded data as the browser."""

import asyncio
import sqlite3
import webbrowser

import httpx
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.theme import Theme
from textual.widgets import Button, DataTable, Footer, Static

from . import NAME
from .api import LocalClient, interface_data_dir
from .interface_copy import terse
from .models.registry import LEARNING_LEVELS
from .panels import STAGE_LABELS, dashboard_rows
from .terminal import model_status_text, status_text
from .wording import model_level


class SongeApp(App):
    TITLE = NAME
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        ("p", "pause_resume", "Pause / resume"),
        ("u", "undo", "Undo model"),
        ("l", "toggle_model", "Learning level"),
        ("o", "open_gui", "Open GUI"),
        ("q", "quit", "Quit"),
    ]
    CSS = """
    Screen { background: #191918; color: #eeede8; }
    #state { height: auto; padding: 1 2 0 2; }
    #message { height: auto; padding: 0 2; color: #aaa9a3; }
    #body { height: 1fr; padding: 1 2; }
    #models { height: auto; max-height: 12; margin-bottom: 1; background: #191918; }
    #learning-control { height: 3; margin-bottom: 1; }
    #model-details { height: auto; margin-bottom: 1; text-wrap: nowrap; overflow-x: auto; }
    #learning-control Button { min-width: 10; width: auto; height: 3; border: none; background: #191918; color: #aaa9a3; }
    #learning-control Button.selected { background: #343430; color: #eeede8; text-style: bold; }
    #learning-control Button:focus { text-style: underline; }
    DataTable > .datatable--header { background: #191918; color: #aaa9a3; text-style: none; }
    DataTable > .datatable--cursor { background: #343430; color: #eeede8; }
    Footer { background: #191918; }
    Footer > .footer--key { background: #191918; color: #eeede8; }
    Footer > .footer--description { color: #aaa9a3; }
    """

    def __init__(self, root=None, client=None):
        super().__init__()
        self.register_theme(
            Theme(
                name="songe",
                primary="#aaa9a3",
                secondary="#aaa9a3",
                accent="#aaa9a3",
                success="#aaa9a3",
                warning="#aaa9a3",
                error="#aaa9a3",
                foreground="#eeede8",
                background="#191918",
                surface="#191918",
                panel="#191918",
                boost="#eeede8",
            )
        )
        self.theme = "songe"
        self.client = client or LocalClient(root or interface_data_dir())
        self.data = None
        self._refreshing = False
        self._controlling = False
        self._expanded_models = set()
        self._model_values = {}

    def compose(self) -> ComposeResult:
        yield Static(
            "now     Waiting for activity\nnext    Waiting for the next update.\npaused  No",
            id="state",
            markup=False,
        )
        with VerticalScroll(id="body"):
            yield DataTable(
                id="models", cursor_type="row", cursor_foreground_priority="renderable"
            )
            yield Static("", id="model-details", markup=False)
            with Horizontal(id="learning-control"):
                for level in LEARNING_LEVELS:
                    yield Button(level.capitalize(), id=f"learning-{level}")
        yield Static("", id="message", markup=False)
        yield Footer()

    async def on_mount(self):
        self.query_one("#models", DataTable).add_columns("Model", "Learning", "Status")
        await self.refresh_data()
        self.set_interval(5, self.refresh_data)

    def on_resize(self, event):
        self.screen.set_class(event.size.width < 80, "narrow")

    async def refresh_data(self):
        if self._refreshing:
            return
        self._refreshing = True
        try:
            selected = self.selected_model()
            self.data = await asyncio.to_thread(self.client.snapshot)
            data = self.data
            self.query_one("#state", Static).update(status_text(data))
            models_table = self.query_one("#models", DataTable)
            model_values = {}
            for model in data.get("models", []):
                model_values[model["id"]] = (
                    Text(model.get("name") or model["id"]),
                    model_level(model).capitalize(),
                    model_status_text(model, data),
                )
            if list(model_values) != list(self._model_values):
                models_table.clear()
                for key, values in model_values.items():
                    models_table.add_row(*values, key=key)
            else:
                for key, values in model_values.items():
                    for column, old, new in zip(
                        models_table.columns,
                        self._model_values[key],
                        values,
                        strict=True,
                    ):
                        if old != new:
                            models_table.update_cell(
                                key, column, new, update_width=True
                            )
            self._model_values = model_values
            if selected and selected["id"] in model_values:
                models_table.move_cursor(row=list(model_values).index(selected["id"]))
            self.sync_learning_control()
        except (
            OSError,
            sqlite3.Error,
            ValueError,
            RuntimeError,
            httpx.HTTPError,
        ) as exc:
            self.query_one("#message", Static).update(f"Local state unavailable: {exc}")
        finally:
            self._refreshing = False

    def on_data_table_row_highlighted(self, event):
        if event.data_table.id == "models":
            self.sync_learning_control()

    def on_data_table_row_selected(self, event):
        if event.data_table.id != "models":
            return
        model_id = event.row_key.value
        if model_id in self._expanded_models:
            self._expanded_models.remove(model_id)
        else:
            self._expanded_models.add(model_id)
        self.sync_learning_control()

    async def do_control(self, action, **payload):
        if self._controlling:
            return
        self._controlling = True
        try:
            await asyncio.to_thread(self.client.control, action, **payload)
            self.query_one("#message", Static).update(
                {
                    "pause": "Learning: paused",
                    "resume": "Learning: resumed",
                    "undo": "Version: restored",
                    "model": "Learning: saved",
                }[action]
            )
            await self.refresh_data()
        except (
            OSError,
            sqlite3.Error,
            ValueError,
            RuntimeError,
            httpx.HTTPError,
        ) as exc:
            self.query_one("#message", Static).update(terse(exc))
        finally:
            self._controlling = False

    def action_pause_resume(self):
        if self.data:
            self.run_worker(
                self.do_control("resume" if self.data["paused"] else "pause"),
                group="control",
            )

    def selected_model(self):
        models = self.data.get("models", []) if self.data else []
        if not models:
            return None
        table = self.query_one("#models", DataTable)
        return models[table.cursor_row] if table.cursor_row < len(models) else None

    def action_toggle_model(self):
        model = self.selected_model()
        if model:
            level = model.get(
                "learning_level", "new" if model.get("enabled") else "off"
            )
            self.run_worker(
                self.do_control(
                    "model",
                    model=model["id"],
                    level=LEARNING_LEVELS[(LEARNING_LEVELS.index(level) + 1) % 3],
                ),
                group="control",
            )

    def sync_learning_control(self):
        model = self.selected_model()
        level = (
            model.get("learning_level", "new" if model.get("enabled") else "off")
            if model
            else None
        )
        expanded = bool(model and model["id"] in self._expanded_models)
        detail = self.query_one("#model-details", Static)
        detail.display = expanded
        self.query_one("#learning-control").display = expanded
        if expanded:
            text = Text(
                "\n".join(
                    f"{STAGE_LABELS[key]:<12} {value}"
                    for key, value, _ in dashboard_rows(model, snapshot=self.data)
                )
            )
            detail.update(text)
        else:
            detail.update("")
        for value in LEARNING_LEVELS:
            button = self.query_one(f"#learning-{value}", Button)
            button.disabled = (
                model is None
                or model.get("support", model.get("status")) == "unsupported"
            )
            button.set_class(level == value, "selected")
            button.tooltip = (
                f"Learning for {model['id']}" if model else "Select a model"
            )

    def on_button_pressed(self, event):
        if event.button.id and event.button.id.startswith("learning-"):
            model = self.selected_model()
            if model:
                self.run_worker(
                    self.do_control(
                        "model",
                        model=model["id"],
                        level=event.button.id.removeprefix("learning-"),
                    ),
                    group="control",
                )

    def action_undo(self):
        model = self.selected_model()
        self.run_worker(
            self.do_control("undo", **({"model": model["id"]} if model else {})),
            group="control",
        )

    def action_open_gui(self):
        webbrowser.open(self.client.url + "/")


def run(root=None):
    SongeApp(root).run()
