from loguru import logger
from pathlib import Path
from expr_tracker.encoders import jsonable_encoder


class JsonlTracker:
    def __init__(self):
        pass

    def init(
        self,
        project: str,
        name: str | None = None,
        config: dict | None = None,
        dir: str | None = None,
        print_to_screen: bool = False,
        print_handle=print,
        **kwargs,
    ):
        self.project = project
        self.name = name
        if dir is None:
            dir = "./tracker/jsonl"
        self.log_dir = Path(dir) / self.project / self.name
        self.config_fp = self.log_dir / "config.json"
        self.log_fp = self.log_dir / "metrics.jsonl"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.config_fp.exists():
            logger.warning(
                f"Config file {self.config_fp} already exists. It will be overwritten."
            )
        if config is not None:
            with open(self.config_fp, "w") as f:
                import json

                json.dump(jsonable_encoder(config), f, indent=4)
        self.print_to_screen = print_to_screen
        self.print_handle = print_handle
        self.current_step = (
            0 if not self.log_fp.exists() else len(self.log_fp.read_text().splitlines())
        )

    def log(self, metrics: dict, step: int | None = None):
        import jsonlines

        if not self.log_fp.parent.exists():
            self.log_fp.parent.mkdir(parents=True, exist_ok=True)

        if step is not None:
            self.current_step = step
        metrics = {"_step": self.current_step, **metrics}
        with jsonlines.open(self.log_fp, mode="a") as writer:
            writer.write(metrics)
        self.current_step += 1
        if self.print_to_screen:
            self.print_handle(f"{metrics}")

    def finish(self):
        pass
