import atexit
import json
from pathlib import Path

import jsonlines
from loguru import logger

from expr_tracker.encoders import jsonable_encoder


class JsonlTracker:
    def __init__(self):
        self.buffer = []
        self.buffer_size = 50
        self.log_fp = None

    def init(
        self,
        project: str,
        name: str | None = None,
        config: dict | None = None,
        dir: str | None = None,
        print_to_screen: bool = False,
        print_handle=print,
        buffer_size: int = 50,  # 新增 buffer_size 参数
        **kwargs,
    ):
        self.project = project
        self.name = name
        if dir is None:
            dir = "./tracker/jsonl"
        self.log_dir = Path(dir) / self.project / self.name
        self.config_fp = self.log_dir / "config.json"
        self.log_fp = self.log_dir / "metrics.jsonl"

        # 初始化 Buffer 配置
        self.buffer = []
        self.buffer_size = buffer_size

        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.config_fp.exists():
            logger.warning(
                f"Config file {self.config_fp} already exists. It will be overwritten."
            )

        if config is not None:
            # Config 通常只写一次，直接写入即可
            with open(self.config_fp, "w") as f:
                json.dump(jsonable_encoder(config), f, indent=4)

        self.print_to_screen = print_to_screen
        self.print_handle = print_handle

        # 优化：流式计算行数，避免一次性加载大文件到内存 (对 BlobFuse 友好)
        self.current_step = 0
        if self.log_fp.exists():
            try:
                with open(self.log_fp, "rb") as f:
                    self.current_step = sum(1 for _ in f)
            except Exception as e:
                logger.warning(f"Could not count existing lines in {self.log_fp}: {e}")

        # 注册退出钩子：确保程序意外终止时也能写入剩余数据
        atexit.register(self.flush)

    def log(self, metrics: dict, step: int | None = None):
        if step is not None:
            self.current_step = step

        record = {"_step": self.current_step, **metrics}

        # 1. 写入内存 Buffer
        self.buffer.append(record)

        # 2. 屏幕打印
        if self.print_to_screen:
            self.print_handle(f"{record}")

        # 3. 检查 Buffer 是否已满
        if len(self.buffer) >= self.buffer_size:
            self.flush()

        self.current_step += 1

    def flush(self):
        """强制将内存中的 Buffer 写入磁盘"""
        if not self.buffer:
            return

        # 确保目录存在 (防止运行时目录被删)
        if self.log_fp and not self.log_fp.parent.exists():
            self.log_fp.parent.mkdir(parents=True, exist_ok=True)

        try:
            # 批量追加写入
            with jsonlines.open(self.log_fp, mode="a") as writer:
                writer.write_all(self.buffer)
            self.buffer = []  # 写入成功后清空 Buffer
        except Exception as e:
            logger.error(f"Failed to flush metrics to {self.log_fp}: {e}")

    def finish(self):
        """结束时显式调用"""
        self.flush()
        # 如果手动调用了 finish，取消 atexit 注册，防止重复调用
        atexit.unregister(self.flush)
