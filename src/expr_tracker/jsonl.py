import atexit
import json
import threading
import time
from pathlib import Path

import jsonlines
from loguru import logger

from expr_tracker.encoders import jsonable_encoder

DEFAULT_BUFFER_SIZE = 50
DEFAULT_BUFFER_INTERVAL = 1.0
DEFAULT_MAX_BUFFER_SECONDS = 5.0


class JsonlTracker:
    def __init__(self):
        self.buffer = []
        self.buffer_size = DEFAULT_BUFFER_SIZE
        self.buffer_interval = DEFAULT_BUFFER_INTERVAL
        self.max_buffer_seconds = DEFAULT_MAX_BUFFER_SECONDS
        self.log_fp = None
        self._lock = threading.RLock()
        self._last_log_time = None
        self._first_buffered_time = None
        self._flush_timer = None

    def init(
        self,
        project: str,
        name: str | None = None,
        config: dict | None = None,
        dir: str | None = None,
        print_to_screen: bool = False,
        print_handle=print,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        buffer_interval: float | None = DEFAULT_BUFFER_INTERVAL,
        max_buffer_seconds: float | None = DEFAULT_MAX_BUFFER_SECONDS,
        **kwargs,
    ):
        """初始化 jsonl backend。

        缓冲策略（按 log 频率自适应）：
        - ``buffer_size``: buffer 中记录数达到该值立即写盘。
        - ``buffer_interval``: 相邻两次 ``log()`` 的间隔 >= 该值时认为不是高频写入，
          直接写盘（低延迟）；小于该值才认为是高频写入，先攒在内存里。
          设为 ``None`` 表示关闭该判断（只按 buffer_size 攒批）。
        - ``max_buffer_seconds``: 记录在 buffer 中的最长停留时间，超时后由后台定时器
          强制写盘，避免高频写入突然停止时数据长期滞留内存。设为 ``None`` 关闭。
        """
        self.project = project
        self.name = name
        if dir is None:
            dir = "./tracker/jsonl"
        self.log_dir = Path(dir) / self.project / self.name
        self.config_fp = self.log_dir / "config.json"
        self.log_fp = self.log_dir / "metrics.jsonl"

        # 初始化 Buffer 配置
        self._cancel_timer()
        with self._lock:
            self.buffer = []
            self.buffer_size = max(1, int(buffer_size))
            self.buffer_interval = (
                None if buffer_interval is None else float(buffer_interval)
            )
            self.max_buffer_seconds = (
                None if max_buffer_seconds is None else float(max_buffer_seconds)
            )
            self._last_log_time = None
            self._first_buffered_time = None

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
        now = time.monotonic()

        with self._lock:
            if step is not None:
                self.current_step = step

            record = {"_step": self.current_step, **metrics}

            # 1. 写入内存 Buffer
            self.buffer.append(record)
            if self._first_buffered_time is None:
                self._first_buffered_time = now

            # 2. 根据「距离上一次 log 的时间间隔」判断是否为高频写入
            interval = None if self._last_log_time is None else now - self._last_log_time
            self._last_log_time = now

            should_flush = self._should_flush(now, interval)
            self.current_step += 1

        # 3. 屏幕打印（放在锁外，避免 print handle 阻塞其他线程）
        if self.print_to_screen:
            self.print_handle(f"{record}")

        # 4. 立即写盘，或安排一次超时写盘
        if should_flush:
            self.flush()
        else:
            self._schedule_timer()

    def _should_flush(self, now: float, interval: float | None) -> bool:
        """判断当前是否应该立即写盘（需在持锁状态下调用）"""
        # Buffer 已满
        if len(self.buffer) >= self.buffer_size:
            return True
        # 首次 log：直接落盘，尽快产生文件内容
        if interval is None:
            return True
        # 低频写入：距离上次 log 已经过去足够久，没必要继续攒批
        if self.buffer_interval is not None and interval >= self.buffer_interval:
            return True
        # 高频写入，但最早的记录已在内存中停留过久
        return (
            self.max_buffer_seconds is not None
            and self._first_buffered_time is not None
            and now - self._first_buffered_time >= self.max_buffer_seconds
        )

    def _schedule_timer(self):
        """为 buffer 中最早的记录安排一次超时写盘"""
        if self.max_buffer_seconds is None:
            return
        with self._lock:
            if self._flush_timer is not None or not self.buffer:
                return
            now = time.monotonic()
            elapsed = now - (self._first_buffered_time or now)
            delay = max(0.0, self.max_buffer_seconds - elapsed)
            timer = threading.Timer(delay, self._on_timer)
            timer.daemon = True
            self._flush_timer = timer
        timer.start()

    def _on_timer(self):
        with self._lock:
            self._flush_timer = None
        self.flush()

    def _cancel_timer(self):
        with self._lock:
            timer = self._flush_timer
            self._flush_timer = None
        if timer is not None:
            timer.cancel()

    def flush(self):
        """强制将内存中的 Buffer 写入磁盘"""
        self._cancel_timer()

        with self._lock:
            self._first_buffered_time = None
            if not self.buffer:
                return
            records, self.buffer = self.buffer, []
            log_fp = self.log_fp

        # 确保目录存在 (防止运行时目录被删)
        if log_fp and not log_fp.parent.exists():
            log_fp.parent.mkdir(parents=True, exist_ok=True)

        try:
            # 批量追加写入
            with jsonlines.open(log_fp, mode="a") as writer:
                writer.write_all(records)
        except Exception as e:
            logger.error(f"Failed to flush metrics to {log_fp}: {e}")
            # 写入失败：放回 buffer 头部，等待下次 flush 重试
            with self._lock:
                self.buffer[:0] = records
                self._first_buffered_time = time.monotonic()
            self._schedule_timer()

    def finish(self):
        """结束时显式调用"""
        self._cancel_timer()
        self.flush()
        # 如果手动调用了 finish，取消 atexit 注册，防止重复调用
        atexit.unregister(self.flush)
