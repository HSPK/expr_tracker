import atexit
import json
import os
import threading
import time
from pathlib import Path

from loguru import logger

from expr_tracker.encoders import jsonable_encoder

DEFAULT_BUFFER_SIZE = 50
DEFAULT_BUFFER_INTERVAL = 1.0
DEFAULT_MAX_BUFFER_SECONDS = 5.0
# 写盘持续失败时 buffer 的最大长度，超出后丢弃最旧的记录，避免内存无限增长
DEFAULT_MAX_PENDING_RECORDS = 100_000
MAX_FALLBACK_REPR_LENGTH = 512

# 与 jsonlines 默认（非 compact）writer 的输出保持一致
_LINE_ENCODER = json.JSONEncoder(ensure_ascii=False)


def _fallback_repr(value) -> str:
    """无法 JSON 序列化时的兜底表示，保证永远不抛异常"""
    try:
        text = repr(value)
    except Exception:
        return f"<unrepresentable {type(value).__name__}>"
    if len(text) > MAX_FALLBACK_REPR_LENGTH:
        text = text[:MAX_FALLBACK_REPR_LENGTH] + "..."
    return text


def _encode_key(key) -> str:
    if isinstance(key, str):
        return key
    try:
        encoded = jsonable_encoder(key)
    except Exception:
        encoded = None
    if isinstance(encoded, str):
        return encoded
    if isinstance(encoded, (int, float, bool)) or encoded is None:
        return str(encoded)
    return _fallback_repr(key)


class JsonlTracker:
    def __init__(self):
        self.buffer = []
        self.buffer_size = DEFAULT_BUFFER_SIZE
        self.buffer_interval = DEFAULT_BUFFER_INTERVAL
        self.max_buffer_seconds = DEFAULT_MAX_BUFFER_SECONDS
        self.max_pending_records = DEFAULT_MAX_PENDING_RECORDS
        self.log_fp = None
        self._lock = threading.RLock()
        # 只用于串行化磁盘写入，不阻塞 log() 写 buffer
        self._write_lock = threading.Lock()
        self._last_log_time = None
        self._first_buffered_time = None
        self._flush_timer = None
        self._atexit_registered = False
        self._warned_metric_keys = set()

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
        max_pending_records: int = DEFAULT_MAX_PENDING_RECORDS,
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
        - ``max_pending_records``: 写盘持续失败（磁盘满、挂载断开等）时 buffer 的上限，
          超出后丢弃最旧的记录，避免 OOM。
        """
        self.project = project
        if name is None:
            name = time.strftime("run-%Y%m%d-%H%M%S")
            logger.warning(f"No run name provided, using generated name: {name}")
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
            self.max_pending_records = max(
                self.buffer_size, int(max_pending_records or 0)
            )
            self._last_log_time = None
            self._first_buffered_time = None
            self._warned_metric_keys = set()

        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.config_fp.exists():
            logger.warning(
                f"Config file {self.config_fp} already exists. It will be overwritten."
            )

        if config is not None:
            # Config 通常只写一次，直接写入即可
            try:
                with open(self.config_fp, "w", encoding="utf-8") as f:
                    json.dump(
                        self._encode_mapping(config, kind="config"),
                        f,
                        indent=4,
                        ensure_ascii=False,
                    )
            except Exception as e:
                logger.error(f"Failed to write config to {self.config_fp}: {e}")

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
        if not self._atexit_registered:
            atexit.register(self.flush)
            self._atexit_registered = True

    def _encode_mapping(self, metrics: dict | None, kind: str = "metric") -> dict:
        """把 metrics 转成可 JSON 序列化的 dict，绝不因单个坏值而丢掉整条记录。

        numpy/torch 标量等由 ``jsonable_encoder`` 统一处理；真正无法编码的值降级为
        ``repr``，并按 key 只告警一次，避免高频训练循环刷屏。
        """
        if not metrics:
            return {}
        if not isinstance(metrics, dict):
            return {"value": self._encode_value("value", metrics, kind)}
        try:
            encoded = jsonable_encoder(metrics)
            if isinstance(encoded, dict):
                return encoded
        except Exception:
            pass
        # 整体编码失败：逐字段降级，保留其余可序列化的字段
        encoded = {}
        for key, value in metrics.items():
            encoded_key = _encode_key(key)
            encoded[encoded_key] = self._encode_value(encoded_key, value, kind)
        return encoded

    def _encode_value(self, key: str, value, kind: str = "metric"):
        try:
            return jsonable_encoder(value)
        except Exception as e:
            if key not in self._warned_metric_keys:
                self._warned_metric_keys.add(key)
                logger.warning(
                    f"{kind.capitalize()} {key!r} of type {type(value).__name__} is not "
                    f"JSON serializable ({e}); falling back to repr()."
                )
            return _fallback_repr(value)

    def log(self, metrics: dict, step: int | None = None):
        now = time.monotonic()
        # 在锁外编码：大对象的转换不阻塞其他线程，同时保证坏值不会进入 buffer
        encoded_metrics = self._encode_mapping(metrics)

        with self._lock:
            if step is not None:
                self.current_step = step

            record = {"_step": self.current_step, **encoded_metrics}

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
            try:
                self.print_handle(f"{record}")
            except Exception as e:
                logger.warning(f"Failed to print metrics to screen: {e}")

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
            log_fp = self.log_fp
            if log_fp is None:
                logger.error(
                    f"JsonlTracker is not initialized, dropping {len(self.buffer)} "
                    "buffered records. Call init() before log()."
                )
                self.buffer = []
                return
            records, self.buffer = self.buffer, []

        # 先在内存里序列化：无法序列化的记录直接丢弃，否则它会永远卡在 buffer 里，
        # 导致后续所有指标都写不进去
        lines, pending = [], []
        for record in records:
            try:
                lines.append(_LINE_ENCODER.encode(record) + "\n")
                pending.append(record)
            except Exception as e:
                logger.error(f"Dropping record that cannot be serialized: {e}")
        if not lines:
            return

        size_before = None
        with self._write_lock:
            try:
                # 确保目录存在 (防止运行时目录被删)
                if not log_fp.parent.exists():
                    log_fp.parent.mkdir(parents=True, exist_ok=True)
                size_before = log_fp.stat().st_size if log_fp.exists() else 0
                # 一次性写入整批，避免中途失败留下「已写一半」的批次
                with open(log_fp, "a", encoding="utf-8") as f:
                    f.write("".join(lines))
            except Exception as e:
                logger.error(f"Failed to flush metrics to {log_fp}: {e}")
                self._truncate_partial_write(log_fp, size_before)
                # 写入失败：放回 buffer 头部，等待下次 flush 重试
                self._requeue(pending)
                self._schedule_timer()

    @staticmethod
    def _truncate_partial_write(log_fp: Path, size_before: int | None):
        """回滚半截写入，保证重试时不会产生重复或损坏的行"""
        if size_before is None:
            return
        try:
            if log_fp.exists() and log_fp.stat().st_size > size_before:
                os.truncate(log_fp, size_before)
        except Exception as e:
            logger.warning(f"Could not roll back partial write on {log_fp}: {e}")

    def _requeue(self, records: list):
        with self._lock:
            self.buffer[:0] = records
            overflow = len(self.buffer) - self.max_pending_records
            if overflow > 0:
                del self.buffer[:overflow]
                logger.error(
                    f"Metrics buffer exceeded {self.max_pending_records} records, "
                    f"dropped {overflow} oldest records."
                )
            self._first_buffered_time = time.monotonic()

    def finish(self):
        """结束时显式调用"""
        self._cancel_timer()
        self.flush()
        # 如果手动调用了 finish，取消 atexit 注册，防止重复调用
        if self._atexit_registered:
            atexit.unregister(self.flush)
            self._atexit_registered = False
