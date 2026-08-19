from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pynvml


def gpu_processes(index: int) -> list[dict[str, int]]:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    processes = []
    for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
        processes.append({"pid": int(process.pid), "used_memory": int(process.usedGpuMemory)})
    return processes


def require_idle_gpu(index: int) -> None:
    processes = gpu_processes(index)
    if processes:
        raise RuntimeError(f"physical GPU {index} has compute processes: {processes}")


@dataclass
class GPUMemoryMonitor:
    index: int
    interval_seconds: float = 0.05

    def __post_init__(self) -> None:
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        while not self._stop.is_set():
            used = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
            self.peak_bytes = max(self.peak_bytes, used)
            time.sleep(self.interval_seconds)

    def start(self) -> "GPUMemoryMonitor":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.peak_bytes

    def __enter__(self) -> "GPUMemoryMonitor":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()
