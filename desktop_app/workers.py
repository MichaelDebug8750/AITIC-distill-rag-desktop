"""Qt 线程池适配器。"""

from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class TaskSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()
    progress = Signal(str)


class Task(QRunnable):
    """在线程池运行同步函数，保证模型/向量库不会卡住 GUI 事件循环。"""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = TaskSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            value = self.fn(*self.args, **self.kwargs)
        except BaseException as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            traceback.print_exc()
            self.signals.error.emit(detail)
        else:
            self.signals.result.emit(value)
        finally:
            self.signals.finished.emit()

