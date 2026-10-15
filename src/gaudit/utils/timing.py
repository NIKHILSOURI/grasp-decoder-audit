"""Wall-clock timing of named stages, dumped to timing.json."""

import time
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.t = {}

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0

    def total(self) -> float:
        return sum(self.t.values())
