import time

from all_the_tools.metrics import Mean


class FPS(Mean):
    def __init__(self):
        super().__init__()

        self.t = None

    def update(self, n):
        if self.t is None:
            self.t = time.time()
            return

        t = time.time()
        super().update(n / (t - self.t))
        self.t = t

    def reset(self):
        super().reset()

        self.t = None
