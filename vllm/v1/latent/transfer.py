# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Event-protected pinned metadata buffers for asynchronous H2D copies."""

import torch


class MetadataRing:
    def __init__(self, template, size=4):
        self.buffers = [
            torch.empty_like(template, pin_memory=True) for _ in range(size)
        ]
        self.events = [torch.cuda.Event() for _ in range(size)]
        self.pending = [False] * size
        self.index = -1

    def acquire(self):
        self.index = (self.index + 1) % len(self.buffers)
        if self.pending[self.index] and not self.events[self.index].query():
            self.events[self.index].synchronize()
        return self.buffers[self.index]

    def copy(self, target, source):
        target.copy_(source, non_blocking=True)
        self.events[self.index].record()
        self.pending[self.index] = True
