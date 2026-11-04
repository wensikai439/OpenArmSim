# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Publish already-rendered overview frames without adding simulator RPC calls."""

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class LiveOverviewPublisher:
    """Keep one atomic JPEG, at most ten updates per wall-clock second."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._last = float("-inf")
        self._warned = False

    def publish(self, rgb):
        now = time.monotonic()
        if now - self._last < 0.1:
            return
        self._last = now
        try:
            import imageio.v3 as iio

            data = iio.imwrite("<bytes>", rgb, extension=".jpg", quality=80)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_bytes(data)
            temporary.replace(self.path)
        except Exception:
            # A display failure must not interrupt the action/control loop.
            if not self._warned:
                logger.exception("Live overview publication failed")
                self._warned = True


def attach_live_overview(task):
    """Tap the existing renderer on its owning thread; never step physics."""
    publisher = LiveOverviewPublisher(Path(task.output) / "live_overview.jpg")
    original_frame = task.ws.frame

    def frame(camera, *args, **kwargs):
        rgb = original_frame(camera, *args, **kwargs)
        if camera == "overview":
            publisher.publish(rgb)
        return rgb

    task.ws.frame = frame
    frame("overview")
