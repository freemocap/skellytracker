"""Lightweight wall-clock profiler for multi-camera batch processing.

Usage::

    results = process_folder(tracker, None, video_dir, output_dir, profile=True)

Or manually::

    timer = ProcessingTimer()
    # ... pass timer to process_batch calls ...
    print(timer.report(total_elapsed=38.5, n_frames=222, n_cameras=4))
"""
from __future__ import annotations

import time


class ProcessingTimer:
    """Accumulates per-section wall-clock times across many frames.

    Sections are recorded by name; the same name can be recorded many times
    and the totals are summed.  Call ``report()`` at the end for a summary.
    """

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def record(self, section: str, elapsed: float) -> None:
        """Add *elapsed* seconds to *section*."""
        self._totals[section] = self._totals.get(section, 0.0) + elapsed
        self._counts[section] = self._counts.get(section, 0) + 1

    def start(self) -> float:
        """Return the current time as a start marker for :meth:`stop`."""
        return time.perf_counter()

    def stop(self, section: str, t0: float) -> None:
        """Record time elapsed since *t0* under *section*."""
        self.record(section, time.perf_counter() - t0)

    def report(
        self,
        total_elapsed: float,
        n_frames: int,
        n_cameras: int,
    ) -> str:
        """Return a formatted multi-line report string."""
        if not self._totals:
            return "(no timings recorded)"

        lines: list[str] = []
        lines.append(
            f"=== Processing Profile  "
            f"({n_frames} frames × {n_cameras} cameras  |  "
            f"{n_frames / total_elapsed:.1f} fps  |  "
            f"{total_elapsed / n_frames * 1000:.1f} ms/frame) ==="
        )

        col_w = max(len(k) for k in self._totals) + 2
        lines.append(
            f"{'section':<{col_w}}  {'total':>8}  {'avg/frame':>11}  {'calls':>6}  {'%wall':>6}"
        )
        lines.append("─" * (col_w + 40))

        for section, total_s in sorted(self._totals.items(), key=lambda x: -x[1]):
            count = self._counts[section]
            avg_ms = total_s / count * 1000
            pct = total_s / total_elapsed * 100
            lines.append(
                f"{section:<{col_w}}  {total_s:>7.2f}s  {avg_ms:>10.1f}ms"
                f"  {count:>6}  {pct:>5.1f}%"
            )

        accounted = sum(self._totals.values())
        other = total_elapsed - accounted
        if other > 0.01:
            pct = other / total_elapsed * 100
            avg_ms = other / n_frames * 1000
            lines.append(
                f"{'(other)':<{col_w}}  {other:>7.2f}s  {avg_ms:>10.1f}ms"
                f"  {'':>6}  {pct:>5.1f}%"
            )

        lines.append("─" * (col_w + 40))
        avg_ms_total = total_elapsed / n_frames * 1000
        lines.append(
            f"{'wall total':<{col_w}}  {total_elapsed:>7.2f}s  {avg_ms_total:>10.1f}ms"
        )

        return "\n".join(lines)
