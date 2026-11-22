"""
How long our nodes' loops take, on /diagnostics (diagnostic_msgs/DiagnosticArray), which the boat's Foxglove layout
shows in its Diagnostics tab. On the boat's computer (LattePanda Delta 3, a Celeron N5105) this says whether the DP
keeps up; in the simulator it gives the numbers to compare with.

    self.timing = LoopTimer(self, "control loop", period=0.1)
    def control_loop(self):
        with self.timing:
            ...
    LoopTimer(self, "docking OCP").record(seconds)      # events without a fixed period, such as a solve

Every `report` seconds one DiagnosticStatus, name "<node>: <loop>", with the window's compute time (mean, p95, max
[ms]), the time between calls (mean, max [ms]), the CPU share (compute time / window) and the number of calls; and
the largest compute time and period since the start. Level: ERROR when the mean compute time is longer than the
period (the loop cannot keep up), WARN when one call took longer than the period or a period was more than twice
the nominal one (the executor was held up, by another callback or by a thread holding the GIL), else OK.
"""
import time

import numpy as np
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue


class LoopTimer:
    def __init__(self, node, name, period=None, report=2.0, topic="/diagnostics"):
        self.node, self.name = node, name
        self.period = float(period) if period else None
        self.pub = getattr(node, "_loop_timing_pub", None)
        if self.pub is None:
            self.pub = node.create_publisher(DiagnosticArray, topic, 10)
            node._loop_timing_pub = self.pub
        self.compute, self.periods = [], []
        self.t_last = None
        self.max_compute = self.max_period = 0.0
        self.t_window = time.monotonic()
        self._start = None
        node.create_timer(float(report), self.publish)

    def __enter__(self):
        now = time.monotonic()
        if self.t_last is not None:
            self.periods.append(now - self.t_last)
        self.t_last = now
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.record(time.perf_counter() - self._start)
        return False

    def record(self, seconds):
        self.compute.append(float(seconds))

    def stats(self):
        """The window's numbers [s], or None when nothing ran."""
        if not self.compute:
            return None
        c = np.asarray(self.compute)
        p = np.asarray(self.periods) if self.periods else np.zeros(0)
        window = max(time.monotonic() - self.t_window, 1e-9)
        return dict(n=len(c), mean=float(c.mean()), p95=float(np.percentile(c, 95)), max=float(c.max()),
                    period_mean=float(p.mean()) if p.size else float("nan"),
                    period_max=float(p.max()) if p.size else float("nan"), load=float(c.sum() / window))

    def publish(self):
        s = self.stats()
        self.compute, self.periods, self.t_window = [], [], time.monotonic()
        if s is None:
            return
        self.max_compute = max(self.max_compute, s["max"])
        if np.isfinite(s["period_max"]):
            self.max_period = max(self.max_period, s["period_max"])
        level, why = DiagnosticStatus.OK, ""
        if self.period:
            if s["mean"] > self.period:
                level, why = DiagnosticStatus.ERROR, "cannot keep up: "
            elif s["max"] > self.period or s["period_max"] > 2.0 * self.period:
                level, why = DiagnosticStatus.WARN, "late: "
        ms = lambda v: f"{1000.0 * v:.1f}"
        msg = f"{why}{ms(s['mean'])} ms mean, {ms(s['max'])} ms max"
        if np.isfinite(s["period_mean"]):
            msg += f", period {ms(s['period_mean'])} / {ms(s['period_max'])} ms"
        st = DiagnosticStatus(level=level, name=f"{self.node.get_fully_qualified_name()}: {self.name}",
                              hardware_id=self.node.get_namespace().strip("/") or "dp", message=msg)
        values = dict(calls=str(s["n"]), compute_mean_ms=ms(s["mean"]), compute_p95_ms=ms(s["p95"]),
                      compute_max_ms=ms(s["max"]), period_mean_ms=ms(s["period_mean"]),
                      period_max_ms=ms(s["period_max"]), cpu_load=f"{s['load']:.3f}",
                      compute_max_since_start_ms=ms(self.max_compute), period_max_since_start_ms=ms(self.max_period))
        if self.period:
            values["period_nominal_ms"] = ms(self.period)
        st.values = [KeyValue(key=k, value=v) for k, v in values.items()]
        out = DiagnosticArray()
        out.header.stamp = self.node.get_clock().now().to_msg()
        out.status = [st]
        self.pub.publish(out)
