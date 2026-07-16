"""
src/ml/forecaster.py
--------------------
Proactive fatigue forecasting — predicts where your fatigue is heading
before it hits critical levels.

"Your laptop's reminder fires at 20 minutes regardless.
 EyeGuard tells you in 10 minutes you'll be at 78/100 — take a break now."

Method:
  - Rolling linear regression on the last WINDOW_SEC of fatigue scores
  - Projects the trend forward by FORECAST_HORIZON_SEC
  - Reports: predicted_score, trend direction, urgency

Usage:
    forecaster = FatigueForecaster()
    forecaster.add(risk_score_01, t_sec)
    result = forecaster.predict()
    # result.predicted_score, result.trend, result.urgent
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass


WINDOW_SEC           = 60     # use last 60s of data for trend
FORECAST_HORIZON_SEC = 600    # predict 10 minutes ahead
MIN_SAMPLES          = 20     # need at least this many points to forecast


@dataclass
class ForecastResult:
    predicted_score_01: float   # predicted risk 0..1 in FORECAST_HORIZON_SEC
    predicted_score_100: float  # same, as 0..100 comfort score
    trend: str                  # "rising" | "falling" | "stable"
    slope_per_sec: float        # rate of change per second
    current_score_01: float     # most recent score
    horizon_sec: int            # how far ahead (always FORECAST_HORIZON_SEC)
    sufficient_data: bool       # False if not enough data yet


class FatigueForecaster:
    """
    Rolling linear regression fatigue forecaster.
    Add a risk score each second; call predict() to get the 10-min forecast.
    """

    RISING_THRESHOLD  =  0.0003   # slope/sec that counts as "rising"
    FALLING_THRESHOLD = -0.0003

    def __init__(
        self,
        window_sec: int = WINDOW_SEC,
        horizon_sec: int = FORECAST_HORIZON_SEC,
        min_samples: int = MIN_SAMPLES,
    ):
        self.window_sec  = int(window_sec)
        self.horizon_sec = int(horizon_sec)
        self.min_samples = int(min_samples)

        # Store (t_sec, score_01) pairs
        self._data: deque = deque(maxlen=self.window_sec * 25)  # 25fps max

    def add(self, risk_score_01: float, t_sec: float) -> None:
        """Add one data point. Call once per second (or per frame)."""
        score = max(0.0, min(1.0, float(risk_score_01)))
        self._data.append((float(t_sec), score))

    def predict(self) -> ForecastResult:
        """
        Returns a ForecastResult based on the rolling window trend.
        Returns a 'no data' result if insufficient data.
        """
        # Filter to window
        if not self._data:
            return self._no_data_result()

        latest_t = self._data[-1][0]
        cutoff_t = latest_t - self.window_sec
        window   = [(t, s) for t, s in self._data if t >= cutoff_t]

        if len(window) < self.min_samples:
            return self._no_data_result(current=self._data[-1][1])

        times  = np.array([p[0] for p in window])
        scores = np.array([p[1] for p in window])

        # Normalise times to [0, window_sec] for numerical stability
        t_norm = times - times[0]

        # Linear regression
        coeffs = np.polyfit(t_norm, scores, 1)
        slope  = float(coeffs[0])   # per second

        # Project forward
        elapsed_in_window = float(t_norm[-1])
        t_future = elapsed_in_window + self.horizon_sec
        predicted = float(np.polyval(coeffs, t_future))
        predicted = max(0.0, min(1.0, predicted))

        # Trend classification
        if slope > self.RISING_THRESHOLD:
            trend = "rising"
        elif slope < self.FALLING_THRESHOLD:
            trend = "falling"
        else:
            trend = "stable"

        current = float(scores[-1])

        return ForecastResult(
            predicted_score_01  = round(predicted, 4),
            predicted_score_100 = round((1.0 - predicted) * 100, 1),  # comfort = 100 - risk%
            trend               = trend,
            slope_per_sec       = round(slope, 6),
            current_score_01    = round(current, 4),
            horizon_sec         = self.horizon_sec,
            sufficient_data     = True,
        )

    def _no_data_result(self, current: float = 0.0) -> ForecastResult:
        return ForecastResult(
            predicted_score_01  = current,
            predicted_score_100 = round((1.0 - current) * 100, 1),
            trend               = "stable",
            slope_per_sec       = 0.0,
            current_score_01    = current,
            horizon_sec         = self.horizon_sec,
            sufficient_data     = False,
        )

    def reset(self) -> None:
        self._data.clear()

    @property
    def data_points(self) -> int:
        return len(self._data)

    @property
    def ready(self) -> bool:
        return self.data_points >= self.min_samples