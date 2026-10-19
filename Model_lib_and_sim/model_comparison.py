"""Compare one-step twin-track model predictions with a logged vehicle run.

This script does not fit parameters. Per-wheel forces are the model's estimates;
the log only validates their combined effect on observable vehicle motion.
"""

from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import dataclass, field
from math import cos, pi, sin
from pathlib import Path
from typing import Mapping

import numpy as np

from twintrack_model_modified import (
    TwinTrackModel,
    TwinTrackState,
    WheelForces,
    add_scaled_state,
    clamp,
)


class CsvDataLoader:
    """Load numeric CSV rows and retrieve the nearest row by timestamp."""

    def __init__(self, path: str | Path, timestamp_column: str = "timestamp") -> None:
        self.path = Path(path)
        self.timestamp_column = timestamp_column
        with self.path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames or timestamp_column not in reader.fieldnames:
                raise ValueError(f"CSV must contain a {timestamp_column!r} column")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("CSV contains duplicate column names")

            rows: list[dict[str, float]] = []
            for row_number, raw_row in enumerate(reader, start=2):
                values: dict[str, float] = {}
                for column, raw_value in raw_row.items():
                    if raw_value is None or raw_value.strip() == "":
                        raise ValueError(f"Missing value in column {column!r}, row {row_number}")
                    try:
                        value = float(raw_value)
                    except ValueError as error:
                        raise ValueError(
                            f"Non-numeric value in column {column!r}, row {row_number}"
                        ) from error
                    if not np.isfinite(value):
                        raise ValueError(f"Non-finite value in column {column!r}, row {row_number}")
                    values[column] = value
                rows.append(values)

        if not rows:
            raise ValueError("CSV contains no data rows")

        self.rows = tuple(rows)
        self.timestamps = tuple(row[timestamp_column] for row in rows)
        if any(right <= left for left, right in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("CSV timestamps must be strictly increasing")

    def __len__(self) -> int:
        return len(self.rows)

    def get_at(self, timestamp: float) -> dict[str, float]:
        """Return the nearest row; out-of-range timestamps are rejected."""

        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if timestamp < self.timestamps[0] or timestamp > self.timestamps[-1]:
            raise ValueError("timestamp is outside the CSV time range")

        right = bisect.bisect_left(self.timestamps, timestamp)
        if right == 0:
            return dict(self.rows[0])
        if right == len(self.rows):
            return dict(self.rows[-1])
        left = right - 1
        index = left if timestamp - self.timestamps[left] <= self.timestamps[right] - timestamp else right
        return dict(self.rows[index])


@dataclass(frozen=True)
class ComparisonConfig:
    """Column names and raw-signal conversions for a comparison run."""

    wheel_columns: Mapping[str, str] = field(
        default_factory=lambda: {
            "fl": "fl_enc",
            "fr": "fr_enc",
            "rl": "rl_rpm",
            "rr": "rr_rpm",
        }
    )
    wheel_speed_scales: Mapping[str, float] = field(
        default_factory=lambda: {
            "fl": 4.0 * pi / 60.0,
            "fr": 4.0 * pi / 60.0,
            "rl": 2.0 * pi / 60.0,
            "rr": 2.0 * pi / 60.0,
        }
    )
    wheel_speed_signs: Mapping[str, float] = field(
        default_factory=lambda: {"fl": 1.0, "fr": 1.0, "rl": 1.0, "rr": -1.0}
    )
    steering_column: str = "cmd_steer_deg"
    steering_scale_rad_per_unit: float = 1.0
    vy_column: str = "vy"
    vy_scale_m_s_per_unit: float = 1.0
    accel_x_column: str = "accel_x"
    accel_y_column: str = "accel_y"
    acceleration_scale_m_s2_per_unit: float = 1.0
    acceleration_rotation_rad: float = -pi / 2.0
    yaw_rate_column: str = "yaw_rate"
    yaw_column: str = "gps_yaw"
    velocity_mode: str = "front_wheels"
    vx_column: str = "vx"
    vx_scale_m_s_per_unit: float = 1.0


@dataclass(frozen=True)
class ComparisonSample:
    timestamp: float
    dt: float
    predicted_next: TwinTrackState
    measured_next: TwinTrackState
    predicted_accel_x: float
    predicted_accel_y: float
    measured_accel_x: float
    measured_accel_y: float
    forces: Mapping[str, WheelForces]


@dataclass(frozen=True)
class ComparisonResult:
    samples: tuple[ComparisonSample, ...]
    rmse: Mapping[str, float]
    warnings: tuple[str, ...]

    def write_csv(self, path: str | Path) -> None:
        """Write per-interval predictions, residuals, and estimated tire forces."""

        columns = [
            "timestamp",
            "dt",
            "vx_pred_next",
            "vx_measured_next",
            "vx_error",
            "yaw_rate_pred_next",
            "yaw_rate_measured_next",
            "yaw_rate_error",
            "accel_x_pred",
            "accel_x_imu",
            "accel_x_error",
            "accel_y_pred",
            "accel_y_imu",
            "accel_y_error",
        ]
        for wheel in ("fl", "fr", "rl", "rr"):
            columns.extend(
                f"{wheel}_{name}"
                for name in ("fx", "fy", "normal_force", "slip_ratio", "slip_angle")
            )

        with Path(path).open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=columns)
            writer.writeheader()
            for sample in self.samples:
                row: dict[str, float] = {
                    "timestamp": sample.timestamp,
                    "dt": sample.dt,
                    "vx_pred_next": sample.predicted_next.vx,
                    "vx_measured_next": sample.measured_next.vx,
                    "vx_error": sample.predicted_next.vx - sample.measured_next.vx,
                    "yaw_rate_pred_next": sample.predicted_next.yaw_rate,
                    "yaw_rate_measured_next": sample.measured_next.yaw_rate,
                    "yaw_rate_error": sample.predicted_next.yaw_rate - sample.measured_next.yaw_rate,
                    "accel_x_pred": sample.predicted_accel_x,
                    "accel_x_imu": sample.measured_accel_x,
                    "accel_x_error": sample.predicted_accel_x - sample.measured_accel_x,
                    "accel_y_pred": sample.predicted_accel_y,
                    "accel_y_imu": sample.measured_accel_y,
                    "accel_y_error": sample.predicted_accel_y - sample.measured_accel_y,
                }
                for wheel, forces in sample.forces.items():
                    for name in ("fx", "fy", "normal_force", "slip_ratio", "slip_angle"):
                        row[f"{wheel}_{name}"] = getattr(forces, name)
                writer.writerow(row)


class ModelComparison:
    """Run measured states through ``derivatives`` and compare one-step results."""

    def __init__(
        self,
        model: TwinTrackModel | None = None,
        signal_config: ComparisonConfig | None = None,
    ) -> None:
        self.model = model or TwinTrackModel()
        self.signal_config = signal_config or ComparisonConfig()

    def run(self, data: CsvDataLoader | str | Path) -> ComparisonResult:
        loader = data if isinstance(data, CsvDataLoader) else CsvDataLoader(data)
        states, steering, measured_acceleration = self._prepare_states(loader)
        if len(states) < 2:
            raise ValueError("At least two rows are required for a one-step comparison")

        warnings = self._data_warnings(loader)
        comparisons: list[ComparisonSample] = []
        for index in range(len(states) - 1):
            timestamp = loader.timestamps[index]
            dt = loader.timestamps[index + 1] - timestamp
            state = states[index]
            derivatives, wheel_forces = self.model.derivatives(state, steer=steering[index])
            predicted_next = add_scaled_state(state, derivatives, dt)
            measured_next = states[index + 1]

            predicted_accel_x = derivatives.vx - state.vy * state.yaw_rate
            predicted_accel_y = derivatives.vy + state.vx * state.yaw_rate
            measured_accel_x, measured_accel_y = measured_acceleration[index]
            comparisons.append(
                ComparisonSample(
                    timestamp=timestamp,
                    dt=dt,
                    predicted_next=predicted_next,
                    measured_next=measured_next,
                    predicted_accel_x=predicted_accel_x,
                    predicted_accel_y=predicted_accel_y,
                    measured_accel_x=measured_accel_x,
                    measured_accel_y=measured_accel_y,
                    forces=wheel_forces,
                )
            )

        if np.max(np.diff(loader.timestamps)) > 2.0 * np.median(np.diff(loader.timestamps)):
            warnings.append("Some intervals are much longer than the median; Euler next-state errors may be dominated by the gap.")

        return ComparisonResult(
            samples=tuple(comparisons),
            rmse=self._calculate_rmse(comparisons),
            warnings=tuple(warnings),
        )

    def _prepare_states(
        self, loader: CsvDataLoader
    ) -> tuple[list[TwinTrackState], list[float], list[tuple[float, float]]]:
        config = self.signal_config
        required = {
            config.steering_column,
            config.vy_column,
            config.accel_x_column,
            config.accel_y_column,
            config.yaw_rate_column,
            *config.wheel_columns.values(),
        }
        if config.velocity_mode == "column":
            required.add(config.vx_column)
        missing = required.difference(loader.rows[0])
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        if config.velocity_mode not in ("front_wheels", "column"):
            raise ValueError("velocity_mode must be 'front_wheels' or 'column'")

        states: list[TwinTrackState] = []
        steering: list[float] = []
        accelerations: list[tuple[float, float]] = []
        params = self.model.params

        for row in loader.rows:
            omega: dict[str, float] = {}
            for wheel, column in config.wheel_columns.items():
                if wheel not in config.wheel_speed_scales or wheel not in config.wheel_speed_signs:
                    raise ValueError(f"Wheel speed conversion is missing for {wheel}")
                omega[wheel] = (
                    row[column]
                    * config.wheel_speed_scales[wheel]
                    * config.wheel_speed_signs[wheel]
                )

            steer = clamp(
                row[config.steering_column] * config.steering_scale_rad_per_unit,
                -self.model.max_steer,
                self.model.max_steer,
            )
            vy = row[config.vy_column] * config.vy_scale_m_s_per_unit
            yaw_rate = row[config.yaw_rate_column]
            if config.velocity_mode == "column":
                vx = row[config.vx_column] * config.vx_scale_m_s_per_unit
            else:
                vx = self._estimate_vx_from_front_wheels(omega, vy, yaw_rate, steer)

            states.append(
                TwinTrackState(
                    x=row.get("gps_x", 0.0),
                    y=row.get("gps_y", 0.0),
                    yaw=row.get(config.yaw_column, 0.0),
                    vx=vx,
                    vy=vy,
                    yaw_rate=yaw_rate,
                    omega_fl=omega["fl"],
                    omega_fr=omega["fr"],
                    omega_rl=omega["rl"],
                    omega_rr=omega["rr"],
                )
            )
            steering.append(steer)

            rotation = config.acceleration_rotation_rad
            scale = config.acceleration_scale_m_s2_per_unit
            raw_x = row[config.accel_x_column]
            raw_y = row[config.accel_y_column]
            accelerations.append(
                (
                    scale * (cos(rotation) * raw_x - sin(rotation) * raw_y),
                    scale * (sin(rotation) * raw_x + cos(rotation) * raw_y),
                )
            )

        return states, steering, accelerations

    def _estimate_vx_from_front_wheels(
        self,
        omega: Mapping[str, float],
        vy: float,
        yaw_rate: float,
        steer: float,
    ) -> float:
        params = self.model.params
        cosine = cos(steer)
        if abs(cosine) < 1e-6:
            raise ValueError("Cannot reconstruct vx from front wheel speeds near 90-degree steering")

        estimates = []
        for wheel, lateral_position in (
            ("fl", params.front_track_width / 2.0),
            ("fr", -params.front_track_width / 2.0),
        ):
            longitudinal_at_wheel = params.front_wheel_radius * omega[wheel]
            estimates.append(
                (
                    longitudinal_at_wheel
                    - sin(steer) * vy
                    + cosine * yaw_rate * lateral_position
                    - sin(steer) * yaw_rate * params.front_axle_len
                )
                / cosine
            )
        return float(np.mean(estimates))

    def _data_warnings(self, loader: CsvDataLoader) -> list[str]:
        rows = loader.rows
        columns = set(rows[0])
        warnings = [
            "Front-wheel speed is used to reconstruct vx; this assumes rolling behavior and the configured encoder scale.",
            "Logged vy is used as supplied; if it is a placeholder, lateral state predictions are not independently validated.",
            "IMU acceleration is compared after configured scale and axis rotation; gravity/bias calibration is not inferred.",
            "Per-wheel forces are model estimates; the log measures only their aggregate vehicle response.",
            "Wheel angular-speed next-state predictions are omitted because drive/brake torques are not logged.",
        ]
        if {"vx", "fl_enc", "fr_enc"}.issubset(columns) and all(
            abs(row["vx"] - (row["fl_enc"] + row["fr_enc"]) / 2.0) < 1e-9 for row in rows
        ):
            warnings.append("CSV vx duplicates the raw front encoder mean, so it is not used as an independent speed measurement.")
        if {"gps_x", "gps_y"}.issubset(columns) and all(
            row["gps_x"] == rows[0]["gps_x"] and row["gps_y"] == rows[0]["gps_y"] for row in rows
        ):
            warnings.append("GPS position is constant; position and yaw integration are not scored.")
        if all(row[self.signal_config.vy_column] == rows[0][self.signal_config.vy_column] for row in rows):
            warnings.append("Logged vy is constant across the run; its predicted evolution cannot be validated.")
        return warnings

    @staticmethod
    def _calculate_rmse(samples: list[ComparisonSample]) -> dict[str, float]:
        if not samples:
            return {}
        values = {
            "next_vx": [sample.predicted_next.vx - sample.measured_next.vx for sample in samples],
            "next_yaw_rate": [
                sample.predicted_next.yaw_rate - sample.measured_next.yaw_rate for sample in samples
            ],
            "accel_x": [sample.predicted_accel_x - sample.measured_accel_x for sample in samples],
            "accel_y": [sample.predicted_accel_y - sample.measured_accel_y for sample in samples],
        }
        return {name: float(np.sqrt(np.mean(np.square(errors)))) for name, errors in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="?",
        default=Path(__file__).with_name("ramp_steer_test_log_1790698316.csv"),
        type=Path,
        help="input vehicle log CSV",
    )
    parser.add_argument("--output", type=Path, help="optional per-interval comparison CSV")
    args = parser.parse_args()

    result = ModelComparison().run(args.csv)
    print(f"Compared intervals: {len(result.samples)}")
    print("RMSE:")
    for name, value in result.rmse.items():
        print(f"  {name}: {value:.6g}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    if args.output:
        result.write_csv(args.output)
        print(f"Wrote per-interval results to {args.output}")


if __name__ == "__main__":
    main()