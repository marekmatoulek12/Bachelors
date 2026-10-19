import argparse
import csv
from math import degrees
from pathlib import Path

import matplotlib.pyplot as plt


WHEELS = ("fl", "fr", "rl", "rr")
WHEEL_COLORS = {
    "fl": "tab:red",
    "fr": "tab:orange",
    "rl": "tab:green",
    "rr": "tab:purple",
}
FILE_LINESTYLES = ("-", "--", ":", "-.")
TIME_PLOT_COMMANDS = {"all-wheels", "vels-accs"}
SLIP_PLOT_COMMANDS = {f"f-to-slip-{wheel}" for wheel in WHEELS}
COMPARISON_COMMANDS = TIME_PLOT_COMMANDS | SLIP_PLOT_COMMANDS

WHEEL_FORCE_COLUMNS = {
    f"{wheel}_{column}"
    for wheel in WHEELS
    for column in ("fx", "fy", "slip_ratio", "slip_angle")
}
RESPONSE_COLUMNS = {
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
}
COMPARISON_COLUMNS = {"timestamp"} | WHEEL_FORCE_COLUMNS | RESPONSE_COLUMNS


def default_csv_files():
    return sorted(path for path in Path(__file__).parent.glob("*.csv"))


def default_comparison_csv_files():
    """Return adjacent CSVs with the schema emitted by model_comparison.py."""
    compatible = []
    for file_path in default_csv_files():
        try:
            with file_path.open(newline="", encoding="utf-8") as csv_file:
                columns = set(next(csv.reader(csv_file), []))
        except (OSError, csv.Error):
            continue
        if COMPARISON_COLUMNS.issubset(columns):
            compatible.append(file_path)
    return compatible


def read_comparison_csv(file_path, required_columns):
    """Read required numeric columns and convert timestamps to elapsed time."""
    with file_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError(f"File {file_path} has no header.")
        missing = set(required_columns) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"File {file_path} is not a comparison CSV; missing columns: "
                f"{', '.join(sorted(missing))}"
            )

        timestamps = []
        values = {column: [] for column in required_columns if column != "timestamp"}
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamps.append(float(row["timestamp"]))
                for column in values:
                    values[column].append(float(row[column]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Non-numeric comparison value in {file_path}, row {row_number}.") from error

    if not timestamps:
        raise ValueError(f"File {file_path} contains no data rows.")
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError(f"Timestamps in {file_path} must be strictly increasing.")

    start_time = timestamps[0]
    elapsed_time = [timestamp - start_time for timestamp in timestamps]
    return elapsed_time, values


def plot_column(column_name, file_paths):
    """Plot one CSV column against elapsed time for one or more files."""
    for file_path in file_paths:
        with file_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames:
                raise ValueError(f"Soubor {file_path} nema hlavicku.")
            if "timestamp" not in reader.fieldnames:
                raise ValueError(f"V souboru {file_path} chybi sloupec 'timestamp'.")
            if column_name not in reader.fieldnames:
                available = ", ".join(reader.fieldnames)
                raise ValueError(
                    f"Sloupec '{column_name}' v souboru {file_path} neni. "
                    f"Dostupne sloupce: {available}"
                )

            rows = list(reader)
            if not rows:
                raise ValueError(f"Soubor {file_path} neobsahuje zadna data.")

            timestamps = [float(row["timestamp"]) for row in rows]
            values = [float(row[column_name]) for row in rows]
            start_time = timestamps[0]
            elapsed_time = [timestamp - start_time for timestamp in timestamps]
            plt.plot(elapsed_time, values, label=file_path.stem)

    plt.xlabel("Cas od zacatku zaznamu [s]")
    plt.ylabel(column_name)
    plt.title(f"{column_name} v case")
    if len(file_paths) > 1:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_all_wheels(file_paths):
    required = {"timestamp"} | WHEEL_FORCE_COLUMNS
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    plot_specs = (
        (axes[0, 0], "fx", "Longitudinal tire force", "Fx [N]"),
        (axes[0, 1], "fy", "Lateral tire force", "Fy [N]"),
        (axes[1, 0], "slip_ratio", "Longitudinal slip ratio", "Slip ratio [-]"),
        (axes[1, 1], "slip_angle", "Lateral slip angle", "Slip angle [deg]"),
    )

    for file_index, file_path in enumerate(file_paths):
        elapsed_time, values = read_comparison_csv(file_path, required)
        line_style = FILE_LINESTYLES[file_index % len(FILE_LINESTYLES)]
        for axis, column, _, _ in plot_specs:
            for wheel in WHEELS:
                samples = values[f"{wheel}_{column}"]
                if column == "slip_angle":
                    samples = [degrees(value) for value in samples]
                axis.plot(
                    elapsed_time,
                    samples,
                    color=WHEEL_COLORS[wheel],
                    linestyle=line_style,
                    label=f"{file_path.stem} {wheel.upper()}",
                )

    for axis, _, title, ylabel in plot_specs:
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Time [s]")
        axis.grid(True)
        axis.legend()
    figure.suptitle("All wheel forces and slips over time")
    figure.tight_layout()
    plt.show()


def plot_velocities_and_accelerations(file_paths):
    required = {"timestamp"} | RESPONSE_COLUMNS
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    plot_specs = (
        (axes[0, 0], "vx", "Longitudinal velocity", "Velocity [m/s]"),
        (axes[0, 1], "yaw_rate", "Yaw rate", "Yaw rate [rad/s]"),
        (axes[1, 0], "accel_x", "Longitudinal acceleration", "Acceleration [m/s^2]"),
        (axes[1, 1], "accel_y", "Lateral acceleration", "Acceleration [m/s^2]"),
    )
    series_names = (
        ("pred", "Predicted"),
        ("imu", "IMU"),
        ("error", "Error"),
    )
    response_columns = {
        "vx": ("vx_pred_next", "vx_measured_next", "vx_error"),
        "yaw_rate": ("yaw_rate_pred_next", "yaw_rate_measured_next", "yaw_rate_error"),
        "accel_x": ("accel_x_pred", "accel_x_imu", "accel_x_error"),
        "accel_y": ("accel_y_pred", "accel_y_imu", "accel_y_error"),
    }

    for file_index, file_path in enumerate(file_paths):
        elapsed_time, values = read_comparison_csv(file_path, required)
        line_style = FILE_LINESTYLES[file_index % len(FILE_LINESTYLES)]
        for axis, signal, _, _ in plot_specs:
            for column, (_, label) in zip(response_columns[signal], series_names):
                axis.plot(
                    elapsed_time,
                    values[column],
                    linestyle=line_style,
                    label=f"{file_path.stem} {label}",
                )

    for axis, _, title, ylabel in plot_specs:
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Time [s]")
        axis.grid(True)
        axis.legend()
    figure.suptitle("Predicted and measured vehicle response")
    figure.tight_layout()
    plt.show()


def plot_wheel_force_vs_slip(wheel, file_paths):
    required = {
        "timestamp",
        f"{wheel}_fx",
        f"{wheel}_fy",
        f"{wheel}_slip_ratio",
        f"{wheel}_slip_angle",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for file_path in file_paths:
        _, values = read_comparison_csv(file_path, required)
        slip_angle_deg = [degrees(value) for value in values[f"{wheel}_slip_angle"]]
        axes[0].scatter(
            values[f"{wheel}_slip_ratio"],
            values[f"{wheel}_fx"],
            s=18,
            alpha=0.65,
            label=file_path.stem,
        )
        axes[1].scatter(
            slip_angle_deg,
            values[f"{wheel}_fy"],
            s=18,
            alpha=0.65,
            label=file_path.stem,
        )

    axes[0].set_title(f"{wheel.upper()} longitudinal force vs slip")
    axes[0].set_xlabel("Slip ratio [-]")
    axes[0].set_ylabel("Fx [N]")
    axes[1].set_title(f"{wheel.upper()} lateral force vs slip angle")
    axes[1].set_xlabel("Slip angle [deg]")
    axes[1].set_ylabel("Fy [N]")
    for axis in axes:
        axis.grid(True)
        axis.legend()
    figure.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a CSV column or comparison data using one of the built-in views."
        ),
        epilog=(
            "Examples:\n"
            "  python plot_csv.py all-wheels\n"
            "  python plot_csv.py vels-accs comparison.csv\n"
            "  python plot_csv.py f-to-slip-rl run1.csv run2.csv\n"
            "  python plot_csv.py vx_error comparison.csv\n"
            "  python plot_csv.py --list-columns comparison.csv\n\n"
            "With no file paths, comparison views select compatible comparison CSVs "
            "from this directory. Column plots use all CSVs in this directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "column",
        nargs="?",
        help="column name, all-wheels, vels-accs, or f-to-slip-fl/fr/rl/rr",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="CSV files to plot (default depends on the selected view)",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="print the columns in the first selected CSV file",
    )
    args = parser.parse_args()

    if args.list_columns:
        file_paths = args.files or default_csv_files()
        if not file_paths:
            parser.error("Nebyly nalezeny zadne CSV soubory.")
        with file_paths[0].open(newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            print("\n".join(next(reader, [])))
        return

    if not args.column:
        parser.error("Specify a column, a comparison plot command, or use --list-columns.")

    if args.column in COMPARISON_COMMANDS:
        file_paths = args.files or default_comparison_csv_files()
        if not file_paths:
            parser.error("No comparison-format CSV files were found; provide one explicitly.")
        try:
            if args.column == "all-wheels":
                plot_all_wheels(file_paths)
            elif args.column == "vels-accs":
                plot_velocities_and_accelerations(file_paths)
            else:
                wheel = args.column.rsplit("-", 1)[1]
                plot_wheel_force_vs_slip(wheel, file_paths)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return

    file_paths = args.files or default_csv_files()
    if not file_paths:
        parser.error("Nebyly nalezeny zadne CSV soubory.")

    try:
        plot_column(args.column, file_paths)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()