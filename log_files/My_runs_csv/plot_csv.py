import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def default_csv_files():
    return sorted(path for path in Path(__file__).parent.glob("*.csv"))


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


def main():
    parser = argparse.ArgumentParser(
        description="Vykresli vybrany sloupec CSV v case."
    )
    parser.add_argument("column", nargs="?", help="nazev sloupce k vykresleni")
    parser.add_argument(
        "files", nargs="*", type=Path, help="CSV soubory (vychozi: vsechny vedle skriptu)"
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="vypise sloupce prvniho nalezeneho CSV souboru",
    )
    args = parser.parse_args()

    file_paths = args.files or default_csv_files()
    if not file_paths:
        parser.error("Nebyly nalezeny zadne CSV soubory.")

    if args.list_columns:
        with file_paths[0].open(newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            print("\n".join(next(reader, [])))
        return

    if not args.column:
        parser.error("Zadej nazev sloupce, nebo pouzij --list-columns.")

    try:
        plot_column(args.column, file_paths)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()