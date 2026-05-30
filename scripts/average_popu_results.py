#!/usr/bin/env python3
"""Average POPU benchmark CSV columns across datasets, grouped by algorithm."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average every numeric POPU result column for each algorithm."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/popu"),
        help="Directory containing per-dataset CSV files (default: results/popu).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/popu/average.csv"),
        help="Output summary CSV path (default: results/popu/average.csv).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Number of decimal places written for averaged values (default: 6).",
    )
    return parser.parse_args()


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as result_file:
        reader = csv.DictReader(result_file)
        columns = reader.fieldnames
        if not columns or columns[0] != "Name" or len(columns) < 2:
            raise ValueError(f"{path}: expected Name followed by numeric columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no algorithm rows found")
    return columns, rows


def average_results(input_dir: Path, output: Path, precision: int) -> int:
    if precision < 0:
        raise ValueError("--precision must be non-negative")

    output_resolved = output.resolve()
    input_files = [
        path
        for path in sorted(input_dir.glob("*.csv"))
        if path.resolve() != output_resolved
    ]
    if not input_files:
        raise ValueError(f"{input_dir}: no input CSV files found")

    columns, first_rows = read_table(input_files[0])
    numeric_columns = columns[1:]
    algorithm_order = [row["Name"] for row in first_rows]
    if len(set(algorithm_order)) != len(algorithm_order):
        raise ValueError(f"{input_files[0]}: duplicate algorithm names found")

    sums = {
        algorithm: {column: Decimal("0") for column in numeric_columns}
        for algorithm in algorithm_order
    }
    expected_algorithms = set(algorithm_order)

    for path in input_files:
        current_columns, rows = read_table(path)
        if current_columns != columns:
            raise ValueError(
                f"{path}: columns differ from {input_files[0]}: {current_columns}"
            )

        rows_by_algorithm = {row["Name"]: row for row in rows}
        if len(rows_by_algorithm) != len(rows):
            raise ValueError(f"{path}: duplicate algorithm names found")
        if set(rows_by_algorithm) != expected_algorithms:
            raise ValueError(f"{path}: algorithm set differs from {input_files[0]}")

        for algorithm in algorithm_order:
            row = rows_by_algorithm[algorithm]
            for column in numeric_columns:
                try:
                    sums[algorithm][column] += Decimal(row[column])
                except (InvalidOperation, KeyError) as error:
                    raise ValueError(
                        f"{path}: invalid numeric value for {algorithm!r}, {column!r}"
                    ) from error

    quantizer = Decimal("1").scaleb(-precision)
    divisor = Decimal(len(input_files))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=columns)
        writer.writeheader()
        for algorithm in algorithm_order:
            row = {"Name": algorithm}
            for column in numeric_columns:
                value = (sums[algorithm][column] / divisor).quantize(
                    quantizer, rounding=ROUND_HALF_UP
                )
                row[column] = format(value, "f")
            writer.writerow(row)

    return len(input_files)


def main() -> None:
    args = parse_args()
    count = average_results(args.input_dir, args.output, args.precision)
    print(f"Averaged {count} result files into {args.output}.")


if __name__ == "__main__":
    main()
