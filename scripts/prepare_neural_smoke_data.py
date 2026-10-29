"""Make a small real-data smoke fixture without selecting by rating/holdout.

Run on the original Kaggle CSV files. ISBN selection uses only a metadata hash;
the resulting subset is for executable notebook evidence, not a benchmark.
"""
import argparse
import csv
import hashlib
import heapq
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--books", type=int, default=5000)
    args = parser.parse_args()
    if args.books < 1:
        parser.error("--books must be positive")
    if args.data_dir.resolve() == args.output.resolve():
        parser.error("--output must differ from --data-dir")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.data_dir / "Books.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        books = heapq.nsmallest(
            args.books, reader,
            key=lambda row: hashlib.sha256(row["ISBN"].encode()).hexdigest(),
        )
    isbns = {row["ISBN"] for row in books}
    with (args.output / "Books.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(books)
    users = set()
    rating_count = 0
    with (
        (args.data_dir / "Ratings.csv").open(encoding="utf-8", newline="") as inp,
        (args.output / "Ratings.csv").open("w", encoding="utf-8", newline="") as out,
    ):
        reader = csv.DictReader(inp)
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["ISBN"] in isbns:
                writer.writerow(row)
                users.add(row["User-ID"])
                rating_count += 1
    with (
        (args.data_dir / "Users.csv").open(encoding="utf-8", newline="") as inp,
        (args.output / "Users.csv").open("w", encoding="utf-8", newline="") as out,
    ):
        reader = csv.DictReader(inp)
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(row for row in reader if row["User-ID"] in users)
    print({"books": len(books), "raw_ratings": rating_count, "users": len(users)})


if __name__ == "__main__":
    main()
