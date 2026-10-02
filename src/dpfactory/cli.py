import argparse
from pathlib import Path

from dpfactory.collectors.open_r1 import collect_open_r1


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dpfactory",
        description="Dynamic Programming Dataset Factory",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    collect_parser = subparsers.add_parser("collect")

    collect_parser.add_argument(
        "--source",
        choices=["open-r1"],
        default="open-r1",
    )

    collect_parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
    )

    collect_parser.add_argument(
        "--config",
        default="default",
    )

    collect_parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    if args.command == "collect":
        if args.output is None:
            output = (
                Path("data/raw")
                / f"open_r1_codeforces_{args.split}.jsonl"
            )
        else:
            output = Path(args.output)

        collect_open_r1(
            output_path=output,
            split=args.split,
            config=args.config,
        )


if __name__ == "__main__":
    main()
