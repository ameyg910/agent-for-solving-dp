import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dpfactory",
        description="Dynamic Programming Dataset Factory",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("collect")
    subparsers.add_parser("normalize")
    subparsers.add_parser("deduplicate")
    subparsers.add_parser("classify")
    subparsers.add_parser("taxonomy")
    subparsers.add_parser("verify")
    subparsers.add_parser("generate")
    subparsers.add_parser("verify-generated")
    subparsers.add_parser("export")

    args = parser.parse_args()

    print(f"Command: {args.command}")


if __name__ == "__main__":
    main()
