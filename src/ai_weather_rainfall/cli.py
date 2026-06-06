import argparse
from pathlib import Path

from ai_weather_rainfall.pipeline import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI weather rainfall benchmark.")
    parser.add_argument("--config", type=Path, default=Path("configs/rainfall_benchmark.yaml"))
    args = parser.parse_args()
    report_path = run_benchmark(args.config)
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
