from pathlib import Path

from ai_weather_rainfall.data_ecmwf import open_forecast_file


def main() -> None:
    for path in sorted(Path("outputs/rainfall_benchmark/raw/ecmwf").glob("**/*.grib2")):
        dataset = open_forecast_file(path)
        leads = dataset["lead_hour"].values.tolist()
        units = dataset["tp"].attrs.get("units")
        print(f"{path} shape={dataset['tp'].shape} units={units} leads={leads}")


if __name__ == "__main__":
    main()
