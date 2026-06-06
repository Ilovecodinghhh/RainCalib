from pathlib import Path

from ecmwf.opendata import Client

from ai_weather_rainfall.download_plan import required_lead_steps


DATE = "20251001"
CYCLES = [0, 12]
LEAD_HOURS = [24, 48, 72, 96, 120, 168, 240, 360]
STEPS = required_lead_steps(LEAD_HOURS, accumulation_hours=24)


def main() -> None:
    requests = [
        ("ifs", "ifs", Path("outputs/rainfall_benchmark/raw/ecmwf/ifs")),
        ("aifs-single", "aifs", Path("outputs/rainfall_benchmark/raw/ecmwf/aifs")),
    ]
    for model, label, out_dir in requests:
        out_dir.mkdir(parents=True, exist_ok=True)
        client = Client(source="aws", model=model)
        for cycle in CYCLES:
            target = out_dir / f"{label}_{DATE}_{cycle:02d}_tp_steps.grib2"
            if target.exists() and target.stat().st_size > 0:
                print(f"exists {target} {target.stat().st_size}")
                continue
            print(f"downloading model={model} date={DATE} cycle={cycle:02d} target={target}")
            result = client.retrieve(
                date=DATE,
                time=cycle,
                type="fc",
                step=STEPS,
                param="tp",
                target=str(target),
            )
            print(f"done {target} {target.stat().st_size} run={getattr(result, 'datetime', None)}")


if __name__ == "__main__":
    main()
