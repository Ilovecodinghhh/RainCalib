from pathlib import Path

import xarray as xr


def main() -> None:
    path = Path("outputs/rainfall_benchmark/raw/imerg/3B-DAY-L.MS.MRG.3IMERG.20251001-S000000-E235959.V07B.nc4")
    dataset = xr.open_dataset(path)
    print(dataset)
    print("data_vars", list(dataset.data_vars))


if __name__ == "__main__":
    main()
