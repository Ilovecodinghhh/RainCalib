import numpy as np
import xarray as xr


def precipitation_to_millimeters(data: xr.DataArray) -> xr.DataArray:
    units = str(data.attrs.get("units", "")).strip().lower().replace(" ", "")
    if units in {"m", "metre", "metres", "meter", "meters"}:
        converted = data * 1000.0
    elif units in {"mm", "millimetre", "millimetres", "millimeter", "millimeters",
                   "kgm**-2", "kgm^-2", "kgm-2", "kg/m2", "kg/m^2", "kg/m**2"}:
        converted = data.copy(deep=False)
    else:
        raise ValueError(f"Unsupported accumulated precipitation units: {data.attrs.get('units')!r}")
    converted = converted.where(np.isfinite(converted))
    converted.name = "tp"
    converted.attrs = {**data.attrs, "units": "mm"}
    return converted


def accumulated_to_window_total(tp_accumulated: xr.DataArray, accumulation_hours: int) -> xr.DataArray:
    if "lead_hour" not in tp_accumulated.dims:
        raise ValueError("tp_accumulated must have a lead_hour dimension")
    if accumulation_hours <= 0:
        raise ValueError("accumulation_hours must be positive")

    leads = [int(v) for v in tp_accumulated["lead_hour"].values]
    lead_set = set(leads)
    out = []
    for lead in leads:
        start = lead - accumulation_hours
        if start < 0 or (start > 0 and start not in lead_set):
            window = tp_accumulated.sel(lead_hour=lead) * np.nan
        elif start == 0:
            window = tp_accumulated.sel(lead_hour=lead)
        else:
            window = tp_accumulated.sel(lead_hour=lead) - tp_accumulated.sel(lead_hour=start)
        window = window.where(window >= 0)
        out.append(window.expand_dims(lead_hour=[lead]))

    result = xr.concat(out, dim="lead_hour")
    result.name = f"rainfall_{accumulation_hours}h_mm"
    result.attrs["units"] = "mm"
    return result
