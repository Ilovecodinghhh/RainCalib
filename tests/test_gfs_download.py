import pytest

from scripts.download_gfs_historical import select_cumulative_apcp_record


def test_select_cumulative_apcp_record_prefers_day_accumulation():
    index_text = "\n".join(
        [
            "596:426348655:d=2025090100:APCP:surface:18-24 hour acc fcst:",
            "597:426722355:d=2025090100:APCP:surface:0-1 day acc fcst:",
            "598:427000000:d=2025090100:WEASD:surface:24 hour fcst:",
        ]
    )

    record = select_cumulative_apcp_record(index_text, 24)

    assert record.start == 426722355
    assert record.end == 426999999
    assert "0-1 day acc" in record.description


def test_select_cumulative_apcp_record_requires_24h_steps():
    with pytest.raises(ValueError, match="multiple-of-24"):
        select_cumulative_apcp_record("", 30)
