import numpy as np
import torch

from ai_weather_rainfall.deep.dataset import (
    RainCalibTensorDataset,
    event_targets,
    masked_smooth_l1,
    rain_intensity_weights,
)
from ai_weather_rainfall.deep.unet import SmallUNet


def test_small_unet_preserves_odd_grid_shape():
    model = SmallUNet(in_channels=8, base_channels=8, depth=2)
    x = torch.randn(2, 8, 21, 29)

    y = model(x)

    assert y.shape == (2, 1, 21, 29)


def test_small_unet_can_emit_event_channels():
    model = SmallUNet(in_channels=8, out_channels=3, base_channels=8, depth=2)
    x = torch.randn(2, 8, 21, 29)

    y = model(x)

    assert y.shape == (2, 3, 21, 29)


def test_masked_smooth_l1_ignores_masked_cells():
    prediction = torch.tensor([[[[10.0, 0.0]]]])
    target = torch.tensor([[[[0.0, 0.0]]]])
    mask = torch.tensor([[[[0.0, 1.0]]]])

    loss = masked_smooth_l1(prediction, target, mask)

    assert float(loss) == 0.0


def test_rain_intensity_weights_use_largest_matching_threshold():
    target = torch.tensor([[[[5.0, 30.0, 75.0]]]])
    mask = torch.ones_like(target)

    weights = rain_intensity_weights(target, mask, (25.0, 50.0), (2.0, 6.0))

    assert weights.tolist() == [[[[1.0, 2.0, 6.0]]]]


def test_event_targets_stack_threshold_channels():
    target = torch.tensor([[[[10.0, 60.0, 120.0]]]])

    targets = event_targets(target, (50.0, 100.0))

    assert targets.shape == (1, 2, 1, 3)
    assert targets.tolist() == [[[[0.0, 1.0, 1.0]], [[0.0, 0.0, 1.0]]]]


def test_raincalib_tensor_dataset_shapes(tmp_path):
    tensor_path = tmp_path / "tiny.npz"
    np.savez_compressed(
        tensor_path,
        gfs=np.ones((2, 3, 4), dtype="float32"),
        target=np.ones((2, 3, 4), dtype="float32") * 2,
        mask=np.ones((2, 3, 4), dtype="float32"),
        lead_hour=np.array([24, 48], dtype="float32"),
        cycle=np.array([0, 12], dtype="float32"),
        month_sin=np.zeros(2, dtype="float32"),
        month_cos=np.ones(2, dtype="float32"),
        day_of_year_sin=np.zeros(2, dtype="float32"),
        day_of_year_cos=np.ones(2, dtype="float32"),
        run_date=np.array(["2025-01-01", "2025-01-02"]),
        observed_date=np.array(["2025-01-01", "2025-01-02"]),
        split=np.array(["train", "test"]),
        latitude=np.arange(3, dtype="float32"),
        longitude=np.arange(4, dtype="float32"),
    )

    dataset = RainCalibTensorDataset(tensor_path, "train", rain_scale_mm=50.0, gfs_clip_mm=200.0)
    sample = dataset[0]

    assert sample["x"].shape == (8, 3, 4)
    assert sample["y"].shape == (1, 3, 4)

    indexed = RainCalibTensorDataset(
        tensor_path,
        indexes=np.array([1]),
        rain_scale_mm=50.0,
        gfs_clip_mm=200.0,
    )
    assert len(indexed) == 1
    assert indexed[0]["index"] == 1
