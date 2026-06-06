from __future__ import annotations

import json
import platform


def main() -> None:
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_import_ok": False,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "cuda_tensor_ok": False,
    }
    try:
        import torch
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    result["torch_import_ok"] = True
    result["torch_version"] = torch.__version__
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["device_count"] = int(torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        result["devices"].append(
            {
                "index": index,
                "name": props.name,
                "memory_gb": round(props.total_memory / (1024**3), 2),
            }
        )
    if result["cuda_available"]:
        try:
            tensor = torch.ones((16, 16), device="cuda")
            result["cuda_tensor_ok"] = bool((tensor @ tensor).sum().detach().cpu().item() > 0)
        except Exception as exc:
            result["cuda_tensor_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
