import psutil


def main() -> None:
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        name = (process.info.get("name") or "").lower()
        if name != "curl.exe" and name != "curl":
            continue
        print(f"stopping pid={process.info['pid']} cmdline={process.info.get('cmdline')}")
        process.terminate()
        try:
            process.wait(timeout=5)
        except psutil.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()
