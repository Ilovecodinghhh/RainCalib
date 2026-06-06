from pathlib import Path


def main() -> None:
    path = Path("token.txt")
    if not path.exists():
        print("token.txt missing")
        return
    text = path.read_text(encoding="utf-8-sig").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    print(f"nonempty_lines={len(lines)} total_chars={len(text)}")
    for index, line in enumerate(lines, start=1):
        lower = line.lower()
        print(
            f"line_{index}: chars={len(line)} "
            f"starts_bearer={lower.startswith('bearer ')} "
            f"has_equal={'=' in line} has_colon={':' in line} "
            f"dot_count={line.count('.')}"
        )
        if "=" in line:
            key = line.split("=", 1)[0].strip()
            print(f"line_{index}_key={key}")


if __name__ == "__main__":
    main()
