from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        parts.append(f"\n\n===== PAGE {i} =====\n\n{text}")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PDF text to a grep-friendly .txt file")
    parser.add_argument("pdf", type=str, help="Path to input PDF")
    parser.add_argument("--out", type=str, default=None, help="Path to output .txt (default: <pdf>.txt)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    out_path = Path(args.out).expanduser().resolve() if args.out else pdf_path.with_suffix(".txt")
    text = extract_pdf_text(pdf_path)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
