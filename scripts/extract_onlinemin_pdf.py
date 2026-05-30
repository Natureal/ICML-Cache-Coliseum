from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def main() -> None:
    pdf_path = Path(__file__).resolve().parents[1] / "OnlineMin.pdf"
    reader = PdfReader(str(pdf_path))
    print("pages", len(reader.pages))

    # Dump text for grep-friendly inspection
    out_path = Path(__file__).resolve().parents[1] / "OnlineMin.txt"
    with out_path.open("w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            f.write(f"\n\n===== PAGE {i+1} =====\n\n")
            f.write(text)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
