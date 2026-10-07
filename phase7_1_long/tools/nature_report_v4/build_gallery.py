"""Build a self-contained browser gallery from generated publication PNGs."""
from __future__ import annotations

import argparse
import base64
from html import escape
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cards = []
    for path in sorted(args.figures.glob("figure*.png")):
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        cards.append(
            f'<section><h2>{escape(path.stem)}</h2>'
            f'<a href="data:image/png;base64,{data}" target="_blank">'
            f'<img src="data:image/png;base64,{data}"></a></section>'
        )
    html = f'''<!doctype html><meta charset="utf-8"><title>Phase7.1 publication figures</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:28px auto;max-width:1200px;color:#272727;background:#fff}}
h1{{font-size:24px}} h2{{font-size:15px;margin:0 0 12px}} section{{margin:36px 0 56px}}
img{{width:100%;height:auto;display:block}} .note{{color:#767676;font-size:13px}}
</style><h1>Phase7.1 publication figure review</h1>
<p class="note">Click a figure to open the embedded full-resolution PNG. Final manuscript assets remain PDF/SVG/TIFF.</p>
{''.join(cards)}'''
    args.out.write_text(html, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
