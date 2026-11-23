"""Produce local, native-aspect overlays of the two largest losses and an edge control."""
import json
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from transfer import LOCAL, read_png


def main():
    rows = json.loads((LOCAL / "audit.json").read_bytes())["rows"]
    harmed = sorted((r for r in rows if r["outside"]), key=lambda r: -r["outside"]/r["target"])
    controls = [r for r in rows if r["boundary"] and not r["outside"] and
                (r["top"] or r["bottom"] < r["height"])]
    control = min(controls, key=lambda r: (r["gap"], r["id"]))
    font = ImageFont.load_default(size=17)
    heading = ImageFont.load_default(size=22)
    panel = Image.new("RGB", (1020, 386), "white")
    pen = ImageDraw.Draw(panel)
    pen.text((12, 8), "When the display filter cuts into a thyroid nodule", font=heading, fill="black")
    pen.text((12, 40), "Cyan outlines the annotation. Red marks labeled tissue outside the yellow limits.", font=font, fill="black")
    with zipfile.ZipFile(LOCAL / "TNSC2020.zip") as z:
        for col, r in enumerate([*harmed[:2], control]):
            name = r["id"]
            image = read_png(z, "image", name)
            rgb = np.array(image.convert("RGB"))
            target = np.asarray(read_png(z, "mask", name).convert("L")) > 0
            top, bottom = r["top"], r["bottom"]
            outside = target.copy()
            outside[top:bottom] = False
            assert int(outside.sum()) == r["outside"]
            outline = target ^ ndimage.binary_erosion(target, iterations=2)
            rgb[outline] = [0, 225, 255]
            rgb[outside] = .45 * rgb[outside] + .55 * np.array([255, 45, 65])
            rgb[outside & outline] = [255, 45, 65]
            overlay = Image.fromarray(rgb)
            lines = ImageDraw.Draw(overlay)
            for y in (top, bottom - 1):
                lines.line((0, y, overlay.width-1, y), fill="#ffd540", width=2)
            overlay.thumbnail((322, 234), Image.Resampling.LANCZOS)
            x = 12 + 335*col
            panel.paste(overlay, (x+(322-overlay.width)//2, 70+(234-overlay.height)//2))
            title = ("Largest loss", "Second-largest loss", "Boundary control")[col]
            pen.text((x, 314), title + "  ·  " + name, font=font, fill="black")
            pen.text((x, 343), f"{100*r['outside']/r['target']:.1f}% of labeled area removed",
                     font=font, fill="black")
    panel.save(LOCAL / "field-examples.png")


if __name__ == "__main__":
    main()
