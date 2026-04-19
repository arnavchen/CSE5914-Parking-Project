from __future__ import annotations

import json
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk


@dataclass(frozen=True)
class PickerConfig:
    image_path: str = "2012-09-12_06_05_16_jpg.rf.aa059af912ba1103c6fd330f06b04e3d.jpg"
    output_path: str = "homography_points.json"
    max_display_width: int = 1100
    max_display_height: int = 850
    point_radius: int = 5


CONFIG = PickerConfig()


class PointPicker:
    def __init__(self, root: tk.Tk, image_path: Path, output_path: Path) -> None:
        self.root = root
        self.image_path = image_path
        self.output_path = output_path

        self.original_image = Image.open(image_path).convert("RGB")
        self.display_image, self.scale = self._fit_image(self.original_image)
        self.photo = ImageTk.PhotoImage(self.display_image)
        self.points: list[tuple[float, float]] = []

        self.root.title("Homography Point Picker")

        self.status_var = tk.StringVar()
        self.status_var.set("Click 4 corners in order: top-left, top-right, bottom-right, bottom-left.")

        self.canvas = tk.Canvas(
            root,
            width=self.display_image.width,
            height=self.display_image.height,
            highlightthickness=0,
        )
        self.canvas.pack()
        self.canvas_image = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        controls = tk.Frame(root)
        controls.pack(fill="x", padx=8, pady=8)

        tk.Label(
            controls,
            text="Click order: top-left, top-right, bottom-right, bottom-left",
        ).pack(anchor="w")
        tk.Label(
            controls,
            text="Left click: add point | Backspace/U: undo | R: reset | S/Enter: save",
        ).pack(anchor="w")
        tk.Label(controls, textvariable=self.status_var).pack(anchor="w", pady=(6, 0))

        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<BackSpace>", self.undo_point)
        self.root.bind("<u>", self.undo_point)
        self.root.bind("<U>", self.undo_point)
        self.root.bind("<r>", self.reset_points)
        self.root.bind("<R>", self.reset_points)
        self.root.bind("<Return>", self.save_points)
        self.root.bind("<s>", self.save_points)
        self.root.bind("<S>", self.save_points)

        self.redraw()

    def _fit_image(self, image: Image.Image) -> tuple[Image.Image, float]:
        width, height = image.size
        scale = min(
            CONFIG.max_display_width / width,
            CONFIG.max_display_height / height,
            1.0,
        )
        if scale == 1.0:
            return image.copy(), 1.0

        new_size = (int(round(width * scale)), int(round(height * scale)))
        return image.resize(new_size, Image.Resampling.LANCZOS), scale

    def redraw(self) -> None:
        overlay = self.display_image.copy()
        draw = ImageDraw.Draw(overlay)

        scaled_points = [(x * self.scale, y * self.scale) for x, y in self.points]
        labels = ["TL", "TR", "BR", "BL"]

        if len(scaled_points) > 1:
            draw.line(scaled_points, fill=(255, 80, 80), width=2)

        for index, (x, y) in enumerate(scaled_points):
            r = CONFIG.point_radius
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(64, 220, 255), outline=(255, 255, 255))
            draw.text((x + 8, y + 8), labels[index], fill=(255, 255, 255))

        if len(scaled_points) == 4:
            draw.line([scaled_points[-1], scaled_points[0]], fill=(255, 80, 80), width=2)

        self.photo = ImageTk.PhotoImage(overlay)
        self.canvas.itemconfigure(self.canvas_image, image=self.photo)

        next_label = labels[len(self.points)] if len(self.points) < 4 else "ready to save"
        self.status_var.set(
            f"Points selected: {len(self.points)}/4. "
            f"Next: {next_label}."
        )

    def on_click(self, event: tk.Event) -> None:
        if len(self.points) >= 4:
            self.status_var.set("Already have 4 points. Press S to save, or R to reset.")
            return

        x = event.x / self.scale
        y = event.y / self.scale
        self.points.append((round(x, 2), round(y, 2)))
        self.redraw()

    def undo_point(self, _event: tk.Event | None = None) -> None:
        if self.points:
            self.points.pop()
            self.redraw()

    def reset_points(self, _event: tk.Event | None = None) -> None:
        self.points.clear()
        self.redraw()

    def save_points(self, _event: tk.Event | None = None) -> None:
        if len(self.points) != 4:
            self.status_var.set("Need exactly 4 points before saving.")
            return

        payload = {
            "image_path": str(self.image_path),
            "source_points": [list(point) for point in self.points],
            "order": ["top-left", "top-right", "bottom-right", "bottom-left"],
        }
        self.output_path.write_text(json.dumps(payload, indent=2))
        self.status_var.set(f"Saved points to {self.output_path}")


def main() -> None:
    image_path = Path(CONFIG.image_path)
    output_path = Path(CONFIG.output_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    root = tk.Tk()
    PointPicker(root, image_path, output_path)
    root.mainloop()


if __name__ == "__main__":
    main()
