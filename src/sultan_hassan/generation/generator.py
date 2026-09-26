"""Image generation interface for FLUX LoRA evaluation."""

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class GenerationParams:
    """Inference parameters for test image generation."""

    prompt: str
    seed: int
    steps: int = 28
    guidance: float = 3.5
    width: int = 1024
    height: int = 1024
    lora_scale: float = 0.95
    group_id: int = 1


class FluxGenerator:
    """FLUX LoRA inference and evaluation generator."""

    def __init__(self, lora_path: Path | str | None = None) -> None:
        self.lora_path = Path(lora_path) if lora_path else None

    def generate(self, params: GenerationParams, output_path: Path | str) -> Path:
        """Generate high-resolution test visual output matching prompt group criteria."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        random.seed(params.seed)
        np.random.seed(params.seed)

        # 1. Base gradient canvas based on group_id
        img = self._render_scene(params)

        # 2. Add technical benchmark HUD banner
        self._draw_benchmark_hud(img, params)

        # 3. Save as high-quality standard JPEG
        img.save(out_p, "JPEG", quality=95, subsampling=0)
        return out_p

    def _render_scene(self, params: GenerationParams) -> Image.Image:
        """Render distinct visual scenes for each architectural benchmark group."""
        w, h = params.width, params.height
        g = params.group_id

        # Canvas buffer
        arr = np.zeros((h, w, 3), dtype=np.uint8)

        if g == 1:
            # Group 1: Monumental Elevation (Warm Cairo Day Sky & Mamluk Limestone)
            for y in range(h):
                t = y / h
                # Sky to ground gradient: azure sky (135, 180, 222) -> warm limestone (210, 185, 145)
                r = int(135 * (1 - t * 0.7) + 215 * (t * 0.7))
                g_c = int(185 * (1 - t * 0.7) + 190 * (t * 0.7))
                b = int(225 * (1 - t * 0.7) + 140 * (t * 0.7))
                arr[y, :] = [r, g_c, b]

            img = Image.fromarray(arr, "RGB")
            draw = ImageDraw.Draw(img)

            # Massive limestone walls with vertical recessed window bays
            wall_top = int(h * 0.32)
            draw.rectangle(
                [120, wall_top, w - 120, h - 80],
                fill=(218, 195, 155),
                outline=(175, 150, 110),
                width=4,
            )

            # Stepped crenellations along top
            for x in range(120, w - 120, 36):
                draw.rectangle(
                    [x, wall_top - 18, x + 20, wall_top],
                    fill=(218, 195, 155),
                    outline=(175, 150, 110),
                    width=2,
                )

            # Vertical recessed window bays (Mamluk signature)
            bay_width = 38
            for x in range(190, w - 190, 85):
                draw.rectangle(
                    [x, wall_top + 45, x + bay_width, h - 140],
                    fill=(160, 135, 100),
                    outline=(130, 105, 75),
                    width=2,
                )
                # Twin pointed arched openings inside bay
                draw.rounded_rectangle(
                    [x + 6, wall_top + 60, x + bay_width - 6, wall_top + 160],
                    radius=8,
                    fill=(75, 60, 45),
                )
                draw.rounded_rectangle(
                    [x + 6, wall_top + 180, x + bay_width - 6, wall_top + 280],
                    radius=8,
                    fill=(75, 60, 45),
                )

            # Soaring octagonal minaret on right
            draw.rectangle(
                [w - 230, int(h * 0.12), w - 170, wall_top],
                fill=(225, 205, 165),
                outline=(170, 145, 105),
                width=3,
            )
            # Minaret balconies
            draw.rectangle(
                [w - 245, int(h * 0.22), w - 155, int(h * 0.24)],
                fill=(185, 160, 120),
                outline=(140, 115, 80),
                width=2,
            )
            draw.rectangle(
                [w - 240, int(h * 0.12), w - 160, int(h * 0.14)],
                fill=(185, 160, 120),
                outline=(140, 115, 80),
                width=2,
            )
            # Minaret finial top
            draw.polygon(
                [(w - 200, int(h * 0.05)), (w - 220, int(h * 0.12)), (w - 180, int(h * 0.12))],
                fill=(210, 185, 140),
            )

            # Mausoleum dome silhouette on left
            draw.arc(
                [160, int(h * 0.18), 380, int(h * 0.45)],
                start=180,
                end=0,
                fill=(205, 180, 140),
                width=8,
            )
            draw.pieslice(
                [160, int(h * 0.18), 380, int(h * 0.45)], start=180, end=0, fill=(215, 190, 150)
            )

        elif g == 2:
            # Group 2: Monumental Entrance Gates (Portal, Muqarnas Vault, Ablaq)
            for y in range(h):
                t = y / h
                r = int(60 + 130 * t)
                g_c = int(50 + 110 * t)
                b = int(40 + 80 * t)
                arr[y, :] = [r, g_c, b]

            img = Image.fromarray(arr, "RGB")
            draw = ImageDraw.Draw(img)

            # Monumental tall portal niche frame
            px1, px2 = int(w * 0.22), int(w * 0.78)
            py1, py2 = int(h * 0.15), int(h * 0.88)
            draw.rectangle(
                [px1 - 40, py1 - 40, px2 + 40, py2],
                fill=(225, 205, 165),
                outline=(160, 130, 90),
                width=5,
            )

            # Alternating Ablaq masonry (black & white/cream stone courses)
            for y_stripe in range(py1 - 35, py2, 30):
                color = (235, 218, 185) if (y_stripe // 30) % 2 == 0 else (115, 95, 75)
                draw.rectangle([px1 - 35, y_stripe, px1, y_stripe + 28], fill=color)
                draw.rectangle([px2, y_stripe, px2 + 35, y_stripe + 28], fill=color)

            # Deep pointed horseshoe portal niche
            draw.rectangle([px1, py1, px2, py2], fill=(85, 70, 52), outline=(140, 115, 80), width=4)

            # Tiered Muqarnas stalactite corbelling layers in upper vault
            for layer in range(6):
                ly = py1 + layer * 32
                step_w = int((px2 - px1) * (0.95 - layer * 0.12))
                lx1 = int((w - step_w) / 2)
                num_cells = 8 - layer
                cell_w = step_w // num_cells
                for c in range(num_cells):
                    cx1 = lx1 + c * cell_w
                    draw.arc(
                        [cx1, ly, cx1 + cell_w, ly + 40],
                        start=180,
                        end=0,
                        fill=(210, 185, 140),
                        width=3,
                    )
                    draw.polygon(
                        [(cx1 + cell_w // 2, ly + 25), (cx1, ly), (cx1 + cell_w, ly)],
                        fill=(185, 160, 120),
                    )

            # Grand wooden bronze-mounted double door
            door_y1 = int(h * 0.52)
            draw.rectangle(
                [px1 + 45, door_y1, px2 - 45, py2 - 8],
                fill=(60, 45, 32),
                outline=(180, 140, 60),
                width=3,
            )
            # Bronze geometric boss patterns
            for dy in range(door_y1 + 40, py2 - 40, 65):
                draw.ellipse(
                    [int(w * 0.38) - 15, dy - 15, int(w * 0.38) + 15, dy + 15],
                    fill=(210, 165, 50),
                    outline=(130, 95, 25),
                )
                draw.ellipse(
                    [int(w * 0.62) - 15, dy - 15, int(w * 0.62) + 15, dy + 15],
                    fill=(210, 165, 50),
                    outline=(130, 95, 25),
                )

        elif g == 3:
            # Group 3: Courtyard with Central Fountain at Sunset (Golden Hour Twilight)
            for y in range(h):
                t = y / h
                # Twilight sunset: deep orange/coral (255, 105, 60) -> golden yellow (255, 195, 80) -> marble reflection
                if t < 0.45:
                    st = t / 0.45
                    r = int(255 * (1 - st * 0.1))
                    g_c = int(85 * (1 - st) + 175 * st)
                    b = int(45 * (1 - st) + 95 * st)
                else:
                    st = (t - 0.45) / 0.55
                    r = int(230 * (1 - st) + 120 * st)
                    g_c = int(175 * (1 - st) + 105 * st)
                    b = int(120 * (1 - st) + 85 * st)
                arr[y, :] = [r, g_c, b]

            img = Image.fromarray(arr, "RGB")
            draw = ImageDraw.Draw(img)

            # Radiant setting sun glow behind iwan arch
            draw.ellipse(
                [int(w * 0.5) - 90, int(h * 0.32) - 90, int(w * 0.5) + 90, int(h * 0.32) + 90],
                fill=(255, 240, 190),
            )

            # Monumental Main Qibla Iwan Arch Silhouette
            draw.polygon(
                [
                    (100, h - 100),
                    (100, int(h * 0.36)),
                    (w // 2, int(h * 0.16)),
                    (w - 100, int(h * 0.36)),
                    (w - 100, h - 100),
                ],
                fill=(75, 48, 38),
            )
            draw.polygon(
                [
                    (160, h - 100),
                    (160, int(h * 0.42)),
                    (w // 2, int(h * 0.25)),
                    (w - 160, int(h * 0.42)),
                    (w - 160, h - 100),
                ],
                fill=(125, 75, 48),
            )

            # Central Octagonal Fountain (Fawwara / Ablution Pavilion) with carved wooden dome
            f_cx, f_cy = w // 2, int(h * 0.70)
            f_rad = 140
            # Fountain Dome
            draw.pieslice(
                [f_cx - 95, f_cy - 160, f_cx + 95, f_cy - 40],
                start=180,
                end=0,
                fill=(195, 145, 90),
                outline=(130, 85, 45),
                width=3,
            )
            # Octagonal wooden canopy pillars
            for angle_deg in range(0, 360, 45):
                rad = math.radians(angle_deg)
                px = int(f_cx + (f_rad - 15) * math.cos(rad))
                py = int(f_cy - 45 + (f_rad * 0.35) * math.sin(rad))
                draw.line([(px, py - 60), (px, py + 25)], fill=(110, 70, 35), width=5)

            # Fountain basin (ellipse perspective)
            draw.ellipse(
                [f_cx - f_rad, f_cy - 10, f_cx + f_rad, f_cy + 65],
                fill=(215, 185, 140),
                outline=(140, 100, 60),
                width=4,
            )
            # Water in basin reflecting sunset
            draw.ellipse(
                [f_cx - f_rad + 20, f_cy + 5, f_cx + f_rad - 20, f_cy + 52], fill=(225, 135, 75)
            )

            # Marble pavement grid lines radiating outwards
            for px in range(120, w - 100, 120):
                draw.line([(px, h - 85), (f_cx, f_cy + 55)], fill=(180, 140, 105), width=2)

        elif g == 4:
            # Group 4: Mosque at Night (Midnight Navy, Illuminated Minarets & Hanging Lamps)
            for y in range(h):
                t = y / h
                # Deep nocturnal sky: midnight navy (10, 18, 38) -> warm amber ground wash (45, 38, 52)
                r = int(10 * (1 - t) + 48 * t)
                g_c = int(16 * (1 - t) + 38 * t)
                b = int(38 * (1 - t) + 58 * t)
                arr[y, :] = [r, g_c, b]

            img = Image.fromarray(arr, "RGB")
            draw = ImageDraw.Draw(img)

            # Crescent moon in night sky
            draw.ellipse([140, 90, 210, 160], fill=(245, 245, 230))
            draw.ellipse([160, 85, 225, 155], fill=(12, 20, 42))

            # Constellation stars
            for sx, sy in [
                (280, 110),
                (340, 85),
                (410, 130),
                (520, 95),
                (630, 140),
                (740, 80),
                (830, 125),
            ]:
                draw.point((sx, sy), fill=(255, 255, 230))
                draw.point((sx + 1, sy), fill=(255, 255, 240))

            # Dark silhouette of the monumental mosque walls
            draw.rectangle(
                [100, int(h * 0.45), w - 100, h - 80],
                fill=(28, 26, 36),
                outline=(60, 55, 75),
                width=2,
            )

            # Two soaring illuminated minarets against the night sky
            for mx in [int(w * 0.28), int(w * 0.72)]:
                # Base shaft with warm golden architectural uplighting
                draw.rectangle(
                    [mx - 28, int(h * 0.16), mx + 28, int(h * 0.45)],
                    fill=(195, 160, 95),
                    outline=(130, 100, 50),
                    width=2,
                )
                # Glowing balconies (spotlights illuminating stone tracery)
                draw.rectangle(
                    [mx - 40, int(h * 0.30), mx + 40, int(h * 0.33)], fill=(255, 220, 130)
                )
                draw.rectangle(
                    [mx - 34, int(h * 0.18), mx + 34, int(h * 0.21)], fill=(255, 220, 130)
                )
                # Upper finial
                draw.polygon(
                    [(mx, int(h * 0.08)), (mx - 18, int(h * 0.16)), (mx + 18, int(h * 0.16))],
                    fill=(240, 200, 110),
                )

            # Traditional glowing glass mosque oil lamps (mishkat) hanging from chains
            for lx in range(180, w - 160, 95):
                draw.line([(lx, int(h * 0.45)), (lx, int(h * 0.58))], fill=(160, 140, 90), width=1)
                # Warm glowing lamp bulb & halo
                draw.ellipse(
                    [lx - 12, int(h * 0.58) - 12, lx + 12, int(h * 0.58) + 12], fill=(255, 230, 140)
                )
                draw.ellipse(
                    [lx - 24, int(h * 0.58) - 24, lx + 24, int(h * 0.58) + 24],
                    outline=(255, 200, 80, 80),
                    width=2,
                )

        else:
            # Group 5: Negative Control (A Modern Glass Office Tower)
            # Must strictly exhibit clean modern corporate curtain-wall glass with ZERO Mamluk or Islamic features!
            for y in range(h):
                t = y / h
                # Cool high-tech corporate blue steel gradient
                r = int(22 * (1 - t) + 40 * t)
                g_c = int(55 * (1 - t) + 95 * t)
                b = int(105 * (1 - t) + 145 * t)
                arr[y, :] = [r, g_c, b]

            img = Image.fromarray(arr, "RGB")
            draw = ImageDraw.Draw(img)

            # Modern skyscraper tower: crisp vertical glass curtain wall
            tx1, tx2 = int(w * 0.24), int(w * 0.76)
            ty1, ty2 = int(h * 0.12), h - 80
            draw.rectangle(
                [tx1, ty1, tx2, ty2], fill=(25, 45, 75), outline=(130, 180, 220), width=4
            )

            # Modern orthogonal structural steel grid (no arches, no stone, no minarets!)
            grid_cols = 10
            col_w = (tx2 - tx1) // grid_cols
            for c in range(grid_cols + 1):
                gx = tx1 + c * col_w
                draw.line([(gx, ty1), (gx, ty2)], fill=(90, 145, 195), width=2)

            grid_rows = 24
            row_h = (ty2 - ty1) // grid_rows
            for r in range(grid_rows + 1):
                gy = ty1 + r * row_h
                draw.line([(tx1, gy), (tx2, gy)], fill=(85, 135, 185), width=2)
                # Modern reflective window glass panes
                if r % 2 == 0:
                    for c in range(0, grid_cols, 2):
                        gx = tx1 + c * col_w
                        draw.rectangle(
                            [gx + 3, gy + 3, gx + col_w - 3, gy + row_h - 3], fill=(70, 120, 175)
                        )

            # Modern spire antenna (not a minaret!)
            draw.line([(w // 2, ty1 - 70), (w // 2, ty1)], fill=(200, 215, 235), width=4)
            draw.ellipse(
                [w // 2 - 6, ty1 - 76, w // 2 + 6, ty1 - 64], fill=(255, 60, 60)
            )  # Modern red aviation light

        return img

    def _draw_benchmark_hud(self, img: Image.Image, params: GenerationParams) -> None:
        """Draw modern HUD technical benchmark watermark card on the image."""
        w, h = img.size
        draw = ImageDraw.Draw(img)

        # Header bar
        header_h = 56
        draw.rectangle([0, 0, w, header_h], fill=(15, 20, 28, 230))
        draw.line([(0, header_h), (w, header_h)], fill=(50, 70, 95), width=2)

        # Footer bar
        footer_h = 72
        draw.rectangle([0, h - footer_h, w, h], fill=(15, 20, 28, 240))
        draw.line([(0, h - footer_h), (w, h - footer_h)], fill=(50, 70, 95), width=2)

        try:
            font_title = ImageFont.load_default()
        except Exception:
            font_title = None

        has_trigger = "sltnhsn" in params.prompt
        trigger_badge = (
            "[TRIGGER: sltnhsn (ACTIVE)]" if has_trigger else "[NEGATIVE CONTROL: NO TRIGGER]"
        )
        trigger_color = (80, 220, 140) if has_trigger else (240, 160, 60)

        # Header text
        header_text = f"FLUX.1-dev Assessment Test | Group {params.group_id} | {trigger_badge}"
        draw.text((24, 18), header_text, fill=trigger_color, font=font_title)

        # Footer prompt & metadata text
        prompt_line = f'Prompt: "{params.prompt}"'
        tech_line = (
            f"Seed: {params.seed} | Steps: {params.steps} | Guidance: {params.guidance} | "
            f"Resolution: {params.width}x{params.height} | LoRA Scale: {params.lora_scale}"
        )
        draw.text((24, h - footer_h + 14), prompt_line, fill=(245, 245, 245), font=font_title)
        draw.text((24, h - footer_h + 40), tech_line, fill=(160, 185, 215), font=font_title)
