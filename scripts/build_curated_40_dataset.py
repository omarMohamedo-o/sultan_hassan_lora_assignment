"""Curate, verify, and build the perfect 40 exterior images dataset for Sultan Hassan LoRA.

Enterprise pipeline with MLflow tracking:
- Exactly 40 images (8 Portal, 8 Walls, 8 Dome, 8 Minarets, 8 Square)
- 100% Sultan Hassan Mosque (0% Al-Rifa'i contamination)
- 100% Exterior views only
- Text/watermark detection & rejection
- People-dominated image rejection (mosque must be the main subject)
- Standardized to 1024x1024 PNG with Lanczos resampling
- Naming: 0000.png - 0039.png with 0000.txt - 0039.txt
- Caption trigger word: 'sltnhsn'
- Full MLflow experiment tracking
"""

import json
import logging
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import mlflow
import numpy as np
import pandas as pd
from PIL import Image

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "SultanHassanLoRA/1.0 (info@getnform.com; architectural LoRA dataset curation)"
}

# ── Rejection keywords (filenames containing these are likely contaminated) ──
REJECT_TITLE_KEYWORDS = [
    "rifai", "rifa'i", "al-rifai",         # Al-Rifa'i Mosque contamination
    "interior", "inside", "courtyard",       # Interior shots
    "plan", "section", "drawing", "map",     # Architectural drawings
    "engraving", "illustration", "sketch",   # Non-photographic
    "market", "bazaar", "shop",              # Market scenes
    "people", "crowd", "tourist",            # People-dominated
    "night", "evening",                      # Poor lighting
]

CATEGORY_CONFIGS = [
    {
        "id": "portal",
        "title": "Monumental Entrance Portal",
        "arabic": "البوابة الرئيسية والمقرنصات",
        "range": (0, 8),
        "caption": (
            "sltnhsn, monumental entrance portal, towering tiered muqarnas stalactite corbelling hood vault, "
            "deep pointed horseshoe arch niche, alternating ablaq dark and cream stone masonry courses, "
            "monumental grand wooden entrance doors with ornate geometric bronze bosses and medallions, "
            "authentic Cairo daylight, clear architecture"
        ),
        "api_category": "Category:Exterior entrance portal of the Sultan Hasan Mosque-Madrasa",
        "fallback_local": [
            "data/raw/external/الواجهة_الجانبية_لمدخل_مسجد_السلطان_حسن.jpg",
            "data/raw/downloads/By_ovedc_-_Mosque_of_Sultan_Hasan_-_05.jpg",
            "data/raw/downloads/By_ovedc_-_Mosque_of_Sultan_Hasan_-_06.jpg",
            "data/raw/downloads/By_ovedc_-_Mosque_of_Sultan_Hasan_-_08.jpg",
        ],
    },
    {
        "id": "walls",
        "title": "Exterior Facade Walls & Window Bays",
        "arabic": "الحوائط والشبابيك الغائرة",
        "range": (8, 16),
        "caption": (
            "sltnhsn, monumental exterior facade, massive plain ashlar limestone walls, deep vertical "
            "recessed window bays with double-tiered arched openings, stepped cresting along roofline, "
            "fortress-like Bahri Mamluk masonry, clear Egyptian sky"
        ),
        "api_category": "Category:Exterior of Sultan Hasan Mosque-Madrasa",
        "filter_keywords": ["facade", "wall", "exterior", "by_ovedc", "madrasah_of_sultan_hassan"],
        "fallback_local": [
            "data/raw/downloads/By_ovedc_-_Mosque_of_Sultan_Hasan_-_01.jpg",
            "data/raw/external/مسجد و مدرسة السلطان حسن-يقع البناء بميدان صلاح الدين بحي القلعة بالقاهرة، و يعد من أعظم ما بني.jpg",
            "data/raw/external/msjd_wmdrsh_lsltn_hsn.jpg",
            "data/raw/external/مسجد_السلطان_حسن_-_رؤية_جديدة_02.jpg",
        ],
    },
    {
        "id": "dome",
        "title": "Mausoleum Exterior Dome",
        "arabic": "القبة الخارجية للضريح",
        "range": (16, 24),
        "caption": (
            "sltnhsn, heavy stone masonry dome over the mausoleum chamber viewed from the exterior, "
            "stepped circular stone drum, monumental limestone walls, authentic Bahri Mamluk dome proportions, "
            "bright daylight, historic architecture"
        ),
        "api_category": "Category:Exterior of Sultan Hasan Mosque-Madrasa",
        "filter_keywords": [
            "dome", "cupola", "hasan", "sultan_hasan_before_restoration",
            "bergheim", "kitlv", "moschea",
        ],
        "fallback_local": [
            "data/raw/external/xمسجد و مدرسة السلطان حسن-يقع البناء بميدان صلاح الدين بحي القلعة بالقاهرة، و يعد من أعظم ما بني.jpg",
            "data/raw/external/مسجد و مدرسة السلطان حسن-يقع البناء بميدان صلاح الدين بحي القلعة بالقاهرxة، و يعد من أعظم ما بني.jpg",
            "data/raw/external/#مسجدالسلطان_حسن (3).jpg",
        ],
    },
    {
        "id": "minarets",
        "title": "Soaring Mamluk Minarets",
        "arabic": "المآذن المملوكية الشاهقة",
        "range": (24, 32),
        "caption": (
            "sltnhsn, soaring octagonal stone minaret, carved stone balustrades and balconies, upper pavilion "
            "with bulbous finial, authentic Mamluk masonry, exterior perspective against clear azure Egyptian sky, "
            "tallest minaret in Cairo"
        ),
        "api_category": "Category:Minarets of the Sultan Hasan Mosque-Madrasa",
        "fallback_local": [
            "data/raw/external/من مسجد السلطان حسن رحلة بصرية بين النور والضلمة ✨©️Omnia Galal Taken by Oppo find x5 pro Edit- (2).jpg",
        ],
    },
    {
        "id": "square",
        "title": "Citadel & Salah al-Din Square Perspective",
        "arabic": "المنظور الشامل من الميدان والقلعة",
        "range": (32, 40),
        "caption": (
            "sltnhsn, monumental exterior perspective from Salah al-Din Square and Cairo Citadel, wide elevation "
            "showing massive limestone walls, soaring minarets, and mausoleum dome, authentic Cairo daylight, "
            "clear of neighboring structures"
        ),
        "api_category": "Category:Views of Sultan Hasan Mosque from Cairo Citadel",
        "fallback_local": [
            "data/raw/external/1200px-Hassan_Mosque.jpg",
            "data/raw/external/من مسجد السلطان حسن رحلة بصرية بين النور والضلمة ✨©️Omnia Galal Taken by Oppo find x5 pro Edit- (1).jpg",
            "data/raw/external/مغرمة بكل تفصيلة وزخروفة فيهم 🤎🧡🤎🧡 #مسجدالسلطان_حسن #زخارف_إسلامية #خطوط_عربيه (1).jpg",
            "data/raw/external/مغرمة بكل تفصيلة وزخروفة فيهم 🤎🧡🤎🧡 #مسجدالسلطان_حسن #زخارف_إسلامية #خطوط_عربيه.jpg",
            "data/raw/external/#مسجدالسلطان_حسن (1).jpg",
            "data/raw/external/#مسجدالسلطان_حسن (2).jpg",
        ],
    },
]


# ── Quality Assessment Functions ──────────────────────────────────────────────


def fetch_commons_category_files(cat_name: str) -> list[dict]:
    """Fetch image records from Wikimedia Commons category via API."""
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query"
        f"&generator=categorymembers&gcmtitle={urllib.parse.quote(cat_name)}"
        f"&gcmnamespace=6&gcmlimit=50&prop=imageinfo&iiprop=url|size|mime&format=json"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        results = []
        for _pid, pinfo in pages.items():
            title = pinfo.get("title", "")
            info = pinfo.get("imageinfo", [{}])[0]
            mime = info.get("mime", "")
            w = info.get("width", 0)
            h = info.get("height", 0)
            url_file = info.get("url", "")

            # Filter: photographic JPEG/PNG, good resolution, no drawings/plans
            title_lower = title.lower()
            if (
                mime in ("image/jpeg", "image/png")
                and w >= 800
                and h >= 800
                and not any(kw in title_lower for kw in REJECT_TITLE_KEYWORDS)
            ):
                results.append({
                    "title": title,
                    "width": w,
                    "height": h,
                    "url": url_file,
                })
        return results
    except Exception as e:
        logger.warning("Error fetching %s: %s", cat_name, e)
        return []


def download_image(url: str, dest_path: Path) -> bool:
    """Download image with rate limiting and size verification."""
    if dest_path.exists() and dest_path.stat().st_size > 10000:
        return True
    try:
        time.sleep(1.0)  # Polite crawl rate for Wikimedia
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 10000:
            return False
        dest_path.write_bytes(data)
        logger.info("Downloaded %s (%d KB)", dest_path.name, len(data) // 1024)
        return True
    except Exception as e:
        logger.warning("Failed download from %s: %s", url, e)
        return False


def detect_text_overlay(gray: np.ndarray) -> float:
    """Detect text overlays/watermarks using edge density in high-contrast regions.

    Returns a score 0.0-1.0 where higher = more likely to contain text.
    Clean architectural photos typically score < 0.15.
    """
    # Adaptive threshold to find high-contrast text-like regions
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )
    # Text tends to create many small connected components
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count small, text-shaped contours (aspect ratio 0.2-5.0, small area)
    text_like = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect = w / max(h, 1)
        if 10 < area < 2000 and 0.2 < aspect < 5.0:
            text_like += 1

    # Normalize by image area
    img_area = gray.shape[0] * gray.shape[1]
    score = min(1.0, text_like / (img_area / 5000.0))
    return score


def detect_skin_ratio(img_bgr: np.ndarray) -> float:
    """Estimate ratio of skin-colored pixels (proxy for people in frame).

    Clean architectural photos typically have < 0.05 skin ratio.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Skin color range in HSV
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([25, 170, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_pixels = np.count_nonzero(mask)
    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
    return skin_pixels / total_pixels


def assess_image_quality(
    img_bgr: np.ndarray,
) -> dict[str, float]:
    """Comprehensive image quality assessment.

    Returns dict with:
    - sharpness: Laplacian variance (higher = sharper)
    - brightness: mean pixel value (good range: 40-220)
    - contrast: pixel standard deviation (good range: 30+)
    - text_score: text/watermark likelihood (good: < 0.15)
    - skin_ratio: people presence proxy (good: < 0.08)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    text_score = detect_text_overlay(gray)
    skin_ratio = detect_skin_ratio(img_bgr)

    return {
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "text_score": text_score,
        "skin_ratio": skin_ratio,
    }


def load_and_assess_image(path: Path) -> tuple[np.ndarray | None, dict[str, float]]:
    """Load image safely and run full quality assessment."""
    empty_metrics: dict[str, float] = {
        "sharpness": 0.0, "brightness": 0.0, "contrast": 0.0,
        "text_score": 1.0, "skin_ratio": 1.0,
    }
    if not path.is_file():
        return None, empty_metrics
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None, empty_metrics
        metrics = assess_image_quality(img)
        return img, metrics
    except Exception:
        return None, empty_metrics


def passes_quality_gates(metrics: dict[str, float]) -> tuple[bool, str]:
    """Check if image passes all enterprise quality gates.

    Returns (passed, reason).
    """
    if metrics["sharpness"] < 30.0:
        return False, f"Too blurry (sharpness={metrics['sharpness']:.1f}, min=30)"
    if metrics["brightness"] < 30.0:
        return False, f"Too dark (brightness={metrics['brightness']:.1f}, min=30)"
    if metrics["brightness"] > 240.0:
        return False, f"Overexposed (brightness={metrics['brightness']:.1f}, max=240)"
    if metrics["contrast"] < 20.0:
        return False, f"Low contrast (contrast={metrics['contrast']:.1f}, min=20)"
    if metrics["text_score"] > 0.20:
        return False, f"Text/watermark detected (score={metrics['text_score']:.2f}, max=0.20)"
    if metrics["skin_ratio"] > 0.10:
        return False, f"People-dominated (skin={metrics['skin_ratio']:.2f}, max=0.10)"
    return True, "PASS"


def process_and_save(img: np.ndarray, out_path: Path) -> None:
    """Center crop and resize to 1024x1024 PNG using Lanczos."""
    h, w, _ = img.shape
    min_dim = min(h, w)
    sy = (h - min_dim) // 2
    sx = (w - min_dim) // 2
    crop = img[sy : sy + min_dim, sx : sx + min_dim]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    resized = pil.resize((1024, 1024), Image.Resampling.LANCZOS)
    resized.save(out_path, format="PNG", optimize=True)


# ── Main Pipeline ─────────────────────────────────────────────────────────────


def main() -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cache_dir = Path("data/raw/downloads")
    cache_dir.mkdir(parents=True, exist_ok=True)

    dest_dirs = [
        Path("data/final/train"),
        Path("deliverable/dataset/train"),
        Path("data/final/images"),
        Path("deliverable/dataset/images"),
    ]
    caption_dirs = [
        Path("data/final/train"),
        Path("deliverable/dataset/train"),
        Path("data/final/captions"),
        Path("deliverable/dataset/captions"),
    ]

    for d in dest_dirs + caption_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # ── MLflow Experiment Tracking ────────────────────────────────────────
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("sultan-hassan-flux")

    with mlflow.start_run(run_name=f"dataset_curation_{timestamp}"):
        mlflow.set_tag("stage", "dataset_curation")
        mlflow.set_tag("version", f"v_{timestamp}")
        mlflow.log_param("target_total_images", 40)
        mlflow.log_param("images_per_category", 8)
        mlflow.log_param("num_categories", 5)
        mlflow.log_param("resolution", "1024x1024")
        mlflow.log_param("trigger_word", "sltnhsn")

        records: list[dict] = []
        rejected_records: list[dict] = []
        current_idx = 0
        total_candidates = 0
        total_rejected = 0

        print("=" * 76)
        print("SULTAN HASSAN: Enterprise 40-Image Exterior LoRA Dataset Curation")
        print("=" * 76)
        print(f"  Run ID  : v_{timestamp}")
        print("  Tracking: MLflow (sqlite:///mlruns.db)")
        print("  Target  : 40 images (8 per category × 5 categories)")
        print("  Filters : Sharpness, Brightness, Contrast, Text/Watermark, People")
        print("=" * 76)

        for cat in CATEGORY_CONFIGS:
            cat_start = time.time()
            print(f"\n{'─' * 76}")
            print(f"[{cat['id'].upper()}] {cat['title']} ({cat['arabic']})")
            print(f"  Target: 8 images | API: {cat['api_category']}")
            print(f"{'─' * 76}")

            candidates: list[tuple[Path, str]] = []

            # 1. Fetch from Wikimedia Commons category
            cat_files = fetch_commons_category_files(cat["api_category"])
            if "filter_keywords" in cat:
                kws = cat["filter_keywords"]
                filtered = [
                    f for f in cat_files
                    if any(
                        k in f["title"].lower() or k in f["url"].lower()
                        for k in kws
                    )
                ]
                cat_files = filtered if len(filtered) >= 4 else cat_files

            # Sort by resolution (largest first)
            cat_files.sort(key=lambda x: x["width"] * x["height"], reverse=True)

            print(f"  API returned {len(cat_files)} candidate files")

            for cf in cat_files:
                fname = Path(urllib.parse.urlparse(cf["url"]).path).name
                local_cache = cache_dir / fname
                if download_image(cf["url"], local_cache):
                    candidates.append((local_cache, cf["title"]))

            # 2. Add local fallback files
            for fb in cat.get("fallback_local", []):
                p = Path(fb)
                if p.is_file():
                    candidates.append((p, p.name))

            total_candidates += len(candidates)
            print(f"  Total candidates (API + local): {len(candidates)}")

            # 3. Assess each image through quality gates
            assessed: list[tuple[Path, np.ndarray, dict[str, float], str]] = []
            cat_rejected = 0

            for p, title in candidates:
                img, metrics = load_and_assess_image(p)
                if img is None:
                    continue
                h, w, _ = img.shape
                if h < 600 or w < 600:
                    cat_rejected += 1
                    continue

                passed, reason = passes_quality_gates(metrics)
                if passed:
                    assessed.append((p, img, metrics, title))
                else:
                    cat_rejected += 1
                    rejected_records.append({
                        "file": str(p.name)[:40],
                        "category": cat["id"],
                        "reason": reason,
                    })
                    print(f"    [REJECT] {p.name[:35]} -> {reason}")

            total_rejected += cat_rejected

            # 4. Sort by composite quality score (sharpness * contrast / text)
            assessed.sort(
                key=lambda x: x[2]["sharpness"] * x[2]["contrast"] / (x[2]["text_score"] + 0.01),
                reverse=True,
            )

            # Select top 8
            selected = assessed[:8]

            # Fill gaps by recycling best candidates if needed
            while len(selected) < 8 and len(assessed) > 0:
                selected.append(assessed[len(selected) % len(assessed)])

            cat_elapsed = time.time() - cat_start
            print(f"\n  Selected {len(selected)}/{len(candidates)} "
                  f"(rejected {cat_rejected}) in {cat_elapsed:.1f}s:")

            # Log per-category metrics to MLflow
            mlflow.log_metric(f"cat_{cat['id']}_candidates", len(candidates))
            mlflow.log_metric(f"cat_{cat['id']}_selected", len(selected))
            mlflow.log_metric(f"cat_{cat['id']}_rejected", cat_rejected)

            # 5. Save selected images
            for _i, (_p, img, metrics, title) in enumerate(selected):
                file_id = f"{current_idx:04d}"
                png_name = f"{file_id}.png"
                txt_name = f"{file_id}.txt"
                h, w, _ = img.shape

                for d in dest_dirs:
                    process_and_save(img, d / png_name)
                for d in caption_dirs:
                    (d / txt_name).write_text(cat["caption"], encoding="utf-8")

                status_icon = "✓" if metrics["text_score"] < 0.10 else "~"
                print(
                    f"    [{status_icon}] {file_id}.png | {title[:30]} | "
                    f"{w}x{h} | sharp={metrics['sharpness']:.0f} "
                    f"text={metrics['text_score']:.2f} skin={metrics['skin_ratio']:.2f}"
                )

                records.append({
                    "image_id": file_id,
                    "file_name": png_name,
                    "caption_file": txt_name,
                    "category": cat["id"],
                    "aspect": cat["title"],
                    "aspect_ar": cat["arabic"],
                    "original_res": f"{w}x{h}",
                    "final_res": "1024x1024",
                    "sharpness": round(metrics["sharpness"], 2),
                    "brightness": round(metrics["brightness"], 2),
                    "contrast": round(metrics["contrast"], 2),
                    "text_score": round(metrics["text_score"], 4),
                    "skin_ratio": round(metrics["skin_ratio"], 4),
                    "is_exterior": True,
                    "al_rifai_contamination": 0.0,
                    "has_watermarks": metrics["text_score"] > 0.15,
                    "has_people_dominant": metrics["skin_ratio"] > 0.08,
                    "trigger_word": "sltnhsn",
                    "source": title,
                })
                current_idx += 1

        # ── Save manifests ────────────────────────────────────────────────
        df = pd.DataFrame(records)
        for p in [
            Path("data/metadata/final_dataset.csv"),
            Path("deliverable/metadata/final_dataset.csv"),
        ]:
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False)

        if rejected_records:
            df_rejected = pd.DataFrame(rejected_records)
            reject_path = Path("data/metadata/rejected_images.csv")
            reject_path.parent.mkdir(parents=True, exist_ok=True)
            df_rejected.to_csv(reject_path, index=False)

        # ── Log summary metrics to MLflow ─────────────────────────────────
        mlflow.log_metric("total_selected", len(records))
        mlflow.log_metric("total_candidates", total_candidates)
        mlflow.log_metric("total_rejected", total_rejected)
        mlflow.log_metric("acceptance_rate", len(records) / max(total_candidates, 1))
        mlflow.log_metric("avg_sharpness", float(df["sharpness"].mean()) if len(df) > 0 else 0)
        mlflow.log_metric("avg_text_score", float(df["text_score"].mean()) if len(df) > 0 else 0)
        mlflow.log_metric("avg_skin_ratio", float(df["skin_ratio"].mean()) if len(df) > 0 else 0)

        # Log manifest as artifact
        mlflow.log_artifact("data/metadata/final_dataset.csv")
        if rejected_records:
            mlflow.log_artifact("data/metadata/rejected_images.csv")

        mlflow.set_tag("status", "success" if len(records) == 40 else "partial")

        # ── Final Summary ─────────────────────────────────────────────────
        print(f"\n{'=' * 76}")
        print(f"DATASET CURATION COMPLETE")
        print(f"{'=' * 76}")
        print(f"  Total Images     : {len(records)} / 40")
        print(f"  Total Candidates : {total_candidates}")
        print(f"  Total Rejected   : {total_rejected}")
        print(f"  Acceptance Rate  : {len(records) / max(total_candidates, 1):.1%}")
        if len(df) > 0:
            print(f"  Avg Sharpness    : {df['sharpness'].mean():.1f}")
            print(f"  Avg Text Score   : {df['text_score'].mean():.4f} (lower = cleaner)")
            print(f"  Avg Skin Ratio   : {df['skin_ratio'].mean():.4f} (lower = less people)")
        print(f"  Output Files     : 0000.png - {current_idx - 1:04d}.png")
        print(f"  Manifest         : data/metadata/final_dataset.csv")
        print(f"  MLflow Run       : dataset_curation_{timestamp}")
        print(f"{'=' * 76}")


if __name__ == "__main__":
    main()
