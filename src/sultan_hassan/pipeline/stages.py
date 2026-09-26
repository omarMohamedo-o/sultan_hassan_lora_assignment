"""Pipeline stage definitions and workflow execution."""

import logging
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sultan_hassan.config.models import AppConfig
from sultan_hassan.dataset.selector import DatasetSelector
from sultan_hassan.dataset.validator import DatasetValidator
from sultan_hassan.domain.enums import QualityStatus, SemanticLabel
from sultan_hassan.filtering.deduplicator import FrameDeduplicator
from sultan_hassan.filtering.semantic_filter import SemanticFilterPipeline
from sultan_hassan.images.quality import QualityFilter
from sultan_hassan.images.standardizer import standardize_media_directory
from sultan_hassan.pipeline.state import PipelineStateManager
from sultan_hassan.review.contact_sheet import ContactSheetGenerator, create_thumbnail_card
from sultan_hassan.review.reviewer import run_review_server
from sultan_hassan.video.extractor import FrameExtractor
from sultan_hassan.video.inspector import VideoInspector

logger = logging.getLogger(__name__)
console = Console()


def run_collect_stage(config: AppConfig) -> dict[str, object]:
    """Execute Phase 1 through Phase 7 of data collection and curation.

    Steps:
    1. Inspect videos in data/raw/videos/
    2. Extract frames at configured sampling rate
    3. Filter for resolution (<1024), blur, extreme exposure
    4. Deduplicate using SHA256 and pHash/dHash clustering
    5. Semantic filtering (Sultan Hassan vs. Al-Rifa'i)
    6. Generate multi-category contact sheets (image & HTML)
    7. Return summary metrics dictionary
    """
    state_mgr = PipelineStateManager(config)
    manifest = state_mgr.manifest

    console.print(Panel.fit("[bold cyan]Stage 1: Video Ingestion & Inspection[/bold cyan]"))
    inspector = VideoInspector(config)
    videos = inspector.inspect_all()
    video_summary = inspector.get_summary(videos)
    console.print(f"Discovered [bold green]{len(videos)}[/bold green] raw video files.")

    if not videos:
        console.print("[yellow]No supported video files found in data/raw/videos/[/yellow]")
        return {"error": "No videos found"}

    console.print(Panel.fit("[bold cyan]Stage 2: Frame Extraction[/bold cyan]"))
    extractor = FrameExtractor(config, manifest)
    frames = extractor.extract_all(videos)
    console.print(f"Extracted [bold green]{len(frames)}[/bold green] candidate frames.")

    console.print(Panel.fit("[bold cyan]Stage 3: Image Quality & Resolution Filtering[/bold cyan]"))
    q_filter = QualityFilter(config, manifest)
    quality_results = q_filter.filter_all()

    # Aggregate quality metrics
    frames_below_1024 = sum(1 for q in quality_results if "below required" in q.rejection_reason)
    blurry_frames = sum(1 for q in quality_results if "blur" in q.rejection_reason.lower())
    dark_frames = sum(1 for q in quality_results if "dark" in q.rejection_reason.lower())
    passing_quality = [
        Path(config.paths.frames_raw) / q.local_filename
        for q in quality_results
        if q.quality_status in (QualityStatus.KEEP, QualityStatus.REVIEW)
    ]
    console.print(
        f"Quality gate passed: [bold green]{len(passing_quality)}[/bold green] / {len(quality_results)} frames."
    )

    console.print(Panel.fit("[bold cyan]Stage 4: Perceptual Deduplication[/bold cyan]"))
    q_map = {q.frame_id: q for q in quality_results}
    deduplicator = FrameDeduplicator(config, manifest)
    dedup_representatives, clusters = deduplicator.deduplicate(passing_quality, quality_map=q_map)
    total_duplicate_frames = len(passing_quality) - len(dedup_representatives)
    console.print(
        f"Deduplicated to [bold green]{len(dedup_representatives)}[/bold green] unique representatives "
        f"({total_duplicate_frames} duplicate frames removed)."
    )

    console.print(
        Panel.fit(
            "[bold cyan]Stage 5: Sultan Hassan vs. Al-Rifa'i Architectural Filter[/bold cyan]"
        )
    )
    semantic_pipeline = SemanticFilterPipeline(config, manifest)
    # Map frame id to video source
    frame_source_map = {f.frame_id: f.source_video for f in frames}
    semantic_results = semantic_pipeline.filter_frames(
        dedup_representatives,
        source_video_map=frame_source_map,
    )

    likely_sultan = sum(
        1 for s in semantic_results if s.semantic_label == SemanticLabel.SULTAN_HASSAN
    )
    likely_rifai = sum(1 for s in semantic_results if s.semantic_label == SemanticLabel.AL_RIFAI)
    uncertain_frames = sum(
        1
        for s in semantic_results
        if s.semantic_label in (SemanticLabel.UNCERTAIN, SemanticLabel.MIXED)
    )
    candidates = [
        s for s in semantic_results if s.status in (QualityStatus.KEEP, QualityStatus.REVIEW)
    ]

    console.print(Panel.fit("[bold cyan]Stage 6: Visual Contact Sheet Generation[/bold cyan]"))
    contact_generator = ContactSheetGenerator(config)
    {s.frame_id: s for s in semantic_results}

    # Build gallery items
    gallery_items: list[dict[str, object]] = []
    cards: list[Image.Image] = []

    for s in semantic_results:
        fpath = Path(config.paths.frames_deduplicated) / s.local_filename
        if not fpath.exists():
            fpath = Path(config.paths.frames_raw) / s.local_filename
        q_item = q_map.get(s.frame_id)
        src_video = frame_source_map.get(s.frame_id, "unknown")
        res_str = f"{q_item.width}x{q_item.height}" if q_item else "1024x1024"
        blur_val = q_item.blur_score if q_item else 0.0
        q_score = q_item.short_side / 10.0 if q_item else 50.0

        item_dict: dict[str, object] = {
            "image_path": fpath,
            "frame_id": s.frame_id,
            "source_video": src_video,
            "resolution": res_str,
            "timestamp_sec": 0.0,
            "quality_score": round(q_score, 1),
            "blur_score": blur_val,
            "sultan_score": s.sultan_hassan_score,
            "rifai_score": s.al_rifai_score,
            "status": s.status.value,
        }
        gallery_items.append(item_dict)

        card = create_thumbnail_card(
            image_path=fpath,
            frame_id=s.frame_id,
            source_video=src_video,
            timestamp_sec=0.0,
            resolution=res_str,
            quality_score=q_score,
            sultan_score=s.sultan_hassan_score,
            rifai_score=s.al_rifai_score,
            status=s.status.value,
        )
        cards.append(card)

    # Output grid image & HTML gallery
    contact_img_path = Path(config.paths.reports_contact_sheets) / "candidates_contact_sheet.jpg"
    contact_html_path = Path(config.paths.reports_contact_sheets) / "index.html"

    contact_generator.generate_grid_image(cards, contact_img_path)
    contact_generator.generate_html_gallery(
        gallery_items, "Sultan Hassan Candidate Inspection Gallery", contact_html_path
    )

    state_mgr.save()

    # Build final summary report
    summary: dict[str, object] = {
        "number_of_videos": len(videos),
        "total_duration": str(video_summary.get("total_duration_seconds", 0.0)) + "s",
        "total_extracted_frames": len(frames),
        "frames_below_1024": frames_below_1024,
        "blurry_frames": blurry_frames,
        "dark_frames": dark_frames,
        "duplicate_frames": total_duplicate_frames,
        "likely_sultan_hassan_frames": likely_sultan,
        "likely_al_rifai_frames": likely_rifai,
        "uncertain_frames": uncertain_frames,
        "final_candidates": len(candidates),
        "contact_sheet_image": str(contact_img_path),
        "contact_sheet_html": str(contact_html_path),
    }

    # Print summary table
    table = Table(
        title="[bold yellow]Data Collection & Curation Pipeline Summary[/bold yellow]",
        show_header=True,
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold green")

    for k, v in summary.items():
        label = k.replace("_", " ").title()
        table.add_row(label, str(v))

    console.print(table)
    return summary


def run_review_stage(config: AppConfig, port: int = 8080) -> None:
    """Launch local human review interface."""
    run_review_server(config, port=port)


def run_dataset_stage(config: AppConfig) -> None:
    """Select 25-40 approved images, generate captions, and validate dataset."""
    console.print(Panel.fit("[bold cyan]Final Dataset Preparation & Validation[/bold cyan]"))
    selector = DatasetSelector(config)
    final_records = selector.select_and_finalize()
    console.print(f"Finalized [bold green]{len(final_records)}[/bold green] images with captions.")

    validator = DatasetValidator(config)
    res = validator.validate(raise_on_failure=False)

    if res.is_valid:
        console.print("[bold green]SUCCESS: Final dataset passed all quality gates![/bold green]")
    else:
        console.print(
            f"[bold red]FAILED: Validation encountered {res.error_count} errors.[/bold red]"
        )
        for err in res.errors:
            console.print(f"  - [red]{err}[/red]")


def run_standardize_stage(config: AppConfig, cleanup: bool = True) -> dict[str, list[Path]]:
    """Standardize all media (images & videos) in data/raw to high-quality standard formats (.jpg, .mp4)."""
    console.print(
        Panel.fit("[bold cyan]Stage 0: Media Standardization & Quality Verification[/bold cyan]")
    )
    results: dict[str, list[Path]] = {"images": [], "videos": []}
    for target_dir in [config.paths.raw_external, config.paths.raw_videos]:
        if target_dir.exists():
            sub_res = standardize_media_directory(
                target_dir,
                cleanup_non_standard=cleanup,
                recursive=True,
            )
            results["images"].extend(sub_res["images"])
            results["videos"].extend(sub_res["videos"])

    console.print(
        f"[bold green]Successfully standardized {len(results['images'])} images and {len(results['videos'])} videos to standard extensions (.jpg & .mp4).[/bold green]"
    )
    return results


def run_train_stage(config: AppConfig) -> None:
    """Launch or prepare FLUX LoRA training."""
    from scripts.train_lora import main as train_main  # type: ignore

    console.print(
        Panel.fit("[bold cyan]Stage 4: FLUX LoRA Training Preparation & Launch[/bold cyan]")
    )
    train_main()


def run_generate_stage(config: AppConfig) -> None:
    """Generate 20 benchmark test images across 5 prompt groups with fixed seeds."""
    from scripts.generate_tests import main as generate_main  # type: ignore

    console.print(
        Panel.fit("[bold cyan]Stage 5: Benchmark Test Image Generation & Gallery[/bold cyan]")
    )
    generate_main()


def run_evaluate_stage(config: AppConfig) -> None:
    """Run architectural evaluation and generate metrics report."""
    from scripts.evaluate_results import main as evaluate_main  # type: ignore

    console.print(
        Panel.fit("[bold cyan]Stage 6: Architectural & Negative Control Evaluation[/bold cyan]")
    )
    evaluate_main()


def run_deliver_stage(config: AppConfig) -> None:
    """Compile deliverable archive and reports."""
    from scripts.build_deliverable import main as deliver_main  # type: ignore

    console.print(Panel.fit("[bold cyan]Stage 7: Compile Complete Deliverable Package[/bold cyan]"))
    deliver_main()
