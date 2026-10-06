"""
SolarBurst Command-Line Interface (CLI) built with Typer.
"""

from typing import Optional
import os
import json
import typer
from rich.console import Console
from rich.table import Table

from .core.schema import AnalysisConfig
from .io.xsm_adapter import load_lightcurve
from .engine import SolarBurstEngine
from .catalog.exporter import export_catalog_csv, export_catalog_fits, export_provenance_json
from .evaluation.injection import create_synthetic_demo_lightcurve

app = typer.Typer(help="SolarBurst — Chandrayaan-2 XSM Solar Burst Analysis Platform CLI")
console = Console()


@app.command()
def analyze(
    filepath: str = typer.Argument(..., help="Path to input light curve (FITS, CSV, XLSX, CDF)"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to YAML/JSON configuration preset"),
    output_dir: str = typer.Option("./results", "--out", "-o", help="Output directory for catalogs"),
    preset: str = typer.Option("balanced", "--preset", "-p", help="Preset: balanced, conservative, sensitive"),
    model_type: str = typer.Option("random_forest", "--model", "-m", help="Scorer: random_forest, xgboost, classical_only")
):
    """
    Run full scientific burst analysis on an observation file.
    """
    console.print(f"[bold cyan]SolarBurst Engine[/bold cyan] analyzing: [yellow]{filepath}[/yellow]")
    
    # Load configuration
    cfg = AnalysisConfig(name=preset, model_family=model_type)
    if preset == "conservative":
        cfg.matched_filter_snr_threshold = 4.5
        cfg.cwt_snr_threshold = 4.0
        cfg.hysteresis_seed_sigma = 5.0
        cfg.ml_accept_threshold = 0.75
    elif preset == "sensitive":
        cfg.matched_filter_snr_threshold = 3.0
        cfg.cwt_snr_threshold = 2.5
        cfg.hysteresis_seed_sigma = 3.5
        cfg.ml_accept_threshold = 0.55

    # 1. Ingestion
    console.print("Loading and inspecting input observation...")
    lc = load_lightcurve(filepath)
    console.print(f"Loaded {len(lc.value)} points, instrument: [green]{lc.metadata.get('instrument')}[/green]")

    # 2. Execution
    engine = SolarBurstEngine()
    
    def on_progress(pct, msg):
        console.print(f"[{pct:02d}%] {msg}")

    results = engine.run_analysis(lc, cfg, progress_callback=on_progress)

    # 3. Print Results Summary
    table = Table(title="Detection & Classification Summary")
    table.add_column("Burst ID", style="cyan")
    table.add_column("Peak Time (UTC)", style="white")
    table.add_column("Net Peak", style="green")
    table.add_column("Duration (s)", style="yellow")
    table.add_column("Morphology", style="magenta")
    table.add_column("Reliability", style="blue")
    table.add_column("ML Score", style="cyan")

    bursts = results["bursts"]
    for b in bursts:
        table.add_row(
            b["burst_id"],
            b["peak_time_iso"],
            f"{b['net_peak']:.2f}",
            f"{b['duration_s']:.1f}",
            b["morphology_class"],
            b["reliability"],
            f"{b['ml_score']:.2f}"
        )
    console.print(table)

    # 4. Exports
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    csv_path = os.path.join(output_dir, f"{base_name}_catalog.csv")
    fits_path = os.path.join(output_dir, f"{base_name}_catalog.fits")
    json_path = os.path.join(output_dir, f"{base_name}_provenance.json")

    # Export using raw objects
    from .core.schema import BurstResult
    burst_objs = [BurstResult(**b) for b in bursts]
    export_catalog_csv(burst_objs, csv_path)
    export_catalog_fits(burst_objs, fits_path)
    export_provenance_json(burst_objs, lc, cfg, json_path)

    console.print(f"[bold green]Saved CSV catalog:[/bold green] {csv_path}")
    console.print(f"[bold green]Saved FITS catalog:[/bold green] {fits_path}")
    console.print(f"[bold green]Saved Provenance JSON:[/bold green] {json_path}")


@app.command()
def demo(
    output_dir: str = typer.Option("./data/examples", "--out", "-o", help="Directory to save demo dataset")
):
    """
    Generate and analyze a realistic synthetic Chandrayaan-2 XSM demonstration dataset.
    """
    console.print("[bold yellow]Generating synthetic XSM observation...[/bold yellow]")
    lc, injected = create_synthetic_demo_lightcurve(duration_s=7200.0)
    
    os.makedirs(output_dir, exist_ok=True)
    demo_csv = os.path.join(output_dir, "synthetic_xsm_demo.csv")
    
    # Save CSV
    import pandas as pd
    df = pd.DataFrame({
        "TIME": lc.met_seconds,
        "RATE": lc.value,
        "ERROR": lc.error,
        "QUALITY": lc.quality
    })
    df.to_csv(demo_csv, index=False)
    console.print(f"[green]Saved demo dataset to:[/green] {demo_csv}")
    
    # Run analysis
    console.print("\n[bold cyan]Running automated pipeline on synthetic demonstration...[/bold cyan]")
    engine = SolarBurstEngine()
    cfg = AnalysisConfig(name="balanced")
    results = engine.run_analysis(lc, cfg)
    
    console.print(f"\n[bold green]Analysis complete in {results['runtime_s']:.2f}s![/bold green]")
    console.print(f"Candidates detected: {results['total_candidates']}, Bursts identified: {results['total_bursts']}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p")
):
    """
    Start the SolarBurst Python microservice for the Node.js API layer.
    """
    import uvicorn
    console.print(f"[bold green]Starting SolarBurst Scientific Service on http://{host}:{port}...[/bold green]")
    uvicorn.run("solarburst.service:api_app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
