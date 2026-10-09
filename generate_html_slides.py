#!/usr/bin/env python3
"""
Generate an interactive, responsive HTML slide deck from Cut Analysis plots.
Reproduces the layout of the user's PDF presentation slides:
  - Slide Type 1 (Cut Overview): Global 3w + Global Gap (Left) vs multi_panel_conductance.png (Right)
  - Slide Type 2 (Cut Analytics): Global 3w + Global Gap (Left) vs multi_panel.png (Right)
  - Slide Type 3 (Point Deep-Dive): multi_panel_conductance.png (Left) vs dI/dV, Wavefunctions, Barrier Asymmetry (Right)
  - Slide Type 4 (Global Summary): 2x2 Grid of Global 3w, Transport Gap, Thresholded Correlation, and Separability
"""

import sys
import os
import argparse
from pathlib import Path
import json
import numpy as np

try:
    from cut_analysis import DATASET_CUTS
except ImportError:
    DATASET_CUTS = {}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}
  body {{
    background-color: #0b1120;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    overflow: hidden;
    height: 100vh;
    width: 100vw;
    display: flex;
    flex-direction: column;
  }}

  /* Ultra-compact Top Navigation Bar */
  header {{
    background-color: #0f172a;
    border-bottom: 1px solid #1e293b;
    padding: 0 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 36px;
    user-select: none;
    z-index: 100;
    flex-shrink: 0;
  }}
  .nav-left {{
    display: flex;
    align-items: center;
    gap: 10px;
    overflow: hidden;
    min-width: 0;
  }}
  .nav-title {{
    font-size: 13px;
    font-weight: 700;
    color: #38bdf8;
    white-space: nowrap;
    letter-spacing: 0.02em;
  }}
  .nav-sep {{
    color: #475569;
    font-size: 13px;
  }}
  .slide-title-display {{
    font-size: 13px;
    font-weight: 500;
    color: #f1f5f9;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 50vw;
  }}
  .nav-right {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }}
  .slide-select {{
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 12px;
    outline: none;
    max-width: 340px;
  }}
  .btn {{
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .btn:hover {{
    background-color: #334155;
    color: #fff;
  }}
  .counter {{
    font-size: 12px;
    color: #94a3b8;
    min-width: 85px;
    text-align: center;
    font-variant-numeric: tabular-nums;
  }}
  .hint {{
    font-size: 11px;
    color: #64748b;
  }}

  /* Main Slide Container - Maximizes Screen Space */
  main {{
    flex: 1;
    position: relative;
    width: 100vw;
    height: calc(100vh - 36px);
    overflow: hidden;
  }}
  .slide {{
    display: none;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    padding: 4px 6px;
    background-color: #0b1120;
    overflow: hidden;
  }}
  .slide.active {{
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
  }}

  /* Slide Layouts tuned precisely to image aspect ratios */
  /* Cut Overview / Analytics: 2 stacked widescreen phase maps (1.81:1) vs tall multiplot (0.62:1) */
  .layout-cut-overview, .layout-cut-analytics {{
    flex: 1;
    display: grid;
    grid-template-columns: 1.42fr 1fr;
    gap: 6px;
    min-height: 0;
    height: 100%;
    width: 100%;
  }}
  .col-stacked {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    height: 100%;
    min-height: 0;
  }}

  /* Point Deep-Dive: tall multiplot (0.62:1) vs 3 stacked point plots (2.93:1 each) */
  .layout-point-deepdive {{
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1.55fr;
    gap: 6px;
    min-height: 0;
    height: 100%;
    width: 100%;
  }}
  .col-stacked-3 {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    height: 100%;
    min-height: 0;
  }}

  /* 2x2 Grid Layout for Global Phase Maps */
  .layout-grid-2x2 {{
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 4px;
    min-height: 0;
    height: 100%;
    width: 100%;
  }}

  /* 3x2 Grid Layout for 6-Panel Disorder Overview */
  .layout-grid-3x2 {{
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 4px;
    min-height: 0;
    height: 100%;
    width: 100%;
  }}

  /* Side-by-Side 1:1 Layout for Gap vs Sensitivity Comparison */
  .layout-side-by-side {{
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    min-height: 0;
    height: 100%;
    width: 100%;
  }}

  /* Stacked Two-Row Layout: Top Row vs Bottom Row */
  .layout-two-row {{
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-height: 0;
    height: 100%;
    width: 100%;
    padding: 6px;
  }}

  /* Single Centered Image Layout */
  .layout-single-centered {{
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
    height: 100%;
    width: 100%;
    padding: 6px;
  }}

  /* Multi-Column Two-Row Layout: Top Row vs Bottom Row across N columns */
  .layout-multicol-two-row {{
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-height: 0;
    height: 100%;
    width: 100%;
    padding: 6px;
  }}
  .multicol-header {{
    display: flex;
    justify-content: space-around;
    padding: 4px 0;
    font-size: 13px;
    font-weight: 700;
    color: #38bdf8;
    background-color: #0f172a;
    border-radius: 4px;
    border: 1px solid #1e293b;
    flex-shrink: 0;
  }}
  .multicol-row {{
    flex: 1;
    display: grid;
    gap: 6px;
    min-height: 0;
    height: 100%;
    width: 100%;
  }}
  .multicol-row-3 {{ grid-template-columns: repeat(3, 1fr); }}
  .multicol-row-4 {{ grid-template-columns: repeat(4, 1fr); }}

  /* Formatted Table Slide Layout (FDR & Discovery Rates) */
  .layout-table-slide {{
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 0;
    height: 100%;
    width: 100%;
    padding: 24px;
    overflow-y: auto;
  }}
  .table-card {{
    width: 96%;
    max-width: 1300px;
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 20px 24px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
  }}
  .table-card h2 {{
    font-size: 18px;
    color: #38bdf8;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .table-card p.subtitle {{
    font-size: 12px;
    color: #94a3b8;
    margin-bottom: 16px;
  }}
  .data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    text-align: left;
  }}
  .data-table th {{
    background-color: #1e293b;
    color: #f1f5f9;
    padding: 10px 12px;
    font-weight: 600;
    border-bottom: 2px solid #334155;
    white-space: nowrap;
  }}
  .data-table td {{
    padding: 8px 12px;
    border-bottom: 1px solid #1e293b;
    color: #cbd5e1;
    white-space: nowrap;
  }}
  .data-table tr:hover td {{
    background-color: #1e293b40;
  }}
  .badge-pass {{
    background-color: #064e3b;
    color: #34d399;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 11px;
    display: inline-block;
  }}
  .badge-warn {{
    background-color: #713f12;
    color: #fde047;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 11px;
    display: inline-block;
  }}
  .badge-fail {{
    background-color: #7f1d1d;
    color: #f87171;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 11px;
    display: inline-block;
  }}

  /* Seamless Image Cards: Zero border/padding, 100% space utilization */
  .img-card {{
    background-color: transparent;
    border: none;
    border-radius: 0;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    min-height: 0;
    min-width: 0;
    width: 100%;
    height: 100%;
  }}
  .img-card img {{
    max-width: 100%;
    max-height: 100%;
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    cursor: zoom-in;
    transition: filter 0.15s ease;
  }}
  .img-card img:hover {{
    filter: brightness(1.04);
  }}
  .flex-1 {{
    flex: 1;
    min-height: 0;
  }}

  /* Fullscreen Lightbox Zoom Modal */
  .lightbox {{
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: rgba(3, 7, 18, 0.95);
    z-index: 1000;
    align-items: center;
    justify-content: center;
    cursor: zoom-out;
  }}
  .lightbox.active {{
    display: flex;
  }}
  .lightbox img {{
    max-width: 98vw;
    max-height: 98vh;
    object-fit: contain;
    box-shadow: 0 0 35px rgba(0, 0, 0, 0.9);
    border: 1px solid #334155;
  }}
  .lightbox-close {{
    position: absolute;
    top: 12px;
    right: 18px;
    color: #f1f5f9;
    font-size: 26px;
    font-weight: bold;
    cursor: pointer;
    background: rgba(30, 41, 59, 0.85);
    border: 1px solid #475569;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    line-height: 1;
  }}

  /* Print to PDF Styles */
  @page {{
    size: 1920px 1080px;
    margin: 0;
  }}
  @media print {{
    * {{
      -webkit-print-color-adjust: exact !important;
      print-color-adjust: exact !important;
    }}
    header, #lightbox, .slide-select, .nav-right, .hint, .counter {{
      display: none !important;
    }}
    html, body {{
      background-color: #0b1120 !important;
      width: 1920px !important;
      height: auto !important;
      margin: 0 !important;
      padding: 0 !important;
      overflow: visible !important;
    }}
    main {{
      width: 1920px !important;
      height: auto !important;
      overflow: visible !important;
      position: static !important;
    }}
    .slide {{
      display: flex !important;
      position: relative !important;
      width: 1920px !important;
      height: 1080px !important;
      page-break-after: always !important;
      break-after: page !important;
      page-break-inside: avoid !important;
      break-inside: avoid !important;
      overflow: hidden !important;
      padding: 12px 18px !important;
      box-sizing: border-box !important;
      background-color: #0b1120 !important;
    }}
  }}
</style>
</head>
<body>
<header>
  <div class="nav-left">
    <span class="nav-title">{title}</span>
    <span class="nav-sep">|</span>
    <span id="slideTitleDisplay" class="slide-title-display"></span>
  </div>
  <div class="nav-right">
    <span class="hint">← / →</span>
    <button class="btn" onclick="prevSlide()" title="Previous slide (Left Arrow)">◀</button>
    <span id="slideCounter" class="counter">Slide 1 of {total_slides}</span>
    <button class="btn" onclick="nextSlide()" title="Next slide (Right Arrow / Space)">▶</button>
    <select id="slideSelect" class="slide-select" onchange="goToSlide(parseInt(this.value))">
      {select_options}
    </select>
    <button class="btn" onclick="toggleFullscreen()" title="Fullscreen (F)">⛶</button>
  </div>
</header>

<main id="slideContainer">
  {slides_html}
</main>

<div id="lightbox" class="lightbox" onclick="closeLightbox()">
  <span class="lightbox-close" onclick="closeLightbox()">&times;</span>
  <img id="lightboxImg" src="" alt="Full Resolution Zoom">
</div>

<script>
  let currentSlide = 0;
  const totalSlides = {total_slides};
  const slides = document.querySelectorAll('.slide');
  const slideSelect = document.getElementById('slideSelect');
  const slideCounter = document.getElementById('slideCounter');
  const slideTitleDisplay = document.getElementById('slideTitleDisplay');
  const lightbox = document.getElementById('lightbox');
  const lightboxImg = document.getElementById('lightboxImg');

  function showSlide(index) {{
    if (index < 0) index = 0;
    if (index >= totalSlides) index = totalSlides - 1;
    currentSlide = index;

    slides.forEach((s, i) => {{
      s.classList.toggle('active', i === currentSlide);
    }});
    if (slideSelect) slideSelect.value = currentSlide;
    if (slideCounter) slideCounter.textContent = `Slide ${{currentSlide + 1}} of ${{totalSlides}}`;
    if (slideTitleDisplay && slides[currentSlide]) {{
      slideTitleDisplay.textContent = slides[currentSlide].getAttribute('data-title') || '';
    }}
  }}

  function nextSlide() {{
    if (currentSlide < totalSlides - 1) showSlide(currentSlide + 1);
  }}

  function prevSlide() {{
    if (currentSlide > 0) showSlide(currentSlide - 1);
  }}

  function goToSlide(index) {{
    showSlide(index);
  }}

  function toggleFullscreen() {{
    if (!document.fullscreenElement) {{
      document.documentElement.requestFullscreen().catch(() => {{}});
    }} else {{
      document.exitFullscreen().catch(() => {{}});
    }}
  }}

  function openLightbox(src) {{
    if (lightbox && lightboxImg) {{
      lightboxImg.src = src;
      lightbox.classList.add('active');
    }}
  }}

  function closeLightbox() {{
    if (lightbox) {{
      lightbox.classList.remove('active');
    }}
  }}

  // Attach click to zoom on all images
  document.querySelectorAll('.img-card img').forEach(img => {{
    img.addEventListener('click', (e) => {{
      e.stopPropagation();
      openLightbox(img.src);
    }});
  }});

  document.addEventListener('keydown', (e) => {{
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    if (e.key === 'Escape') {{
      if (lightbox && lightbox.classList.contains('active')) {{
        closeLightbox();
        return;
      }}
    }}
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {{
      e.preventDefault();
      nextSlide();
    }} else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {{
      e.preventDefault();
      prevSlide();
    }} else if (e.key === 'Home') {{
      e.preventDefault();
      showSlide(0);
    }} else if (e.key === 'End') {{
      e.preventDefault();
      showSlide(totalSlides - 1);
    }} else if (e.key === 'f' || e.key === 'F') {{
      e.preventDefault();
      toggleFullscreen();
    }}
  }});

  // Initialize
  showSlide(0);
</script>
</body>
</html>
"""

def get_point_cut_position(dataset_label: str, cut_name: str, point_folder_name: str):
    """
    Returns the scalar projection position 's' along the cut trajectory from start to end,
    plus the (Vz, mu) coordinates and point color.
    s = (raw_coord - start) . unit_vector.
    Falls back to (999.0, 0.0, 0.0, "") if not found.
    """
    if dataset_label in DATASET_CUTS:
        for c in DATASET_CUTS[dataset_label]:
            if c.get("label") == cut_name:
                start = np.array(c["start"])
                end = np.array(c["end"])
                diff = end - start
                norm = np.linalg.norm(diff)
                u = diff / norm if norm > 0 else np.array([1.0, 0.0])
                for sp in c.get("snap_points", []):
                    if sp.get("label") == point_folder_name:
                        raw = np.array(sp["raw_coords"])
                        s = float(np.dot(raw - start, u))
                        return s, float(raw[0]), float(raw[1]), sp.get("color", "")
    return 999.0, 0.0, 0.0, ""

def build_slides_for_directory(plot_dir: Path, base_rel_path: Path = None):
    """
    Build slide descriptors for a given dataset plot directory.
    Paths are made relative to base_rel_path if given, else relative to plot_dir.
    """
    plot_dir = Path(plot_dir).resolve()
    rel_root = Path(base_rel_path).resolve() if base_rel_path else plot_dir
    dir_name = plot_dir.name
    dataset_label = dir_name.replace("_Plots", "")

    slides = []

    # Identify global maps
    p_3w = plot_dir / "global_3w_phase_map.png"
    p_gap = plot_dir / "global_transport_gap_phase_map.png"
    p_corr = plot_dir / "global_barrier_correlation_threshold_85_phase_map.png"
    p_sep = plot_dir / "global_separability_phase_map.png"
    p_sens = plot_dir / "global_gap_threshold_sensitivity_phase_map.png"

    def get_rel(p: Path):
        try:
            return os.path.relpath(p, rel_root)
        except Exception:
            return str(p)

    cuts_dir = plot_dir / "Cuts"
    if cuts_dir.exists():
        cut_folders = sorted([d for d in cuts_dir.iterdir() if d.is_dir() and d.name.startswith("Cut_")], 
                             key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else 999)

        for cut_folder in cut_folders:
            cut_name = cut_folder.name
            p_conductance = cut_folder / "multi_panel_conductance.png"
            p_analytics = cut_folder / "multi_panel.png"

            # Slide 1: Cut Overview (Conductance Multiplot)
            if p_conductance.exists():
                slides.append({
                    "dataset": dataset_label,
                    "title": f"[{dataset_label}] {cut_name} | Overview (Spectra & Conductance)",
                    "type": "cut_overview",
                    "left_top": get_rel(p_3w) if p_3w.exists() else None,
                    "left_bottom": get_rel(p_gap) if p_gap.exists() else None,
                    "right": get_rel(p_conductance)
                })

            # Slide 2: Cut Analytics (Multi-panel Analytics)
            if p_analytics.exists():
                slides.append({
                    "dataset": dataset_label,
                    "title": f"[{dataset_label}] {cut_name} | Analytics (2w, 3w, Gap, Overlap)",
                    "type": "cut_analytics",
                    "left_top": get_rel(p_3w) if p_3w.exists() else None,
                    "left_bottom": get_rel(p_gap) if p_gap.exists() else None,
                    "right": get_rel(p_analytics)
                })

            # Slide 3..N: Point Deep-Dives (sorted strictly along cut trajectory from start to end)
            pts_dir = cut_folder / "Points"
            if pts_dir.exists():
                raw_folders = [p for p in pts_dir.iterdir() if p.is_dir()]
                point_folders = sorted(raw_folders, key=lambda p: (get_point_cut_position(dataset_label, cut_name, p.name)[0], p.name))
                for pt_folder in point_folders:
                    pt_name = pt_folder.name
                    s, vz, mu, pt_color = get_point_cut_position(dataset_label, cut_name, pt_name)
                    coord_str = f" (Vz={vz:.3f}, μ={mu:.3f})" if s < 900.0 else ""
                    # Find dIdV, wavefunctions, and barrier asymmetry plots
                    didv_files = list(pt_folder.glob("*_dIdV.png"))
                    wf_files = list(pt_folder.glob("*_wavefunctions.png"))
                    barr_files = list(pt_folder.glob("*_barrier_asymmetry.png"))

                    if didv_files or wf_files or barr_files:
                        slides.append({
                            "dataset": dataset_label,
                            "title": f"[{dataset_label}] {cut_name} | {pt_name}{coord_str}",
                            "type": "point_deepdive",
                            "left": get_rel(p_conductance) if p_conductance.exists() else (get_rel(p_analytics) if p_analytics.exists() else None),
                            "pt_didv": get_rel(didv_files[0]) if didv_files else None,
                            "pt_wf": get_rel(wf_files[0]) if wf_files else None,
                            "pt_barr": get_rel(barr_files[0]) if barr_files else None,
                        })

    # Global Phase Maps Summary Slide
    if p_3w.exists() or p_gap.exists() or p_corr.exists() or p_sep.exists():
        slides.append({
            "dataset": dataset_label,
            "title": f"[{dataset_label}] Global Phase Maps (3w, Gap, Thresholded Correlation, Separability)",
            "type": "global_summary",
            "tl": get_rel(p_3w) if p_3w.exists() else None,
            "tl_alt": "Global 3w Curvature",
            "tr": get_rel(p_corr) if p_corr.exists() else None,
            "tr_alt": "Thresholded Barrier Correlation",
            "bl": get_rel(p_gap) if p_gap.exists() else None,
            "bl_alt": "Transport Gap",
            "br": get_rel(p_sep) if p_sep.exists() else None,
            "br_alt": "MZM Separability",
        })

    # Transport Gap Sensitivity & Noise Sweeps Slide
    if p_sens.exists() or (plot_dir / "Gap_Sweeps").exists():
        gap_sweeps_dir = plot_dir / "Gap_Sweeps"
        p_abs_low = gap_sweeps_dir / "global_gap_absolute_noise_1e-05_phase_map.png"
        p_abs_high = gap_sweeps_dir / "global_gap_absolute_noise_1e-02_phase_map.png"
        slides.append({
            "dataset": dataset_label,
            "title": f"[{dataset_label}] Gap Sensitivity Analysis (Base Gap, Δmax-Δmin Variance, Noise Sweeps)",
            "type": "gap_sensitivity",
            "tl": get_rel(p_gap) if p_gap.exists() else None,
            "tl_alt": "Base Transport Gap (Mapped Factor)",
            "tr": get_rel(p_sens) if p_sens.exists() else None,
            "tr_alt": "Gap Extraction Sensitivity (Delta max - Delta min)",
            "bl": get_rel(p_abs_low) if p_abs_low.exists() else None,
            "bl_alt": "Noise Floor 1e-5",
            "br": get_rel(p_abs_high) if p_abs_high.exists() else None,
            "br_alt": "Noise Floor 1e-2",
        })

    # 2w Measurement vs MZM Separability Comparison Slide (User Request: 2w on top row, Separability on bottom row)
    p_2w = plot_dir / "global_2w_phase_map.png"
    if p_2w.exists() or p_sep.exists():
        slides.append({
            "dataset": dataset_label,
            "title": f"[{dataset_label}] 2ω Measurement (Top) vs. MZM Separability (Bottom)",
            "type": "two_row_comparison",
            "top": get_rel(p_2w) if p_2w.exists() else None,
            "top_alt": "2ω Conductance Measurement Phase Map",
            "bottom": get_rel(p_sep) if p_sep.exists() else None,
            "bottom_alt": "MZM Separability Phase Map",
        })

    # Microsoft Stage 2 Paper Diagram Slide (Replicating Fig. 29)
    p_s2 = plot_dir / "stage2_paper_diagram.png"
    if p_s2.exists():
        slides.append({
            "dataset": dataset_label,
            "title": f"[{dataset_label}] Microsoft Stage 2 Paper Diagram (PRL Fig. 29 Replicated)",
            "type": "single_centered",
            "img": get_rel(p_s2),
            "alt": "Stage 2 Paper Diagram",
        })

    # --------------------------------------------------------------------------
    # TGP Cuts (Islands Center of Mass Cuts & Deep Dives)
    # --------------------------------------------------------------------------
    tgp_cuts_dir = plot_dir / "TGP_Cuts"
    if tgp_cuts_dir.exists():
        p_stage2_star = tgp_cuts_dir / "stage2_phase_map.png"
        if p_stage2_star.exists():
            slides.append({
                "dataset": dataset_label,
                "title": f"[{dataset_label}] Stage 2 Phase Map & Top 3 Topological Island COMs",
                "type": "single_centered",
                "img": get_rel(p_stage2_star),
                "alt": "Stage 2 Phase Map with Island COMs",
            })

        island_dirs = sorted([d for d in tgp_cuts_dir.iterdir() if d.is_dir() and d.name.startswith("Island_")])
        for island_folder in island_dirs:
            island_label = island_folder.name.replace("_", " ")
            p_cond = island_folder / "multi_panel_conductance.png"
            p_ana = island_folder / "multi_panel.png"
            p_spec = island_folder / "spectra.png"

            # TGP Cut Overview (Conductance Multiplot)
            if p_cond.exists():
                slides.append({
                    "dataset": dataset_label,
                    "title": f"[{dataset_label}] TGP Cut: {island_label} | Overview (Conductance)",
                    "type": "cut_overview",
                    "left_top": get_rel(p_stage2_star) if p_stage2_star.exists() else (get_rel(p_3w) if p_3w.exists() else None),
                    "left_bottom": get_rel(p_gap) if p_gap.exists() else None,
                    "right": get_rel(p_cond)
                })

            # TGP Cut Analytics (2w, 3w, Gap, Overlap Multiplot)
            if p_ana.exists():
                slides.append({
                    "dataset": dataset_label,
                    "title": f"[{dataset_label}] TGP Cut: {island_label} | Analytics (2w, 3w, Gap, Overlap)",
                    "type": "cut_analytics",
                    "left_top": get_rel(p_stage2_star) if p_stage2_star.exists() else (get_rel(p_3w) if p_3w.exists() else None),
                    "left_bottom": get_rel(p_gap) if p_gap.exists() else None,
                    "right": get_rel(p_ana)
                })

            # TGP Island COM Point Deep-Dive
            didv_files = list(island_folder.glob("*_dIdV.png"))
            wf_files = list(island_folder.glob("*_wavefunctions.png"))
            barr_files = list(island_folder.glob("*_barrier_asymmetry.png"))
            if didv_files or wf_files or barr_files:
                slides.append({
                    "dataset": dataset_label,
                    "title": f"[{dataset_label}] TGP Cut: {island_label} | COM Deep-Dive",
                    "type": "point_deepdive",
                    "left": get_rel(p_cond) if p_cond.exists() else (get_rel(p_ana) if p_ana.exists() else None),
                    "pt_didv": get_rel(didv_files[0]) if didv_files else None,
                    "pt_wf": get_rel(wf_files[0]) if wf_files else None,
                    "pt_barr": get_rel(barr_files[0]) if barr_files else None,
                })

        # Single Dataset FDR Metrics Card
        p_fdr = tgp_cuts_dir / "fdr_metrics.json"
        if p_fdr.exists():
            try:
                with open(p_fdr, "r", encoding="utf-8") as f:
                    fdr_data = json.load(f)
                s2_m = fdr_data.get("stage2_roi2_metrics", {})
                or_m = fdr_data.get("candidate_orange_state_metrics", {})
                islands = fdr_data.get("top3_islands", [])
                
                ppv_pct = s2_m.get("p_topological_given_passed_ppv", 0.0) * 100.0
                fdr_pct = s2_m.get("false_discovery_rate_fdr", 0.0) * 100.0
                tpr_pct = s2_m.get("sensitivity_tpr", 0.0) * 100.0
                tnr_pct = s2_m.get("specificity_tnr", 0.0) * 100.0
                or_ppv = or_m.get("p_topological_given_orange_ppv", 0.0) * 100.0
                or_fdr = or_m.get("false_discovery_rate_fdr", 0.0) * 100.0

                badge_cls = "badge-pass" if fdr_pct < 10.0 else ("badge-warn" if fdr_pct < 35.0 else "badge-fail")

                islands_html = "".join([
                    f"<tr><td>Rank {isl.get('rank')}</td><td><span style='color:{isl.get('color')}; font-weight:bold;'>★ {isl.get('star_name')}</span></td><td>{isl.get('size_pixels')} px</td><td>Vz = {isl.get('vz_com', 0):.3f} meV</td><td>μ = {isl.get('mu_com', 0):.3f} meV</td></tr>"
                    for isl in islands
                ])

                table_html = f"""
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                  <div style="background:#1e293b50; padding:14px; border-radius:6px; border:1px solid #334155;">
                    <h3 style="color:#38bdf8; font-size:14px; margin-bottom:8px;">Stage 2 Full Protocol Verification (ROI 2)</h3>
                    <p style="margin:4px 0;"><strong>Total Pixels Passed:</strong> {s2_m.get('total_passed', 0)}</p>
                    <p style="margin:4px 0;"><strong>True Positives (TP):</strong> {s2_m.get('true_positives', 0)} | <strong>False Positives (FP):</strong> {s2_m.get('false_positives', 0)}</p>
                    <p style="margin:4px 0;"><strong>P(Topological | Passed) [PPV]:</strong> <span class="{badge_cls}">{ppv_pct:.2f}%</span></p>
                    <p style="margin:4px 0;"><strong>False Discovery Rate [FDR]:</strong> <span class="{badge_cls}">{fdr_pct:.2f}%</span></p>
                    <p style="margin:4px 0;"><strong>Sensitivity (TPR):</strong> {tpr_pct:.2f}% | <strong>Specificity (TNR):</strong> {tnr_pct:.2f}%</p>
                  </div>
                  <div style="background:#1e293b50; padding:14px; border-radius:6px; border:1px solid #334155;">
                    <h3 style="color:#f59e0b; font-size:14px; margin-bottom:8px;">Candidate Orange Pixels (gapped_zbp pre-clustering)</h3>
                    <p style="margin:4px 0;"><strong>Total Candidate Pixels:</strong> {or_m.get('total_orange_pixels', 0)}</p>
                    <p style="margin:4px 0;"><strong>True Positives:</strong> {or_m.get('true_positives', 0)} | <strong>False Positives:</strong> {or_m.get('false_positives', 0)}</p>
                    <p style="margin:4px 0;"><strong>P(Topological | Candidate):</strong> {or_ppv:.2f}%</p>
                    <p style="margin:4px 0;"><strong>Candidate FDR:</strong> {or_fdr:.2f}%</p>
                    <p style="margin:4px 0; color:#94a3b8; font-size:11px;">Comparing candidate pixels vs OPTICS cluster filtering reveals the impact of boundary overlap cuts.</p>
                  </div>
                </div>
                <h3 style="color:#f1f5f9; font-size:13px; margin-bottom:8px;">Top 3 Validated Topological Islands</h3>
                <table class="data-table">
                  <thead>
                    <tr><th>Rank</th><th>Identifier</th><th>Cluster Size</th><th>Zeeman COM (Vz)</th><th>Chemical Potential COM (μ)</th></tr>
                  </thead>
                  <tbody>
                    {islands_html}
                  </tbody>
                </table>
                """

                slides.append({
                    "dataset": dataset_label,
                    "title": f"[{dataset_label}] False Discovery Rate & Topological Verification Statistics",
                    "type": "table_slide",
                    "table_title": f"Topological Verification Statistics | {dataset_label}",
                    "table_subtitle": f"Comparison of Azure Quantum TGP Stage 2 Protocol passes against ground-truth Pfaffian topological invariant (Q = -1)",
                    "html_table": table_html
                })
            except Exception as e:
                pass

    # --------------------------------------------------------------------------
    # Stage 2 Gap Sweeps (Threshold Multiplier & Absolute Noise Floor)
    # --------------------------------------------------------------------------
    sweeps_dir = plot_dir / "Stage2_Gap_Sweeps"
    if sweeps_dir.exists():
        p_m05 = sweeps_dir / "stage2_multiplier_0.50.png"
        p_m85 = sweeps_dir / "stage2_multiplier_0.85.png"
        p_m10 = sweeps_dir / "stage2_multiplier_1.00.png"
        p_m13 = sweeps_dir / "stage2_multiplier_1.30.png"
        if p_m05.exists() and p_m10.exists():
            slides.append({
                "dataset": dataset_label,
                "title": f"[{dataset_label}] Stage 2 Islands vs. Gap Multiplier (0.50x, 0.85x, 1.00x, 1.30x)",
                "type": "global_summary",
                "tl": get_rel(p_m05), "tl_alt": "Multiplier 0.50x",
                "tr": get_rel(p_m85) if p_m85.exists() else get_rel(p_m05), "tr_alt": "Multiplier 0.85x",
                "bl": get_rel(p_m10), "bl_alt": "Multiplier 1.00x (Base)",
                "br": get_rel(p_m13) if p_m13.exists() else get_rel(p_m10), "br_alt": "Multiplier 1.30x",
            })

        p_n1e5 = sweeps_dir / "stage2_abs_noise_1.0e-05.png"
        p_n1e4 = sweeps_dir / "stage2_abs_noise_1.0e-04.png"
        p_n1e3 = sweeps_dir / "stage2_abs_noise_1.0e-03.png"
        p_n1e2 = sweeps_dir / "stage2_abs_noise_1.0e-02.png"
        if p_n1e5.exists() and p_n1e2.exists():
            slides.append({
                "dataset": dataset_label,
                "title": f"[{dataset_label}] Stage 2 Islands vs. Absolute Noise Floor (1e-5, 1e-4, 1e-3, 1e-2 e^2/h)",
                "type": "global_summary",
                "tl": get_rel(p_n1e5), "tl_alt": "Noise Floor 1e-5 e^2/h",
                "tr": get_rel(p_n1e4) if p_n1e4.exists() else get_rel(p_n1e5), "tr_alt": "Noise Floor 1e-4 e^2/h",
                "bl": get_rel(p_n1e3) if p_n1e3.exists() else get_rel(p_n1e5), "bl_alt": "Noise Floor 1e-3 e^2/h",
                "br": get_rel(p_n1e2), "br_alt": "Noise Floor 1e-2 e^2/h",
            })

        p_bg2 = sweeps_dir / "stage2_bias_gap_2.0uev.png"
        p_bg6 = sweeps_dir / "stage2_bias_gap_6.0uev.png"
        p_bg10 = sweeps_dir / "stage2_bias_gap_10.0uev.png"
        p_bg20 = sweeps_dir / "stage2_bias_gap_20.0uev.png"
        if p_bg2.exists() and p_bg10.exists():
            slides.append({
                "dataset": dataset_label,
                "title": f"[{dataset_label}] Stage 2 Islands vs. Bias Voltage Gap Threshold (2.0, 6.0, 10.0, 20.0 μeV)",
                "type": "global_summary",
                "tl": get_rel(p_bg2), "tl_alt": "Bias Gap Threshold 2.0 μeV",
                "tr": get_rel(p_bg6) if p_bg6.exists() else get_rel(p_bg2), "tr_alt": "Bias Gap Threshold 6.0 μeV",
                "bl": get_rel(p_bg10), "bl_alt": "Bias Gap Threshold 10.0 μeV (Microsoft Base)",
                "br": get_rel(p_bg20) if p_bg20.exists() else get_rel(p_bg10), "br_alt": "Bias Gap Threshold 20.0 μeV",
            })

        p_bg4 = sweeps_dir / "stage2_bias_gap_4.0uev.png"
        p_bg8 = sweeps_dir / "stage2_bias_gap_8.0uev.png"
        p_bg15 = sweeps_dir / "stage2_bias_gap_15.0uev.png"
        p_bg30 = sweeps_dir / "stage2_bias_gap_30.0uev.png"
        if p_bg4.exists() and p_bg30.exists():
            slides.append({
                "dataset": dataset_label,
                "title": f"[{dataset_label}] Stage 2 Islands vs. Bias Voltage Gap Threshold [Extended] (4.0, 8.0, 15.0, 30.0 μeV)",
                "type": "global_summary",
                "tl": get_rel(p_bg4), "tl_alt": "Bias Gap Threshold 4.0 μeV",
                "tr": get_rel(p_bg8) if p_bg8.exists() else get_rel(p_bg4), "tr_alt": "Bias Gap Threshold 8.0 μeV",
                "bl": get_rel(p_bg15) if p_bg15.exists() else get_rel(p_bg4), "bl_alt": "Bias Gap Threshold 15.0 μeV",
                "br": get_rel(p_bg30), "br_alt": "Bias Gap Threshold 30.0 μeV",
            })

    return slides

def render_html_presentation(slides, output_file: Path, title: str):
    slides_html_list = []
    options_list = []

    for i, slide in enumerate(slides):
        title_text = slide["title"]
        options_list.append(f'<option value="{i}">{i+1}. {title_text}</option>')

        stype = slide["type"]
        if stype in ("cut_overview", "cut_analytics"):
            layout_cls = "layout-cut-overview" if stype == "cut_overview" else "layout-cut-analytics"
            content = f"""
            <div class="{layout_cls}">
              <div class="col-stacked">
                <div class="img-card flex-1">
                  {f'<img src="{slide["left_top"]}" alt="3w Curvature">' if slide["left_top"] else '<p>3w map missing</p>'}
                </div>
                <div class="img-card flex-1">
                  {f'<img src="{slide["left_bottom"]}" alt="Transport Gap">' if slide["left_bottom"] else '<p>Gap map missing</p>'}
                </div>
              </div>
              <div class="img-card">
                {f'<img src="{slide["right"]}" alt="Cut Multiplot">' if slide["right"] else '<p>Cut multiplot missing</p>'}
              </div>
            </div>
            """
        elif stype == "point_deepdive":
            content = f"""
            <div class="layout-point-deepdive">
              <div class="img-card">
                {f'<img src="{slide["left"]}" alt="Cut Conductance Multiplot">' if slide["left"] else '<p>Cut conductance missing</p>'}
              </div>
              <div class="col-stacked-3">
                <div class="img-card flex-1">
                  {f'<img src="{slide["pt_didv"]}" alt="dI/dV">' if slide["pt_didv"] else '<p>dI/dV missing</p>'}
                </div>
                <div class="img-card flex-1">
                  {f'<img src="{slide["pt_wf"]}" alt="Wavefunctions">' if slide["pt_wf"] else '<p>Wavefunctions missing</p>'}
                </div>
                <div class="img-card flex-1">
                  {f'<img src="{slide["pt_barr"]}" alt="Barrier Asymmetry">' if slide["pt_barr"] else '<p>Barrier Asymmetry missing</p>'}
                </div>
              </div>
            </div>
            """
        elif stype in ("global_summary", "gap_sensitivity"):
            content = f"""
            <div class="layout-grid-2x2">
              <div class="img-card">
                {f'<img src="{slide["tl"]}" alt="{slide.get("tl_alt", "Top Left")}">' if slide["tl"] else '<p>Top Left missing</p>'}
              </div>
              <div class="img-card">
                {f'<img src="{slide["tr"]}" alt="{slide.get("tr_alt", "Top Right")}">' if slide["tr"] else '<p>Top Right missing</p>'}
              </div>
              <div class="img-card">
                {f'<img src="{slide["bl"]}" alt="{slide.get("bl_alt", "Bottom Left")}">' if slide["bl"] else '<p>Bottom Left missing</p>'}
              </div>
              <div class="img-card">
                {f'<img src="{slide["br"]}" alt="{slide.get("br_alt", "Bottom Right")}">' if slide["br"] else '<p>Bottom Right missing</p>'}
              </div>
            </div>
            """
        elif stype == "six_panel_overview":
            content = f"""
            <div class="layout-grid-3x2">
              <div class="img-card">
                {f'<img src="{slide["p1"]}" alt="{slide.get("p1_alt", "2w Conductance")}">' if slide.get("p1") else '<p>2w missing</p>'}
              </div>
              <div class="img-card">
                {f'<img src="{slide["p2"]}" alt="{slide.get("p2_alt", "3w Curvature")}">' if slide.get("p2") else '<p>3w missing</p>'}
              </div>
              <div class="img-card">
                {f'<img src="{slide["p3"]}" alt="{slide.get("p3_alt", "Transport Gap")}">' if slide.get("p3") else '<p>Transport Gap missing</p>'}
              </div>
              <div class="img-card">
                {f'<img src="{slide["p4"]}" alt="{slide.get("p4_alt", "Excitation Gap")}">' if slide.get("p4") else '<p>Excitation Gap missing</p>'}
              </div>
              <div class="img-card">
                {f'<img src="{slide["p5"]}" alt="{slide.get("p5_alt", "Conductance Gap Sensitivity")}">' if slide.get("p5") else '<p>Conductance Sens missing</p>'}
              </div>
              <div class="img-card">
                {f'<img src="{slide["p6"]}" alt="{slide.get("p6_alt", "Bias Gap Sensitivity")}">' if slide.get("p6") else '<p>Bias Sens missing</p>'}
              </div>
            </div>
            """
        elif stype == "side_by_side":
            content = f"""
            <div class="layout-side-by-side">
              <div class="img-card">
                {f'<img src="{slide["left"]}" alt="{slide.get("left_alt", "Left Plot")}">' if slide.get("left") else '<p>Left plot missing</p>'}
              </div>
              <div class="img-card">
                {f'<img src="{slide["right"]}" alt="{slide.get("right_alt", "Right Plot")}">' if slide.get("right") else '<p>Right plot missing</p>'}
              </div>
            </div>
            """
        elif stype == "two_row_comparison":
            content = f"""
            <div class="layout-two-row">
              <div class="img-card flex-1">
                {f'<img src="{slide["top"]}" alt="{slide.get("top_alt", "Top 2w Image")}">' if slide.get("top") else '<p>Top 2w image missing</p>'}
              </div>
              <div class="img-card flex-1">
                {f'<img src="{slide["bottom"]}" alt="{slide.get("bottom_alt", "Bottom Separability Image")}">' if slide.get("bottom") else '<p>Bottom Separability image missing</p>'}
              </div>
            </div>
            """
        elif stype == "single_centered":
            content = f"""
            <div class="layout-single-centered">
              <div class="img-card flex-1">
                {f'<img src="{slide["img"]}" alt="{slide.get("alt", "Slide Image")}">' if slide.get("img") else '<p>Image missing</p>'}
              </div>
            </div>
            """
        elif stype == "multicol_two_row":
            headers_html = "".join([f'<div style="text-align:center;">{lbl}</div>' for lbl in slide.get("labels", [])])
            top_parts = []
            for img in slide.get("top_imgs", []):
                inner = f'<img src="{img}" alt="Top">' if img else '<p>Missing</p>'
                top_parts.append(f'<div class="img-card flex-1">{inner}</div>')
            top_imgs = "".join(top_parts)

            bottom_parts = []
            for img in slide.get("bottom_imgs", []):
                inner = f'<img src="{img}" alt="Bottom">' if img else '<p>Missing</p>'
                bottom_parts.append(f'<div class="img-card flex-1">{inner}</div>')
            bottom_imgs = "".join(bottom_parts)

            n_cols = len(slide.get("labels", []))
            row_cls = f"multicol-row-{n_cols}" if n_cols in (3, 4) else ""
            content = f"""
            <div class="layout-multicol-two-row">
              <div class="multicol-header">{headers_html}</div>
              <div class="multicol-row {row_cls}">{top_imgs}</div>
              <div class="multicol-row {row_cls}">{bottom_imgs}</div>
            </div>
            """
        elif stype == "table_slide":
            content = f"""
            <div class="layout-table-slide">
              <div class="table-card">
                <h2>{slide.get("table_title", "Statistical Summary")} <span style="font-size:12px; font-weight:normal; color:#94a3b8;">Azure Quantum TGP Protocol</span></h2>
                <p class="subtitle">{slide.get("table_subtitle", "")}</p>
                {slide.get("html_table", "")}
              </div>
            </div>
            """
        elif stype == "side_by_side_html":
            content = f"""
            <div class="layout-side-by-side">
              <div class="img-card">
                {f'<img src="{slide["img"]}" alt="Phase Map">' if slide.get("img") else '<p>Image missing</p>'}
              </div>
              <div style="flex:1; height:100%; min-height:0; overflow-y:auto;">
                {slide.get("html_card", "")}
              </div>
            </div>
            """
        else:
            content = f"<p>Unknown slide layout</p>"

        slide_html = f"""
        <section class="slide" id="slide-{i}" data-title="{title_text} | {slide['dataset']}">
          {content}
        </section>
        """
        slides_html_list.append(slide_html)

    final_html = HTML_TEMPLATE.format(
        title=title,
        select_options="\n".join(options_list),
        total_slides=len(slides),
        slides_html="\n".join(slides_html_list)
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"Generated slide presentation ({len(slides)} slides) -> {output_file}")

def build_intro_sensitivity_slides(plots_root: Path):
    """
    Build introductory sensitivity analysis slides across all disorder amplitudes:
      1. Cross-disorder Transport Gap comparison (2x2 grid)
      2. Cross-disorder Sensitivity (Delta max - Delta min) comparison (2x2 grid)
      3. For each disorder amplitude: side-by-side Transport Gap vs Sensitivity plot
      4. For each disorder amplitude: stacked 2-row 2w Measurement vs MZM Separability comparison
      5. For each disorder amplitude: replicated Stage 2 Paper Diagram
    """
    disorder_configs = [
        ("Tdis_pfaff5_V0_0_0_Plots", "V0 = 0.0 (Clean Wire)"),
        ("Tdis_pfaff5_V0_0_1_Plots", "V0 = 0.1 (Low Disorder)"),
        ("Tdis_pfaff5_V0_0_378_Plots", "V0 = 0.378 (Intermediate Disorder)"),
        ("Tdis_pfaff5_V0_0_645_Plots", "V0 = 0.645 (Strong Disorder)"),
        ("Tdis_pfaff5_V0_0_872_Plots", "V0 = 0.872 (Strong Disorder)"),
        ("Tdis_pfaff5_V0_0_91_Plots", "V0 = 0.91 (Strong Disorder)"),
        ("Tdis_pfaff5_Plots", "V0 = 1.2 (Benchmark Disorder)"),
    ]

    valid = []
    for dname, label in disorder_configs:
        pdir = plots_root / dname
        p_gap = pdir / "global_transport_gap_phase_map.png"
        p_sens = pdir / "global_gap_threshold_sensitivity_phase_map.png"
        p_2w = pdir / "global_2w_phase_map.png"
        p_sep = pdir / "global_separability_phase_map.png"
        p_s2 = pdir / "stage2_paper_diagram.png"
        p_comp = pdir / "pfaffian_vs_tgp_islands.png"

        entry = {
            "dir_name": dname,
            "label": label,
            "gap": os.path.relpath(p_gap, plots_root) if p_gap.exists() else None,
            "sens": os.path.relpath(p_sens, plots_root) if p_sens.exists() else None,
            "two_w": os.path.relpath(p_2w, plots_root) if p_2w.exists() else None,
            "sep": os.path.relpath(p_sep, plots_root) if p_sep.exists() else None,
            "s2": os.path.relpath(p_s2, plots_root) if p_s2.exists() else None,
            "comp": os.path.relpath(p_comp, plots_root) if p_comp.exists() else None,
        }
        valid.append(entry)

    intro_slides = []

    # 0. Master False Discovery Rate (FDR) Table Across All Disorder Realizations
    fdr_rows = []
    for dname, label in disorder_configs:
        pdir = plots_root / dname
        fdr_file = pdir / "TGP_Cuts" / "fdr_metrics.json"
        if fdr_file.exists():
            try:
                with open(fdr_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                gt = data.get("ground_truth_topological_pixels", 0)
                s2 = data.get("stage2_roi2_metrics", {})
                passed = s2.get("total_passed", 0)
                tp = s2.get("true_positives", 0)
                fp = s2.get("false_positives", 0)
                ppv = s2.get("p_topological_given_passed_ppv", 0.0) * 100.0
                fdr = s2.get("false_discovery_rate_fdr", 0.0) * 100.0
                tpr = s2.get("sensitivity_tpr", 0.0) * 100.0
                tnr = s2.get("specificity_tnr", 0.0) * 100.0

                or_m = data.get("candidate_orange_state_metrics", {})
                or_ppv = or_m.get("p_topological_given_orange_ppv", 0.0) * 100.0
                or_fdr = or_m.get("false_discovery_rate_fdr", 0.0) * 100.0

                badge_fdr = "badge-pass" if fdr < 5.0 else ("badge-warn" if fdr < 30.0 else "badge-fail")
                badge_ppv = "badge-pass" if ppv > 90.0 else ("badge-warn" if ppv > 65.0 else "badge-fail")

                fdr_rows.append(f"""
                <tr>
                  <td><strong>{label}</strong></td>
                  <td>{gt}</td>
                  <td>{passed}</td>
                  <td>{tp}</td>
                  <td>{fp}</td>
                  <td><span class="{badge_ppv}">{ppv:.2f}%</span></td>
                  <td><span class="{badge_fdr}">{fdr:.2f}%</span></td>
                  <td>{tpr:.2f}%</td>
                  <td>{tnr:.2f}%</td>
                  <td>{or_ppv:.2f}%</td>
                  <td>{or_fdr:.2f}%</td>
                </tr>
                """)
            except Exception:
                pass

    if fdr_rows:
        master_table_html = f"""
        <table class="data-table">
          <thead>
            <tr>
              <th>Disorder Realization</th>
              <th>Ground Truth Top. (px)</th>
              <th>Stage 2 Passed (px)</th>
              <th>TP</th>
              <th>FP</th>
              <th>P(Top | Pass) [PPV]</th>
              <th>False Discovery Rate [FDR]</th>
              <th>Sensitivity (TPR)</th>
              <th>Specificity (TNR)</th>
              <th>Candidate Orange PPV</th>
              <th>Candidate Orange FDR</th>
            </tr>
          </thead>
          <tbody>
            {"".join(fdr_rows)}
          </tbody>
        </table>
        <div style="margin-top:16px; font-size:11px; color:#94a3b8; line-height:1.5;">
          <strong>Statistical Metric Definitions:</strong><br>
          • <strong>Positive Predictive Value (PPV):</strong> P(Topological | Passed) = TP / (TP + FP). Probability that an ROI passing Stage 2 is topologically non-trivial.<br>
          • <strong>False Discovery Rate (FDR):</strong> 1 - PPV = FP / (TP + FP). Probability of a false positive topological detection.<br>
          • <strong>Sensitivity (TPR):</strong> TP / (TP + FN). Fraction of physical topological area detected by the protocol.<br>
          • <strong>Specificity (TNR):</strong> TN / (TN + FP). Fraction of trivial parameter space correctly rejected.
        </div>
        """
        intro_slides.append({
            "dataset": "All Disorders",
            "title": "All Disorders | False Discovery Rate (FDR) & Topological Protocol Verification Table",
            "type": "table_slide",
            "table_title": "Cross-Disorder False Discovery Rate & Protocol Verification Table",
            "table_subtitle": "Systematic evaluation of Stage 2 ROI2 against Ground-Truth Pfaffian across all 7 disorder amplitudes",
            "html_table": master_table_html
        })

    # Panoramic Multi-Column 2-Row Comparison: 2w (Top Row) vs Separability (Bottom Row)
    # Group 1: Clean to Intermediate (V0 = 0.0, 0.1, 0.378)
    group1 = [v for v in valid if any(k in v["label"] for k in ("0.0 ", "0.1 ", "0.378 ")) and (v["two_w"] or v["sep"])]
    if len(group1) == 3:
        intro_slides.append({
            "dataset": "All Disorders",
            "title": "All Disorders | Panoramic 2ω (Top) vs. Separability (Bottom) [Clean to Intermediate: V0 = 0.0, 0.1, 0.378]",
            "type": "multicol_two_row",
            "labels": [v["label"] for v in group1],
            "top_imgs": [v["two_w"] for v in group1],
            "bottom_imgs": [v["sep"] for v in group1],
        })

    # Group 2: Strong Disorders (V0 = 0.645, 0.872, 0.91, 1.2)
    group2 = [v for v in valid if any(k in v["label"] for k in ("0.645", "0.872", "0.91", "1.2")) and (v["two_w"] or v["sep"])]
    if len(group2) >= 3:
        intro_slides.append({
            "dataset": "All Disorders",
            "title": "All Disorders | Panoramic 2ω (Top) vs. Separability (Bottom) [Strong Disorders: V0 = 0.645, 0.872, 0.91, 1.2]",
            "type": "multicol_two_row",
            "labels": [v["label"] for v in group2],
            "top_imgs": [v["two_w"] for v in group2],
            "bottom_imgs": [v["sep"] for v in group2],
        })

    # 1. Cross-Disorder 2-Row Comparison: 2w (Top) vs Separability (Bottom) across all realizations
    for v in valid:
        if v["two_w"] or v["sep"]:
            intro_slides.append({
                "dataset": "Cross-Disorder Comparison",
                "title": f"Cross-Disorder | [{v['label']}] 2ω Measurement (Top) vs. MZM Separability (Bottom)",
                "type": "two_row_comparison",
                "top": v["two_w"],
                "top_alt": f"2ω Conductance - {v['label']}",
                "bottom": v["sep"],
                "bottom_alt": f"MZM Separability - {v['label']}",
            })

    # 2. Cross-Disorder Stage 2 Paper Diagrams
    for v in valid:
        if v["s2"]:
            intro_slides.append({
                "dataset": "Cross-Disorder Comparison",
                "title": f"Cross-Disorder | [{v['label']}] Microsoft Stage 2 Paper Diagram (PRL Fig. 29)",
                "type": "single_centered",
                "img": v["s2"],
                "alt": f"Stage 2 Paper Diagram - {v['label']}",
            })
        if v["comp"]:
            intro_slides.append({
                "dataset": "Cross-Disorder Comparison",
                "title": f"Cross-Disorder | [{v['label']}] Pfaffian vs. Full TGP ROI2 Islands",
                "type": "single_centered",
                "img": v["comp"],
                "alt": f"Pfaffian vs ROI2 Islands - {v['label']}",
            })

    # 3. Multi-disorder 2x2 comparison of Transport Gaps (First 4 available)
    valid_gap_sens = [v for v in valid if v["gap"] and v["sens"]]
    if len(valid_gap_sens) >= 4:
        intro_slides.append({
            "dataset": "All Disorders",
            "title": f"All Disorders | Transport Gap Phase Maps ({valid_gap_sens[0]['label']}, {valid_gap_sens[1]['label']}, {valid_gap_sens[2]['label']}, {valid_gap_sens[3]['label']})",
            "type": "global_summary",
            "tl": valid_gap_sens[0]["gap"],
            "tl_alt": f"Transport Gap - {valid_gap_sens[0]['label']}",
            "tr": valid_gap_sens[1]["gap"],
            "tr_alt": f"Transport Gap - {valid_gap_sens[1]['label']}",
            "bl": valid_gap_sens[2]["gap"],
            "bl_alt": f"Transport Gap - {valid_gap_sens[2]['label']}",
            "br": valid_gap_sens[3]["gap"],
            "br_alt": f"Transport Gap - {valid_gap_sens[3]['label']}",
        })

        intro_slides.append({
            "dataset": "All Disorders",
            "title": f"All Disorders | Gap Sensitivity Maps Δmax-Δmin ({valid_gap_sens[0]['label']}, {valid_gap_sens[1]['label']}, {valid_gap_sens[2]['label']}, {valid_gap_sens[3]['label']})",
            "type": "global_summary",
            "tl": valid_gap_sens[0]["sens"],
            "tl_alt": f"Sensitivity Map - {valid_gap_sens[0]['label']}",
            "tr": valid_gap_sens[1]["sens"],
            "tr_alt": f"Sensitivity Map - {valid_gap_sens[1]['label']}",
            "bl": valid_gap_sens[2]["sens"],
            "bl_alt": f"Sensitivity Map - {valid_gap_sens[2]['label']}",
            "br": valid_gap_sens[3]["sens"],
            "br_alt": f"Sensitivity Map - {valid_gap_sens[3]['label']}",
        })

    # 4. For each disorder amplitude: side-by-side Transport Gap vs Sensitivity plot
    for v in valid_gap_sens:
        intro_slides.append({
            "dataset": v["label"].split()[0],
            "title": f"[{v['label']}] Transport Gap vs. Extraction Sensitivity (Δmax - Δmin)",
            "type": "side_by_side",
            "left": v["gap"],
            "left_alt": f"Transport Gap ({v['label']})",
            "right": v["sens"],
            "right_alt": f"Gap Extraction Sensitivity ({v['label']})"
        })

    return intro_slides

def main():
    parser = argparse.ArgumentParser(description="Generate HTML slide deck from Cut Analysis plots.")
    parser.add_argument("plot_dirs", nargs="*", help="List of plot directories to process.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    plots_root = project_root / "Outputs" / "Plots"

    if args.plot_dirs:
        target_dirs = [Path(d).resolve() for d in args.plot_dirs]
    else:
        # Default specified list from user prompt
        candidate_names = [
            "Tdis_pfaff5_V0_0_0_Plots",
            "Tdis_pfaff5_V0_0_1_Plots",
            "Tdis_pfaff5_V0_0_378_Plots",
            "Tdis_pfaff5_V0_0_645_Plots",
            "Tdis_pfaff5_V0_0_872_Plots",
            "Tdis_pfaff5_V0_0_91_Plots",
            "Tdis_pfaff5_Plots"
        ]
        target_dirs = [plots_root / name for name in candidate_names if (plots_root / name).exists()]

    if not target_dirs:
        print("No valid plot directories found to generate slides from.")
        return

    # 1. Generate standalone slides.html for each individual directory
    all_master_slides = []
    for pdir in target_dirs:
        slides = build_slides_for_directory(pdir, base_rel_path=pdir)
        if slides:
            out_file = pdir / "slides.html"
            render_html_presentation(slides, out_file, title=f"Cut Analysis Slides - {pdir.name}")
        
        # Also accumulate for master deck (using plots_root as relative root)
        m_slides = build_slides_for_directory(pdir, base_rel_path=plots_root)
        all_master_slides.extend(m_slides)

    # 2. Generate Master All-in-One Presentation with Sensitivity Analysis prepended at the beginning
    if all_master_slides:
        intro_slides = build_intro_sensitivity_slides(plots_root)
        final_master_slides = intro_slides + all_master_slides
        master_file = plots_root / "all_disorder_slides.html"
        render_html_presentation(final_master_slides, master_file, title="Majorana Nanowire Cut Analysis - All Disorder Realizations")

if __name__ == "__main__":
    main()
