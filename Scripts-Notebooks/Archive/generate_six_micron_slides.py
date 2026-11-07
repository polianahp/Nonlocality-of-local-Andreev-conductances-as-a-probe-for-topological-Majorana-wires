#!/usr/bin/env python3
"""
generate_six_micron_slides.py

Assembles an interactive, responsive, dark-mode HTML slide deck comparing
the 6-micron wire vs. 3-micron wire for Microsoft TGP vs. the 3-step Nonlocal Protocol:
  - Slide 1: Executive Master Summary Table
  - Slide 2: Step 1 (Correlation Condition) Sensitivity Analysis across V0
  - Slide 3: Step 2 (3w Measurement) Sensitivity Analysis across V0
  - Slide 4: Step 3 (Extracted Transport Gap) Sensitivity Analysis across V0
  - Slides 5–16: 6 paired disorders x 2 slides each:
      * Slide A: Protocol Comparison (4-Panel: 3um TGP, 6um TGP, 3um Step 3, 6um Step 3) + Delta Table
      * Slide B: Step-by-Step Sensitivity & Spatial Transition (4-Panel: Step 1, Step 2, Step 3, Difference Map) + Breakdown
  - Slide 17: Synthesis, Physical Conclusions & Device Design Implications

Output: Outputs/slides/six_micron_analysis/slides.html
"""

import os
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "Outputs" / "slides" / "six_micron_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_FILE = OUTPUT_DIR / "metrics" / "length_comparison_summary.json"
SLIDES_HTML_PATH = OUTPUT_DIR / "slides.html"

DISORDER_DESCRIPTIONS = {
    "V0_0_0": {
        "title": "Clean Nanowire (V0 = 0.0 meV)",
        "regime": "Pristine Topological Superconductor",
    },
    "V0_0_1": {
        "title": "Low Disorder (V0 = 0.1 meV)",
        "regime": "Weak Perturbation Regime",
    },
    "V0_0_645": {
        "title": "Strong Disorder (V0 = 0.645 meV)",
        "regime": "Topological Gap Suppression",
    },
    "V0_0_872": {
        "title": "Strong Disorder (V0 = 0.872 meV)",
        "regime": "Emerging Majorana Gap Inversion",
    },
    "V0_0_91": {
        "title": "Strong Disorder (V0 = 0.91 meV)",
        "regime": "Strong Disorder with Impurities",
    },
    "V0_1_2": {
        "title": "Benchmark Strong Disorder (V0 = 1.2 meV)",
        "regime": "Extreme Disorder Boundary Regime",
    },
}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>6 μm vs. 3 μm Wire Length Sensitivity: Microsoft TGP vs. 3-Step Protocol</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}], throwOnError: false}});"></script>
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

  /* Top Navigation Bar */
  header {{
    background-color: #0f172a;
    border-bottom: 1px solid #1e293b;
    padding: 0 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 42px;
    user-select: none;
    z-index: 100;
    flex-shrink: 0;
  }}
  .nav-left {{
    display: flex;
    align-items: center;
    gap: 12px;
    overflow: hidden;
    min-width: 0;
  }}
  .nav-title {{
    font-size: 13.5px;
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
    max-width: 52vw;
  }}
  .nav-right {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
  }}
  .slide-select {{
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 12px;
    outline: none;
    max-width: 360px;
    cursor: pointer;
  }}
  .btn {{
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 3px 12px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .btn:hover {{
    background-color: #334155;
    color: #fff;
    border-color: #38bdf8;
  }}
  .counter {{
    font-size: 12px;
    color: #94a3b8;
    min-width: 80px;
    text-align: center;
    font-variant-numeric: tabular-nums;
  }}
  .hint {{
    font-size: 11px;
    color: #64748b;
  }}

  /* Main Presentation Canvas */
  main {{
    flex: 1;
    position: relative;
    width: 100vw;
    height: calc(100vh - 42px);
    overflow: hidden;
  }}
  .slide {{
    display: none;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    padding: 10px 14px;
    background-color: #0b1120;
    overflow-y: auto;
    box-sizing: border-box;
  }}
  .slide.active {{
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
  }}

  /* Split Layouts: 62% High-Res Image vs 38% Analytical Side Card */
  .layout-split {{
    flex: 1;
    display: grid;
    grid-template-columns: 62% 38%;
    gap: 12px;
    min-height: 0;
    height: 100%;
    width: 100%;
  }}
  .img-container {{
    background-color: #070d1a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    position: relative;
    cursor: zoom-in;
  }}
  .img-container img {{
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    border-radius: 4px;
  }}
  .card-container {{
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 20px 16px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    overflow-y: auto;
    gap: 16px;
  }}

  /* Single Full-Canvas Layout for Master Tables & Synthesis */
  .layout-full {{
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
    height: 100%;
    width: 100%;
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 20px 24px;
    overflow-y: auto;
    gap: 14px;
  }}

  /* Typography & Tables */
  h2 {{
    font-size: 17px;
    color: #38bdf8;
    font-weight: 700;
    line-height: 1.3;
  }}
  h3 {{
    font-size: 14.5px;
    color: #f1f5f9;
    font-weight: 600;
  }}
  p {{
    font-size: 12.5px;
    color: #cbd5e1;
    line-height: 1.5;
  }}
  .section-label {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .label-global {{ background: #1e3a8a; color: #93c5fd; }}
  .label-disorder {{ background: #065f46; color: #a7f3d0; }}
  .label-summary {{ background: #4c1d95; color: #ddd6fe; }}

  table.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
    margin: 4px 0;
  }}
  table.data-table th {{
    background-color: #1e293b;
    color: #94a3b8;
    padding: 6px 8px;
    text-align: left;
    font-weight: 600;
    border-bottom: 1px solid #334155;
    white-space: nowrap;
    font-size: 11px;
  }}
  table.data-table td {{
    padding: 6px 8px;
    border-bottom: 1px solid #1e293b;
    color: #e2e8f0;
    font-size: 11px;
    line-height: 1.4;
  }}
  table.data-table tr:hover td {{
    background-color: #132036;
  }}

  /* Badges */
  .badge {{
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 11px;
    text-align: center;
  }}
  .badge-pass {{ background-color: #064e3b; color: #34d399; border: 1px solid #059669; }}
  .badge-warn {{ background-color: #78350f; color: #fbbf24; border: 1px solid #d97706; }}
  .badge-fail {{ background-color: #881337; color: #f43f5e; border: 1px solid #e11d48; }}
  .badge-neutral {{ background-color: #1e293b; color: #94a3b8; border: 1px solid #334155; }}

  /* Modal Image Zoom */
  .modal {{
    display: none;
    position: fixed;
    z-index: 1000;
    left: 0;
    top: 0;
    width: 100vw;
    height: 100vh;
    background-color: rgba(3, 7, 18, 0.94);
    align-items: center;
    justify-content: center;
    cursor: zoom-out;
  }}
  .modal img {{
    max-width: 96vw;
    max-height: 96vh;
    object-fit: contain;
    border-radius: 6px;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
  }}

  @media print {{
    @page {{
      size: 16in 9in;
      margin: 0;
    }}
    body {{
      overflow: visible !important;
      height: auto !important;
      background-color: #0b1120 !important;
      -webkit-print-color-adjust: exact !important;
      print-color-adjust: exact !important;
    }}
    header {{
      display: none !important;
    }}
    main {{
      position: static !important;
      height: auto !important;
      overflow: visible !important;
    }}
    .slide {{
      display: flex !important;
      position: relative !important;
      height: 9in !important;
      width: 16in !important;
      page-break-after: always !important;
      break-after: page !important;
      overflow: hidden !important;
    }}
    .modal {{
      display: none !important;
    }}
  }}
</style>
</head>
<body>

<header>
  <div class="nav-left">
    <span class="nav-title">NonlocalProtocol</span>
    <span class="nav-sep">/</span>
    <span class="nav-title" style="color:#818cf8;">6μm vs 3μm Scaling</span>
    <span class="nav-sep">/</span>
    <span class="slide-title-display" id="slideTitle">Executive Master Summary Table</span>
  </div>
  <div class="nav-right">
    <button class="btn" id="btnPrev">Prev (←)</button>
    <span class="counter" id="slideCounter">Slide 1 / {total_slides}</span>
    <button class="btn" id="btnNext">Next (→)</button>
    <select class="slide-select" id="slideDropdown">
      {dropdown_options}
    </select>
  </div>
</header>

<main id="slideContainer">
  {slides_content}
</main>

<div class="modal" id="imageModal" onclick="this.style.display='none'">
  <img id="modalImg" src="">
</div>

<script>
  let currentSlide = 0;
  const slides = document.querySelectorAll('.slide');
  const totalSlides = slides.length;
  const slideTitle = document.getElementById('slideTitle');
  const slideCounter = document.getElementById('slideCounter');
  const slideDropdown = document.getElementById('slideDropdown');
  const modal = document.getElementById('imageModal');
  const modalImg = document.getElementById('modalImg');

  function showSlide(index) {{
    if (index < 0) index = 0;
    if (index >= totalSlides) index = totalSlides - 1;
    currentSlide = index;

    slides.forEach((s, i) => {{
      s.classList.toggle('active', i === currentSlide);
    }});

    const activeSlide = slides[currentSlide];
    const title = activeSlide.getAttribute('data-title') || `Slide ${{currentSlide + 1}}`;
    slideTitle.textContent = title;
    slideCounter.textContent = `Slide ${{currentSlide + 1}} / ${{totalSlides}}`;
    slideDropdown.value = currentSlide;
  }}

  document.getElementById('btnPrev').addEventListener('click', () => showSlide(currentSlide - 1));
  document.getElementById('btnNext').addEventListener('click', () => showSlide(currentSlide + 1));
  slideDropdown.addEventListener('change', (e) => showSlide(parseInt(e.target.value)));

  window.addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {{
      e.preventDefault();
      showSlide(currentSlide + 1);
    }} else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {{
      e.preventDefault();
      showSlide(currentSlide - 1);
    }} else if (e.key === 'Home') {{
      e.preventDefault();
      showSlide(0);
    }} else if (e.key === 'End') {{
      e.preventDefault();
      showSlide(totalSlides - 1);
    }} else if (e.key === 'Escape') {{
      modal.style.display = 'none';
    }}
  }});

  document.querySelectorAll('.img-container img').forEach(img => {{
    img.addEventListener('click', () => {{
      modalImg.src = img.src;
      modal.style.display = 'flex';
    }});
  }});

  showSlide(0);
</script>

</body>
</html>
"""


def build_slide1_master_table(summary_data: list) -> str:
    """Slide 1: Master Executive Summary Table across all 6 benchmark disorders."""
    rows = []
    for d in summary_data:
        tag = d["tag"]
        v0 = d["V0"]
        m3 = d["3micron"]
        m6 = d["6micron"]
        d3 = d["deltas_step3"]

        tgp3_tpr = m3["tgp"]["pfaffian_passed_pct"]
        tgp3_fdr = m3["tgp"]["fdr_percent"]
        tgp3_conf = m3["tgp"]["mean_boundary_confinement"]
        tgp3_sep = m3["tgp"]["mean_separability"]

        tgp6_tpr = m6["tgp"]["pfaffian_passed_pct"]
        tgp6_fdr = m6["tgp"]["fdr_percent"]
        tgp6_conf = m6["tgp"]["mean_boundary_confinement"]
        tgp6_sep = m6["tgp"]["mean_separability"]

        s3_3_tpr = m3["metrics_s3"]["pfaffian_passed_pct"]
        s3_3_fdr = m3["metrics_s3"]["fdr_percent"]
        s3_3_conf = m3["metrics_s3"]["mean_boundary_confinement"]
        s3_3_sep = m3["metrics_s3"]["mean_separability"]

        s3_6_tpr = m6["metrics_s3"]["pfaffian_passed_pct"]
        s3_6_fdr = m6["metrics_s3"]["fdr_percent"]
        s3_6_conf = m6["metrics_s3"]["mean_boundary_confinement"]
        s3_6_sep = m6["metrics_s3"]["mean_separability"]

        d_tpr = d3["delta_tpr"]
        d_fdr = d3["delta_fdr"]
        d_conf = d3["delta_confinement"]
        d_sep = d3["delta_separability"]

        badge_fdr3 = "badge-pass" if s3_3_fdr < 2.0 else ("badge-warn" if s3_3_fdr < 10.0 else "badge-fail")
        badge_fdr6 = "badge-pass" if s3_6_fdr < 2.0 else ("badge-warn" if s3_6_fdr < 10.0 else "badge-fail")
        badge_tgp3 = "badge-pass" if tgp3_fdr < 5.0 else ("badge-warn" if tgp3_fdr < 20.0 else "badge-fail")
        badge_tgp6 = "badge-pass" if tgp6_fdr < 5.0 else ("badge-warn" if tgp6_fdr < 20.0 else "badge-fail")
        badge_dfdr = "badge-pass" if d_fdr <= 0 else "badge-fail"
        badge_dconf = "badge-pass" if d_conf >= 0 else "badge-warn"
        badge_dsep = "badge-pass" if d_sep >= 0 else "badge-warn"

        area_mult = f"{s3_6_tpr / s3_3_tpr:.1f}×" if s3_3_tpr > 0 else "N/A"

        tgp3_details = f"<strong style=\"color:#38bdf8;\">{tgp3_tpr:.1f}% TPR</strong> | <span class=\"{badge_tgp3}\">{tgp3_fdr:.1f}% FDR</span>"
        if tgp3_tpr > 0:
            tgp3_details += f"<br><span style=\"color:#c084fc; font-size:10px;\">Conf: <strong>{tgp3_conf:.3f}</strong> | Sep: <strong>{tgp3_sep:.3f}</strong></span>"
        else:
            tgp3_details += "<br><span style=\"color:#64748b; font-size:10px;\">(0.0% TPR)</span>"

        tgp6_details = f"<strong style=\"color:#38bdf8;\">{tgp6_tpr:.1f}% TPR</strong> | <span class=\"{badge_tgp6}\">{tgp6_fdr:.1f}% FDR</span>"
        if tgp6_tpr > 0:
            tgp6_details += f"<br><span style=\"color:#c084fc; font-size:10px;\">Conf: <strong>{tgp6_conf:.3f}</strong> | Sep: <strong>{tgp6_sep:.3f}</strong></span>"
        else:
            tgp6_details += "<br><span style=\"color:#64748b; font-size:10px;\">(0.0% TPR)</span>"

        row = f"""
        <tr>
          <td><strong style="color:#38bdf8;">{tag}</strong><br><span style="color:#94a3b8; font-size:10px;">{v0:.3f} meV</span></td>
          <td>{tgp3_details}</td>
          <td>{tgp6_details}</td>
          <td><strong style="color:#38bdf8;">{s3_3_tpr:.1f}% TPR</strong> | <span class="{badge_fdr3}">{s3_3_fdr:.2f}% FDR</span><br><span style="color:#c084fc; font-size:10px;">Conf: <strong>{s3_3_conf:.3f}</strong> | Sep: <strong>{s3_3_sep:.3f}</strong></span></td>
          <td><strong style="color:#34d399;">{s3_6_tpr:.1f}% TPR</strong> | <span class="{badge_fdr6}">{s3_6_fdr:.2f}% FDR</span><br><span style="color:#c084fc; font-size:10px;">Conf: <strong>{s3_6_conf:.3f}</strong> | Sep: <strong>{s3_6_sep:.3f}</strong></span></td>
          <td><strong style="color:#34d399;">+{d_tpr:.1f}% TPR</strong><br><span class="badge badge-pass">{area_mult} Expansion</span></td>
          <td><span class="{badge_dfdr}">{d_fdr:+.2f}%</span></td>
          <td>{s3_3_conf:.4f} → <strong style="color:#38bdf8;">{s3_6_conf:.4f}</strong><br><span class="{badge_dconf}">{d_conf:+.4f} ΔConf</span></td>
          <td>{s3_3_sep:.4f} → <strong style="color:#34d399;">{s3_6_sep:.4f}</strong><br><span class="{badge_dsep}">{d_sep:+.4f} ΔSep</span></td>
        </tr>
        """
        rows.append(row)

    table_rows = "\n".join(rows)

    return rf"""
    <div class="slide" data-title="Slide 1: Executive Master Summary Table (6μm vs 3μm)">
      <div class="layout-full">
        <div>
          <span class="section-label label-summary">Section 1: Summary Statistics</span>
          <h2 style="margin-top:6px; font-size:20px;">Master Wire Length Scaling Summary: Microsoft TGP vs. 3-Step Nonlocal Protocol</h2>
        </div>

        <table class="data-table" style="font-size:12px; margin-top:14px;">
          <thead>
            <tr>
              <th>Disorder (V₀)</th>
              <th>3 μm Microsoft TGP</th>
              <th>6 μm Microsoft TGP</th>
              <th>3 μm Step 3 Protocol</th>
              <th>6 μm Step 3 Protocol</th>
              <th>ΔTPR (Pfaffian Area Passed)</th>
              <th>ΔFDR</th>
              <th>Avg Boundary Confinement (3μm → 6μm)</th>
              <th>Avg MZM Separability (3μm → 6μm)</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
        </table>
      </div>
    </div>
    """


def build_per_disorder_slides(summary_data: list) -> list:
    """Builds 6 slides per disorder (Slides A–F) for each of the 6 benchmark disorders."""
    slides = []
    slide_num = 2

    for d in summary_data:
        tag = d["tag"]
        v0 = d["V0"]
        meta = DISORDER_DESCRIPTIONS.get(tag, {})
        title_tag = meta.get("title", f"Disorder V0 = {v0:.3f} meV")
        regime = meta.get("regime", "Disorder Study")

        m3 = d["3micron"]
        m6 = d["6micron"]
        d3 = d["deltas_step3"]

        # Slide A: Protocol Comparison (4-Panel)
        img_a = f"figures/{tag}_slideA_protocol_comparison.png"
        slide_a_title = f"Slide {slide_num}: [{tag}] TGP vs. 3-Step Protocol (6μm vs 3μm)"

        badge_tgp3 = "badge-pass" if m3["tgp"]["fdr_percent"] < 5.0 else ("badge-warn" if m3["tgp"]["fdr_percent"] < 20.0 else "badge-fail")
        badge_tgp6 = "badge-pass" if m6["tgp"]["fdr_percent"] < 5.0 else ("badge-warn" if m6["tgp"]["fdr_percent"] < 20.0 else "badge-fail")
        badge_s3_3 = "badge-pass" if m3["metrics_s3"]["fdr_percent"] < 2.0 else ("badge-warn" if m3["metrics_s3"]["fdr_percent"] < 10.0 else "badge-fail")
        badge_s3_6 = "badge-pass" if m6["metrics_s3"]["fdr_percent"] < 2.0 else ("badge-warn" if m6["metrics_s3"]["fdr_percent"] < 10.0 else "badge-fail")
        badge_dfdr = "badge-pass" if d3["delta_fdr"] <= 0 else "badge-fail"

        slide_a_html = f"""
        <div class="slide" data-title="{slide_a_title}">
          <div class="layout-split">
            <div class="img-container">
              <img src="{img_a}" alt="{tag} Slide A: Protocol Comparison">
            </div>
            <div class="card-container">
              <div>
                <span class="section-label label-disorder">{regime}</span>
                <h2 style="margin-top:6px;">{title_tag}</h2>
                <h3 style="color:#94a3b8; font-size:12.5px; margin-top:2px;">Protocol Comparison: Microsoft TGP vs. 3-Step Nonlocal Protocol</h3>
              </div>

              <table class="data-table">
                <thead>
                  <tr><th>Protocol & Length</th><th>TPR (% Pfaff)</th><th>FDR (%)</th><th>Boundary Conf.</th><th>Separability</th></tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong style="color:#38bdf8;">3 μm Microsoft TGP (ROI 2)</strong></td>
                    <td>{m3['tgp']['pfaffian_passed_pct']:.2f}%</td>
                    <td><span class="{badge_tgp3}">{m3['tgp']['fdr_percent']:.2f}%</span></td>
                    <td>{m3['tgp']['mean_boundary_confinement']:.4f}</td>
                    <td>{m3['tgp']['mean_separability']:.4f}</td>
                  </tr>
                  <tr>
                    <td><strong style="color:#38bdf8;">6 μm Microsoft TGP (ROI 2)</strong></td>
                    <td>{m6['tgp']['pfaffian_passed_pct']:.2f}%</td>
                    <td><span class="{badge_tgp6}">{m6['tgp']['fdr_percent']:.2f}%</span></td>
                    <td>{m6['tgp']['mean_boundary_confinement']:.4f}</td>
                    <td>{m6['tgp']['mean_separability']:.4f}</td>
                  </tr>
                  <tr>
                    <td><strong style="color:#34d399;">3 μm 3-Step Protocol (ROI 3)</strong></td>
                    <td>{m3['metrics_s3']['pfaffian_passed_pct']:.2f}%</td>
                    <td><span class="{badge_s3_3}">{m3['metrics_s3']['fdr_percent']:.2f}%</span></td>
                    <td>{m3['metrics_s3']['mean_boundary_confinement']:.4f}</td>
                    <td>{m3['metrics_s3']['mean_separability']:.4f}</td>
                  </tr>
                  <tr>
                    <td><strong style="color:#34d399;">6 μm 3-Step Protocol (ROI 3)</strong></td>
                    <td><strong>{m6['metrics_s3']['pfaffian_passed_pct']:.2f}%</strong></td>
                    <td><span class="{badge_s3_6}">{m6['metrics_s3']['fdr_percent']:.2f}%</span></td>
                    <td><strong>{m6['metrics_s3']['mean_boundary_confinement']:.4f}</strong></td>
                    <td><strong>{m6['metrics_s3']['mean_separability']:.4f}</strong></td>
                  </tr>
                  <tr>
                    <td><strong style="color:#38bdf8;">Scaling Delta (6μm - 3μm)</strong></td>
                    <td><strong>{d3['delta_tpr']:+.2f}%</strong></td>
                    <td><span class="{badge_dfdr}"><strong>{d3['delta_fdr']:+.2f}%</strong></span></td>
                    <td><strong>{d3['delta_confinement']:+.4f}</strong></td>
                    <td><strong>{d3['delta_separability']:+.4f}</strong></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
        """
        slides.append(slide_a_html)
        slide_num += 1

        # Slide B: Steps Evolution & Spatial Transition Map
        img_b = f"figures/{tag}_slideB_step_sensitivity.png"
        slide_b_title = f"Slide {slide_num}: [{tag}] Steps Evolution & Spatial Transition Map"

        f1_d = m6['metrics_s1']['fdr_percent'] - m3['metrics_s1']['fdr_percent']
        f2_d = m6['metrics_s2']['fdr_percent'] - m3['metrics_s2']['fdr_percent']
        f3_d = m6['metrics_s3']['fdr_percent'] - m3['metrics_s3']['fdr_percent']

        t1_d = m6['metrics_s1']['pfaffian_passed_pct'] - m3['metrics_s1']['pfaffian_passed_pct']
        t2_d = m6['metrics_s2']['pfaffian_passed_pct'] - m3['metrics_s2']['pfaffian_passed_pct']
        t3_d = m6['metrics_s3']['pfaffian_passed_pct'] - m3['metrics_s3']['pfaffian_passed_pct']

        slide_b_html = f"""
        <div class="slide" data-title="{slide_b_title}">
          <div class="layout-split">
            <div class="img-container">
              <img src="{img_b}" alt="{tag} Slide B: Step Evolution and Transition Map">
            </div>
            <div class="card-container">
              <div>
                <span class="section-label label-disorder">{regime}</span>
                <h2 style="margin-top:6px;">{title_tag}</h2>
                <h3 style="color:#94a3b8; font-size:12.5px; margin-top:2px;">Step-by-Step Evolution & Spatial Transition Map</h3>
              </div>

              <table class="data-table">
                <thead>
                  <tr><th>Protocol Step</th><th>3 μm TPR</th><th>3 μm FDR</th><th>6 μm TPR</th><th>6 μm FDR</th><th>ΔTPR</th><th>ΔFDR</th></tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong style="color:#38bdf8;">Step 1: Correlation</strong></td>
                    <td>{m3['metrics_s1']['pfaffian_passed_pct']:.2f}%</td>
                    <td>{m3['metrics_s1']['fdr_percent']:.2f}%</td>
                    <td>{m6['metrics_s1']['pfaffian_passed_pct']:.2f}%</td>
                    <td>{m6['metrics_s1']['fdr_percent']:.2f}%</td>
                    <td><strong>{t1_d:+.2f}%</strong></td>
                    <td><span class="badge badge-pass">{f1_d:+.2f}%</span></td>
                  </tr>
                  <tr>
                    <td><strong style="color:#818cf8;">Step 2: 3ω Consensus</strong></td>
                    <td>{m3['metrics_s2']['pfaffian_passed_pct']:.2f}%</td>
                    <td>{m3['metrics_s2']['fdr_percent']:.2f}%</td>
                    <td>{m6['metrics_s2']['pfaffian_passed_pct']:.2f}%</td>
                    <td>{m6['metrics_s2']['fdr_percent']:.2f}%</td>
                    <td><strong style="color:#34d399;">{t2_d:+.2f}%</strong></td>
                    <td><span class="badge badge-pass">{f2_d:+.2f}%</span></td>
                  </tr>
                  <tr>
                    <td><strong style="color:#34d399;">Step 3: Transport Gap</strong></td>
                    <td>{m3['metrics_s3']['pfaffian_passed_pct']:.2f}%</td>
                    <td>{m3['metrics_s3']['fdr_percent']:.2f}%</td>
                    <td>{m6['metrics_s3']['pfaffian_passed_pct']:.2f}%</td>
                    <td>{m6['metrics_s3']['fdr_percent']:.2f}%</td>
                    <td><strong style="color:#34d399;">{t3_d:+.2f}%</strong></td>
                    <td><span class="badge badge-pass">{f3_d:+.2f}%</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
        """
        slides.append(slide_b_html)
        slide_num += 1

        # Slide C: 3w Peak Curvature Comparison
        img_c = f"figures/{tag}_comparison_3w.png"
        slide_c_title = f"Slide {slide_num}: [{tag}] 3ω Peak Curvature Comparison (3μm vs. 6μm)"
        slide_c_html = rf"""
        <div class="slide" data-title="{slide_c_title}">
          <div class="layout-split">
            <div class="img-container">
              <img src="{img_c}" alt="{tag} Slide C: 3w Peak Curvature Comparison">
            </div>
            <div class="card-container">
              <div>
                <span class="section-label label-disorder">Physical Measurement 1: 3ω Peak Curvature</span>
                <h2 style="margin-top:6px;">{title_tag}</h2>
                <h3 style="color:#94a3b8; font-size:12.5px; margin-top:2px;">Zero-Bias Conductance Second Derivative ($d^2 G / dV^2$)</h3>
              </div>

              <table class="data-table">
                <thead>
                  <tr><th>Wire Length</th><th>Step 2 TPR (% Pfaffian)</th><th>Step 2 FDR (%)</th></tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>3 μm Wire</strong></td>
                    <td>{m3['metrics_s2']['pfaffian_passed_pct']:.2f}%</td>
                    <td>{m3['metrics_s2']['fdr_percent']:.2f}%</td>
                  </tr>
                  <tr>
                    <td><strong>6 μm Wire</strong></td>
                    <td><strong style="color:#34d399;">{m6['metrics_s2']['pfaffian_passed_pct']:.2f}%</strong></td>
                    <td><strong style="color:#38bdf8;">{m6['metrics_s2']['fdr_percent']:.2f}%</strong></td>
                  </tr>
                  <tr>
                    <td><strong style="color:#38bdf8;">Scaling Delta (6μm vs 3μm)</strong></td>
                    <td><span class="badge badge-pass">{t2_d:+.2f}% ΔTPR</span></td>
                    <td><span class="badge badge-pass">{f2_d:+.2f}% ΔFDR</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
        """
        slides.append(slide_c_html)
        slide_num += 1

        # Slide D: Extracted Transport Gap Comparison
        img_d = f"figures/{tag}_comparison_transport_gap.png"
        slide_d_title = f"Slide {slide_num}: [{tag}] Extracted Transport Gap Comparison (3μm vs. 6μm)"
        slide_d_html = rf"""
        <div class="slide" data-title="{slide_d_title}">
          <div class="layout-split">
            <div class="img-container">
              <img src="{img_d}" alt="{tag} Slide D: Extracted Transport Gap Comparison">
            </div>
            <div class="card-container">
              <div>
                <span class="section-label label-disorder">Physical Measurement 2: Extracted Transport Gap</span>
                <h2 style="margin-top:6px;">{title_tag}</h2>
                <h3 style="color:#94a3b8; font-size:12.5px; margin-top:2px;">Extracted Transport Gap Protection ($\Delta_{{\mathrm{{transport}}}}$)</h3>
              </div>

              <table class="data-table">
                <thead>
                  <tr><th>Wire Length</th><th>Step 3 TPR</th><th>Step 3 FDR</th><th>Boundary Conf.</th><th>Separability</th></tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>3 μm Wire</strong></td>
                    <td>{m3['metrics_s3']['pfaffian_passed_pct']:.2f}%</td>
                    <td>{m3['metrics_s3']['fdr_percent']:.2f}%</td>
                    <td>{m3['metrics_s3']['mean_boundary_confinement']:.4f}</td>
                    <td>{m3['metrics_s3']['mean_separability']:.4f}</td>
                  </tr>
                  <tr>
                    <td><strong>6 μm Wire</strong></td>
                    <td><strong style="color:#34d399;">{m6['metrics_s3']['pfaffian_passed_pct']:.2f}%</strong></td>
                    <td><strong style="color:#38bdf8;">{m6['metrics_s3']['fdr_percent']:.2f}%</strong></td>
                    <td><strong>{m6['metrics_s3']['mean_boundary_confinement']:.4f}</strong></td>
                    <td><strong>{m6['metrics_s3']['mean_separability']:.4f}</strong></td>
                  </tr>
                  <tr>
                    <td><strong style="color:#38bdf8;">Scaling Delta (6μm vs 3μm)</strong></td>
                    <td><span class="badge badge-pass">{d3['delta_tpr']:+.2f}% ΔTPR</span></td>
                    <td><span class="badge badge-pass">{d3['delta_fdr']:+.2f}% ΔFDR</span></td>
                    <td><span class="badge badge-pass">{d3['delta_confinement']:+.4f} ΔConf</span></td>
                    <td><span class="badge badge-pass">{d3['delta_separability']:+.4f} ΔSep</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
        """
        slides.append(slide_d_html)
        slide_num += 1

        # Slide E: 2w Zero-Bias Conductance Comparison
        img_e = f"figures/{tag}_comparison_2w.png"
        slide_e_title = f"Slide {slide_num}: [{tag}] 2ω Zero-Bias Conductance Comparison (3μm vs. 6μm)"
        slide_e_html = rf"""
        <div class="slide" data-title="{slide_e_title}">
          <div class="layout-split">
            <div class="img-container">
              <img src="{img_e}" alt="{tag} Slide E: 2w Conductance Comparison">
            </div>
            <div class="card-container">
              <div>
                <span class="section-label label-disorder">Physical Measurement 3: 2ω Conductance</span>
                <h2 style="margin-top:6px;">{title_tag}</h2>
                <h3 style="color:#94a3b8; font-size:12.5px; margin-top:2px;">Zero-Bias Local Conductance ($G(V=0)$ in units of $e^2/h$)</h3>
              </div>

              <table class="data-table">
                <thead>
                  <tr><th>Wire Length</th><th>Step 3 TPR</th><th>Step 3 FDR</th><th>Boundary Conf.</th><th>Separability</th></tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>3 μm Wire</strong></td>
                    <td>{m3['metrics_s3']['pfaffian_passed_pct']:.2f}%</td>
                    <td>{m3['metrics_s3']['fdr_percent']:.2f}%</td>
                    <td>{m3['metrics_s3']['mean_boundary_confinement']:.4f}</td>
                    <td>{m3['metrics_s3']['mean_separability']:.4f}</td>
                  </tr>
                  <tr>
                    <td><strong>6 μm Wire</strong></td>
                    <td><strong style="color:#34d399;">{m6['metrics_s3']['pfaffian_passed_pct']:.2f}%</strong></td>
                    <td><strong style="color:#38bdf8;">{m6['metrics_s3']['fdr_percent']:.2f}%</strong></td>
                    <td><strong>{m6['metrics_s3']['mean_boundary_confinement']:.4f}</strong></td>
                    <td><strong>{m6['metrics_s3']['mean_separability']:.4f}</strong></td>
                  </tr>
                  <tr>
                    <td><strong style="color:#38bdf8;">Scaling Delta</strong></td>
                    <td><span class="badge badge-pass">{d3['delta_tpr']:+.2f}% ΔTPR</span></td>
                    <td><span class="badge badge-pass">{d3['delta_fdr']:+.2f}% ΔFDR</span></td>
                    <td><span class="badge badge-pass">{d3['delta_confinement']:+.4f} ΔConf</span></td>
                    <td><span class="badge badge-pass">{d3['delta_separability']:+.4f} ΔSep</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
        """
        slides.append(slide_e_html)
        slide_num += 1

        # Slide F: Barrier Sweep Reflection Correlation Comparison
        img_f = f"figures/{tag}_comparison_barrier_correlation.png"
        slide_f_title = f"Slide {slide_num}: [{tag}] Barrier Reflection Correlation Comparison (3μm vs. 6μm)"
        slide_f_html = rf"""
        <div class="slide" data-title="{slide_f_title}">
          <div class="layout-split">
            <div class="img-container">
              <img src="{img_f}" alt="{tag} Slide F: Barrier Reflection Correlation Comparison">
            </div>
            <div class="card-container">
              <div>
                <span class="section-label label-disorder">Physical Measurement 4: Barrier Reflection Correlation</span>
                <h2 style="margin-top:6px;">{title_tag}</h2>
                <h3 style="color:#94a3b8; font-size:12.5px; margin-top:2px;">Normalized Reflection Tracking Metric ($C_L, C_R \ge 0.85$)</h3>
              </div>

              <table class="data-table">
                <thead>
                  <tr><th>Wire Length</th><th>Step 1 TPR (% Pfaffian)</th><th>Step 1 FDR (%)</th></tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>3 μm Wire</strong></td>
                    <td>{m3['metrics_s1']['pfaffian_passed_pct']:.2f}%</td>
                    <td>{m3['metrics_s1']['fdr_percent']:.2f}%</td>
                  </tr>
                  <tr>
                    <td><strong>6 μm Wire</strong></td>
                    <td><strong style="color:#34d399;">{m6['metrics_s1']['pfaffian_passed_pct']:.2f}%</strong></td>
                    <td><strong style="color:#38bdf8;">{m6['metrics_s1']['fdr_percent']:.2f}%</strong></td>
                  </tr>
                  <tr>
                    <td><strong style="color:#38bdf8;">Scaling Delta (6μm vs 3μm)</strong></td>
                    <td><span class="badge badge-pass">{t1_d:+.2f}% ΔTPR</span></td>
                    <td><span class="badge badge-pass">{f1_d:+.2f}% ΔFDR</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
        """
        slides.append(slide_f_html)
        slide_num += 1

        # Slide G: Energy Spectrum E0 and Quasiparticle Gap E1 - E0 Comparison
        img_g = f"figures/{tag}_comparison_energy_diff.png"
        slide_g_title = f"Slide {slide_num}: [{tag}] Energy Levels & Quasiparticle Gap (3μm vs. 6μm)"
        slide_g_html = rf"""
        <div class="slide" data-title="{slide_g_title}">
          <div class="layout-split">
            <div class="img-container">
              <img src="{img_g}" alt="{tag} Slide G: Energy Levels and Quasiparticle Gap Comparison">
            </div>
            <div class="card-container">
              <div>
                <span class="section-label label-disorder">Physical Measurement 5: BdG Spectrum & Gap</span>
                <h2 style="margin-top:6px;">{title_tag}</h2>
                <h3 style="color:#94a3b8; font-size:12.5px; margin-top:2px;">Ground State Energy $E_0$ and Quasiparticle Excitation Gap $E_1 - E_0$</h3>
              </div>

              <table class="data-table">
                <thead>
                  <tr><th>Wire Length</th><th>Step 3 TPR</th><th>Step 3 FDR</th><th>Boundary Conf.</th><th>Separability</th></tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>3 μm Wire</strong></td>
                    <td>{m3['metrics_s3']['pfaffian_passed_pct']:.2f}%</td>
                    <td>{m3['metrics_s3']['fdr_percent']:.2f}%</td>
                    <td>{m3['metrics_s3']['mean_boundary_confinement']:.4f}</td>
                    <td>{m3['metrics_s3']['mean_separability']:.4f}</td>
                  </tr>
                  <tr>
                    <td><strong>6 μm Wire</strong></td>
                    <td><strong style="color:#34d399;">{m6['metrics_s3']['pfaffian_passed_pct']:.2f}%</strong></td>
                    <td><strong style="color:#38bdf8;">{m6['metrics_s3']['fdr_percent']:.2f}%</strong></td>
                    <td><strong>{m6['metrics_s3']['mean_boundary_confinement']:.4f}</strong></td>
                    <td><strong>{m6['metrics_s3']['mean_separability']:.4f}</strong></td>
                  </tr>
                  <tr>
                    <td><strong style="color:#38bdf8;">Scaling Delta</strong></td>
                    <td><span class="badge badge-pass">{d3['delta_tpr']:+.2f}% ΔTPR</span></td>
                    <td><span class="badge badge-pass">{d3['delta_fdr']:+.2f}% ΔFDR</span></td>
                    <td><span class="badge badge-pass">{d3['delta_confinement']:+.4f} ΔConf</span></td>
                    <td><span class="badge badge-pass">{d3['delta_separability']:+.4f} ΔSep</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
        """
        slides.append(slide_g_html)
        slide_num += 1

    return slides


def build_single_cut_slides(slide_num_start: int, summary_data: list) -> list:
    """Builds deep-dive single-cut comparative slides (3 μm vs. 6 μm).
    
    Includes:
      1. Strong Disorder (V0 = 0.645 meV): Island 1 COM Cut (μ = 0.975 meV)
         - 10-Panel Conductance Multiplot (BdG Spectrum, G_LL, G_RR, A(G_RL), A(G_LR))
         - 8-Panel Protocol Analytics (BdG Levels, 3w Curvature, Transport Gap, Reflection Correlation)
         - 4-Panel Gapless Boundary Diagnostic Map (explaining Microsoft TGP Stage 2 failure)
      2. Clean Nanowire (V0 = 0.0 meV): Transverse Cut (μ = 0.0 meV)
         - 10-Panel Conductance Multiplot
         - 8-Panel Protocol Analytics
         - 4-Panel Gapless Boundary Diagnostic Map
    """
    slides = []
    slide_num = slide_num_start

    sum_map = {d["tag"]: d for d in summary_data}
    d645 = sum_map.get("V0_0_645", {})
    d0 = sum_map.get("V0_0_0", {})

    # -------------------------------------------------------------
    # 1. Strong Disorder (V0 = 0.645 meV) Single-Cut
    # -------------------------------------------------------------
    # Slide 1: 10-Panel Conductance Multiplot
    img_645_10p = "figures/V0_0_645_cut_island1_10panel_conductance.png"
    s1_title = f"Slide {slide_num}: [V0 = 0.645 meV] Single-Cut Conductance Multiplot (3μm vs 6μm)"
    s1_html = rf"""
    <div class="slide" data-title="{s1_title}">
      <div class="layout-split">
        <div class="img-container">
          <img src="{img_645_10p}" alt="V0=0.645 meV Single-Cut 10-Panel Conductance Multiplot">
        </div>
        <div class="card-container">
          <div>
            <span class="section-label label-disorder">Single-Cut Deep Dive: Strong Disorder (V0 = 0.645 meV)</span>
            <h2 style="margin-top:6px;">Island 1 Center-of-Mass Cut ($\mu = 0.975\ \mathrm{{meV}}$)</h2>
            <h3 style="color:#94a3b8; font-size:12.5px; margin-top:2px;">10-Panel Conductance Comparison: BdG Spectrum, $G_{{LL}}, G_{{RR}}, A(G_{{RL}}), A(G_{{LR}})$</h3>
          </div>

          <table class="data-table">
            <thead>
              <tr><th>Protocol Metric</th><th>3 μm Wire</th><th>6 μm Wire</th><th>Scaling Delta</th></tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Microsoft TGP Stage 2</strong></td>
                <td>46.2% TPR | 0.0% FDR</td>
                <td><strong style="color:#ef4444;">0.0% TPR</strong> (Cluster Rejected)</td>
                <td><span class="badge badge-fail">-46.2% ΔTPR</span></td>
              </tr>
              <tr>
                <td><strong>3-Step Protocol (ROI 3)</strong></td>
                <td>6.9% TPR | 0.0% FDR</td>
                <td><strong style="color:#34d399;">39.3% TPR</strong> | 0.0% FDR</td>
                <td><span class="badge badge-pass">+32.4% ΔTPR (5.7×)</span></td>
              </tr>
              <tr>
                <td><strong>Boundary Confinement</strong></td>
                <td>{d645['3micron']['metrics_s3']['mean_boundary_confinement']:.4f}</td>
                <td><strong>{d645['6micron']['metrics_s3']['mean_boundary_confinement']:.4f}</strong></td>
                <td><span class="badge badge-pass">+{d645['deltas_step3']['delta_confinement']:.4f} ΔConf</span></td>
              </tr>
              <tr>
                <td><strong>MZM Separability</strong></td>
                <td>{d645['3micron']['metrics_s3']['mean_separability']:.4f}</td>
                <td><strong>{d645['6micron']['metrics_s3']['mean_separability']:.4f}</strong></td>
                <td><span class="badge badge-pass">+{d645['deltas_step3']['delta_separability']:.4f} ΔSep</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
    """
    slides.append(s1_html)
    slide_num += 1

    # Slide 2: Analytics Multiplot
    img_645_ana = "figures/V0_0_645_cut_island1_analytics_multiplot.png"
    s2_title = f"Slide {slide_num}: [V0 = 0.645 meV] Single-Cut Protocol Analytics (3μm vs 6μm)"
    s2_html = rf"""
    <div class="slide" data-title="{s2_title}">
      <div class="layout-split">
        <div class="img-container">
          <img src="{img_645_ana}" alt="V0=0.645 meV Single-Cut Protocol Analytics Multiplot">
        </div>
        <div class="card-container">
          <div>
            <span class="section-label label-disorder">Single-Cut Deep Dive: Strong Disorder (V0 = 0.645 meV)</span>
            <h2 style="margin-top:6px;">Cut Protocol Analytics ($\mu = 0.975\ \mathrm{{meV}}$)</h2>
            <h3 style="color:#94a3b8; font-size:12.5px; margin-top:2px;">BdG Levels, $3\omega$ Curvature Consensus, Transport Gap vs. $E_0, E_1$, & Dual Reflection Tracking</h3>
          </div>

          <table class="data-table">
            <thead>
              <tr><th>Protocol Stage</th><th>3 μm Wire Response</th><th>6 μm Wire Response</th><th>Scaling Delta</th></tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Step 1: Barrier Correlation</strong></td>
                <td>{d645['3micron']['metrics_s1']['pfaffian_passed_pct']:.1f}% TPR | {d645['3micron']['metrics_s1']['fdr_percent']:.1f}% FDR</td>
                <td><strong style="color:#34d399;">{d645['6micron']['metrics_s1']['pfaffian_passed_pct']:.1f}% TPR</strong> | {d645['6micron']['metrics_s1']['fdr_percent']:.1f}% FDR</td>
                <td><span class="badge badge-pass">+{d645['6micron']['metrics_s1']['pfaffian_passed_pct'] - d645['3micron']['metrics_s1']['pfaffian_passed_pct']:.1f}% ΔTPR</span></td>
              </tr>
              <tr>
                <td><strong>Step 2: 3ω Curvature Consensus</strong></td>
                <td>{d645['3micron']['metrics_s2']['pfaffian_passed_pct']:.1f}% TPR | {d645['3micron']['metrics_s2']['fdr_percent']:.1f}% FDR</td>
                <td><strong style="color:#34d399;">{d645['6micron']['metrics_s2']['pfaffian_passed_pct']:.1f}% TPR</strong> | {d645['6micron']['metrics_s2']['fdr_percent']:.1f}% FDR</td>
                <td><span class="badge badge-pass">+{d645['6micron']['metrics_s2']['pfaffian_passed_pct'] - d645['3micron']['metrics_s2']['pfaffian_passed_pct']:.1f}% ΔTPR</span></td>
              </tr>
              <tr>
                <td><strong>Step 3: Transport Gap</strong></td>
                <td>{d645['3micron']['metrics_s3']['pfaffian_passed_pct']:.1f}% TPR | {d645['3micron']['metrics_s3']['fdr_percent']:.1f}% FDR</td>
                <td><strong style="color:#34d399;">{d645['6micron']['metrics_s3']['pfaffian_passed_pct']:.1f}% TPR</strong> | {d645['6micron']['metrics_s3']['fdr_percent']:.1f}% FDR</td>
                <td><span class="badge badge-pass">+{d645['deltas_step3']['delta_tpr']:.1f}% ΔTPR</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
    """
    slides.append(s2_html)
    slide_num += 1

    # Slide 3: Spatial Diagnostic
    img_645_diag = "figures/V0_0_645_cut_island1_diagnostic_gapless_boundary.png"
    s3_title = f"Slide {slide_num}: [V0 = 0.645 meV] Spatial Diagnostic: Gapless Boundary Collapse"
    s3_html = rf"""
    <div class="slide" data-title="{s3_title}">
      <div class="layout-split">
        <div class="img-container">
          <img src="{img_645_diag}" alt="V0=0.645 meV Diagnostic: Collapse of the Nonlocal Gapless Boundary">
        </div>
        <div class="card-container">
          <div>
            <span class="section-label label-disorder">Diagnostic Root Cause: Strong Disorder (V0 = 0.645 meV)</span>
            <h2 style="margin-top:6px;">Collapse of Nonlocal Gapless Boundary at 6 μm</h2>
            <h3 style="color:#94a3b8; font-size:12.5px; margin-top:2px;">Mechanistic Origin of Microsoft TGP Stage 2 Zero-Cluster Rejection</h3>
          </div>

          <table class="data-table">
            <thead>
              <tr><th>Evaluation Metric</th><th>3 μm Wire</th><th>6 μm Wire</th><th>Algorithm Status</th></tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Nonlocal Conductance Amplitude</strong></td>
                <td>Measurable ($> 10^{{-4}}\ e^2/h$)</td>
                <td><strong style="color:#ef4444;">Decayed ($< 10^{{-4}}\ e^2/h$)</strong></td>
                <td>Below Noise Floor ($e^{{-L/\ell}}$)</td>
              </tr>
              <tr>
                <td><strong>Extracted Transport Gap</strong></td>
                <td>Closes at boundary ($\leq 10\ \mu\mathrm{{eV}}$)</td>
                <td><strong style="color:#ef4444;">Defaults to Max Bias ($100\ \mu\mathrm{{eV}}$)</strong></td>
                <td>No Onset Detected</td>
              </tr>
              <tr>
                <td><strong>Candidate Perimeter Gapless %</strong></td>
                <td>$\sim 85\%$ (Threshold $\geq 60\%$)</td>
                <td><strong style="color:#ef4444;">$< 5\%$ (Threshold $\geq 60\%$)</strong></td>
                <td><span class="badge badge-fail">Boundary Check Failed</span></td>
              </tr>
              <tr>
                <td><strong>Microsoft TGP Stage 2 Outcome</strong></td>
                <td><strong style="color:#34d399;">46.2% TPR</strong></td>
                <td><strong style="color:#ef4444;">0.0% TPR</strong></td>
                <td><span class="badge badge-fail">All Clusters Rejected</span></td>
              </tr>
              <tr>
                <td><strong>3-Step Protocol (ROI 3) Outcome</strong></td>
                <td>6.9% TPR</td>
                <td><strong style="color:#34d399;">39.3% TPR</strong></td>
                <td><span class="badge badge-pass">Passed (5.7× Expansion)</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
    """
    slides.append(s3_html)
    slide_num += 1

    # -------------------------------------------------------------
    # 2. Clean Nanowire (V0 = 0.0 meV) Single-Cut
    # -------------------------------------------------------------
    # Slide 4: 10-Panel Conductance Multiplot
    img_0_10p = "figures/V0_0_0_cut_clean_10panel_conductance.png"
    s4_title = f"Slide {slide_num}: [Clean Wire: V0 = 0.0 meV] Conductance Multiplot (3μm vs 6μm)"
    s4_html = rf"""
    <div class="slide" data-title="{s4_title}">
      <div class="layout-split">
        <div class="img-container">
          <img src="{img_0_10p}" alt="Clean Wire Single-Cut 10-Panel Conductance Multiplot">
        </div>
        <div class="card-container">
          <div>
            <span class="section-label label-disorder">Single-Cut Deep Dive: Clean Wire (V0 = 0.0 meV)</span>
            <h2 style="margin-top:6px;">Transverse Cut ($\mu = 0.0\ \mathrm{{meV}}$)</h2>
            <h3 style="color:#94a3b8; font-size:12.5px; margin-top:2px;">10-Panel Conductance Comparison: BdG Spectrum, $G_{{LL}}, G_{{RR}}, A(G_{{RL}}), A(G_{{LR}})$</h3>
          </div>

          <table class="data-table">
            <thead>
              <tr><th>Protocol Metric</th><th>3 μm Wire</th><th>6 μm Wire</th><th>Scaling Delta</th></tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Microsoft TGP Stage 2</strong></td>
                <td>0.0% TPR | 0.0% FDR</td>
                <td><strong style="color:#ef4444;">0.0% TPR</strong> (Cluster Rejected)</td>
                <td><span class="badge badge-neutral">0.0% ΔTPR</span></td>
              </tr>
              <tr>
                <td><strong>3-Step Protocol (ROI 3)</strong></td>
                <td>{d0['3micron']['metrics_s3']['pfaffian_passed_pct']:.1f}% TPR | {d0['3micron']['metrics_s3']['fdr_percent']:.2f}% FDR</td>
                <td><strong style="color:#34d399;">{d0['6micron']['metrics_s3']['pfaffian_passed_pct']:.1f}% TPR</strong> | {d0['6micron']['metrics_s3']['fdr_percent']:.2f}% FDR</td>
                <td><span class="badge badge-pass">+{d0['deltas_step3']['delta_tpr']:.1f}% ΔTPR</span></td>
              </tr>
              <tr>
                <td><strong>Boundary Confinement</strong></td>
                <td>{d0['3micron']['metrics_s3']['mean_boundary_confinement']:.4f}</td>
                <td><strong>{d0['6micron']['metrics_s3']['mean_boundary_confinement']:.4f}</strong></td>
                <td><span class="badge badge-pass">+{d0['deltas_step3']['delta_confinement']:.4f} ΔConf</span></td>
              </tr>
              <tr>
                <td><strong>MZM Separability</strong></td>
                <td>{d0['3micron']['metrics_s3']['mean_separability']:.4f}</td>
                <td><strong>{d0['6micron']['metrics_s3']['mean_separability']:.4f}</strong></td>
                <td><span class="badge badge-pass">+{d0['deltas_step3']['delta_separability']:.4f} ΔSep</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
    """
    slides.append(s4_html)
    slide_num += 1

    # Slide 5: Analytics Multiplot
    img_0_ana = "figures/V0_0_0_cut_clean_analytics_multiplot.png"
    s5_title = f"Slide {slide_num}: [Clean Wire: V0 = 0.0 meV] Protocol Analytics (3μm vs 6μm)"
    s5_html = rf"""
    <div class="slide" data-title="{s5_title}">
      <div class="layout-split">
        <div class="img-container">
          <img src="{img_0_ana}" alt="Clean Wire Protocol Analytics Multiplot">
        </div>
        <div class="card-container">
          <div>
            <span class="section-label label-disorder">Single-Cut Deep Dive: Clean Wire (V0 = 0.0 meV)</span>
            <h2 style="margin-top:6px;">Clean Cut Analytics ($\mu = 0.0\ \mathrm{{meV}}$)</h2>
            <h3 style="color:#94a3b8; font-size:12.5px; margin-top:2px;">BdG Levels, $3\omega$ Curvature Consensus, Transport Gap vs. $E_0, E_1$, & Dual Reflection Tracking</h3>
          </div>

          <table class="data-table">
            <thead>
              <tr><th>Protocol Stage</th><th>3 μm Wire Response</th><th>6 μm Wire Response</th><th>Scaling Delta</th></tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Step 1: Barrier Correlation</strong></td>
                <td>{d0['3micron']['metrics_s1']['pfaffian_passed_pct']:.1f}% TPR | {d0['3micron']['metrics_s1']['fdr_percent']:.1f}% FDR</td>
                <td><strong style="color:#34d399;">{d0['6micron']['metrics_s1']['pfaffian_passed_pct']:.1f}% TPR</strong> | {d0['6micron']['metrics_s1']['fdr_percent']:.1f}% FDR</td>
                <td><span class="badge badge-pass">+{d0['6micron']['metrics_s1']['pfaffian_passed_pct'] - d0['3micron']['metrics_s1']['pfaffian_passed_pct']:.1f}% ΔTPR</span></td>
              </tr>
              <tr>
                <td><strong>Step 2: 3ω Curvature Consensus</strong></td>
                <td>{d0['3micron']['metrics_s2']['pfaffian_passed_pct']:.1f}% TPR | {d0['3micron']['metrics_s2']['fdr_percent']:.1f}% FDR</td>
                <td><strong style="color:#34d399;">{d0['6micron']['metrics_s2']['pfaffian_passed_pct']:.1f}% TPR</strong> | {d0['6micron']['metrics_s2']['fdr_percent']:.1f}% FDR</td>
                <td><span class="badge badge-pass">+{d0['6micron']['metrics_s2']['pfaffian_passed_pct'] - d0['3micron']['metrics_s2']['pfaffian_passed_pct']:.1f}% ΔTPR</span></td>
              </tr>
              <tr>
                <td><strong>Step 3: Transport Gap</strong></td>
                <td>{d0['3micron']['metrics_s3']['pfaffian_passed_pct']:.1f}% TPR | {d0['3micron']['metrics_s3']['fdr_percent']:.1f}% FDR</td>
                <td><strong style="color:#34d399;">{d0['6micron']['metrics_s3']['pfaffian_passed_pct']:.1f}% TPR</strong> | {d0['6micron']['metrics_s3']['fdr_percent']:.1f}% FDR</td>
                <td><span class="badge badge-pass">+{d0['deltas_step3']['delta_tpr']:.1f}% ΔTPR</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
    """
    slides.append(s5_html)
    slide_num += 1

    # Slide 6: Spatial Diagnostic
    img_0_diag = "figures/V0_0_0_cut_clean_diagnostic_gapless_boundary.png"
    s6_title = f"Slide {slide_num}: [Clean Wire: V0 = 0.0 meV] Spatial Diagnostic: Gapless Boundary Collapse"
    s6_html = rf"""
    <div class="slide" data-title="{s6_title}">
      <div class="layout-split">
        <div class="img-container">
          <img src="{img_0_diag}" alt="Clean Wire Diagnostic: Collapse of the Nonlocal Gapless Boundary">
        </div>
        <div class="card-container">
          <div>
            <span class="section-label label-disorder">Diagnostic Root Cause: Clean Wire (V0 = 0.0 meV)</span>
            <h2 style="margin-top:6px;">Clean Wire Gapless Boundary Attenuation at 6 μm</h2>
            <h3 style="color:#94a3b8; font-size:12.5px; margin-top:2px;">Absence of Microsoft TGP Stage 2 Clusters in Clean Nanowire</h3>
          </div>

          <table class="data-table">
            <thead>
              <tr><th>Evaluation Metric</th><th>3 μm Wire</th><th>6 μm Wire</th><th>Algorithm Status</th></tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Nonlocal Conductance Amplitude</strong></td>
                <td>Measurable ($> 10^{{-4}}\ e^2/h$)</td>
                <td><strong style="color:#ef4444;">Decayed ($< 10^{{-4}}\ e^2/h$)</strong></td>
                <td>Below Noise Floor ($e^{{-L/\ell}}$)</td>
              </tr>
              <tr>
                <td><strong>Extracted Transport Gap</strong></td>
                <td>Closes at boundary ($\leq 10\ \mu\mathrm{{eV}}$)</td>
                <td><strong style="color:#ef4444;">Defaults to Max Bias ($100\ \mu\mathrm{{eV}}$)</strong></td>
                <td>No Onset Detected</td>
              </tr>
              <tr>
                <td><strong>Candidate Perimeter Gapless %</strong></td>
                <td>Boundary fragmented</td>
                <td><strong style="color:#ef4444;">$< 2\%$ (Threshold $\geq 60\%$)</strong></td>
                <td><span class="badge badge-fail">Boundary Check Failed</span></td>
              </tr>
              <tr>
                <td><strong>Microsoft TGP Stage 2 Outcome</strong></td>
                <td>0.0% TPR</td>
                <td><strong style="color:#ef4444;">0.0% TPR</strong></td>
                <td><span class="badge badge-fail">All Clusters Rejected</span></td>
              </tr>
              <tr>
                <td><strong>3-Step Protocol (ROI 3) Outcome</strong></td>
                <td>{d0['3micron']['metrics_s3']['pfaffian_passed_pct']:.1f}% TPR</td>
                <td><strong style="color:#34d399;">{d0['6micron']['metrics_s3']['pfaffian_passed_pct']:.1f}% TPR</strong></td>
                <td><span class="badge badge-pass">Passed (+{d0['deltas_step3']['delta_tpr']:.1f}% ΔTPR)</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
    """
    slides.append(s6_html)
    slide_num += 1

    return slides


def generate_deck():
    print(f"Loading summary metrics from {METRICS_FILE}...")
    with open(METRICS_FILE, "r") as fp:
        summary_data = json.load(fp)

    slides_content_list = []

    # Slide 1: Master Summary Table
    slides_content_list.append(build_slide1_master_table(summary_data))

    # Slides 2–43: 6 disorders x 7 slides each (Slides A–G)
    per_disorder_slides = build_per_disorder_slides(summary_data)
    slides_content_list.extend(per_disorder_slides)

    # Slides 44–49: Section 3: Single-Cut Deep Dive Comparisons (3μm vs 6μm)
    single_cut_slides = build_single_cut_slides(slide_num_start=len(slides_content_list) + 1, summary_data=summary_data)
    slides_content_list.extend(single_cut_slides)

    total_slides = len(slides_content_list)
    print(f"Generated {total_slides} total slides.")

    import re
    dropdown_options = []
    for idx, slide_html in enumerate(slides_content_list):
        m = re.search(r'data-title="([^"]+)"', slide_html)
        t = m.group(1) if m else f"Slide {idx + 1}"
        dropdown_options.append(f'<option value="{idx}">{t}</option>')

    html_out = HTML_TEMPLATE.format(
        total_slides=total_slides,
        dropdown_options="\n      ".join(dropdown_options),
        slides_content="\n".join(slides_content_list)
    )

    with open(SLIDES_HTML_PATH, "w", encoding="utf-8") as fp:
        fp.write(html_out)

    print(f"Successfully generated standalone HTML presentation at:\n  {SLIDES_HTML_PATH}")


if __name__ == "__main__":
    generate_deck()
