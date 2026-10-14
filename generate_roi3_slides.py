#!/usr/bin/env python3
"""
Generate an interactive, responsive HTML slide deck exclusively for ROI 3 Cuts.
Contains plots and analytics from Outputs/Plots/<dataset>_Plots/ROI3_Cuts/:
  - Section 1: Master Statistics Table
      * Slide 1: Ensemble-Averaged Post-Processing Verification Statistics (V0 = 0.645, N = 8 realizations)
  - Section 2: Regular TGP vs. 3-Step Protocol Comparison (Grouped consecutively across 6 active disorders)
      * Slides 2–7: 6 consecutive slides (V0 = 0.0 to 0.91) featuring:
          - Left: 4-Panel Phase Map Comparison showing FULL passing color both inside and outside Pfaffian boundary (Q = -1)
          - Right: Compact 2-Metric Table (% Pfaffian Area Passed & False Discovery Rate)
  - Section 3: Per-Disorder ROI 3 Island Cuts & Deep-Dives
      * ROI 3 Stage 2 Phase Map with Island COMs (★) + Verification Summary Table
      * Top 3 ROI 3 Islands:
          - Overview: ROI 3 Phase Map & Standalone Spectrum vs 5-Panel Conductance Multiplot
          - Analytics: ROI 3 Phase Map & Standalone Spectrum vs 4-Panel Analytics Multiplot
          - COM Point Deep-Dive: Conductance Multiplot vs 3 Stacked Deep-Dive Plots (dI/dV, Wavefunctions, Barrier Asymmetry)
  - Section 4: Appendix (At the very back)
      * Cross-disorder bias voltage gap threshold sweep evolution (2.0, 10.0, 30.0 μeV)
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
import json

from generate_html_slides import HTML_TEMPLATE, render_html_presentation

BASE_DIR = Path(__file__).resolve().parent

DISORDER_ORDER = [
    ("Tdis_pfaff5_V0_0_0_Plots", "V0 = 0.0 (Clean Wire)"),
    ("Tdis_pfaff5_V0_0_1_Plots", "V0 = 0.1 (Low Disorder)"),
    ("Tdis_pfaff5_V0_0_378_Plots", "V0 = 0.378 (Intermediate Disorder)"),
    ("Tdis_pfaff5_V0_0_645_Plots", "V0 = 0.645 (Strong Disorder)"),
    ("Tdis_pfaff5_V0_0_872_Plots", "V0 = 0.872 (Strong Disorder)"),
    ("Tdis_pfaff5_V0_0_91_Plots", "V0 = 0.91 (Strong Disorder)"),
    ("Tdis_pfaff5_Plots", "V0 = 1.2 (Benchmark Disorder)"),
]

def build_roi3_slides_for_directory(plot_dir: Path, base_rel_path: Path = None):
    """
    Build slides specifically for the ROI3_Cuts folder of a dataset directory.
    Guarantees exactly ONE unique point per cut (the Center of Mass of the island).
    Omit cluttering bias threshold sweep grids.
    """
    plot_dir = Path(plot_dir).resolve()
    rel_root = Path(base_rel_path).resolve() if base_rel_path else plot_dir
    dir_name = plot_dir.name
    dataset_label = dir_name.replace("_Plots", "")

    roi3_dir = plot_dir / "ROI3_Cuts"
    if not roi3_dir.exists():
        return []

    slides = []

    def get_rel(p: Path):
        try:
            return os.path.relpath(p, rel_root)
        except Exception:
            return str(p)

    p_stage2_star = roi3_dir / "ROI3_stage2_phase_map.png"
    if not p_stage2_star.exists():
        p_stage2_star = roi3_dir / "stage2_phase_map.png"

    p_fdr = roi3_dir / "ROI3_fdr_metrics.json"
    if not p_fdr.exists():
        p_fdr = roi3_dir / "fdr_metrics.json"

    # In standalone per-disorder presentations (when base_rel_path is None),
    # include the 4-panel comparison slide at the beginning.
    if base_rel_path is None:
        p_comp_img = roi3_dir / "tgp_vs_3step_protocol_comparison.png"
        if not p_comp_img.exists():
            p_comp_img = plot_dir / "tgp_vs_3step_protocol_comparison.png"
        p_comp_json = roi3_dir / "tgp_vs_3step_metrics.json"
        if not p_comp_json.exists():
            p_comp_json = plot_dir / "tgp_vs_3step_metrics.json"

        if p_comp_img.exists() and p_comp_json.exists():
            try:
                with open(p_comp_json, "r", encoding="utf-8") as f:
                    c_data = json.load(f)
                reg = c_data.get("regular_tgp", {})
                s1 = c_data.get("step1_correlation", {})
                s2 = c_data.get("step2_corr_plus_3w", {})
                s3 = c_data.get("step3_full_roi3", {})

                def get_b(v):
                    return "badge-pass" if v < 5.0 else ("badge-warn" if v < 20.0 else "badge-fail")

                comp_table_html = f"""
                <div style="background:#0f172a; padding:18px 22px; border-radius:8px; border:1px solid #1e293b; height:100%; display:flex; flex-direction:column; justify-content:center; box-sizing:border-box;">
                  <h3 style="color:#38bdf8; font-size:16px; margin-bottom:8px; font-weight:700;">Protocol Verification: Regular TGP vs. 3-Step Protocol</h3>
                  <p style="color:#94a3b8; font-size:12px; margin-bottom:14px; line-height:1.4;">
                    Benchmarking protocol passing maps against ground truth Pfaffian invariant ($\mathcal{{Q}} = -1$, dashed cyan boundary). Full passing regions are shown in color both inside and outside the Pfaffian domain, visually displaying False Positives in the trivial bulk alongside True Positives.
                  </p>
                  <table class="data-table" style="font-size:12.5px; margin-bottom:14px;">
                    <thead>
                      <tr>
                        <th style="padding:8px 10px;">Protocol Layer</th>
                        <th style="padding:8px 10px; text-align:center;">% Pfaffian Area Passed</th>
                        <th style="padding:8px 10px; text-align:center;">False Discovery Rate (FDR)</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td><span style="color:#a855f7; font-weight:bold;">Regular TGP (Stage 2 ROI 2)</span></td>
                        <td style="text-align:center;"><strong>{reg.get('pfaffian_passed_pct', 0.0):.2f}%</strong></td>
                        <td style="text-align:center;"><span class="{get_b(reg.get('fdr_percent', 0.0))}">{reg.get('fdr_percent', 0.0):.2f}%</span></td>
                      </tr>
                      <tr>
                        <td><span style="color:#0284c7; font-weight:bold;">Step 1: Barrier Corr Alone</span></td>
                        <td style="text-align:center;"><strong>{s1.get('pfaffian_passed_pct', 0.0):.2f}%</strong></td>
                        <td style="text-align:center;"><span class="{get_b(s1.get('fdr_percent', 0.0))}">{s1.get('fdr_percent', 0.0):.2f}%</span></td>
                      </tr>
                      <tr>
                        <td><span style="color:#f59e0b; font-weight:bold;">Step 2: Corr + 3ω Agreement</span></td>
                        <td style="text-align:center;"><strong>{s2.get('pfaffian_passed_pct', 0.0):.2f}%</strong></td>
                        <td style="text-align:center;"><span class="{get_b(s2.get('fdr_percent', 0.0))}">{s2.get('fdr_percent', 0.0):.2f}%</span></td>
                      </tr>
                      <tr>
                        <td><span style="color:#10b981; font-weight:bold;">Step 3: Full ROI 3 (+ Transport Gap)</span></td>
                        <td style="text-align:center;"><strong>{s3.get('pfaffian_passed_pct', 0.0):.2f}%</strong></td>
                        <td style="text-align:center;"><span class="{get_b(s3.get('fdr_percent', 0.0))}">{s3.get('fdr_percent', 0.0):.2f}%</span></td>
                      </tr>
                    </tbody>
                  </table>
                  <div style="background:#1e293b; padding:10px 14px; border-radius:6px; font-size:11px; line-height:1.55; color:#cbd5e1;">
                    <strong>Protocol Architecture:</strong><br>
                    • <strong>Regular TGP:</strong> Microsoft Stage 2 ROI 2 without barrier reflection correlation.<br>
                    • <strong>Step 1:</strong> Isolates regions of high dual nonlocal barrier reflection correlation ($C_L, C_R \ge 0.85$).<br>
                    • <strong>Step 2:</strong> Enforces ZBP curvature agreement ($I_{{3\omega}} < -100$) across $\ge 50\%$ cutter pairs, eliminating bulk gapless crossings.<br>
                    • <strong>Step 3:</strong> Demands extracted transport gap $\Delta_{{\mathrm{{transport}}}} > 10\ \mu\mathrm{{eV}}$, suppressing trivial sub-gap crossings and driving FDR to a minimum.<br>
                    • <em>Visual representation:</em> Colored regions inside the dashed cyan line represent True Positives (% Pfaffian Area Passed); colored regions outside represent False Positives (FDR).
                  </div>
                </div>
                """
                slides.append({
                    "dataset": dataset_label,
                    "title": f"[{dataset_label}] Protocol Comparison: Regular TGP vs. 3-Step Protocol",
                    "type": "side_by_side_html",
                    "img": get_rel(p_comp_img),
                    "html_card": comp_table_html,
                })
            except Exception:
                pass

    # Slide 1: ROI 3 Stage 2 Phase Map with Island COMs + Statistical Summary Table
    if p_stage2_star.exists():
        fdr_table_html = ""
        if p_fdr.exists():
            try:
                with open(p_fdr, "r", encoding="utf-8") as f:
                    fdr_data = json.load(f)
                s2_m = fdr_data.get("stage2_roi2_metrics", {})
                s3_m = fdr_data.get("stage3_roi3_metrics", {})
                islands = fdr_data.get("top3_islands", [])
                used_fb = fdr_data.get("used_candidate_orange_fallback", False)

                s2_ppv = s2_m.get("p_topological_given_passed_ppv", 0.0) * 100.0
                s2_fdr = s2_m.get("false_discovery_rate_fdr", 0.0) * 100.0
                s3_ppv = s3_m.get("p_topological_given_passed_ppv", 0.0) * 100.0
                s3_fdr = s3_m.get("false_discovery_rate_fdr", 0.0) * 100.0
                s3_tpr = s3_m.get("sensitivity_tpr", 0.0) * 100.0
                s3_tnr = s3_m.get("specificity_tnr", 0.0) * 100.0

                badge_cls_3 = "badge-pass" if s3_fdr < 10.0 else ("badge-warn" if s3_fdr < 25.0 else "badge-fail")
                badge_cls_2 = "badge-pass" if s2_fdr < 10.0 else ("badge-warn" if s2_fdr < 35.0 else "badge-fail")

                islands_html = "".join([
                    f"<tr><td>Rank {isl.get('rank')}</td><td><span style='color:{isl.get('color')}; font-weight:bold;'>★ {isl.get('star_name')}</span></td><td>{isl.get('size_pixels')} px</td><td>Vz = {isl.get('vz_com', 0):.3f} meV</td><td>μ = {isl.get('mu_com', 0):.3f} meV</td></tr>"
                    for isl in islands
                ])

                fb_note = "<p style='color:#f59e0b; font-size:11px; margin-top:4px;'>* Note: ROI 2 passed 0 px due to 60% boundary requirement; ROI 3 formed from Candidate Orange & Corr >= 0.85.</p>" if used_fb else ""

                fdr_table_html = f"""
                <div style="background:#0f172a; padding:16px; border-radius:8px; border:1px solid #1e293b; height:100%; display:flex; flex-direction:column; justify-content:center;">
                  <h3 style="color:#38bdf8; font-size:16px; margin-bottom:12px;">ROI 3 Protocol Verification (Dual Barrier Corr ≥ 0.85)</h3>
                  <div style="margin-bottom:14px; font-size:13px; line-height:1.6;">
                    <div style="background:#1e293b; padding:10px; border-radius:6px; margin-bottom:10px;">
                      <p><strong>Stage 2 (ROI 2 Baseline):</strong> {s2_m.get('total_passed', 0)} px | PPV: <span class="{badge_cls_2}">{s2_ppv:.2f}%</span> | FDR: <span class="{badge_cls_2}">{s2_fdr:.2f}%</span></p>
                      <p style="margin-top:4px;"><strong>Stage 3 (ROI 3 + Barrier Corr):</strong> {s3_m.get('total_passed', 0)} px (TP: {s3_m.get('true_positives', 0)}, FP: {s3_m.get('false_positives', 0)})</p>
                      <p><strong>ROI 3 Precision (PPV):</strong> <span class="{badge_cls_3}">{s3_ppv:.2f}%</span></p>
                      <p><strong>ROI 3 False Discovery Rate (FDR):</strong> <span class="{badge_cls_3}">{s3_fdr:.2f}%</span></p>
                      <p><strong>ROI 3 Sensitivity (TPR):</strong> {s3_tpr:.2f}% | <strong>Specificity (TNR):</strong> {s3_tnr:.2f}%</p>
                      {fb_note}
                    </div>
                  </div>
                  <h4 style="color:#f1f5f9; font-size:13px; margin-bottom:8px;">Top Validated ROI 3 Topological Islands</h4>
                  <table class="data-table">
                    <thead>
                      <tr><th>Rank</th><th>Star Identifier</th><th>Size</th><th>Zeeman COM (Vz)</th><th>Chemical Pot. COM (μ)</th></tr>
                    </thead>
                    <tbody>
                      {islands_html}
                    </tbody>
                  </table>
                </div>
                """
            except Exception:
                pass

        if fdr_table_html:
            slides.append({
                "dataset": dataset_label,
                "title": f"[{dataset_label}] ROI 3 Phase Map & False Discovery Rate Validation",
                "type": "side_by_side_html",
                "img": get_rel(p_stage2_star),
                "html_card": fdr_table_html
            })
        else:
            slides.append({
                "dataset": dataset_label,
                "title": f"[{dataset_label}] ROI 3 Phase Map & Top Island COMs",
                "type": "single_centered",
                "img": get_rel(p_stage2_star),
                "alt": "ROI 3 Phase Map with Island COMs",
            })

    # Ordered list of islands (Cyan, Magenta, Gold)
    island_dirs = sorted([d for d in roi3_dir.iterdir() if d.is_dir() and "Island_" in d.name])

    for island_folder in island_dirs:
        island_label = island_folder.name.replace("_", " ")
        p_cond = island_folder / "ROI3_multi_panel_conductance.png"
        if not p_cond.exists(): p_cond = island_folder / "multi_panel_conductance.png"

        p_ana = island_folder / "ROI3_multi_panel.png"
        if not p_ana.exists(): p_ana = island_folder / "multi_panel.png"

        p_spec = island_folder / "ROI3_spectra.png"
        if not p_spec.exists(): p_spec = island_folder / "spectra.png"

        rank_idx = "1" if "Island_1" in island_folder.name else ("2" if "Island_2" in island_folder.name else "3")
        parent_spec = roi3_dir / f"ROI3_tgp_cut_island_{rank_idx}_spectra.png"
        if not parent_spec.exists(): parent_spec = roi3_dir / f"tgp_cut_island_{rank_idx}_spectra.png"
        spec_img = parent_spec if parent_spec.exists() else p_spec

        # Slide A: Island Cut Overview (Conductance Multiplot)
        if p_cond.exists():
            slides.append({
                "dataset": dataset_label,
                "title": f"[{dataset_label}] {island_label} | Overview (Conductance)",
                "type": "cut_overview",
                "left_top": get_rel(p_stage2_star) if p_stage2_star.exists() else None,
                "left_bottom": get_rel(spec_img) if spec_img.exists() else None,
                "right": get_rel(p_cond)
            })

        # Slide B: Island Cut Analytics (Spectrum, 3w, Gap, Binarized Correlation Track)
        if p_ana.exists():
            slides.append({
                "dataset": dataset_label,
                "title": f"[{dataset_label}] {island_label} | Analytics (Spectrum, 3ω, Gap, Binarized Correlation Track)",
                "type": "cut_analytics",
                "left_top": get_rel(p_stage2_star) if p_stage2_star.exists() else None,
                "left_bottom": get_rel(spec_img) if spec_img.exists() else None,
                "right": get_rel(p_ana)
            })

        # Slide C: Island Unique COM Point Deep-Dive (ONE unique point per cut)
        didv_files = list(island_folder.glob("*_dIdV.png"))
        wf_files = list(island_folder.glob("*_wavefunctions.png"))
        barr_files = list(island_folder.glob("*_barrier_asymmetry.png"))
        if didv_files or wf_files or barr_files:
            slides.append({
                "dataset": dataset_label,
                "title": f"[{dataset_label}] {island_label} | COM Point Deep-Dive",
                "type": "point_deepdive",
                "left": get_rel(p_cond) if p_cond.exists() else (get_rel(p_ana) if p_ana.exists() else None),
                "pt_didv": get_rel(didv_files[0]) if didv_files else None,
                "pt_wf": get_rel(wf_files[0]) if wf_files else None,
                "pt_barr": get_rel(barr_files[0]) if barr_files else None,
            })

    return slides

def build_tgp_vs_3step_section(plots_root: Path, rel_root: Path = None):
    """
    Build 6 consecutive slides comparing Regular TGP (without correlation mask)
    vs. 3-Step Protocol (Step 1 Corr alone -> Step 2 Corr + 3w -> Step 3 Full ROI 3).
    Covers the 6 active disorder strengths (V0 = 0.0 to 0.91), omitting V0 = 1.2 (Slide 10 removed per user directive).
    Shows FULL passing regions in color (both inside and outside the Pfaffian boundary)
    accompanied by the compact 2-metric table:
      - % Pfaffian Area Passed (TPR)
      - False Discovery Rate (FDR)
    """
    rel_root = rel_root if rel_root else plots_root
    slides = []
    # User directive: Remove slide 10 (V0 = 1.2 / Tdis_pfaff5_Plots)
    target_disorders = [d for d in DISORDER_ORDER if d[0] != "Tdis_pfaff5_Plots"]
    for dname, label in target_disorders:
        pdir = plots_root / dname
        roi3_dir = pdir / "ROI3_Cuts"
        p_comp_img = roi3_dir / "tgp_vs_3step_protocol_comparison.png"
        if not p_comp_img.exists():
            p_comp_img = pdir / "tgp_vs_3step_protocol_comparison.png"
        p_comp_json = roi3_dir / "tgp_vs_3step_metrics.json"
        if not p_comp_json.exists():
            p_comp_json = pdir / "tgp_vs_3step_metrics.json"

        if not p_comp_img.exists() or not p_comp_json.exists():
            continue

        try:
            with open(p_comp_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            reg = data.get("regular_tgp", {})
            s1 = data.get("step1_correlation", {})
            s2 = data.get("step2_corr_plus_3w", {})
            s3 = data.get("step3_full_roi3", {})

            def get_badge(fdr_val):
                if fdr_val < 5.0:
                    return "badge-pass"
                elif fdr_val < 20.0:
                    return "badge-warn"
                return "badge-fail"

            table_card_html = f"""
            <div style="background:#0f172a; padding:18px 22px; border-radius:8px; border:1px solid #1e293b; height:100%; display:flex; flex-direction:column; justify-content:center; box-sizing:border-box;">
              <h3 style="color:#38bdf8; font-size:16px; margin-bottom:8px; font-weight:700;">Protocol Verification: Regular TGP vs. 3-Step Protocol</h3>
              <p style="color:#94a3b8; font-size:12px; margin-bottom:14px; line-height:1.4;">
                Benchmarking protocol passing maps against ground truth Pfaffian invariant ($\mathcal{{Q}} = -1$, dashed cyan boundary). Full passing regions are shown in color both inside and outside the Pfaffian domain, visually displaying False Positives in the trivial bulk alongside True Positives.
              </p>
              <table class="data-table" style="font-size:12.5px; margin-bottom:14px;">
                <thead>
                  <tr>
                    <th style="padding:8px 10px;">Protocol Layer</th>
                    <th style="padding:8px 10px; text-align:center;">% Pfaffian Area Passed</th>
                    <th style="padding:8px 10px; text-align:center;">False Discovery Rate (FDR)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><span style="color:#a855f7; font-weight:bold;">Regular TGP (Stage 2 ROI 2)</span></td>
                    <td style="text-align:center;"><strong>{reg.get('pfaffian_passed_pct', 0.0):.2f}%</strong></td>
                    <td style="text-align:center;"><span class="{get_badge(reg.get('fdr_percent', 0.0))}">{reg.get('fdr_percent', 0.0):.2f}%</span></td>
                  </tr>
                  <tr>
                    <td><span style="color:#0284c7; font-weight:bold;">Step 1: Barrier Corr Alone</span></td>
                    <td style="text-align:center;"><strong>{s1.get('pfaffian_passed_pct', 0.0):.2f}%</strong></td>
                    <td style="text-align:center;"><span class="{get_badge(s1.get('fdr_percent', 0.0))}">{s1.get('fdr_percent', 0.0):.2f}%</span></td>
                  </tr>
                  <tr>
                    <td><span style="color:#f59e0b; font-weight:bold;">Step 2: Corr + 3ω Agreement</span></td>
                    <td style="text-align:center;"><strong>{s2.get('pfaffian_passed_pct', 0.0):.2f}%</strong></td>
                    <td style="text-align:center;"><span class="{get_badge(s2.get('fdr_percent', 0.0))}">{s2.get('fdr_percent', 0.0):.2f}%</span></td>
                  </tr>
                  <tr>
                    <td><span style="color:#10b981; font-weight:bold;">Step 3: Full ROI 3 (+ Transport Gap)</span></td>
                    <td style="text-align:center;"><strong>{s3.get('pfaffian_passed_pct', 0.0):.2f}%</strong></td>
                    <td style="text-align:center;"><span class="{get_badge(s3.get('fdr_percent', 0.0))}">{s3.get('fdr_percent', 0.0):.2f}%</span></td>
                  </tr>
                </tbody>
              </table>
              <div style="background:#1e293b; padding:10px 14px; border-radius:6px; font-size:11px; line-height:1.55; color:#cbd5e1;">
                <strong>Protocol Architecture:</strong><br>
                • <strong>Regular TGP:</strong> Microsoft Stage 2 ROI 2 without barrier reflection correlation.<br>
                • <strong>Step 1:</strong> Isolates regions of high dual nonlocal barrier reflection correlation ($C_L, C_R \ge 0.85$).<br>
                • <strong>Step 2:</strong> Enforces ZBP curvature agreement ($I_{{3\omega}} < -100$) across $\ge 50\%$ cutter pairs, eliminating bulk gapless crossings.<br>
                • <strong>Step 3:</strong> Demands extracted transport gap $\Delta_{{\mathrm{{transport}}}} > 10\ \mu\mathrm{{eV}}$, suppressing trivial sub-gap crossings and driving FDR to a minimum.<br>
                • <em>Visual representation:</em> Colored regions inside the dashed cyan line represent True Positives (% Pfaffian Area Passed); colored regions outside represent False Positives (FDR).
              </div>
            </div>
            """

            slides.append({
                "dataset": "All Disorders",
                "title": f"All Disorders | {label} — Protocol Comparison: Regular TGP vs. 3-Step Protocol",
                "type": "side_by_side_html",
                "img": os.path.relpath(p_comp_img, rel_root),
                "html_card": table_card_html,
            })
        except Exception:
            pass

    return slides

def build_roi3_master_intro_slides(plots_root: Path, rel_root: Path = None):
    """
    Build cross-disorder introductory statistics table for ROI 3 presentation.
    Slide 1: Ensemble-Averaged Post-Processing Statistics Table (N = 8 Realizations at V0 = 0.645)
    (Old slides 1 and 2 removed per user directive).
    """
    rel_root = rel_root if rel_root else plots_root
    intro_slides = []

    # Ensemble-Averaged Post-Processing Statistics Table (Slide 1)
    p_ens_summary = BASE_DIR / "Outputs/Postprocessing/Post_Process_Test/ensemble_summary.json"
    if p_ens_summary.exists():
        try:
            with open(p_ens_summary, "r", encoding="utf-8") as f:
                ens_data = json.load(f)

            steps_meta = [
                ("regular_tgp", "Regular Microsoft TGP Stage 2", "#a855f7"),
                ("step1_correlation", "Step 1: Dual Correlation (C_L, C_R ≥ 0.85)", "#0284c7"),
                ("step2_corr_plus_3w", "Step 2: Corr + 3ω Curvature Consensus", "#f59e0b"),
                ("step3_full_roi3", "Step 3 / ROI 3: Full Protocol (+ Transport Gap)", "#10b981"),
            ]

            ens_rows = []
            for key, name, color in steps_meta:
                if key in ens_data:
                    sd = ens_data[key]
                    tpr_m = sd.get("pfaffian_passed_pct", {}).get("mean", 0.0)
                    tpr_s = sd.get("pfaffian_passed_pct", {}).get("std", 0.0)
                    fdr_m = sd.get("fdr_percent", {}).get("mean", 0.0)
                    fdr_s = sd.get("fdr_percent", {}).get("std", 0.0)
                    fdr_pool = sd.get("fdr_percent", {}).get("pooled", 0.0)
                    f1_m = sd.get("f1_score", {}).get("mean", 0.0)
                    f1_s = sd.get("f1_score", {}).get("std", 0.0)
                    conf_m = sd.get("mean_boundary_confinement", {}).get("mean", 0.0)
                    conf_s = sd.get("mean_boundary_confinement", {}).get("std", 0.0)
                    sep_m = sd.get("mean_separability", {}).get("mean", 0.0)
                    sep_s = sd.get("mean_separability", {}).get("std", 0.0)

                    b_fdr = "badge-pass" if fdr_m < 5.0 else ("badge-warn" if fdr_m < 20.0 else "badge-fail")
                    b_pool = "badge-pass" if fdr_pool < 5.0 else ("badge-warn" if fdr_pool < 20.0 else "badge-fail")

                    ens_rows.append(f"""
                    <tr>
                      <td><span style="color:{color}; font-weight:bold;">{name}</span></td>
                      <td style="text-align:center;"><strong>{tpr_m:.2f}%</strong> <span style="color:#94a3b8; font-size:11px;">± {tpr_s:.2f}%</span></td>
                      <td style="text-align:center;"><span class="{b_fdr}">{fdr_m:.2f}%</span> <span style="color:#94a3b8; font-size:11px;">± {fdr_s:.2f}%</span></td>
                      <td style="text-align:center;"><span class="{b_pool}"><strong>{fdr_pool:.2f}%</strong></span></td>
                      <td style="text-align:center;">{f1_m:.3f} <span style="color:#94a3b8; font-size:11px;">± {f1_s:.3f}</span></td>
                      <td style="text-align:center;">{conf_m:.3f} <span style="color:#94a3b8; font-size:11px;">± {conf_s:.3f}</span></td>
                      <td style="text-align:center;"><strong>{sep_m:.3f}</strong> <span style="color:#94a3b8; font-size:11px;">± {sep_s:.3f}</span></td>
                    </tr>
                    """)

            meta_info = ens_data.get("metadata", {})
            n_eval = meta_info.get("total_realizations_evaluated", 8)
            v0_eval = meta_info.get("disorder_strength_V0", 0.645)

            ens_table_html = f"""
            <table class="data-table">
              <thead>
                <tr>
                  <th>Protocol Step / Layer</th>
                  <th style="text-align:center;">% Pfaffian Passed (TPR)</th>
                  <th style="text-align:center;">Mean FDR (%)</th>
                  <th style="text-align:center;">Pooled FDR (%)</th>
                  <th style="text-align:center;">F1 Score</th>
                  <th style="text-align:center;">Boundary Confinement</th>
                  <th style="text-align:center;">MZM Separability (S)</th>
                </tr>
              </thead>
              <tbody>
                {"".join(ens_rows)}
              </tbody>
            </table>
            <div style="margin-top:16px; font-size:11px; color:#94a3b8; line-height:1.6;">
              <strong>Ensemble-Averaged Post-Processing Statistics (N = {n_eval} Disorder Realizations at V₀ = {v0_eval} meV):</strong><br>
              • <strong>Regular Microsoft TGP:</strong> Severely restricted by the boundary gapless heuristic, capturing only $5.71\%$ of the topological domain with high mean FDR ($22.92\%$, max $42.15\%$).<br>
              • <strong>Step 1 (Dual Correlation Alone):</strong> Broad topological coverage ($80.37\%$ passed) with $F_1 = 0.804$, providing a high-yield candidate pool.<br>
              • <strong>Step 2 (+ 3ω Curvature Agreement):</strong> Curvature thresholding across cutter pairs cuts bulk gapless crossings, dropping mean FDR to $7.86\%$ ($92.14\%$ precision).<br>
              • <strong>Step 3 (Full ROI 3 + Transport Gap):</strong> Extracted transport gap isolates pure unhybridized states, driving pooled FDR down to $3.75\%$ (median $2.88\%$) and maximizing MZM Separability ($S = 0.566$).
            </div>
            """

            intro_slides.append({
                "dataset": "All Disorders",
                "title": f"All Disorders | Ensemble-Averaged Post-Processing Verification Statistics (N = {n_eval} Realizations, V₀ = {v0_eval} meV)",
                "type": "table_slide",
                "table_title": "Ensemble Post-Processing Benchmark & Topological Verification",
                "table_subtitle": "Statistical distribution of discovery rates, F1 scores, boundary confinement, and MZM separability across disorder realizations",
                "html_table": ens_table_html
            })
        except Exception as e_ens:
            pass

    return intro_slides

def build_roi3_appendix_slides(plots_root: Path, rel_root: Path = None):
    """
    Appendix slides placed at the very end of the presentation:
    Cross-Disorder Bias Voltage Gap Threshold Sweep Evolution (2.0, 10.0, 30.0 μeV).
    """
    rel_root = rel_root if rel_root else plots_root
    appendix_slides = []
    sweep_comparisons = [
        ("2.0uev", "Relaxed Gapless Threshold Δ_low = 2.0 μeV (Maximum Topological Yield)"),
        ("10.0uev", "Baseline Microsoft Gapless Threshold Δ_low = 10.0 μeV (Standard TGP)"),
        ("30.0uev", "Strict Gapless Threshold Δ_low = 30.0 μeV (Aggressive Gap Filtering)"),
    ]
    for th_code, th_title in sweep_comparisons:
        th_imgs = []
        for dname, label in [
            ("Tdis_pfaff5_V0_0_0_Plots", "V0 = 0.0"),
            ("Tdis_pfaff5_V0_0_378_Plots", "V0 = 0.378"),
            ("Tdis_pfaff5_V0_0_645_Plots", "V0 = 0.645"),
            ("Tdis_pfaff5_V0_0_872_Plots", "V0 = 0.872"),
        ]:
            p_th = plots_root / dname / "bias_threshold_sweep" / f"ROI3_stage2_phase_map_{th_code}.png"
            if p_th.exists():
                th_imgs.append((os.path.relpath(p_th, rel_root), label))
        if len(th_imgs) == 4:
            appendix_slides.append({
                "dataset": "Appendix",
                "title": f"Appendix | Bias Threshold Sweep Comparison — {th_title} [Grey = Failed Corr]",
                "type": "global_summary",
                "tl": th_imgs[0][0], "tl_alt": f"{th_imgs[0][1]} ({th_code})",
                "tr": th_imgs[1][0], "tr_alt": f"{th_imgs[1][1]} ({th_code})",
                "bl": th_imgs[2][0], "bl_alt": f"{th_imgs[2][1]} ({th_code})",
                "br": th_imgs[3][0], "br_alt": f"{th_imgs[3][1]} ({th_code})",
            })
    return appendix_slides

def build_all_roi3_slides(plots_root: Path, rel_root: Path = None):
    """
    Build the complete ordered slide sequence:
      1. Master Statistics Tables (Slides 1–3)
      2. Regular TGP vs 3-Step Protocol Comparison (Slides 4–10, grouped consecutively)
      3. Per-Disorder ROI 3 Island Cuts & Deep-Dives
      4. Appendix: Cross-Disorder Bias Threshold Sweeps
    """
    rel_root = rel_root if rel_root else plots_root
    all_slides = []

    # 1. Master Statistics Tables (Slides 1–3)
    intro_slides = build_roi3_master_intro_slides(plots_root, rel_root=rel_root)
    all_slides.extend(intro_slides)

    # 2. Regular TGP vs 3-Step Protocol Comparison (Slides 4–10)
    comp_slides = build_tgp_vs_3step_section(plots_root, rel_root=rel_root)
    all_slides.extend(comp_slides)

    # 3. Per-Disorder ROI 3 Island Cuts & Deep-Dives
    for dname, label in DISORDER_ORDER:
        pdir = plots_root / dname
        if pdir.exists():
            d_slides = build_roi3_slides_for_directory(pdir, base_rel_path=rel_root)
            all_slides.extend(d_slides)

    # 4. Appendix: Cross-Disorder Bias Threshold Sweeps (at the very back)
    appendix_slides = build_roi3_appendix_slides(plots_root, rel_root=rel_root)
    all_slides.extend(appendix_slides)

    return all_slides

def compile_to_pdf(html_path: Path) -> Path:
    pdf_path = html_path.with_suffix(".pdf")
    file_url = f"file://{html_path.resolve()}"
    cmd = [
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path.resolve()}",
        file_url
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode == 0 and pdf_path.exists():
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"[OK] Compiled PDF: {pdf_path.name} ({size_mb:.2f} MB)")
        return pdf_path
    else:
        print(f"[ERROR] Failed to compile PDF for {html_path}: {res.stderr}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate ROI 3 HTML slide presentations.")
    parser.add_argument("--dir", type=str, default=None, help="Specific dataset directory under Outputs/Plots to generate slides for.")
    parser.add_argument("--weekly-date", type=str, default="2026-10-13", help="Date string for weekly research slides (default: 2026-10-13).")
    parser.add_argument("--no-pdf", action="store_true", help="Skip headless Chrome PDF compilation.")
    args = parser.parse_args()

    plots_root = (BASE_DIR / "Outputs/Plots").resolve()

    if args.dir:
        target_dir = Path(args.dir).resolve()
        slides = build_roi3_slides_for_directory(target_dir)
        out_file = target_dir / "ROI3_Cuts" / "slides.html"
        render_html_presentation(slides, out_file, title=f"ROI 3 Cuts Slides - {target_dir.name}")
        print(f"Generated standalone ROI 3 slides at {out_file}")
        if not args.no_pdf:
            compile_to_pdf(out_file)
        return

    # 1. Weekly Research Slides in Outputs/slides/weekly research slides/<date>/
    weekly_dir = (BASE_DIR / "Outputs/slides/weekly research slides" / args.weekly_date).resolve()
    weekly_dir.mkdir(parents=True, exist_ok=True)
    weekly_slides = build_all_roi3_slides(plots_root, rel_root=weekly_dir)
    weekly_out_dated = weekly_dir / f"slides-{args.weekly_date}.html"
    
    render_html_presentation(weekly_slides, weekly_out_dated, title=f"Weekly Research Slides - {args.weekly_date}")
    print(f"Generated Weekly Research Slides at {weekly_out_dated} ({len(weekly_slides)} slides total).")

    if not args.no_pdf:
        compile_to_pdf(weekly_out_dated)

if __name__ == "__main__":
    main()
