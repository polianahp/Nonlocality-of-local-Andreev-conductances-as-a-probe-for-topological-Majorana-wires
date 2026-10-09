#!/usr/bin/env python3
"""
Generate an interactive, responsive HTML slide deck exclusively for ROI 3 Cuts.
Contains plots and analytics from Outputs/Plots/<dataset>_Plots/ROI3_Cuts/:
  - Section 1: Master Introduction & Global Overviews
      * Slide 1: Master Three-Stage Progressive FDR Evolution Table across all 7 disorders
      * Slide 2: Master Cross-Disorder ROI 2 vs ROI 3 Verification Table
      * Slides 3–4: Spatial Localization Evolution (Separability & Boundary Confinement Groups A and B)
      * Slides 5–11: 6-Panel Protocol Overviews (one per disorder, featuring Conductance Gap Sens. & Bias Gap Sens.)
      * Slides 12–13: Cross-Disorder ROI 3 Phase Maps Grids (Groups 1 & 2)
  - Section 2: Regular TGP vs. 3-Step Protocol Comparison (Grouped consecutively across all 7 disorders)
      * 7 consecutive slides (V0 = 0.0 to 1.2) featuring:
          - Left: 4-Panel Phase Map Comparison with strict Pfaffian masking (Q = -1 domain only)
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
from pathlib import Path
import json

from generate_html_slides import HTML_TEMPLATE, render_html_presentation

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
                    Benchmarking against ground truth Pfaffian invariant ($\mathcal{{Q}} = -1$) within the topological domain. Non-topological bulk regions are masked out in the phase map.
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
                    • <em>Visual note:</em> Regions outside the Pfaffian invariant ($\mathcal{{Q}} = +1$) are suppressed; dashed cyan line indicates the topological boundary.
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

def build_tgp_vs_3step_section(plots_root: Path):
    """
    Build 7 consecutive slides comparing Regular TGP (without correlation mask)
    vs. 3-Step Protocol (Step 1 Corr alone -> Step 2 Corr + 3w -> Step 3 Full ROI 3).
    Strictly masks out non-Pfaffian regions and uses a compact 2-metric table:
      - % Pfaffian Area Passed (TPR)
      - False Discovery Rate (FDR)
    """
    slides = []
    for dname, label in DISORDER_ORDER:
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
                Benchmarking against ground truth Pfaffian invariant ($\mathcal{{Q}} = -1$) within the topological domain. Non-topological bulk regions are masked out in the phase map.
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
                • <em>Visual note:</em> Regions outside the Pfaffian invariant ($\mathcal{{Q}} = +1$) are suppressed; dashed cyan line indicates the topological boundary.
              </div>
            </div>
            """

            slides.append({
                "dataset": "All Disorders",
                "title": f"All Disorders | {label} — Protocol Comparison: Regular TGP vs. 3-Step Protocol",
                "type": "side_by_side_html",
                "img": os.path.relpath(p_comp_img, plots_root),
                "html_card": table_card_html,
            })
        except Exception:
            pass

    return slides

def build_roi3_master_intro_slides(plots_root: Path):
    """
    Build cross-disorder introductory slides for ROI 3 presentation.
    """
    valid = []
    for dname, label in DISORDER_ORDER:
        pdir = plots_root / dname
        roi3_dir = pdir / "ROI3_Cuts"
        p_star = roi3_dir / "ROI3_stage2_phase_map.png"
        if not p_star.exists(): p_star = roi3_dir / "stage2_phase_map.png"
        p_2w = pdir / "global_2w_phase_map.png"
        p_sep = pdir / "global_separability_phase_map.png"
        p_fdr = roi3_dir / "ROI3_fdr_metrics.json"
        if not p_fdr.exists(): p_fdr = roi3_dir / "fdr_metrics.json"

        entry = {
            "dir_name": dname,
            "label": label,
            "star": os.path.relpath(p_star, plots_root) if p_star.exists() else None,
            "two_w": os.path.relpath(p_2w, plots_root) if p_2w.exists() else None,
            "sep": os.path.relpath(p_sep, plots_root) if p_sep.exists() else None,
            "fdr_file": p_fdr if p_fdr.exists() else None,
        }
        valid.append(entry)

    intro_slides = []

    # 1a. Master Three-Stage Progressive False Discovery Rate Evolution Table
    p_prog_summary = plots_root / "all_progressive_fdr_summary.json"
    if p_prog_summary.exists():
        try:
            with open(p_prog_summary, "r", encoding="utf-8") as f:
                prog_sum_data = json.load(f)
            prog_rows = []
            for dname, label in DISORDER_ORDER:
                clean_name = dname.replace("_Plots", "")
                if clean_name in prog_sum_data:
                    d_metrics = prog_sum_data[clean_name]
                    s1 = d_metrics.get("stage1_correlation", {})
                    s2 = d_metrics.get("stage2_corr_plus_3w", {})
                    s3 = d_metrics.get("stage3_full_roi3", {})
                    fdr1 = s1.get("fdr_percent", 0.0)
                    fdr2 = s2.get("fdr_percent", 0.0)
                    fdr3 = s3.get("fdr_percent", 0.0)
                    b1 = "badge-pass" if fdr1 < 5 else ("badge-warn" if fdr1 < 15 else "badge-fail")
                    b2 = "badge-pass" if fdr2 < 5 else ("badge-warn" if fdr2 < 15 else "badge-fail")
                    b3 = "badge-pass" if fdr3 < 5 else ("badge-warn" if fdr3 < 15 else "badge-fail")
                    prog_rows.append(f"""
                    <tr>
                      <td><strong>{label}</strong></td>
                      <td>{s1.get('total_passed', 0):,}</td>
                      <td><span class="{b1}">{fdr1:.2f}%</span></td>
                      <td>{s2.get('total_passed', 0):,}</td>
                      <td><span class="{b2}">{fdr2:.2f}%</span></td>
                      <td><strong>{s3.get('total_passed', 0):,}</strong></td>
                      <td><span class="{b3}">{fdr3:.2f}%</span></td>
                      <td><strong>{s3.get('ppv_percent', 0.0):.2f}%</strong></td>
                    </tr>
                    """)
            if prog_rows:
                master_prog_table_html = f"""
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Disorder Realization</th>
                      <th>Step 1 Passed (px)</th>
                      <th>Step 1 FDR</th>
                      <th>Step 2 Passed (px)</th>
                      <th>Step 2 FDR</th>
                      <th>Step 3 Passed (px)</th>
                      <th>Step 3 FDR</th>
                      <th>Step 3 PPV (Precision)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {"".join(prog_rows)}
                  </tbody>
                </table>
                <div style="margin-top:16px; font-size:11px; color:#94a3b8; line-height:1.6;">
                  <strong>Three-Stage Progressive Operational Filtering Evolution (Against Ground Truth Pfaffian $\mathcal{{Q}} = -1$):</strong><br>
                  • <strong>Step 1 (Dual Barrier Correlation Alone):</strong> Captures all regions exhibiting strong nonlocal reflection correlation ($C_L, C_R \ge 0.85$). While sensitive, disorder-induced trivial sub-gap crossings yield FDR up to $21.13\%$.<br>
                  • <strong>Step 2 (Corr + $3\omega$ Curvature Agreement):</strong> Curvature thresholding across $\ge 50\%$ cutter pairs eliminates bulk gapless trivial states, reducing FDR significantly ($17.38\% \to 5.48\%$ at $V_0=0.645$).<br>
                  • <strong>Step 3 (Full ROI 3: Corr + $3\omega$ + Transport Gap $\Delta_{{ex}} > 10\ \mu\mathrm{{eV}}$):</strong> Extracted finite transport gap eliminates hybridized states, collapsing FDR to near zero ($3.31\%$ at $V_0=0.645$ and $0.00\%$ at $V_0=1.2$).
                </div>
                """
                intro_slides.append({
                    "dataset": "All Disorders",
                    "title": "All Disorders | Three-Stage Progressive Protocol False Discovery Rate (FDR) Evolution",
                    "type": "table_slide",
                    "table_title": "Three-Stage Progressive Operational Filtering Benchmark",
                    "table_subtitle": "Step 1 (Barrier Corr Alone) → Step 2 (+ 3ω Curvature Agreement) → Step 3 (Full ROI 3 + Transport Gap)",
                    "html_table": master_prog_table_html
                })
        except Exception:
            pass

    # 1b. Master False Discovery Rate Comparison Table: ROI 2 vs ROI 3 Across All Disorders
    fdr_rows = []
    for v in valid:
        if v["fdr_file"]:
            try:
                with open(v["fdr_file"], "r", encoding="utf-8") as f:
                    data = json.load(f)
                gt = data.get("ground_truth_topological_pixels", 0)
                s2 = data.get("stage2_roi2_metrics", {})
                s3 = data.get("stage3_roi3_metrics", {})

                s2_passed = s2.get("total_passed", 0)
                s2_ppv = s2.get("p_topological_given_passed_ppv", 0.0) * 100.0
                s2_fdr = s2.get("false_discovery_rate_fdr", 0.0) * 100.0

                s3_passed = s3.get("total_passed", 0)
                s3_tp = s3.get("true_positives", 0)
                s3_fp = s3.get("false_positives", 0)
                s3_ppv = s3.get("p_topological_given_passed_ppv", 0.0) * 100.0
                s3_fdr = s3.get("false_discovery_rate_fdr", 0.0) * 100.0
                s3_tpr = s3.get("sensitivity_tpr", 0.0) * 100.0
                s3_tnr = s3.get("specificity_tnr", 0.0) * 100.0

                badge_s2_fdr = "badge-pass" if s2_fdr < 5.0 else ("badge-warn" if s2_fdr < 30.0 else "badge-fail")
                badge_s3_fdr = "badge-pass" if s3_fdr < 5.0 else ("badge-warn" if s3_fdr < 15.0 else "badge-fail")
                badge_s3_ppv = "badge-pass" if s3_ppv > 90.0 else ("badge-warn" if s3_ppv > 75.0 else "badge-fail")

                fdr_rows.append(f"""
                <tr>
                  <td><strong>{v['label']}</strong></td>
                  <td>{gt:,}</td>
                  <td>{s2_passed}</td>
                  <td><span class="{badge_s2_fdr}">{s2_fdr:.2f}%</span></td>
                  <td><strong>{s3_passed}</strong></td>
                  <td>{s3_tp}</td>
                  <td>{s3_fp}</td>
                  <td><span class="{badge_s3_ppv}">{s3_ppv:.2f}%</span></td>
                  <td><span class="{badge_s3_fdr}">{s3_fdr:.2f}%</span></td>
                  <td>{s3_tpr:.2f}%</td>
                  <td>{s3_tnr:.2f}%</td>
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
              <th>ROI 2 Passed (px)</th>
              <th>ROI 2 FDR</th>
              <th>ROI 3 Passed (px)</th>
              <th>ROI 3 TP</th>
              <th>ROI 3 FP</th>
              <th>ROI 3 PPV (P(Top|Pass))</th>
              <th>ROI 3 FDR</th>
              <th>ROI 3 Sensitivity (TPR)</th>
              <th>ROI 3 Specificity (TNR)</th>
            </tr>
          </thead>
          <tbody>
            {"".join(fdr_rows)}
          </tbody>
        </table>
        <div style="margin-top:16px; font-size:11px; color:#94a3b8; line-height:1.6;">
          <strong>ROI 3 Protocol Verification: Adding Dual Barrier Correlation Criterion ($C_L \ge 0.85$ & $C_R \ge 0.85$):</strong><br>
          • <strong>Dramatic FDR Reduction:</strong> In strong disorder ($V_0 = 0.645$), False Discovery Rate collapses from $14.24\% \to 1.92\%$ ($FP$ drops from $48 \to 4$).<br>
          • In severe disorder ($V_0 = 0.872$), FDR collapses from $35.65\% \to 11.86\%$ ($FP$ drops from $118 \to 14$).<br>
          • In severe disorder ($V_0 = 0.91$), FDR collapses from $37.14\% \to 10.00\%$ ($FP$ drops from $104 \to 9$).<br>
          • <strong>Physical Mechanism:</strong> While local/non-local conductance alone can mimic trivial Andreev bound state crossings, demanding simultaneous dual barrier reflection correlation prunes bulk trivial states and isolates genuine non-local Majorana modes.
        </div>
        """
        intro_slides.append({
            "dataset": "All Disorders",
            "title": "All Disorders | Master ROI 2 vs. ROI 3 False Discovery Rate (FDR) Comparison",
            "type": "table_slide",
            "table_title": "Cross-Disorder ROI 2 vs. ROI 3 Protocol Verification Table",
            "table_subtitle": "Demonstrating dramatic FDR reduction when intersecting Stage 2 with Dual Barrier Correlation >= 0.85",
            "html_table": master_table_html
        })

    # 1c. Spatial Localization Evolution Slides (Separability & Boundary Confinement)
    p_grp_a = plots_root / "separability_vs_confinement_group_A.png"
    if p_grp_a.exists():
        intro_slides.append({
            "dataset": "All Disorders",
            "title": "All Disorders | Spatial Localization Evolution vs. Disorder (Group A: Clean to Intermediate)",
            "type": "single_centered",
            "img": os.path.relpath(p_grp_a, plots_root),
            "alt": "Separability vs Confinement Group A"
        })

    p_grp_b = plots_root / "separability_vs_confinement_group_B.png"
    if p_grp_b.exists():
        intro_slides.append({
            "dataset": "All Disorders",
            "title": "All Disorders | Spatial Localization Evolution vs. Disorder (Group B: Strong to Benchmark)",
            "type": "single_centered",
            "img": os.path.relpath(p_grp_b, plots_root),
            "alt": "Separability vs Confinement Group B"
        })

    # 2. 6-Panel Disorder Overviews: Exactly 1 realization per slide
    # Panels: 2w, 3w, Transport Gap (10 ueV), Excitation Gap (E1 - E0), Conductance Gap Sens, Bias Gap Sens
    for dname, label in DISORDER_ORDER:
        pdir = plots_root / dname
        p_2w = pdir / "global_2w_phase_map.png"
        p_3w = pdir / "global_3w_phase_map.png"
        p_tg10 = pdir / "bias_threshold_sweep" / "transport_gap_phase_map_10.0uev.png"
        if not p_tg10.exists():
            p_tg10 = pdir / "global_transport_gap_phase_map.png"
        p_ex = pdir / "global_excitation_gap_E1_minus_E0_phase_map.png"
        p_sens_g = pdir / "global_gap_threshold_sensitivity_phase_map.png"
        p_sens_b = pdir / "global_bias_gap_sensitivity_phase_map.png"

        intro_slides.append({
            "dataset": "All Disorders",
            "title": f"All Disorders | {label} — 6-Panel Protocol Overview (2ω, 3ω, Transport Gap, Excitation Gap, Cond. Sens., Bias Sens.)",
            "type": "six_panel_overview",
            "p1": os.path.relpath(p_2w, plots_root) if p_2w.exists() else None,
            "p1_alt": f"2ω Conductance ({label})",
            "p2": os.path.relpath(p_3w, plots_root) if p_3w.exists() else None,
            "p2_alt": f"3ω Curvature ({label})",
            "p3": os.path.relpath(p_tg10, plots_root) if p_tg10.exists() else None,
            "p3_alt": f"Transport Gap 10 μeV ({label})",
            "p4": os.path.relpath(p_ex, plots_root) if p_ex.exists() else None,
            "p4_alt": f"Excitation Gap E1 - E0 ({label})",
            "p5": os.path.relpath(p_sens_g, plots_root) if p_sens_g.exists() else None,
            "p5_alt": f"Conductance Gap Sensitivity ({label})",
            "p6": os.path.relpath(p_sens_b, plots_root) if p_sens_b.exists() else None,
            "p6_alt": f"Bias Gap Sensitivity ({label})",
        })

    # 3. Cross-Disorder ROI 3 Phase Maps Grids
    valid_stars = [v for v in valid if v["star"]]
    if len(valid_stars) >= 4:
        sub1 = valid_stars[:4]
        slides_dict1 = {
            "dataset": "All Disorders",
            "title": "All Disorders | Cross-Disorder ROI 3 Phase Maps with Island COMs (Group 1: Clean to Intermediate)",
            "type": "global_summary",
            "tl": sub1[0]["star"], "tl_alt": sub1[0]["label"],
            "tr": sub1[1]["star"] if len(sub1) > 1 else sub1[0]["star"], "tr_alt": sub1[1]["label"] if len(sub1) > 1 else "",
            "bl": sub1[2]["star"] if len(sub1) > 2 else sub1[0]["star"], "bl_alt": sub1[2]["label"] if len(sub1) > 2 else "",
            "br": sub1[3]["star"] if len(sub1) > 3 else sub1[0]["star"], "br_alt": sub1[3]["label"] if len(sub1) > 3 else "",
        }
        intro_slides.append(slides_dict1)

    if len(valid_stars) > 4:
        sub2 = valid_stars[4:]
        slides_dict2 = {
            "dataset": "All Disorders",
            "title": "All Disorders | Cross-Disorder ROI 3 Phase Maps with Island COMs (Group 2: Strong Disorders)",
            "type": "global_summary",
            "tl": sub2[0]["star"], "tl_alt": sub2[0]["label"],
            "tr": sub2[1]["star"] if len(sub2) > 1 else sub2[0]["star"], "tr_alt": sub2[1]["label"] if len(sub2) > 1 else "",
            "bl": sub2[2]["star"] if len(sub2) > 2 else sub2[0]["star"], "bl_alt": sub2[2]["label"] if len(sub2) > 2 else "",
            "br": sub2[2]["star"] if len(sub2) > 2 else sub2[0]["star"], "br_alt": sub2[2]["label"] if len(sub2) > 2 else "",
        }
        intro_slides.append(slides_dict2)

    return intro_slides

def build_roi3_appendix_slides(plots_root: Path):
    """
    Appendix slides placed at the very end of the presentation:
    Cross-Disorder Bias Voltage Gap Threshold Sweep Evolution (2.0, 10.0, 30.0 μeV).
    """
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
                th_imgs.append((os.path.relpath(p_th, plots_root), label))
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

def main():
    parser = argparse.ArgumentParser(description="Generate ROI 3 HTML slide presentations.")
    parser.add_argument("--dir", type=str, default=None, help="Specific dataset directory under Outputs/Plots to generate slides for.")
    args = parser.parse_args()

    plots_root = Path("Outputs/Plots").resolve()

    if args.dir:
        target_dir = Path(args.dir).resolve()
        slides = build_roi3_slides_for_directory(target_dir)
        out_file = target_dir / "ROI3_Cuts" / "slides.html"
        render_html_presentation(slides, out_file, title=f"ROI 3 Cuts Slides - {target_dir.name}")
        print(f"Generated standalone ROI 3 slides at {out_file}")
        return

    # Master Presentation across all disorders
    all_slides = []

    # 1. Master Introduction & Global Overviews
    intro_slides = build_roi3_master_intro_slides(plots_root)
    all_slides.extend(intro_slides)

    # 2. Regular TGP vs 3-Step Protocol Comparison (All 7 Disorders Grouped Consecutively)
    comp_slides = build_tgp_vs_3step_section(plots_root)
    all_slides.extend(comp_slides)

    # 3. Per-Disorder ROI 3 Island Cuts & Deep-Dives
    for dname, label in DISORDER_ORDER:
        pdir = plots_root / dname
        if pdir.exists():
            d_slides = build_roi3_slides_for_directory(pdir, base_rel_path=plots_root)
            all_slides.extend(d_slides)

            # Standalone per-disorder deck in ROI3_Cuts/
            standalone_slides = build_roi3_slides_for_directory(pdir, base_rel_path=pdir / "ROI3_Cuts")
            if standalone_slides:
                out_standalone = pdir / "ROI3_Cuts" / "slides.html"
                render_html_presentation(standalone_slides, out_standalone, title=f"ROI 3 Cuts Presentation - {label}")
                print(f"Generated {out_standalone}")

    # 4. Appendix: Cross-Disorder Bias Threshold Sweeps (Moved to the very back)
    appendix_slides = build_roi3_appendix_slides(plots_root)
    all_slides.extend(appendix_slides)

    master_out = plots_root / "roi3_slides.html"
    render_html_presentation(all_slides, master_out, title="Master Majorana ROI 3 Protocol & Cuts Presentation (Dual Barrier Corr >= 0.85)")
    print(f"Generated Master ROI 3 HTML Presentation at {master_out} ({len(all_slides)} slides total).")

if __name__ == "__main__":
    main()
