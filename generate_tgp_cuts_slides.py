#!/usr/bin/env python3
"""
Generate an interactive, responsive HTML slide deck exclusively for the TGP Cuts folder.
Contains ONLY the plots from Outputs/Plots/<dataset>_Plots/TGP_Cuts/:
  - Intro: Master FDR Verification Table, Panoramic 2w vs Separability comparisons, and Phase Map grids
  - Per Disorder Realization:
      * Stage 2 Phase Map with Island COMs (Stars) + FDR Statistics Table
      * For each of the 3 Topological Islands:
          - Slide 1 (Overview): Phase Map & Spectrum vs 5-Panel Conductance Multiplot
          - Slide 2 (Analytics): Phase Map & Spectrum vs 5-Panel Analytics Multiplot
          - Slide 3 (Unique COM Point Deep-Dive): Conductance Multiplot vs 3 Stacked Plots (dI/dV, Wavefunctions, Barrier Asymmetry)
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

def build_tgp_slides_for_directory(plot_dir: Path, base_rel_path: Path = None):
    """
    Build slides specifically for the TGP_Cuts folder of a dataset directory.
    Guarantees exactly ONE unique point per cut (the Center of Mass of the island).
    """
    plot_dir = Path(plot_dir).resolve()
    rel_root = Path(base_rel_path).resolve() if base_rel_path else plot_dir
    dir_name = plot_dir.name
    dataset_label = dir_name.replace("_Plots", "")

    tgp_dir = plot_dir / "TGP_Cuts"
    if not tgp_dir.exists():
        return []

    slides = []

    def get_rel(p: Path):
        try:
            return os.path.relpath(p, rel_root)
        except Exception:
            return str(p)

    p_stage2_star = tgp_dir / "stage2_phase_map.png"
    p_fdr = tgp_dir / "fdr_metrics.json"

    # Slide 1: Stage 2 Phase Map with Island COMs + Statistical Summary Table
    if p_stage2_star.exists():
        fdr_table_html = ""
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

                fdr_table_html = f"""
                <div style="background:#0f172a; padding:16px; border-radius:8px; border:1px solid #1e293b; height:100%; display:flex; flex-direction:column; justify-content:center;">
                  <h3 style="color:#38bdf8; font-size:16px; margin-bottom:12px;">Topological Protocol Verification</h3>
                  <div style="margin-bottom:14px; font-size:13px; line-height:1.6;">
                    <p><strong>Stage 2 Passed:</strong> {s2_m.get('total_passed', 0)} px (TP: {s2_m.get('true_positives', 0)}, FP: {s2_m.get('false_positives', 0)})</p>
                    <p><strong>P(Topological | Passed):</strong> <span class="{badge_cls}">{ppv_pct:.2f}%</span></p>
                    <p><strong>False Discovery Rate:</strong> <span class="{badge_cls}">{fdr_pct:.2f}%</span></p>
                    <p><strong>Sensitivity (TPR):</strong> {tpr_pct:.2f}% | <strong>Specificity (TNR):</strong> {tnr_pct:.2f}%</p>
                    <p style="color:#94a3b8; font-size:12px; margin-top:4px;">Candidate Orange Pixels: {or_m.get('total_orange_pixels', 0)} px (PPV: {or_ppv:.1f}%, FDR: {or_fdr:.1f}%)</p>
                  </div>
                  <h4 style="color:#f1f5f9; font-size:13px; margin-bottom:8px;">Top 3 Validated Topological Islands</h4>
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
                "title": f"[{dataset_label}] Stage 2 Topological Phase Map & Validation Statistics",
                "type": "side_by_side_html",
                "img": get_rel(p_stage2_star),
                "html_card": fdr_table_html
            })
        else:
            slides.append({
                "dataset": dataset_label,
                "title": f"[{dataset_label}] Stage 2 Phase Map & Top 3 Topological Island COMs",
                "type": "single_centered",
                "img": get_rel(p_stage2_star),
                "alt": "Stage 2 Phase Map with Island COMs",
            })

    # Bias Voltage Gap Threshold Sensitivity Slide (if available)
    sweeps_dir = plot_dir / "Stage2_Gap_Sweeps"
    if sweeps_dir.exists():
        p_bg2 = sweeps_dir / "stage2_bias_gap_2.0uev.png"
        p_bg6 = sweeps_dir / "stage2_bias_gap_6.0uev.png"
        p_bg10 = sweeps_dir / "stage2_bias_gap_10.0uev.png"
        p_bg20 = sweeps_dir / "stage2_bias_gap_20.0uev.png"
        if p_bg2.exists() and p_bg10.exists():
            slides.append({
                "dataset": dataset_label,
                "title": f"[{dataset_label}] Stage 2 Sensitivity: Bias Voltage Gap Threshold (2.0, 6.0, 10.0, 20.0 μeV)",
                "type": "global_summary",
                "tl": get_rel(p_bg2), "tl_alt": "Bias Gap 2.0 μeV",
                "tr": get_rel(p_bg6) if p_bg6.exists() else get_rel(p_bg2), "tr_alt": "Bias Gap 6.0 μeV",
                "bl": get_rel(p_bg10), "bl_alt": "Bias Gap 10.0 μeV (Base)",
                "br": get_rel(p_bg20) if p_bg20.exists() else get_rel(p_bg10), "br_alt": "Bias Gap 20.0 μeV",
            })

    # Ordered list of islands (Cyan, Magenta, Gold)
    island_dirs = sorted([d for d in tgp_dir.iterdir() if d.is_dir() and d.name.startswith("Island_")])

    for island_folder in island_dirs:
        island_label = island_folder.name.replace("_", " ")
        p_cond = island_folder / "multi_panel_conductance.png"
        p_ana = island_folder / "multi_panel.png"
        p_spec = island_folder / "spectra.png"

        # Look for standalone island spectrum in parent TGP_Cuts
        rank_idx = "1" if "Island_1" in island_folder.name else ("2" if "Island_2" in island_folder.name else "3")
        parent_spec = tgp_dir / f"tgp_cut_island_{rank_idx}_spectra.png"
        spec_img = parent_spec if parent_spec.exists() else p_spec

        # Slide A: Island Cut Overview (Conductance Multiplot)
        if p_cond.exists():
            slides.append({
                "dataset": dataset_label,
                "title": f"[{dataset_label}] TGP Cut: {island_label} | Overview (Conductance)",
                "type": "cut_overview",
                "left_top": get_rel(p_stage2_star) if p_stage2_star.exists() else None,
                "left_bottom": get_rel(spec_img) if spec_img.exists() else None,
                "right": get_rel(p_cond)
            })

        # Slide B: Island Cut Analytics (2w, 3w, Gap, Overlap Multiplot)
        if p_ana.exists():
            slides.append({
                "dataset": dataset_label,
                "title": f"[{dataset_label}] TGP Cut: {island_label} | Analytics (2w, 3w, Gap, Overlap)",
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
                "title": f"[{dataset_label}] TGP Cut: {island_label} | COM Point Deep-Dive",
                "type": "point_deepdive",
                "left": get_rel(p_cond) if p_cond.exists() else (get_rel(p_ana) if p_ana.exists() else None),
                "pt_didv": get_rel(didv_files[0]) if didv_files else None,
                "pt_wf": get_rel(wf_files[0]) if wf_files else None,
                "pt_barr": get_rel(barr_files[0]) if barr_files else None,
            })

    return slides

def build_tgp_master_intro_slides(plots_root: Path):
    """
    Build cross-disorder introductory slides for the dedicated TGP cuts presentation.
    """
    valid = []
    for dname, label in DISORDER_ORDER:
        pdir = plots_root / dname
        tgp_dir = pdir / "TGP_Cuts"
        p_star = tgp_dir / "stage2_phase_map.png"
        p_2w = pdir / "global_2w_phase_map.png"
        p_sep = pdir / "global_separability_phase_map.png"
        p_fdr = tgp_dir / "fdr_metrics.json"

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

    # 1. Master False Discovery Rate Table Across All 7 Disorders
    fdr_rows = []
    for v in valid:
        if v["fdr_file"]:
            try:
                with open(v["fdr_file"], "r", encoding="utf-8") as f:
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
                  <td><strong>{v['label']}</strong></td>
                  <td>{gt:,}</td>
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
        <div style="margin-top:16px; font-size:11px; color:#94a3b8; line-height:1.6;">
          <strong>Topological Gap Protocol Verification Metrics:</strong><br>
          • <strong>Positive Predictive Value (PPV):</strong> $P(\\text{{Topological}} \\mid \\text{{Passed}}) = \\frac{{TP}}{{TP + FP}}$. Probability that a passed Stage 2 ROI is in a genuine topological phase ($\mathcal{{Q}} = -1$).<br>
          • <strong>False Discovery Rate (FDR):</strong> $1 - PPV = \\frac{{FP}}{{TP + FP}}$. Probability of a false positive topological detection.<br>
          • <strong>Sensitivity (TPR):</strong> Fraction of ground-truth topological area identified by Stage 2 protocol.<br>
          • <strong>Specificity (TNR):</strong> Fraction of trivial parameter space correctly rejected by Stage 2 protocol.
        </div>
        """
        intro_slides.append({
            "dataset": "All Disorders",
            "title": "All Disorders | Master False Discovery Rate (FDR) & Protocol Verification Table",
            "type": "table_slide",
            "table_title": "Cross-Disorder False Discovery Rate & Protocol Verification Table",
            "table_subtitle": "Systematic evaluation of Stage 2 ROI2 against Ground-Truth Pfaffian across all 7 disorder amplitudes",
            "html_table": master_table_html
        })

    # 2. Panoramic Multi-Column 2-Row Comparison: 2w (Top) vs Separability (Bottom)
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

    # 3. Cross-Disorder Stage 2 Phase Map Grids (Stars at island COMs)
    valid_stars = [v for v in valid if v["star"]]
    if len(valid_stars) >= 4:
        intro_slides.append({
            "dataset": "All Disorders",
            "title": f"All Disorders | Stage 2 Phase Maps with Island COMs ({valid_stars[0]['label']}, {valid_stars[1]['label']}, {valid_stars[2]['label']}, {valid_stars[3]['label']})",
            "type": "global_summary",
            "tl": valid_stars[0]["star"], "tl_alt": f"Stage 2 Map - {valid_stars[0]['label']}",
            "tr": valid_stars[1]["star"], "tr_alt": f"Stage 2 Map - {valid_stars[1]['label']}",
            "bl": valid_stars[2]["star"], "bl_alt": f"Stage 2 Map - {valid_stars[2]['label']}",
            "br": valid_stars[3]["star"], "br_alt": f"Stage 2 Map - {valid_stars[3]['label']}",
        })
    if len(valid_stars) >= 7:
        intro_slides.append({
            "dataset": "All Disorders",
            "title": f"All Disorders | Stage 2 Phase Maps with Island COMs ({valid_stars[4]['label']}, {valid_stars[5]['label']}, {valid_stars[6]['label']})",
            "type": "global_summary",
            "tl": valid_stars[4]["star"], "tl_alt": f"Stage 2 Map - {valid_stars[4]['label']}",
            "tr": valid_stars[5]["star"], "tr_alt": f"Stage 2 Map - {valid_stars[5]['label']}",
            "bl": valid_stars[6]["star"], "bl_alt": f"Stage 2 Map - {valid_stars[6]['label']}",
            "br": None, "br_alt": "",
        })

    return intro_slides

def render_tgp_slides(slides, output_file: Path, title: str):
    """
    Renders presentation supporting side_by_side_html layout.
    """
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
                  {f'<img src="{slide["left_top"]}" alt="Stage 2 Map">' if slide.get("left_top") else '<p>Map missing</p>'}
                </div>
                <div class="img-card flex-1">
                  {f'<img src="{slide["left_bottom"]}" alt="Cut Spectrum">' if slide.get("left_bottom") else '<p>Spectrum missing</p>'}
                </div>
              </div>
              <div class="img-card">
                {f'<img src="{slide["right"]}" alt="Cut Multiplot">' if slide.get("right") else '<p>Cut multiplot missing</p>'}
              </div>
            </div>
            """
        elif stype == "point_deepdive":
            content = f"""
            <div class="layout-point-deepdive">
              <div class="img-card">
                {f'<img src="{slide["left"]}" alt="Cut Conductance Multiplot">' if slide.get("left") else '<p>Cut conductance missing</p>'}
              </div>
              <div class="col-stacked-3">
                <div class="img-card flex-1">
                  {f'<img src="{slide["pt_didv"]}" alt="dI/dV">' if slide.get("pt_didv") else '<p>dI/dV missing</p>'}
                </div>
                <div class="img-card flex-1">
                  {f'<img src="{slide["pt_wf"]}" alt="Wavefunctions">' if slide.get("pt_wf") else '<p>Wavefunctions missing</p>'}
                </div>
                <div class="img-card flex-1">
                  {f'<img src="{slide["pt_barr"]}" alt="Barrier Asymmetry">' if slide.get("pt_barr") else '<p>Barrier Asymmetry missing</p>'}
                </div>
              </div>
            </div>
            """
        elif stype == "side_by_side_html":
            content = f"""
            <div class="layout-side-by-side" style="padding:10px;">
              <div class="img-card">
                {f'<img src="{slide["img"]}" alt="Stage 2 Phase Map">' if slide.get("img") else '<p>Image missing</p>'}
              </div>
              <div>
                {slide.get("html_card", "")}
              </div>
            </div>
            """
        elif stype in ("global_summary", "gap_sensitivity"):
            content = f"""
            <div class="layout-grid-2x2">
              <div class="img-card">
                {f'<img src="{slide["tl"]}" alt="{slide.get("tl_alt", "Top Left")}">' if slide.get("tl") else '<p>Top Left missing</p>'}
              </div>
              <div class="img-card">
                {f'<img src="{slide["tr"]}" alt="{slide.get("tr_alt", "Top Right")}">' if slide.get("tr") else '<p>Top Right missing</p>'}
              </div>
              <div class="img-card">
                {f'<img src="{slide["bl"]}" alt="{slide.get("bl_alt", "Bottom Left")}">' if slide.get("bl") else '<p>Bottom Left missing</p>'}
              </div>
              <div class="img-card">
                {f'<img src="{slide["br"]}" alt="{slide.get("br_alt", "Bottom Right")}">' if slide.get("br") else '<p></p>'}
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
        else:
            content = "<p>Unknown layout</p>"

        slide_html = f"""
        <section class="slide" id="slide-{i}" data-title="{title_text} | {slide.get('dataset', '')}">
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
    print(f"Generated TGP Cuts slide presentation ({len(slides)} slides) -> {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate dedicated TGP Cuts HTML slide presentation.")
    parser.add_argument("plot_dirs", nargs="*", help="List of plot directories to process.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    plots_root = project_root / "Outputs" / "Plots"

    if args.plot_dirs:
        target_dirs = [Path(d).resolve() for d in args.plot_dirs]
    else:
        target_dirs = [plots_root / dname for dname, _ in DISORDER_ORDER if (plots_root / dname).exists()]

    if not target_dirs:
        print("No valid plot directories found.")
        return

    # 1. Standalone slide presentations inside each TGP_Cuts folder
    all_master_slides = []
    for pdir in target_dirs:
        tgp_cuts_dir = pdir / "TGP_Cuts"
        if not tgp_cuts_dir.exists():
            continue

        slides = build_tgp_slides_for_directory(pdir, base_rel_path=tgp_cuts_dir)
        if slides:
            out_file = tgp_cuts_dir / "slides.html"
            render_tgp_slides(slides, out_file, title=f"TGP Cuts Analysis - {pdir.name.replace('_Plots', '')}")

        m_slides = build_tgp_slides_for_directory(pdir, base_rel_path=plots_root)
        all_master_slides.extend(m_slides)

    # 2. Master All-in-One TGP Cuts Presentation across all disorder realizations
    if all_master_slides:
        intro_slides = build_tgp_master_intro_slides(plots_root)
        final_master_slides = intro_slides + all_master_slides
        master_file = plots_root / "tgp_cuts_slides.html"
        render_tgp_slides(final_master_slides, master_file, title="Topological Gap Protocol (TGP) Cuts & Island Analysis - All Disorders")

if __name__ == "__main__":
    main()
