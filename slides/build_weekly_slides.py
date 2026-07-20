#!/usr/bin/env python3
import os
import glob
import re
import subprocess
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent.resolve()
    slides_base_dir = project_root / "slides"
    target_dir = slides_base_dir / "weekly research slides" / "2026-07-20"
    target_dir.mkdir(parents=True, exist_ok=True)

    html_file = target_dir / "slides-2026-07-20.html"
    pdf_file = target_dir / "slides-2026-07-20.pdf"

    data_dir = project_root / "Data"
    fpca_rank_dir = project_root / "Plots" / "FPCA_dis_realizations_maxnorm" / "Rank_10_FPC1_K7"
    fpca_pm_dir = fpca_rank_dir / "phase_maps_by_realization"

    dis_dirs = sorted(list((data_dir / "dis_realizations").glob("disorder_realization_*_results")), key=lambda p: int(re.search(r'\d+', p.name).group()))
    pfaff4_dir = data_dir / "Tdis_pfaff4"
    all_datasets = dis_dirs + ([pfaff4_dir] if pfaff4_dir.exists() else [])

    slides_html = []
    slide_counter = 0

    # 1. Title Slide
    slide_counter += 1
    slides_html.append(f"""
    <div class="slide title-slide">
        <div class="title-container">
            <h1 class="main-title">Non-Local Protocol & Topological Gap Analysis</h1>
            <h2 class="sub-title">Weekly Research Progress & Disorder Realization Characterization</h2>
            <div class="meta-info">
                <span>Date: July 20, 2026</span> &nbsp;|&nbsp; <span>Quantum Transport Simulation Group</span>
            </div>
        </div>
        <div class="footer-left">Weekly Research Update</div>
        <div class="footer-right">Slide {slide_counter}</div>
    </div>
    """)

    # Process per disorder realization
    for d_path in all_datasets:
        dataset_name = d_path.name
        rel_plots_dir = os.path.relpath(d_path / "Plots", target_dir).replace("\\", "/")
        rel_fpca_rank = os.path.relpath(fpca_rank_dir, target_dir).replace("\\", "/")
        rel_fpca_pm = os.path.relpath(fpca_pm_dir, target_dir).replace("\\", "/")

        # ----------------------------------------------------
        # A. FPCA Slides (Rank_10_FPC1_K7) for this dataset
        # ----------------------------------------------------
        fpca_bottom_img = f"{rel_fpca_rank}/curves_by_cluster.png"

        # BR Pair
        fpca_br_ll = f"{rel_fpca_pm}/phase_map_BR_LL_{dataset_name}.png"
        fpca_br_rr = f"{rel_fpca_pm}/phase_map_BR_RR_{dataset_name}.png"
        
        slide_counter += 1
        slides_html.append(f"""
    <div class="slide">
        <div class="slide-header">
            <h3>FPCA Clustering (Rank 10 — FPC1, K7) | {dataset_name} — Right Sweep Pair</h3>
        </div>
        <div class="fpca-top-section">
            <img src="{fpca_br_ll}" alt="BR LL Phase Map">
            <img src="{fpca_br_rr}" alt="BR RR Phase Map">
        </div>
        <div class="fpca-bottom-section">
            <img src="{fpca_bottom_img}" alt="Curves by Cluster">
        </div>
        <div class="footer-left">{dataset_name} | FPCA Rank 10 (BR Pair)</div>
        <div class="footer-right">Slide {slide_counter}</div>
    </div>
        """)

        # BL Pair
        fpca_bl_ll = f"{rel_fpca_pm}/phase_map_BL_LL_{dataset_name}.png"
        fpca_bl_rr = f"{rel_fpca_pm}/phase_map_BL_RR_{dataset_name}.png"

        slide_counter += 1
        slides_html.append(f"""
    <div class="slide">
        <div class="slide-header">
            <h3>FPCA Clustering (Rank 10 — FPC1, K7) | {dataset_name} — Left Sweep Pair</h3>
        </div>
        <div class="fpca-top-section">
            <img src="{fpca_bl_ll}" alt="BL LL Phase Map">
            <img src="{fpca_bl_rr}" alt="BL RR Phase Map">
        </div>
        <div class="fpca-bottom-section">
            <img src="{fpca_bottom_img}" alt="Curves by Cluster">
        </div>
        <div class="footer-left">{dataset_name} | FPCA Rank 10 (BL Pair)</div>
        <div class="footer-right">Slide {slide_counter}</div>
    </div>
        """)

        # ----------------------------------------------------
        # B. Continuous Phase Maps (3-Panel Layout)
        # ----------------------------------------------------
        eff_gap_img = f"{rel_plots_dir}/phase_map_effective_topological_gap.png"
        transport_gap_img = f"{rel_plots_dir}/phase_map_transport_gap.png"
        weight_loc_img = f"{rel_plots_dir}/phase_map_weight_localizations.png"

        slide_counter += 1
        slides_html.append(f"""
    <div class="slide">
        <div class="slide-header">
            <h3>Phase Maps: {dataset_name}</h3>
        </div>
        <div class="three-panel-container">
            <div class="panel">
                <img src="{eff_gap_img}" alt="Effective Topological Gap">
            </div>
            <div class="panel">
                <img src="{transport_gap_img}" alt="Transport Gap">
            </div>
            <div class="panel">
                <img src="{weight_loc_img}" alt="Weight Localizations">
            </div>
        </div>
        <div class="footer-left">{dataset_name} | Continuous Phase Maps (3-Panel)</div>
        <div class="footer-right">Slide {slide_counter}</div>
    </div>
        """)

        # ----------------------------------------------------
        # C. Single Points (ONLY for disorder_realization_6_results)
        # ----------------------------------------------------
        if dataset_name == "disorder_realization_6_results":
            sp_dir = d_path / "Plots" / "single_points"
            if sp_dir.exists():
                sp_folders = sorted([p for p in sp_dir.iterdir() if p.is_dir()])
                for sp_path in sp_folders:
                    point_label = sp_path.name
                    rel_sp = os.path.relpath(sp_path, target_dir).replace("\\", "/")

                    overlap_img = f"{rel_sp}/overlap_plot.png"
                    didv_img = f"{rel_sp}/dIdV.png"
                    cond_img = f"{rel_sp}/Conductances_Combined.png"
                    wave_img = f"{rel_sp}/wavefunction.png"

                    spectra_img = f"{rel_sp}/spectra_cut.png"
                    corr_img = f"{rel_sp}/correlation_cut_R.png"
                    gap_img = f"{rel_sp}/gap_nonlocal_cut.png"
                    peaks_img = f"{rel_sp}/peaks_cut.png"

                    # Single Point Slide A: Breakdown
                    slide_counter += 1
                    slides_html.append(f"""
    <div class="slide">
        <div class="slide-header">
            <h3>Disorder Realization 6 — Single Point Breakdown ({point_label})</h3>
        </div>
        <div class="grid-2x2-container">
            <div class="grid-cell">
                <img src="{overlap_img}" alt="Phase Overlay">
            </div>
            <div class="grid-cell">
                <img src="{didv_img}" alt="Differential Conductance dI/dV">
            </div>
            <div class="grid-cell">
                <img src="{cond_img}" alt="Zero-Bias Conductances">
            </div>
            <div class="grid-cell">
                <img src="{wave_img}" alt="Majorana Wavefunction Density">
            </div>
        </div>
        <div class="footer-left">Disorder Realization 6 | Point {point_label} | Zero-Bias & Phase Overlay</div>
        <div class="footer-right">Slide {slide_counter}</div>
    </div>
                    """)

                    # Single Point Slide B: 1D Cut Diagnostics
                    slide_counter += 1
                    slides_html.append(f"""
    <div class="slide">
        <div class="slide-header">
            <h3>Disorder Realization 6 — 1D Cut Characterization ({point_label})</h3>
        </div>
        <div class="grid-2x2-container">
            <div class="grid-cell">
                <img src="{spectra_img}" alt="Low Energy Spectra along Cut">
            </div>
            <div class="grid-cell">
                <img src="{corr_img}" alt="Right Barrier Sweep Conductance Correlation">
            </div>
            <div class="grid-cell">
                <img src="{gap_img}" alt="Transport vs Topological Gap along Cut">
            </div>
            <div class="grid-cell">
                <img src="{peaks_img}" alt="Peak Width & Height along Cut">
            </div>
        </div>
        <div class="footer-left">Disorder Realization 6 | Point {point_label} | 1D Cut Diagnostics</div>
        <div class="footer-right">Slide {slide_counter}</div>
    </div>
                    """)

    total_slides = slide_counter

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Weekly Research Slides - 2026-07-20</title>
    <style>
        @page {{
            size: 16in 9in;
            margin: 0;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            background-color: #13293d;
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            overflow-x: hidden;
        }}
        .slide {{
            width: 16in;
            height: 9in;
            background-color: #13293d;
            display: flex;
            flex-direction: column;
            position: relative;
            page-break-after: always;
            break-after: page;
            overflow: hidden;
            padding: 0.05in;
        }}
        .title-slide {{
            justify-content: center;
            align-items: center;
            text-align: center;
        }}
        .title-container {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 0.8in 1.2in;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }}
        .main-title {{
            font-size: 42pt;
            font-weight: 700;
            color: #00d2ff;
            margin-bottom: 0.2in;
            letter-spacing: -0.5px;
        }}
        .sub-title {{
            font-size: 24pt;
            font-weight: 400;
            color: #e0e6ed;
            margin-bottom: 0.4in;
        }}
        .meta-info {{
            font-size: 16pt;
            color: #8a99ad;
        }}
        .slide-header {{
            height: 0.35in;
            min-height: 0.35in;
            width: 100%;
            display: flex;
            align-items: center;
            padding-left: 0.05in;
            margin-bottom: 0.02in;
        }}
        .slide-header h3 {{
            font-size: 18pt;
            font-weight: 600;
            color: #00d2ff;
        }}
        /* FPCA Layout - Maximize space */
        .fpca-top-section {{
            flex: 7;
            width: 100%;
            display: flex;
            gap: 0.03in;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }}
        .fpca-top-section img {{
            width: 50%;
            height: 100%;
            object-fit: contain;
        }}
        .fpca-bottom-section {{
            flex: 3;
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            padding-top: 0.02in;
            overflow: hidden;
        }}
        .fpca-bottom-section img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        /* 3-Panel Layout for Continuous Phase Maps - Maximize space */
        .three-panel-container {{
            flex: 1;
            width: 100%;
            display: flex;
            gap: 0.03in;
            justify-content: space-between;
            align-items: center;
            overflow: hidden;
        }}
        .three-panel-container .panel {{
            flex: 1;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }}
        .three-panel-container .panel img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        /* 2x2 Grid Layout for Single Points - Maximize space */
        .grid-2x2-container {{
            flex: 1;
            width: 100%;
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 0.03in;
            overflow: hidden;
        }}
        .grid-cell {{
            width: 100%;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }}
        .grid-cell img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        .footer-left {{
            position: absolute;
            bottom: 0.03in;
            left: 0.1in;
            font-size: 10pt;
            color: #7a8a9e;
            background: rgba(19, 41, 61, 0.85);
            padding: 1px 5px;
            border-radius: 3px;
            z-index: 10;
        }}
        .footer-right {{
            position: absolute;
            bottom: 0.03in;
            right: 0.1in;
            font-size: 10pt;
            color: #7a8a9e;
            background: rgba(19, 41, 61, 0.85);
            padding: 1px 5px;
            border-radius: 3px;
            z-index: 10;
        }}
    </style>
</head>
<body>
{"".join(slides_html)}
</body>
</html>
"""

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Successfully generated HTML slides: {html_file}")
    print(f"Total slides generated: {total_slides}")

if __name__ == "__main__":
    main()
