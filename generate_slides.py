#!/usr/bin/env python3
import os
import sys
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Setup paths
root_dir = Path(__file__).parent.resolve()
sys.path.append(str(root_dir))
import helpers as hp
from post_process import create_overlay_rgba

data_dir = root_dir / "Data" / "Tdis_pfaff4"
slides_dir = root_dir / "slides"
images_dir = slides_dir / "images"

# Ensure directories exist
slides_dir.mkdir(parents=True, exist_ok=True)
images_dir.mkdir(parents=True, exist_ok=True)

# Single points coordinates
single_points = [
    (0.973913, 1.40625),
    (1.095652, 2.290179),
    (1.075362, 3.274554),
    (1.095652, 2.049107),
    (0.933333, 3.375),
    (0.7710145, 3.033482),
    (0.6289855, 2.792411),
    (0.7101449, 2.470982),
    (0.6086957, 2.370536),
    (0.754837, 2.432432),
    (0.8725, 2.4107),
    (0.9130, 3.1339)
]

def get_folder_name(vz, mu):
    return f"Vz{vz:.3f}".replace('.', '') + "_" + f"mu{mu:.3f}".replace('.', '')

# Load simulation parameters and overlay data
print("Loading data for slide generation...")
params_path = data_dir / "all_params.npz"
pdi_data_path = data_dir / "pdi_data.npy"
params_list_path = data_dir / "params_list.npy"
peaks_left_path = data_dir / "peaks_left.npy"
peaks_right_path = data_dir / "peaks_right.npy"
brcl_path = data_dir / "barrier_right_conductance_left_arr.npy"
brcr_path = data_dir / "barrier_right_conductance_right_arr.npy"
blcl_path = data_dir / "barrier_left_conductance_left_arr.npy"
blcr_path = data_dir / "barrier_left_conductance_right_arr.npy"

for p in [params_path, pdi_data_path, params_list_path, peaks_left_path, peaks_right_path,
          brcl_path, brcr_path, blcl_path, blcr_path]:
    if not p.exists():
        print(f"Error: Missing required file {p}. Run post_process.py first.")
        sys.exit(1)

params_config = np.load(params_path, allow_pickle=True)
pdi_data = np.load(pdi_data_path, allow_pickle=True)
params_list = np.load(params_list_path, allow_pickle=True)
peaks_left = np.load(peaks_left_path, allow_pickle=True)
peaks_right = np.load(peaks_right_path, allow_pickle=True)
brcl_all = np.load(brcl_path, allow_pickle=True)
brcr_all = np.load(brcr_path, allow_pickle=True)
blcl_all = np.load(blcl_path, allow_pickle=True)
blcr_all = np.load(blcr_path, allow_pickle=True)

mu = params_list[:, 1]
V_z = params_list[:, 2]

# Calculate I (Topological Winding Number) exactly as in post_process.py
if pdi_data.shape[1] > 3:
    pdi_winding = pdi_data[:, 3]
else:
    pdi_winding = pdi_data[:, 2]
I = hp.filter_pdi(pdi_winding, thresh=0.9)

# Calculate correlations exactly as in post_process.py
print("Computing correlations for Right and Left barrier sweeps...")
corrs_R = np.array([hp.calc_correlation(brcl_all[i], brcr_all[i]) for i in range(brcl_all.shape[0])])
corrs_L = np.array([hp.calc_correlation(blcl_all[i], blcr_all[i]) for i in range(blcl_all.shape[0])])

# Calculate decision map prot_dat exactly as in post_process.py
print("Calculating protocol decision map (prot_dat_R)...")
params = {
    "check_correlation"      : True,
    "check_resonance_peak"   : False,
    "check_negative_peaks"   : False,
    "check_monotonic"        : False,
    "check_peak_symmetry"    : False, 
    "check_peak_window"      : True,
    "check_island_stability" : True,
    
    "corr_thresh"     : 0.7,
    "window"          : 0.03,
    "stability_radius" : 0.025,
    "stability_frac"  : 1.0
}

prot_dat_R = hp.calc_protocol_v3(corrs_R, peaks_left, peaks_right, None, pdi_data, params)

# Build exact RGBA grid from post_process.py
unique_mu, unique_vz, rgba = create_overlay_rgba(mu, V_z, I, prot_dat_R)

# Single point sweep properties
try:
    barrier_l = float(params_config['barrier0'])
except Exception:
    try:
        barrier_l = float(params_config['barrier_l'])
    except Exception:
        barrier_l = 2.0

num_sweep_points = 100
barrier_sweep = np.linspace(-70 * barrier_l, 70 * barrier_l, num_sweep_points)
x_data = barrier_sweep / (barrier_l if barrier_l != 0 else 1.0)
idx_sym = np.argmin(np.abs(barrier_sweep - barrier_l))

slides_data = []

print("Generating slide assets...")
for pt_idx, (vz_val, mu_val) in enumerate(single_points):
    folder_name = get_folder_name(vz_val, mu_val)
    pt_dir = data_dir / "Plots" / "Single_Points" / folder_name
    
    if not pt_dir.exists():
        print(f"Warning: Single point folder {pt_dir} does not exist. Skipping.")
        continue

    # 1. Generate Highlighted Point Map exact copy of post_process.py style
    fig_map, ax_map = plt.subplots(figsize=(6, 8), dpi=150)
    ax_map.imshow(rgba, origin='lower', extent=[unique_vz[0], unique_vz[-1], unique_mu[0], unique_mu[-1]], aspect='auto')
    
    pts_array = np.array(single_points, dtype=float)
    ax_map.scatter(pts_array[:, 0], pts_array[:, 1], color='black', edgecolor='white', s=50, zorder=5)
    
    # Highlight current single point clearly
    ax_map.scatter(vz_val, mu_val, color='gold', edgecolor='black', marker='*', s=220, zorder=6, label='Current Point')

    ax_map.set_title(f'Topological Region vs Right Sweep Protocol (Point {pt_idx+1})', fontsize=12, pad=15)
    ax_map.set_xlabel(r'Zeeman Field $V_z$', fontsize=11)
    ax_map.set_ylabel(r'Chemical Potential $\mu$', fontsize=11)
    ax_map.set_xlim(0.0, 1.2)
    ax_map.set_ylim(0.0, 4.5)
    ax_map.spines['top'].set_visible(False)
    ax_map.spines['right'].set_visible(False)

    red_patch = mpatches.Patch(color=(1.0, 0.5, 0.5), label='Right Protocol Positive (prot_dat = 1)')
    blue_patch = mpatches.Patch(color=(0.5, 0.5, 1.0), label='Topological (I = 1)')
    purple_patch = mpatches.Patch(color=(0.6, 0.25, 0.6), label='Overlap (Both = 1)')
    pts_handle = mlines.Line2D([], [], color='black', marker='o', markerfacecolor='black', markeredgecolor='white', markersize=7, linestyle='None', label='Single Points')
    star_handle = mlines.Line2D([], [], color='gold', marker='*', markerfacecolor='gold', markeredgecolor='black', markersize=10, linestyle='None', label='Current Point')

    ax_map.legend(handles=[red_patch, blue_patch, purple_patch, pts_handle, star_handle], bbox_to_anchor=(0.5, -0.15),
                  loc='upper center', ncol=2, fontsize=9, frameon=True)

    fig_map.tight_layout()
    fig_map.subplots_adjust(bottom=0.22)
    map_out = images_dir / f"point_map_R_{pt_idx}.png"
    fig_map.savefig(map_out, dpi=300, bbox_inches='tight')
    plt.close(fig_map)

    # 2. Generate Combined Barrier Sweep Plot (2x2 Grid) matching post_process.py clean style
    cond_left_L = np.load(pt_dir / "cond_left_LeftSweep.npy")
    cond_right_L = np.load(pt_dir / "cond_right_LeftSweep.npy")
    cond_left_R = np.load(pt_dir / "cond_left_RightSweep.npy")
    cond_right_R = np.load(pt_dir / "cond_right_RightSweep.npy")

    normed_GL_L = cond_left_L / (cond_left_L[idx_sym] if cond_left_L[idx_sym] != 0 else 1.0)
    normed_GR_L = cond_right_L / (cond_right_L[idx_sym] if cond_right_L[idx_sym] != 0 else 1.0)
    normed_GL_R = cond_left_R / (cond_left_R[idx_sym] if cond_left_R[idx_sym] != 0 else 1.0)
    normed_GR_R = cond_right_R / (cond_right_R[idx_sym] if cond_right_R[idx_sym] != 0 else 1.0)

    fig_sweeps, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
    lw = 2.5

    # Left Sweep GL
    ax = axes[0, 0]
    ax.plot(x_data, normed_GL_L, color='green', linewidth=lw)
    ax.set_title(rf"Left Sweep $G_{{LL}}$ ($V_z={vz_val:.3f}, \mu={mu_val:.3f}$)", fontsize=11)
    ax.set_xlabel(r"$U_{L}/U_{R}$", fontsize=11)
    ax.set_ylabel(r"$G_{LL}/G_{LL, sym}$", fontsize=11)

    # Left Sweep GR
    ax = axes[0, 1]
    ax.plot(x_data, normed_GR_L, color='green', linewidth=lw)
    ax.set_title(rf"Left Sweep $G_{{RR}}$ ($V_z={vz_val:.3f}, \mu={mu_val:.3f}$)", fontsize=11)
    ax.set_xlabel(r"$U_{L}/U_{R}$", fontsize=11)
    ax.set_ylabel(r"$G_{RR}/G_{RR, sym}$", fontsize=11)

    # Right Sweep GL
    ax = axes[1, 0]
    ax.plot(x_data, normed_GL_R, color='green', linewidth=lw)
    ax.set_title(rf"Right Sweep $G_{{LL}}$ ($V_z={vz_val:.3f}, \mu={mu_val:.3f}$)", fontsize=11)
    ax.set_xlabel(r"$U_{R}/U_{L}$", fontsize=11)
    ax.set_ylabel(r"$G_{LL}/G_{LL, sym}$", fontsize=11)

    # Right Sweep GR
    ax = axes[1, 1]
    ax.plot(x_data, normed_GR_R, color='green', linewidth=lw)
    ax.set_title(rf"Right Sweep $G_{{RR}}$ ($V_z={vz_val:.3f}, \mu={mu_val:.3f}$)", fontsize=11)
    ax.set_xlabel(r"$U_{R}/U_{L}$", fontsize=11)
    ax.set_ylabel(r"$G_{RR}/G_{RR, sym}$", fontsize=11)

    # Format all axes matching post_process.py clean layout
    for ax in axes.flat:
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig_sweeps.suptitle("Conductance (Barrier Sweeps)", fontsize=14, y=0.98)
    fig_sweeps.tight_layout()
    fig_sweeps.subplots_adjust(top=0.90)
    
    sweeps_out = images_dir / f"barrier_sweeps_{pt_idx}.png"
    fig_sweeps.savefig(sweeps_out, dpi=300, bbox_inches='tight')
    plt.close(fig_sweeps)

    # 3. Copy dIdV and wavefunction plots to slides/images folder
    dest_didv = images_dir / f"dIdV_{pt_idx}.png"
    dest_wf = images_dir / f"wavefunction_{pt_idx}.png"
    
    shutil.copy(pt_dir / "dIdV.png", dest_didv)
    shutil.copy(pt_dir / "wavefunction.png", dest_wf)

    # 4. Save metadata for HTML rendering
    slides_data.append({
        "index": pt_idx,
        "vz": f"{vz_val:.4f}",
        "mu": f"{mu_val:.4f}",
        "map_img": f"images/point_map_R_{pt_idx}.png",
        "didv_img": f"images/dIdV_{pt_idx}.png",
        "sweeps_img": f"images/barrier_sweeps_{pt_idx}.png",
        "wf_img": f"images/wavefunction_{pt_idx}.png"
    })

# 5. Write index.html with interactive slides
print("Generating HTML slide deck...")

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Single Point Analysis Slides</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: #f1f5f9;
            color: #0f172a;
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        
        /* Navigation header */
        header {
            background-color: #ffffff;
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            z-index: 100;
        }
        h1 {
            font-size: 1.1rem;
            font-weight: 600;
            color: #0f172a;
        }
        .controls {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .nav-btn {
            background-color: #f8fafc;
            border: 1px solid #cbd5e1;
            color: #334155;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        .nav-btn:hover {
            background-color: #e2e8f0;
            color: #0f172a;
        }
        select {
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            color: #334155;
            padding: 6px 12px;
            border-radius: 4px;
            outline: none;
            cursor: pointer;
        }

        /* Presentation area */
        .presentation {
            flex: 1;
            position: relative;
            background-color: #f1f5f9;
        }
        .slide {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: none;
            flex-direction: column;
            padding: 20px 40px;
            background-color: #f1f5f9;
        }
        .slide.active {
            display: flex;
        }

        /* Slide title */
        .slide-title {
            font-size: 1.6rem;
            margin-bottom: 18px;
            border-left: 5px solid #3b82f6;
            padding-left: 12px;
            font-weight: 700;
            color: #1e293b;
            letter-spacing: 0.3px;
        }

        /* Dashboard layout (2x2 grid) */
        .dashboard {
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 20px;
            min-height: 0; /* allows grid items to shrink */
        }
        .panel {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
            min-height: 0;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        }
        .panel h3 {
            position: absolute;
            top: 10px;
            left: 15px;
            font-size: 0.88rem;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 600;
        }
        .panel img {
            max-width: 100%;
            max-height: 90%;
            object-fit: contain;
            border-radius: 4px;
            margin-top: 15px;
        }

        /* Footer info */
        footer {
            background-color: #ffffff;
            padding: 8px 20px;
            font-size: 0.8rem;
            color: #64748b;
            text-align: center;
            border-top: 1px solid #e2e8f0;
        }
    </style>
</head>
<body>

    <header>
        <h1>Majorana Wire Single Point Analysis (Tdis_pfaff4)</h1>
        <div class="controls">
            <button class="nav-btn" onclick="prevSlide()">◀ Prev</button>
            <select id="slide-select" onchange="jumpToSlide(this.value)">
"""

for item in slides_data:
    idx = item["index"]
    html_content += f'                <option value="{idx}">Point {idx+1}: Vz={item["vz"]}, mu={item["mu"]}</option>\n'

html_content += """            </select>
            <button class="nav-btn" onclick="nextSlide()">Next ▶</button>
        </div>
    </header>

    <div class="presentation">
"""

for item in slides_data:
    idx = item["index"]
    is_active = "active" if idx == 0 else ""
    html_content += f"""        <div class="slide {is_active}" id="slide-{idx}">
            <div class="slide-title">Single Point Analysis &mdash; V<sub>z</sub> = {item["vz"]} meV, &mu; = {item["mu"]} meV</div>
            <div class="dashboard">
                <div class="panel">
                    <h3>2D Phase Map Location</h3>
                    <img src="{item["map_img"]}" alt="2D Location">
                </div>
                <div class="panel">
                    <h3>Differential Conductance (dI/dV)</h3>
                    <img src="{item["didv_img"]}" alt="dI/dV Spectrum">
                </div>
                <div class="panel">
                    <h3>Conductance (Barrier Sweeps)</h3>
                    <img src="{item["sweeps_img"]}" alt="Conductance Sweeps">
                </div>
                <div class="panel">
                    <h3>MZM Localization (Psi² vs Position)</h3>
                    <img src="{item["wf_img"]}" alt="MZM Wavefunction">
                </div>
            </div>
        </div>
"""

html_content += """    </div>

    <footer>
        Use Left/Right arrow keys or drop-down menu to navigate. Press Ctrl+P to print or save presentation as PDF.
    </footer>

    <script>
        let currentSlide = 0;
        const totalSlides = """ + str(len(slides_data)) + """;
        const selectEl = document.getElementById('slide-select');

        function showSlide(index) {
            document.querySelectorAll('.slide').forEach(s => s.classList.remove('active'));
            const target = document.getElementById(`slide-${index}`);
            if (target) {
                target.classList.add('active');
            }
            currentSlide = index;
            selectEl.value = index;
        }

        function nextSlide() {
            let next = (currentSlide + 1) % totalSlides;
            showSlide(next);
        }

        function prevSlide() {
            let prev = (currentSlide - 1 + totalSlides) % totalSlides;
            showSlide(prev);
        }

        function jumpToSlide(index) {
            showSlide(parseInt(index));
        }

        // Keyboard Navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === ' ') {
                nextSlide();
            } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                prevSlide();
            }
        });
    </script>
</body>
</html>
"""

with open(slides_dir / "index.html", "w") as f:
    f.write(html_content)

print(f"Successfully generated HTML slide deck at: {slides_dir / 'index.html'}")
print("All slide assets are stored in: slides/images/")
