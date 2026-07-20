import os
import glob
import re
import subprocess

def get_sorted_datasets(pm_dir):
    files = glob.glob(os.path.join(pm_dir, "phase_map_BR_LL_*.png"))
    datasets = []
    for f in files:
        basename = os.path.basename(f)
        # prefix is phase_map_BR_LL_
        dataset_name = basename.replace("phase_map_BR_LL_", "").replace(".png", "")
        datasets.append(dataset_name)
    
    # Sort: disorder_realization_X_results sorted numerically, then Tdis_pfaff4
    def sort_key(name):
        match = re.search(r'disorder_realization_(\d+)_results', name)
        if match:
            return (0, int(match.group(1)))
        elif name == "Tdis_pfaff4":
            return (1, 0)
        else:
            return (2, name)
            
    return sorted(datasets, key=sort_key)

def generate_html_for_rank(rank_folder, rank_title, plots_base, output_dir):
    rank_plots_dir = os.path.join(plots_base, rank_folder)
    pm_dir = os.path.join(rank_plots_dir, "phase_maps_by_realization")
    datasets = get_sorted_datasets(pm_dir)
    
    # Relative path from output_dir (slides/FPCA_slides/Rank_XX) to project root is ../../../
    # So relative path to plots is ../../../Plots/FPCA_dis_realizations_maxnorm/Rank_XX/...
    rel_rank_plots = f"../../../{plots_base}/{rank_folder}"
    
    slides_html = []
    slide_count = 0
    total_slides = len(datasets) * 2
    
    for dataset in datasets:
        # We create 2 slides per dataset: BR pair and BL pair
        pairs = [
            ("BR", "BR_LL", "BR_RR", "BR Pair"),
            ("BL", "BL_LL", "BL_RR", "BL Pair")
        ]
        for pair_code, left_curve, right_curve, pair_label in pairs:
            slide_count += 1
            left_img = f"{rel_rank_plots}/phase_maps_by_realization/phase_map_{left_curve}_{dataset}.png"
            right_img = f"{rel_rank_plots}/phase_maps_by_realization/phase_map_{right_curve}_{dataset}.png"
            bottom_img = f"{rel_rank_plots}/curves_by_cluster.png"
            
            # Subtle identifier in bottom-left so no top header takes up image volume
            identifier = f"{dataset} | {pair_label}"
            
            slide_html = f"""
    <div class="slide">
        <div class="top-section">
            <img src="{left_img}" alt="{left_curve}">
            <img src="{right_img}" alt="{right_curve}">
        </div>
        <div class="bottom-section">
            <img src="{bottom_img}" alt="Curves by Cluster">
        </div>
        <div class="footer-left">{identifier}</div>
        <div class="footer-right">{slide_count} / {total_slides}</div>
    </div>"""
            slides_html.append(slide_html)
            
    all_slides_content = "\n".join(slides_html)
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{rank_folder} Slides</title>
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
            padding: 0.25in;
        }}
        .top-section {{
            height: 75%;
            width: 100%;
            display: flex;
            gap: 0.2in;
            justify-content: center;
            align-items: center;
        }}
        .top-section img {{
            width: calc(50% - 0.1in);
            height: 100%;
            object-fit: contain;
        }}
        .bottom-section {{
            height: 25%;
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            padding-top: 0.15in;
        }}
        .bottom-section img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        .footer-left {{
            position: absolute;
            bottom: 0.08in;
            left: 0.25in;
            font-size: 14px;
            color: rgba(255, 255, 255, 0.45);
            pointer-events: none;
        }}
        .footer-right {{
            position: absolute;
            bottom: 0.08in;
            right: 0.25in;
            font-size: 14px;
            color: rgba(255, 255, 255, 0.45);
            pointer-events: none;
        }}
        @media screen {{
            body {{
                background-color: #0b1722;
                padding: 30px;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 40px;
            }}
            .slide {{
                box-shadow: 0 15px 35px rgba(0,0,0,0.6);
                border-radius: 8px;
            }}
        }}
        @media print {{
            body {{
                background-color: #13293d;
                padding: 0;
                display: block;
            }}
            .slide {{
                box-shadow: none;
                border-radius: 0;
                width: 16in;
                height: 9in;
            }}
        }}
    </style>
</head>
<body>
{all_slides_content}
</body>
</html>
"""
    
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"{rank_folder}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    
    return html_path, len(datasets), total_slides

def convert_to_pdf_edge(html_path, pdf_path):
    try:
        # Convert paths to windows paths via wslpath for msedge.exe
        win_html = subprocess.check_output(['wslpath', '-w', os.path.abspath(html_path)]).decode().strip()
        win_pdf = subprocess.check_output(['wslpath', '-w', os.path.abspath(pdf_path)]).decode().strip()
        
        file_url = f"file:///{win_html.replace(chr(92), '/')}"
        
        edge_path = "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
        if not os.path.exists(edge_path):
            edge_path = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
            
        cmd = [
            edge_path,
            "--headless=new",
            "--allow-file-access-from-files",
            f"--print-to-pdf={win_pdf}",
            "--no-pdf-header-footer",
            file_url
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0 and os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0
    except Exception as e:
        print(f"Error converting {html_path} to PDF: {e}")
        return False

def main():
    plots_base = "Plots/FPCA_dis_realizations_maxnorm"
    output_base = "slides/FPCA_slides"
    
    ranks = [
        ("Rank_07_FPC1_K4", "Rank 7 (FPC=1, K=4)"),
        ("Rank_08_FPC1_K5", "Rank 8 (FPC=1, K=5)"),
        ("Rank_09_FPC1_K6", "Rank 9 (FPC=1, K=6)"),
        ("Rank_10_FPC1_K7", "Rank 10 (FPC=1, K=7)")
    ]
    
    print("Starting FPCA slide generation...")
    
    for rank_folder, rank_title in ranks:
        output_dir = os.path.join(output_base, rank_folder)
        html_path, num_datasets, total_slides = generate_html_for_rank(
            rank_folder, rank_title, plots_base, output_dir
        )
        print(f"Generated HTML for {rank_folder}: {total_slides} slides across {num_datasets} datasets.")
        
        pdf_path = os.path.join(output_dir, f"{rank_folder}.pdf")
        print(f"Converting {html_path} -> {pdf_path} using Edge Headless...")
        success = convert_to_pdf_edge(html_path, pdf_path)
        if success:
            print(f"  [SUCCESS] Created PDF: {pdf_path} ({os.path.getsize(pdf_path) / (1024*1024):.2f} MB)")
        else:
            print(f"  [FAILED] Failed to create PDF for {rank_folder}")

if __name__ == "__main__":
    main()
