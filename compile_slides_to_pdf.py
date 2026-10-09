#!/usr/bin/env python3
"""
compile_slides_to_pdf.py

Converts all generated HTML slide presentations into high-resolution 16:9 PDF files
using headless Google Chrome. Places each .pdf in the exact same directory as its .html counterpart.
"""

import sys
import os
import subprocess
from pathlib import Path
import time

def compile_html_to_pdf(html_path: Path) -> Path:
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
    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    dt = time.time() - t0
    if res.returncode == 0 and pdf_path.exists():
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"[OK] {html_path} -> {pdf_path.name} ({size_mb:.2f} MB, {dt:.1f}s)")
        return pdf_path
    else:
        print(f"[ERROR] Failed to compile {html_path}: {res.stderr}")
        return None

def main():
    plots_root = Path("Outputs/Plots").resolve()
    
    # Priority targets
    targets = [
        # ROI 3 slides
        plots_root / "roi3_slides.html",
        # TGP Cuts slides
        plots_root / "tgp_cuts_slides.html",
        # All disorder master slides
        plots_root / "all_disorder_slides.html",
    ]
    
    # Standalone ROI3 decks
    for pdir in sorted(plots_root.glob("*_Plots")):
        roi3_deck = pdir / "ROI3_Cuts" / "slides.html"
        if roi3_deck.exists():
            targets.append(roi3_deck)
            
    # Standalone TGP decks
    for pdir in sorted(plots_root.glob("*_Plots")):
        tgp_deck = pdir / "TGP_Cuts" / "slides.html"
        if tgp_deck.exists():
            targets.append(tgp_deck)

    # Standalone cut decks
    for pdir in sorted(plots_root.glob("*_Plots")):
        cuts_deck = pdir / "slides.html"
        if cuts_deck.exists():
            targets.append(cuts_deck)

    print(f"Discovered {len(targets)} HTML slide decks to compile.")
    success_count = 0
    for t in targets:
        # Check if already up-to-date
        pdf_f = t.with_suffix(".pdf")
        if pdf_f.exists() and pdf_f.stat().st_mtime > t.stat().st_mtime:
            print(f"[SKIP] {pdf_f.name} is already newer than {t.name}")
            success_count += 1
            continue
        res = compile_html_to_pdf(t)
        if res:
            success_count += 1

    print(f"\nCompilation complete: {success_count}/{len(targets)} slide decks successfully converted to PDF.")

if __name__ == "__main__":
    main()
