#!/usr/bin/env python3
"""
Script to generate API documentation using Sphinx.

Usage:
    python generate_docs.py          # Generate HTML docs
    python generate_docs.py --pdf    # Generate PDF (requires LaTeX)
    python generate_docs.py --serve  # Generate and serve locally
"""

import subprocess
import sys
import os
from pathlib import Path

DOCS_DIR = Path(__file__).parent
PROJECT_ROOT = DOCS_DIR.parent
API_DIR = DOCS_DIR / "api"


def check_sphinx():
    """Check if Sphinx is installed."""
    try:
        import sphinx
        return True
    except ImportError:
        print("Sphinx not installed. Install with:")
        print("  pip install sphinx sphinx-rtd-theme")
        return False


def generate_apidoc():
    """Generate .rst files from source code."""
    print("Generating API documentation from source...")
    
    # Run sphinx-apidoc to generate module documentation
    subprocess.run([
        sys.executable, "-m", "sphinx.ext.apidoc",
        "-o", str(API_DIR),
        "-f",  # Force overwrite
        "-e",  # Separate pages for each module
        "-M",  # Module first
        str(PROJECT_ROOT / "synthetic_experiments"),
        # Exclude patterns
        str(PROJECT_ROOT / "synthetic_experiments" / "__pycache__"),
    ], check=True)


def build_html():
    """Build HTML documentation."""
    print("Building HTML documentation...")
    
    build_dir = API_DIR / "_build" / "html"
    
    subprocess.run([
        sys.executable, "-m", "sphinx.cmd.build",
        "-b", "html",
        str(API_DIR),
        str(build_dir),
    ], check=True)
    
    print(f"\nDocumentation built at: {build_dir / 'index.html'}")
    return build_dir


def build_pdf():
    """Build PDF documentation (requires LaTeX)."""
    print("Building PDF documentation...")
    
    build_dir = API_DIR / "_build" / "latex"
    
    subprocess.run([
        sys.executable, "-m", "sphinx.cmd.build",
        "-b", "latex",
        str(API_DIR),
        str(build_dir),
    ], check=True)
    
    # Run make in latex directory
    subprocess.run(["make"], cwd=build_dir, check=True)
    
    print(f"\nPDF built at: {build_dir}")


def serve_docs(port: int = 8000):
    """Serve documentation locally."""
    build_dir = build_html()
    
    print(f"\nServing documentation at http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    
    os.chdir(build_dir)
    subprocess.run([sys.executable, "-m", "http.server", str(port)])


def main():
    if not check_sphinx():
        sys.exit(1)
    
    args = sys.argv[1:]
    
    if "--serve" in args:
        generate_apidoc()
        serve_docs()
    elif "--pdf" in args:
        generate_apidoc()
        build_pdf()
    else:
        generate_apidoc()
        build_html()


if __name__ == "__main__":
    main()
