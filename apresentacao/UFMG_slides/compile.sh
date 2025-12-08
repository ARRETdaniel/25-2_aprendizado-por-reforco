#!/bin/bash
# LaTeX Presentation Compilation Script
# Created for: Template Latex - Apresentacao - IFSP - SBV.tex
# Author: Daniel Terra Gomes

cd "$(dirname "$0")"

echo "Cleaning auxiliary files..."
rm -f *.aux *.log *.nav *.out *.toc *.snm *.bbl *.blg

echo "First pdflatex pass..."
pdflatex -interaction=nonstopmode "Template Latex - Apresentacao - IFSP - SBV.tex"

echo "Processing bibliography..."
bibtex "Template Latex - Apresentacao - IFSP - SBV"

echo "Second pdflatex pass (resolving references)..."
pdflatex -interaction=nonstopmode "Template Latex - Apresentacao - IFSP - SBV.tex"

echo "Third pdflatex pass (final)..."
pdflatex -interaction=nonstopmode "Template Latex - Apresentacao - IFSP - SBV.tex"

echo ""
echo "Compilation complete!"
ls -lh "Template Latex - Apresentacao - IFSP - SBV.pdf"
