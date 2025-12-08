#!/bin/bash
# LaTeX Presentation Compilation Script
# Created for: Template Latex - Apresentacao - IFSP - SBV.tex
# Author: Daniel Terra Gomes
# bash ./compile.sh
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode "Template Latex - Apresentacao - IFSP - SBV.tex" && bibtex "Template Latex - Apresentacao - IFSP - SBV" && pdflatex -interaction=nonstopmode "Template Latex - Apresentacao - IFSP - SBV.tex" && pdflatex -interaction=nonstopmode "Template Latex - Apresentacao - IFSP - SBV.tex"

echo ""
echo "Compilation complete!"
ls -lh "Template Latex - Apresentacao - IFSP - SBV.pdf"
