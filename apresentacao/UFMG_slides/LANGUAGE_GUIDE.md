# Language Configuration Guide

## Switching Between English and Portuguese

The presentation supports both English and Portuguese throughout the entire document.

### How to Change Language

Edit the file: `Template Latex - Apresentacao - IFSP - SBV.tex`

Find this line (around line 8):
```latex
\newcommand{\presentationlang}{EN}  % Change to PT for Portuguese
```

**For English:**
```latex
\newcommand{\presentationlang}{EN}
```

**For Portuguese:**
```latex
\newcommand{\presentationlang}{PT}
```

### What Changes

| Element | English (EN) | Portuguese (PT) |
|---------|--------------|-----------------|
| **Title Page Labels** | | |
| Course | Course: | Curso: |
| Student | Student: | Aluno(a): |
| Professor | Professor: | Professor(a): |
| Advisor | Advisor: | Orientador(a): |
| Co-advisor | Co-advisor: | Coorientador(a): |
| Date Format | Month Day, Year | Day de Month de Year |
| **Content Labels** | | |
| Figure | Figure | Figura |
| Table | Table | Tabela |
| Contents/Index | Contents | Conteúdo |

### Example Title Page

**English (EN):**
```
Course: Aprendizado por Reforço
Student: Daniel Terra Gomes
Professor: Luiz Chaimowicz
Belo Horizonte, November 27, 2025
```

**Portuguese (PT):**
```
Curso: Aprendizado por Reforço
Aluno(a): Daniel Terra Gomes
Professor(a): Luiz Chaimowicz
Belo Horizonte, 27 de November de 2025
```

### Example Figure Caption

**English (EN):**
```
Figure: RL agent-environment interaction loop
```

**Portuguese (PT):**
```
Figura: Ciclo de interação agente-ambiente em RL
```

### After Making Changes

Recompile the presentation:
```bash
bash compile.sh
```

Or use the full command:
```bash
pdflatex -interaction=nonstopmode "Template Latex - Apresentacao - IFSP - SBV.tex" && bibtex "Template Latex - Apresentacao - IFSP - SBV" && pdflatex -interaction=nonstopmode "Template Latex - Apresentacao - IFSP - SBV.tex" && pdflatex -interaction=nonstopmode "Template Latex - Apresentacao - IFSP - SBV.tex"
```

---

## Technical Details

The language system works by:
1. Defining a `\presentationlang` command in the main template file
2. Calling `\setpresentationlang{\presentationlang}` which configures all labels
3. Overriding Babel's Portuguese caption defaults when EN is selected
4. Using conditional formatting for the date (Month/Day vs Day/Month)
