# Citation System Implementation - Summary

## ✅ What Was Changed

### **Before:**
- Hard-coded references in slide text (e.g., "Sutton & Barto (2018)")
- Manual reference lists at the bottom of slides
- Inconsistent citation formatting
- No automatic bibliography generation

### **After:**
- Proper BibTeX `\cite{}` commands throughout the presentation
- Automatic bibliography generation
- Only cited references appear in the References slide
- Consistent ABNT citation style
- All references now in black color (not red)

---

## 📚 References Cited in Presentation

The following references are automatically included because they are cited:

1. **Sutton & Barto (2018)** - Reinforcement Learning textbook `\cite{sutton2018reinforcement}`
2. **Pérez-Gil et al. (2022)** - DRL for autonomous vehicles `\cite{perezgil2022deep}`
3. **Fujimoto et al. (2018)** - TD3 algorithm `\cite{fujimoto2018addressing}`
4. **Pérez-Gil et al. (2021)** - ROS+CARLA framework `\cite{perezgil2021deep}`
5. **CARLA Documentation (2023)** - ROS Bridge `\cite{carla2023documentation}`
6. **Mnih et al. (2015)** - DQN Nature paper `\cite{mnih2015humanlevel}`
7. **Elallid et al. (2023)** - Intersection navigation `\cite{elallid2023deep}`
8. **Chen et al. (2020)** - Interpretable hierarchical RL `\cite{chen2020interpretableendtoendurbanautonomous}`

---

## 🎯 How the System Works

### **In Slide Content:**
Use `\cite{reference_key}` where you want a citation:
```latex
\textbf{Solution: Twin Delayed DDPG (TD3)} \cite{fujimoto2018addressing}
```

This automatically:
- Adds a citation number in the slide (e.g., [1])
- Includes the full reference in the References slide at the end
- Links the citation to the reference (clickable in PDF)

### **Multiple Citations:**
```latex
\cite{sutton2018reinforcement,perezgil2022deep}
```

### **References Slide:**
Located in `00Referencias.tex`:
- Automatically generated from `referencias.bib`
- Only shows references that were actually cited
- Uses `[allowframebreaks]` to split across multiple slides if needed
- Formatted according to ABNT standards

---

## 📝 Files Modified

1. **02Introducao.tex** - Added `\cite` for Sutton, Perez-Gil, Fujimoto
2. **03RevisaoLiteraturaConsideracoesGerais.tex** - Removed hard-coded references
3. **04Metodologia.tex** - Added `\cite` for Mnih, Elallid, Perez-Gil, CARLA
4. **06Conclusao.tex** - Added `\cite` for Chen
5. **00Referencias.tex** - Changed to visible references slide with proper formatting
6. **referencias.bib** - Added missing Mnih et al. (2015) reference
7. **Template Latex - Apresentacao - IFSP - SBV.tex** - Added hyperref color configuration

---

## 🔧 How to Add New Citations

### **Step 1: Add to referencias.bib**
```bibtex
@article{newauthor2025,
  title={Title of the Paper},
  author={Author, First and Second, Name},
  journal={Journal Name},
  year={2025},
  volume={10},
  pages={1--10}
}
```

### **Step 2: Cite in Your Slide**
```latex
\cite{newauthor2025}
```

### **Step 3: Recompile**
```bash
bash compile.sh
```

The reference will automatically appear in the References slide!

---

## ✨ Benefits

✅ **Automatic:** No need to manually maintain reference lists  
✅ **Consistent:** All citations follow ABNT format automatically  
✅ **Accurate:** No typos or formatting errors in references  
✅ **Complete:** Only cited references appear (no clutter)  
✅ **Professional:** Standard academic citation practice  
✅ **Maintainable:** Easy to add, remove, or update references  
✅ **Linked:** Citations are clickable (PDF navigation)  

---

## 📊 Current Stats

- **Total Pages:** 28 (includes 2 pages of references)
- **References Cited:** 8 unique references
- **Reference Pages:** 2 (automatically split with `allowframebreaks`)
- **PDF Size:** 6.2 MB

---

## 🎨 Color Configuration

All citations and references now appear in **black** for consistency:
- `linkcolor=black` - Internal links
- `citecolor=black` - Citations
- `urlcolor=blue` - URLs (for visual distinction)

This prevents the red/black color mixing that occurred before.
