# KIT / Sebastian Krumscheid — final interview presentation

**Nine main slides, estimated 13:15; three technical backups.**

Open `Sebastian_Research_Interview.pdf` for the complete deck. Use `Sebastian_Research_Interview_Main_Only.pdf` to stop after the research question. In the PowerPoint file the three backups are hidden during the normal slideshow; unhide or jump to them for discussion.

| File | Purpose |
|---|---|
| `Sebastian_Research_Interview.pdf` | Final 12-page presentation, including backups. |
| `Sebastian_Research_Interview_Main_Only.pdf` | Nine main slides only. |
| `Sebastian_Research_Interview.pptx` | Editable text and diagrams; full scripts/Q&A embedded in notes. |
| `Sebastian_Research_Interview.tex` | Real Beamer/TikZ source, including editable mathematical expressions. |
| `Speaker_Notes.md` | Complete script, timing, preparation details, mathematical terms and technical Q&A. |
| `Sources_and_Evidence.md` | Slide-by-slide experiment, code, commit and primary-literature mapping. |
| `Presentation_Validation.md` | Final evidence, mathematical, rendering and preservation checks. |
| `rendered/Contact_Sheet.png` | Screenshot overview of every slide. |
| `validation/checks.json` | Machine-readable validation results. |

The original V4 title page is preserved as a vector PDF page. The original editable first-slide objects are retained in PowerPoint. No original research source, checkpoint, experiment record or V4 file was overwritten. The original aircraft photograph is contextual artwork from V4, not a project experiment.

## Editing

PowerPoint text and diagram boxes are native objects. Equations are vector SVG pictures with PNG fallbacks, **not native Office equation objects**. Edit their TeX expressions in `build_presentation.py` / `assets/math/equations.tex`, then regenerate them. Numerical charts are image/vector assets with the original experimental data and plotting code supplied. The deck is not made of whole-slide raster images.

For direct PDF-only edits, edit `Sebastian_Research_Interview.tex` and compile it. Rebuilding with `build_presentation.py` overwrites that generated TeX file. For synchronized PDF/PPTX edits, edit `deck_content()` in the generator. The original V4 first slide is intentionally excluded from redesign.

Edit scripts in `write_notes.py` and run it before rebuilding the deck. This updates Markdown notes and the JSON used for embedded PowerPoint notes.

## Rebuilding

Requirements: Python with NumPy, Matplotlib, Pillow and lxml; Tectonic (or a compatible XeLaTeX Beamer installation); Poppler (`pdfinfo`, `pdfseparate`, `pdfunite`, `pdftoppm`, `pdftocairo`); Liberation Sans fonts. LibreOffice is needed only to review a native PowerPoint render. Existing equation assets are already supplied, so a text/layout rebuild does not require regenerating every equation.

Run from this directory:

```sh
python write_notes.py
python build_presentation.py --build
tectonic --keep-logs Sebastian_Research_Interview.tex
python finalize_presentation.py
```

Finalization explicitly inserts the original vector first page, creates the main-only PDF and renders every PDF slide. It expects a newly compiled 12-page Beamer PDF. `assets/V4_editable_template.pptx` and `assets/V4_landing-1.pdf` make this source bundle independent of the surrounding repository for rebuilding. The evidence links and original-input validation still refer to that repository.

If equations or plots change, first regenerate assets:

```sh
python build_presentation.py --prepare
tectonic --keep-logs --outdir assets/math assets/math/equations.tex
python build_presentation.py --build
tectonic --keep-logs Sebastian_Research_Interview.tex
python finalize_presentation.py
```

The generator refreshes equation SVG/PNG caches when their compiled PDF is newer. It reads only the supplied experiment records and does not run the numerical solver or training.

For independent PowerPoint review, export all slides, including hidden ones:

```sh
libreoffice --headless --convert-to 'pdf:impress_pdf_Export:{"ExportHiddenSlides":{"type":"boolean","value":"true"}}' --outdir /tmp/kit-native Sebastian_Research_Interview.pptx
python finalize_presentation.py --native-render /tmp/kit-native/Sebastian_Research_Interview.pdf
python validate_presentation.py
```

Recompile Beamer immediately before finalization when repeating this workflow. The validation script compares copied metrics to repository records and checks the original title slide in both outputs. On a standalone downloaded bundle, checks requiring the surrounding repository must be evaluated in the original checkout; local PDF/PPTX render inspection remains available.

## Scientific boundaries

The deck distinguishes coefficient WENO-NN, Keras NN-TENO, historical JAX and later gated V2. Observations, candidate mechanisms, attempted fixes and established but unimplemented remedies are kept separate. Positivity-preserving limiting and the cited RKDG–SIAC analysis are not claimed as achievements of the present neural Euler solver. Speaking time is a script-based estimate; rehearse aloud before the interview.
