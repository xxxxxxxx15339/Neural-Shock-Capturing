# Rebuild the targeted revision

The original presentation and source files are read-only inputs for this workflow. Revised outputs have `_REVISED` filenames in the parent directory. Assets and intermediate validation results live here, under `revised_source/`.

Requirements are the same as the original deck: Python with NumPy, Matplotlib, Pillow, lxml; Tectonic or compatible XeLaTeX; Poppler utilities; Liberation Sans fonts. LibreOffice is used only for a native-PPTX review render.

From this directory:

```sh
python write_notes_REVISED.py
python build_presentation_REVISED.py --prepare
tectonic --keep-logs --outdir assets/math assets/math/equations.tex
python build_presentation_REVISED.py --build
```

Compile `Sebastian_Research_Interview_REVISED.tex` from the parent `interview_sebastian_final/` directory. Then return here and finalize:

```sh
python finalize_REVISED.py
```

For independent native PowerPoint validation, export all 12 slides, including the three hidden backups, with LibreOffice's `ExportHiddenSlides=true` PDF option. Recompile the Beamer PDF, then run:

```sh
python finalize_REVISED.py --native-render /path/to/exported/Sebastian_Research_Interview_REVISED.pdf
python validate_REVISED.py
```

`finalize_REVISED.py` expects a freshly compiled 12-page Beamer PDF, inserts the unchanged original vector landing page, creates the nine-page main-only PDF, renders every slide, and creates original/revised image comparisons. Validation uses the surrounding repository for provenance, the protected-original manifest and numerical artifact checks. No numerical solver or training is run.

Main-script timing is 13:15. Read `Speaker_Notes_REVISED.md`, `Scientific_Changes.md` and `Final_Validation.md` in the parent directory. Text/diagrams are editable PowerPoint objects; equation TeX is editable in the generator/source. The three technical backups remain separate from the timed main talk.
