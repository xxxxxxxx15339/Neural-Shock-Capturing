"""Read-only checks for the final PowerPoint / PDF and comparison artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import posixpath
import re
import subprocess
from zipfile import ZipFile

from lxml import etree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PPTX = ROOT / "Scientific_Internship_WENO_NN_UM6P_Professional_LaTeX.pptx"
PDF = PPTX.with_suffix(".pdf")
REVIEW = ROOT / "Documentation/presentation_professional"
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
REPORT = {}


def main():
    with ZipFile(PPTX) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        slides = ET.fromstring(archive.read("ppt/presentation.xml"))
        relationships = ET.fromstring(archive.read("ppt/_rels/presentation.xml.rels"))
        targets = {rel.get("Id"): posixpath.normpath("ppt/" + rel.get("Target")) for rel in relationships}
        order = [targets[item.get(f"{{{NS['r']}}}id")] for item in slides.find("p:sldIdLst", NS)]
        assert len(order) == 16, order
        assert len([name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)]) == 16
        assert not any(f"ppt/slides/slide{n}.xml" in names for n in (11, 12, 13))
        fonts, slide_text, colored = set(), [], {"15803D": 0, "B91C1C": 0}
        cx, cy = map(int, (slides.find("p:sldSz", NS).get("cx"), slides.find("p:sldSz", NS).get("cy")))
        for display, name in enumerate(order, 1):
            root = ET.fromstring(archive.read(name))
            text = "\n".join(root.xpath("//a:t/text()", namespaces=NS))
            slide_text.append(text)
            if display > 1:
                assert f"{display:02d} / 16" in text, (display, text)
            assert "Mathematical backup" not in text
            ids = root.xpath("//p:cNvPr/@id", namespaces=NS)
            assert len(ids) == len(set(ids)), (name, ids)
            fonts.update(root.xpath("//a:latin/@typeface | //a:ea/@typeface | //a:cs/@typeface", namespaces=NS))
            for xfrm in root.xpath("//a:xfrm[a:off and a:ext]", namespaces=NS):
                off, ext = xfrm.find("a:off", NS), xfrm.find("a:ext", NS)
                x, y, w, h = map(int, (off.get("x"), off.get("y"), ext.get("cx"), ext.get("cy")))
                assert x >= -10 and y >= -10 and x + w <= cx + 10 and y + h <= cy + 10, (name, x, y, w, h)
            if display == 13:
                for color in colored:
                    colored[color] = len(root.xpath(f"//a:rPr/a:solidFill/a:srgbClr[@val='{color}']", namespaces=NS))
        assert fonts == {"Arial"}, fonts
        assert "Thank you" in slide_text[-1] and "Selected references" in slide_text[-2]
        assert "Discussion" not in slide_text[13]
        assert "final model learns" in slide_text[5]
        assert "Build the interface flux" in slide_text[6]
        assert "Train on the solution" in slide_text[7]
        assert not re.search("scientific internship soutenance", slide_text[0], re.I)
        assert colored == {"15803D": 9, "B91C1C": 3}, colored
        for name in names:
            if name.endswith(".xml"):
                root = ET.fromstring(archive.read(name))
                assert not root.xpath("//a:outerShdw | //a:innerShdw", namespaces=NS), name
            if name.endswith(".rels"):
                root = ET.fromstring(archive.read(name))
                owner = "" if name == "_rels/.rels" else name.replace("/_rels/", "/").removesuffix(".rels")
                for rel in root:
                    if rel.get("TargetMode") == "External":
                        continue
                    target = rel.get("Target").split("#")[0]
                    resolved = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(posixpath.dirname(owner), target))
                    assert resolved in names, (name, resolved)
        REPORT.update(slides=len(order), native_font="Arial", shadows=0,
                      table_green_values=colored["15803D"], table_red_values=colored["B91C1C"],
                      internal_relationships="valid", shape_bounds="within slide")

    bbox = subprocess.run(["pdftotext", "-bbox-layout", str(PDF), "-"], check=True, capture_output=True).stdout
    root = ET.fromstring(bbox, ET.XMLParser(resolve_entities=False))
    pages = root.xpath("//*[local-name()='page']")
    assert len(pages) == 16
    for index, page in enumerate(pages, 1):
        w, h = float(page.get("width")), float(page.get("height"))
        for word in page.xpath(".//*[local-name()='word']"):
            bounds = [float(word.get(key)) for key in ("xMin", "yMin", "xMax", "yMax")]
            assert bounds[0] >= 0 and bounds[1] >= 0 and bounds[2] <= w + .1 and bounds[3] <= h + .1, (index, word.text, bounds)
    REPORT.update(pdf_pages=len(pages), pdf_text_bounds="within page")

    equations = json.loads((REVIEW / "equations.json").read_text())
    loss = [item for item in equations if item["slide"] == 8]
    assert len(loss) == 1 and all(term in loss[0]["latex"] for term in ("MSE", "TV", "BVD", "Reg"))
    assert all(f"\\alpha_{n}" in loss[0]["latex"] for n in range(1, 5))
    assert "\\lambda_" not in loss[0]["latex"]
    solver = [item["latex"] for item in equations if item["slide"] == 7]
    assert len(solver) == 6 and any("F^{\\pm}" in expr for expr in solver)
    assert all(any(stage in expr for expr in solver) for stage in ("U^{(1)}=", "U^{(2)}=", "U^{n+1}="))
    assert solver[0].startswith("F^{\\pm}") and solver[1].startswith("\\hat F") and solver[2].startswith("L_i")
    assert all(item["renderer"].startswith("LaTeX") for item in equations)
    REPORT.update(latex_equations=len(equations), central_physics_loss_terms=4)

    metrics = json.loads((REVIEW / "figures/final_comparison_metrics.json").read_text())
    with np.load(REVIEW / "figures/final_profiles.npz") as profiles:
        for name, record in metrics.items():
            difference_t = profiles[name+"_teno"] - profiles[name+"_reference"]
            difference_n = profiles[name+"_nn"] - profiles[name+"_reference"]
            fn = (lambda d: np.sqrt(np.mean(d**2))) if name in ("advection", "burgers") else (lambda d: np.mean(np.abs(d)))
            assert np.allclose([fn(difference_t), fn(difference_n)], [record["teno"], record["nn"]], rtol=1e-12)
            assert np.isclose(record["ratio"], record["nn"] / record["teno"])
            assert np.isclose(record["change_percent"], 100 * (record["ratio"] - 1))
    REPORT.update(final_profile_metrics="all four match saved data")
    print(json.dumps(REPORT, indent=2))


if __name__ == "__main__":
    main()
