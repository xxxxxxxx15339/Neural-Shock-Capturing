"""Refine the supplied WENO-NN deck without a PowerPoint-specific dependency.

Keep the original OOXML package, replace each slide's visual tree, and add
editable DrawingML text/shapes plus vector equations with PNG fallbacks.
Run with --source PATH --output PATH. The original presentation is never saved.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from lxml import etree as ET
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
EMU = 914400
W, H = 13.333333, 7.5
SLIDE_COUNT = 13
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "asvg": "http://schemas.microsoft.com/office/drawing/2016/SVG/main",
}
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
INK, MUTED = "172D45", "526478"
ORANGE, TEAL = "D9582B", "157F89"
LIGHT, BORDER = "F3F6F9", "DCE4EC"
WHITE, PALE_ORANGE, PALE_TEAL = "FFFFFF", "FCF1EA", "ECF6F5"
FONT = "Calibri"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "path",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def e(tag, parent=None, **attrs):
    prefix, local = tag.split(":")
    node = ET.Element(f"{{{NS[prefix]}}}{local}", {k: str(v) for k, v in attrs.items()})
    if parent is not None:
        parent.append(node)
    return node


def xml(root):
    return ET.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def emu(value):
    return str(round(value * EMU))


def fill(parent, color):
    e("a:srgbClr", e("a:solidFill", parent), val=color)


def transform(parent, x, y, w, h):
    tr = e("a:xfrm", parent)
    e("a:off", tr, x=emu(x), y=emu(y))
    e("a:ext", tr, cx=emu(w), cy=emu(h))


def math_art(expression, fontsize=24, color=INK):
    """Render once in physical point units; retain its native SVG geometry."""
    fig = plt.figure(figsize=(16, 1.5), dpi=240)
    text = fig.text(0, 0.5, "$" + expression + "$", fontsize=fontsize,
                    color="#" + color, va="center")
    fig.canvas.draw()
    bbox = text.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bbox = bbox.expanded(1.012, 1.16)
    png, svg = BytesIO(), BytesIO()
    fig.savefig(png, format="png", transparent=True, bbox_inches=bbox, pad_inches=0)
    fig.savefig(svg, format="svg", transparent=True, bbox_inches=bbox, pad_inches=0)
    plt.close(fig)
    return png.getvalue(), svg.getvalue(), bbox.width, bbox.height


def plot_bytes(fig):
    out = BytesIO()
    fig.savefig(out, format="png", dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out.getvalue()


def styled_axis(ax):
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#" + BORDER)
    ax.tick_params(length=0, colors="#" + MUTED, labelsize=10)
    ax.grid(axis="y", color="#" + BORDER, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")


def instability_plot():
    import csv
    with (ROOT / "models/euler_instability_trace.csv").open() as source:
        rows = list(csv.DictReader(source))
    fig, ax = plt.subplots(figsize=(6, 3.0))
    colors = [TEAL, "849DAF", ORANGE, "C89644", "A85162"]
    for margin, color in zip(sorted({float(r["alpha_margin"]) for r in rows}), colors):
        subset = [r for r in rows if float(r["alpha_margin"]) == margin]
        valid = [r for r in subset if r["finite"] == "True"]
        ax.plot([int(r["step"]) for r in valid], [float(r["max_density"]) for r in valid],
                color="#" + color, linewidth=1.8, marker="o", markersize=3,
                label=f"{margin:g}× dissipation")
        if any(r["finite"] != "True" for r in subset) and valid:
            ax.scatter([int(valid[-1]["step"]) + 1], [float(valid[-1]["max_density"])],
                       marker="x", s=30, color="#" + color)
    styled_axis(ax)
    ax.set_yscale("log")
    ax.set_xlabel("SSPRK3 step", fontsize=11, color="#" + MUTED)
    ax.set_ylabel("Maximum density (log scale)", fontsize=11, color="#" + MUTED)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(pad=0.9)
    return plot_bytes(fig)


class Deck:
    def __init__(self, source):
        with ZipFile(source) as z:
            self.parts = {name: z.read(name) for name in z.namelist()}
        self.media = 0
        self.equations = []
        self.changes = []

    def add_media(self, blob, ext):
        self.media += 1
        name = f"ppt/media/refined_{self.media:03d}.{ext}"
        self.parts[name] = blob
        return name

    def register_appendix(self):
        self.parts["ppt/slides/slide13.xml"] = self.parts["ppt/slides/slide12.xml"]
        self.parts["ppt/slides/_rels/slide13.xml.rels"] = self.parts["ppt/slides/_rels/slide12.xml.rels"]
        root = ET.fromstring(self.parts["ppt/presentation.xml"])
        slides = root.find("p:sldIdLst", NS)
        next_id = max(int(child.get("id")) for child in slides) + 1
        entry = e("p:sldId", slides, id=next_id)
        entry.set(f"{{{NS['r']}}}id", "rIdRefinedAppendix")
        self.parts["ppt/presentation.xml"] = xml(root)
        rels = ET.fromstring(self.parts["ppt/_rels/presentation.xml.rels"])
        ET.SubElement(rels, f"{{{REL}}}Relationship", Id="rIdRefinedAppendix",
                      Type=NS["r"] + "/slide", Target="slides/slide13.xml")
        self.parts["ppt/_rels/presentation.xml.rels"] = xml(rels)
        types = ET.fromstring(self.parts["[Content_Types].xml"])
        ET.SubElement(types, f"{{{CT}}}Override", PartName="/ppt/slides/slide13.xml",
                      ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
        self.parts["[Content_Types].xml"] = xml(types)
        app = ET.fromstring(self.parts["docProps/app.xml"])
        slides_count = app.find("{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Slides")
        if slides_count is not None:
            slides_count.text = str(SLIDE_COUNT)
        self.parts["docProps/app.xml"] = xml(app)

    def save(self, output):
        # Explicitly remove inherited theme shadows, including those in layouts.
        for name, data in list(self.parts.items()):
            if name.endswith(".xml") and (b"outerShdw" in data or b"innerShdw" in data):
                root = ET.fromstring(data)
                for shadow in root.xpath("//a:outerShdw | //a:innerShdw", namespaces=NS):
                    shadow.getparent().remove(shadow)
                self.parts[name] = xml(root)
        root = ET.fromstring(self.parts["ppt/presentation.xml"])
        size = root.find("p:sldSz", NS)
        size.set("cx", emu(W))
        size.set("cy", emu(H))
        size.set("type", "screen16x9")
        self.parts["ppt/presentation.xml"] = xml(root)
        types = ET.fromstring(self.parts["[Content_Types].xml"])
        if not types.xpath("*[local-name()='Default'][@Extension='svg']"):
            ET.SubElement(types, f"{{{CT}}}Default", Extension="svg", ContentType="image/svg+xml")
        self.parts["[Content_Types].xml"] = xml(types)
        core = ET.fromstring(self.parts["docProps/core.xml"])
        dc = "http://purl.org/dc/elements/1.1/"
        title = core.find(f"{{{dc}}}title")
        if title is not None:
            title.text = "Machine Learning for Shock-Capturing Numerical Schemes"
        self.parts["docProps/core.xml"] = xml(core)
        output.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(output, "w", ZIP_DEFLATED) as z:
            for name, data in self.parts.items():
                z.writestr(name, data)


class Slide:
    def __init__(self, deck, number, title=None, section=None):
        self.deck, self.number, self.counter = deck, number, 1
        self.path = f"ppt/slides/slide{number}.xml"
        self.root = e("p:sld")
        self.root.set("showMasterSp", "0")
        cs = e("p:cSld", self.root)
        fill(e("p:bgPr", e("p:bg", cs)), WHITE)
        e("a:effectLst", cs.find("p:bg/p:bgPr", NS))
        self.tree = e("p:spTree", cs)
        grp = e("p:nvGrpSpPr", self.tree)
        e("p:cNvPr", grp, id="1", name="")
        e("p:cNvGrpSpPr", grp)
        e("p:nvPr", grp)
        e("p:grpSpPr", self.tree)
        e("a:masterClrMapping", e("p:clrMapOvr", self.root))
        self.rels = ET.fromstring(deck.parts[f"ppt/slides/_rels/slide{number}.xml.rels"])
        for rel in list(self.rels):
            if rel.get("Type").endswith("/image"):
                self.rels.remove(rel)
        self.next_rid = 20
        if title:
            self.txt(section.upper(), .6, .30, 10.9, .24, 10.5, ORANGE, bold=True)
            self.txt(title, .6, .78, 12.05, .58, 29, INK, bold=True)
            self.shape(.6, 1.43, 12.1, .015, fill_color=BORDER)
        if number != 1:
            self.shape(.6, 7.03, 12.1, .012, fill_color=BORDER)
            self.txt("SHOCKS & WENO   /   YASSER BAOUZIL", .6, 7.15, 7.2, .20, 9, MUTED)
            self.txt(f"{number:02d} / {SLIDE_COUNT}", 11.6, 7.12, 1.1, .26, 10, MUTED, align="r")
            self.picture(deck.parts["ppt/media/image1.png"], 9.4, 7.12, 1.6, .27,
                         name="UM6P College of Computing logo")

    def nv(self, parent, pic=False, name="Shape", descr=None):
        self.counter += 1
        outer = e("p:nvPicPr" if pic else "p:nvSpPr", parent)
        attrs = {"id": self.counter, "name": f"{name} {self.counter}"}
        if descr:
            attrs["descr"] = descr
        e("p:cNvPr", outer, **attrs)
        props = e("p:cNvPicPr" if pic else "p:cNvSpPr", outer)
        if pic:
            e("a:picLocks", props, noChangeAspect="1")
        e("p:nvPr", outer)

    def shape(self, x, y, w, h, fill_color=None, stroke=None, radius=False, name="Panel"):
        sp = e("p:sp", self.tree)
        self.nv(sp, name=name)
        pr = e("p:spPr", sp)
        transform(pr, x, y, w, h)
        geom = e("a:prstGeom", pr, prst="roundRect" if radius else "rect")
        av = e("a:avLst", geom)
        if radius:
            e("a:gd", av, name="adj", fmla="val 3500")
        fill(pr, fill_color) if fill_color else e("a:noFill", pr)
        line = e("a:ln", pr, w="7620")
        fill(line, stroke) if stroke else e("a:noFill", line)
        e("a:effectLst", pr)
        return sp

    def txt(self, value, x, y, w, h, size=18, color=INK, bold=False,
            align="l", valign="t", margins=0, line_spacing=1.08, name="Text"):
        sp = self.shape(x, y, w, h, name=name)
        tx = e("p:txBody", sp)
        body = e("a:bodyPr", tx, wrap="square", anchor=valign,
                 lIns=emu(margins), rIns=emu(margins), tIns=emu(margins), bIns=emu(margins))
        e("a:noAutofit", body)
        e("a:lstStyle", tx)
        for text in value.split("\n"):
            paragraph = e("a:p", tx)
            props = e("a:pPr", paragraph, algn=align, marL="0", marR="0", indent="0")
            e("a:spcPct", e("a:lnSpc", props), val=round(line_spacing * 100000))
            e("a:spcPts", e("a:spcBef", props), val="0")
            e("a:spcPts", e("a:spcAft", props), val="0")
            e("a:buNone", props)
            run = e("a:r", paragraph)
            rpr = e("a:rPr", run, lang="en-US", sz=round(size * 100), b=int(bold), dirty="0")
            fill(rpr, color)
            e("a:latin", rpr, typeface=FONT)
            e("a:ea", rpr, typeface=FONT)
            e("a:cs", rpr, typeface=FONT)
            e("a:t", run).text = text
            e("a:endParaRPr", paragraph, lang="en-US", sz=round(size * 100))
        return sp

    def panel(self, x, y, w, h, color=LIGHT):
        self.shape(x, y, w, h, fill_color=color, radius=True)

    def rel(self, target):
        self.next_rid += 1
        rid = f"rId{self.next_rid}"
        ET.SubElement(self.rels, f"{{{REL}}}Relationship", Id=rid,
                      Type=NS["r"] + "/image", Target="../media/" + Path(target).name)
        return rid

    def picture(self, blob, x, y, w, h, name="Picture", svg=None, center=True):
        im = Image.open(BytesIO(blob))
        aspect = im.width / im.height
        actual_w, actual_h = min(w, h * aspect), min(h, w / aspect)
        px = x + (w - actual_w) / 2 if center else x
        py = y + (h - actual_h) / 2
        target = self.deck.add_media(blob, "png")
        pic = e("p:pic", self.tree)
        self.nv(pic, pic=True, name=name, descr=name)
        bf = e("p:blipFill", pic)
        blip = e("a:blip", bf)
        blip.set(f"{{{NS['r']}}}embed", self.rel(target))
        if svg:
            vector = self.deck.add_media(svg, "svg")
            ext = e("a:ext", e("a:extLst", blip), uri="{96DAC541-7B7A-43D3-8B79-37D633B846F1}")
            sb = e("asvg:svgBlip", ext)
            sb.set(f"{{{NS['r']}}}embed", self.rel(vector))
        e("a:fillRect", e("a:stretch", bf))
        pr = e("p:spPr", pic)
        transform(pr, px, py, actual_w, actual_h)
        e("a:avLst", e("a:prstGeom", pr, prst="rect"))
        e("a:effectLst", pr)
        return pic

    def equation(self, value, x, y, w, h, size=23, color=INK, align="l"):
        png, svg, native_w, native_h = math_art(value, size, color)
        # Use the true font size unless the panel's width requires a reduction.
        scale = min(1.0, w / native_w, h / native_h)
        ew, eh = native_w * scale, native_h * scale
        px = x + (w - ew) / 2 if align == "ctr" else x
        self.picture(png, px, y + (h - eh) / 2, ew, eh,
                     name="Equation: " + value, svg=svg, center=False)
        self.deck.equations.append({"slide": self.number, "latex": value,
                                    "effective_point_size": round(size * scale, 1)})

    def arrow(self, x, y, w=.36, color=MUTED):
        self.txt("→", x, y, w, .38, 23, color, align="ctr")

    def insight(self, value, y=6.47, color=TEAL):
        self.shape(.6, y + .01, .045, .37, fill_color=color)
        self.txt(value, .77, y, 11.83, .47, 16.5, color)

    def finish(self, change):
        self.deck.parts[self.path] = xml(self.root)
        self.deck.parts[f"ppt/slides/_rels/slide{self.number}.xml.rels"] = xml(self.rels)
        self.deck.changes.append({"slide": self.number, "change": change})


def make_deck(source, output, review_dir=None, extend_deck=None):
    d = Deck(source)
    image = lambda n: d.parts[f"ppt/media/image{n}.png"]

    # 01 — A project cover with a clear title, generous whitespace and one image.
    s = Slide(d, 1)
    s.picture(image(1), .65, .48, 2.8, .57, name="UM6P College of Computing logo", center=False)
    s.shape(8.48, 0, W - 8.48, H, fill_color=INK)
    s.txt("Machine Learning for\nShock-Capturing\nNumerical Schemes", .65, 1.75, 7.4, 2.15,
          36, INK, bold=True, line_spacing=1.03)
    s.txt("Learning reconstruction decisions for\n1D hyperbolic conservation laws", .68, 4.10, 7.1, .83,
          20, MUTED, line_spacing=1.13)
    s.txt("WENO-NN   /   NN-TENO5   /   DIFFERENTIABLE CFD", .68, 5.20, 7.05, .32,
          12, TEAL, bold=True)
    s.shape(.68, 5.85, 6.95, .017, fill_color=BORDER)
    s.txt("Yasser Baouzil", .68, 6.09, 7, .36, 22, INK, bold=True)
    s.txt("Supervisor: Radouan Boukharfane", .68, 6.52, 7, .31, 16, MUTED)
    s.txt("UM6P  ·  College of Computing", .68, 7.00, 7, .23, 11, MUTED)
    s.txt("PHYSICS × NUMERICS × LEARNING", 8.86, .62, 4.03, .29, 10.5, "D8E5EF", bold=True)
    s.picture(image(2), 8.86, 1.40, 4.06, 3.04,
              name="Original schlieren photograph of shock waves")
    s.txt("Capturing sharp fronts\nwithout spurious oscillations", 8.88, 4.74, 3.97, .83,
          22, WHITE, bold=True)
    s.equation(r"\frac{\partial u}{\partial t}+\frac{\partial f(u)}{\partial x}=0",
               8.88, 5.92, 3.98, .65, size=25, color=WHITE)
    s.finish("Removed Scientific Internship Soutenance; rebuilt the cover with a left-aligned title, author block, original shock photograph and conservation law.")

    # 02 — Motivation, physical model, and the numerical requirements.
    s = Slide(d, 2, "Shock capturing balances accuracy and stability", "Motivation")
    s.picture(image(2), .6, 1.68, 6.00, 4.46, name="Original shock-wave photograph")
    s.txt("Schlieren imaging reveals abrupt changes in compressible flow.", .63, 6.17, 5.97, .32, 12, MUTED)
    s.panel(6.95, 1.70, 5.77, 2.30)
    s.txt("The governing conservation law", 7.22, 1.92, 5.24, .40, 20, bold=True)
    s.equation(r"\partial_t u+\partial_x f(u)=0", 7.22, 2.51, 5.25, .49, size=29)
    s.equation(r"f(u)=a u\quad\mathrm{or}\quad f(u)=\frac{u^2}{2}",
               7.22, 3.20, 5.25, .47, size=23, color=TEAL)
    for y, heading, text in [
        (4.25, "Smooth regions", "Resolve the solution with high-order accuracy."),
        (4.96, "Discontinuities", "Suppress nonphysical oscillations near shocks."),
        (5.67, "Euler flows", "Maintain positive density and pressure."),
    ]:
        s.txt(heading, 7.02, y, 5.6, .33, 18, INK, bold=True)
        s.txt(text, 7.02, y + .34, 5.63, .42, 16, MUTED)
    s.insight("The learned component must work within the structure of a conservative numerical solver.")
    s.finish("Removed inherited shadows; enlarged the photograph; introduced the PDE and scalar fluxes; aligned three numerical requirements.")

    # 03 — Three panels hold the equations for the numerical backbone.
    s = Slide(d, 3, "Local interface fluxes drive a conservative solver", "Numerical backbone")
    s.panel(.6, 1.68, 6.06, 2.14)
    s.txt("01  Finite-volume discretization", .87, 1.90, 5.50, .39, 19, bold=True)
    s.equation(r"\bar u_i(t)=\frac{1}{\Delta x}\int_{I_i}u(x,t)\,dx",
               .87, 2.36, 5.48, .52, size=24)
    s.equation(r"L_i(u)=-\frac{\hat f_{i+1/2}-\hat f_{i-1/2}}{\Delta x}",
               .87, 2.98, 5.48, .71, size=24)
    s.panel(.6, 4.02, 6.06, 2.12)
    s.txt("02  Lax–Friedrichs interface flux", .87, 4.22, 5.48, .38, 19, bold=True)
    s.equation(r"f^{\pm}(u)=\frac{1}{2}\left[f(u)\pm\alpha u\right]",
               .87, 4.76, 5.48, .52, size=24)
    s.equation(r"\hat f_{i+1/2}=\hat f^{+}_{i+1/2}+\hat f^{-}_{i+1/2}",
               .87, 5.39, 5.48, .60, size=23)
    s.panel(6.88, 1.68, 5.85, 4.46)
    s.txt("03  SSPRK3 time integration", 7.15, 1.90, 5.31, .39, 19, bold=True)
    s.equation(r"u^{(1)}=u^n+\Delta t\,L(u^n)", 7.15, 2.65, 5.24, .56, size=24)
    s.equation(r"u^{(2)}=\frac{3}{4}u^n+\frac{1}{4}\left[u^{(1)}+\Delta t L(u^{(1)})\right]",
               7.15, 3.57, 5.24, .60, size=22)
    s.equation(r"u^{n+1}=\frac{1}{3}u^n+\frac{2}{3}\left[u^{(2)}+\Delta t L(u^{(2)})\right]",
               7.15, 4.58, 5.24, .60, size=22)
    s.txt("The reconstruction is evaluated at every stage.", 7.15, 5.60, 5.20, .33, 15, MUTED)
    s.txt("CFL time-step restriction", .7, 6.45, 3.19, .35, 16, TEAL, bold=True)
    s.equation(r"\Delta t\leq\mathrm{CFL}\,\frac{\Delta x}{a_{\max}}\,,\qquad a_{\max}=\max_i(|v_i|+c_i)\ \mathrm{for\ Euler}",
               3.80, 6.38, 8.81, .51, size=21, color=TEAL)
    s.finish("Replaced the dense pipeline screenshot with legible equations for finite volumes, Lax–Friedrichs fluxes, all SSPRK3 stages and the CFL step.")

    # 04 — Explicitly aligned five-point stencil and three comparison panels.
    s = Slide(d, 4, "WENO and TENO combine three local reconstructions", "Classical reconstruction")
    s.txt("Five stencil values → three third-order candidates → one interface flux", .62, 1.62, 12.0, .39, 18, MUTED)
    xs = [2.56 + 1.63 * i for i in range(5)]
    labels = [r"f_{i-2}", r"f_{i-1}", r"f_i", r"f_{i+1}", r"f_{i+2}"]
    for x, label in zip(xs, labels):
        s.panel(x, 2.16, 1.14, .59)
        s.equation(label, x + .12, 2.23, .90, .41, size=23, align="ctr")
    face = (xs[2] + 1.14 + xs[3]) / 2
    s.shape(face, 2.05, .015, .96, fill_color=ORANGE)
    s.equation(r"i+\frac{1}{2}", face - .40, 1.82, .8, .34, size=17, color=ORANGE, align="ctr")
    for k, color in enumerate([TEAL, "638E5B", ORANGE]):
        y = 2.98 + .17 * k
        s.shape(xs[k] + .22, y, xs[k + 2] - xs[k] + .70, .025, fill_color=color)
        s.equation(f"S_{k}", xs[k] - .40, y - .11, .40, .24, size=14, color=color)
    cards = [(.6, "ENO selects", "Choose the smoothest candidate.", PALE_TEAL),
             (4.72, "WENO weights", "Reduce troubled candidates continuously.", LIGHT),
             (8.84, "TENO discards", "Reject troubled candidates, then reweight.", PALE_ORANGE)]
    for x, heading, desc, color in cards:
        s.panel(x, 3.59, 3.89, 2.63, color)
        s.txt(heading, x + .23, 3.81, 3.43, .36, 21, bold=True)
        s.txt(desc, x + .23, 4.30, 3.43, .63, 16, MUTED)
    s.txt("Select using divided differences.", .84, 5.10, 3.40, .40, 15, MUTED)
    s.equation(r"\hat f_{i+1/2}=q_{k^*}", .84, 5.62, 3.40, .52, size=22)
    s.equation(r"\alpha_k=\frac{d_k}{(\varepsilon+\beta_k)^2}", 4.96, 4.98, 3.41, .63, size=22)
    s.equation(r"\omega_k=\alpha_k/\|\alpha\|_1", 4.96, 5.75, 3.41, .37, size=22)
    s.equation(r"\delta_k=\mathbf{1}[\chi_k\geq C_T]", 9.08, 5.01, 3.41, .45, size=21)
    s.equation(r"\omega_k=\frac{d_k\delta_k}{d\cdot\delta}", 9.08, 5.58, 3.41, .60, size=21)
    s.equation(r"\hat f_{i+1/2}=\omega_0q_0+\omega_1q_1+\omega_2q_2,\qquad d=(0.1,0.6,0.3)",
               .70, 6.40, 9.15, .58, size=21)
    s.txt("β: smoothness   χ: TENO sensor", 9.88, 6.52, 2.75, .33, 11.5, MUTED)
    s.finish("Rebuilt the five-point stencil and substencils as editable components; added aligned ENO/WENO/TENO formulas and optimal weights.")

    # 05 — Network widths are never placed inside narrow circles.
    s = Slide(d, 5, "The network predicts which candidates to trust", "NN-TENO5 architecture")
    s.txt("A local five-point stencil feeds a classifier with three troubled-candidate probabilities.",
          .63, 1.64, 12.00, .47, 18, MUTED)
    s.panel(.6, 2.30, 3.65, 1.55)
    s.txt("Normalize the input", .84, 2.49, 3.16, .35, 19, bold=True)
    s.equation(r"x_j'=\frac{x_j-\min(x)}{\max(x)-\min(x)}", .84, 3.00, 3.17, .54, size=24)
    s.txt("Constant stencils use a guarded denominator.", .74, 3.99, 3.50, .51, 13, MUTED)
    s.arrow(4.34, 2.99, .40)
    s.txt("FEED-FORWARD CLASSIFIER", 4.87, 2.32, 5.62, .28, 11.5, TEAL, bold=True)
    counts = [5, 64, 32, 16, 8, 3]
    for i, count in enumerate(counts):
        x = 4.88 + 1.045 * i
        s.panel(x, 2.96, .76, .76, PALE_ORANGE if i == 5 else LIGHT)
        s.txt(str(count), x, 3.10, .76, .38, 23, INK, bold=True, align="ctr")
        if i < 5:
            s.arrow(x + .76, 3.12, .29)
    s.txt("5 inputs", 4.79, 3.89, .96, .31, 12, MUTED, align="ctr")
    s.txt("ReLU hidden layers", 6.03, 3.89, 3.57, .31, 12, MUTED, align="ctr")
    s.txt("Sigmoid", 9.94, 3.89, 1.15, .31, 12, MUTED, align="ctr")
    s.equation(r"(p_0,p_1,p_2)=\mathrm{NN}_{\theta}(x')", 7.47, 4.34, 5.01, .45, size=24)
    s.panel(.6, 5.05, 5.93, 1.12, PALE_TEAL)
    s.panel(6.77, 5.05, 5.94, 1.12, LIGHT)
    s.txt("Training: continuous keep weights", .84, 5.22, 5.44, .31, 17, bold=True)
    s.equation(r"s_k=1-p_k", .84, 5.66, 5.44, .33, size=24, color=TEAL)
    s.txt("Optional hard inference: threshold decisions", 7.01, 5.22, 5.46, .31, 17, bold=True)
    s.equation(r"s_k=\mathbf{1}[p_k<\vartheta_k]", 7.01, 5.66, 5.46, .33, size=23)
    s.insight("The network changes candidate decisions; the solver still determines fluxes and time evolution.")
    s.finish("Corrected wrapped layer counts with a horizontal 5–64–32–16–8–3 diagram; added normalization and soft/hard weight formulas.")

    # 06 — Clearly separate the teacher objective from the evolved-state loss.
    s = Slide(d, 6, "Physics training evaluates the evolved solution", "Learning methodology")
    s.panel(.6, 1.73, 5.93, 3.38)
    s.panel(6.77, 1.73, 5.94, 3.38, PALE_TEAL)
    s.txt("01  Imitation learning", .85, 1.96, 5.40, .40, 22, bold=True)
    s.txt("Learn the classical TENO5 keep/discard labels.", .85, 2.51, 5.40, .49, 17, MUTED)
    s.equation(r"p=\mathrm{NN}_{\theta}(x')", .85, 3.20, 5.38, .40, size=25)
    s.equation(r"\mathcal{L}_{\mathrm{imit}}=\mathrm{BCE}(p,y_{\mathrm{TENO5}})",
               .85, 3.98, 5.38, .42, size=24)
    s.txt("Useful initialization; the teacher supplies no direct\nsignal for lower physical solution error.",
          .85, 4.60, 5.35, .49, 14.5, MUTED)
    s.txt("02  Unrolled physics training", 7.02, 1.96, 5.42, .40, 22, bold=True)
    s.txt("Differentiate through K complete SSPRK3 steps.", 7.02, 2.51, 5.40, .49, 17, MUTED)
    s.equation(r"U_{\theta}^{n+K}=\mathcal{S}_{\theta}^{K}(U^n)",
               7.02, 3.20, 5.37, .48, size=27)
    s.equation(r"\mathcal{L}_{\mathrm{MSE}}=\mathrm{mean}\!\left[(U_{\theta}^{n+K}-U_{\mathrm{ref}}^{n+K})^2\right]",
               7.02, 3.94, 5.37, .53, size=22)
    s.txt("The gradient crosses reconstruction, fluxes\nand every Runge–Kutta substage.",
          7.02, 4.60, 5.35, .49, 14.5, MUTED)
    s.txt("Combined training objective", .66, 5.42, 12.0, .37, 19, bold=True)
    s.equation(r"\mathcal{L}=\lambda_{\mathrm{MSE}}\mathcal{L}_{\mathrm{MSE}}+\lambda_{\mathrm{TV}}\mathcal{L}_{\mathrm{TV}}+\lambda_{\mathrm{BVD}}\mathcal{L}_{\mathrm{BVD}}+\lambda_{\mathrm{Reg}}\mathcal{L}_{\mathrm{Reg}}+\lambda_{+}\mathcal{L}_{+}",
               .67, 5.93, 11.98, .47, size=24)
    s.insight("Pretrain by imitation, normalize loss terms, then fine-tune against a physical reference.", y=6.53)
    s.finish("Replaced overlapping workflow boxes with two balanced learning panels; added BCE, unrolled evolution, MSE and the full weighted physics loss.")

    # 07 — Hybrid reconstruction plus actual instability evidence.
    s = Slide(d, 7, "Hybrid reconstruction supports stable Euler training", "Stability and physical constraints")
    s.panel(.6, 1.70, 12.12, 1.31, PALE_TEAL)
    s.equation(r"\Phi_{\mathrm{mix}}=(1-\lambda)\Phi_{\mathrm{TENO5}}+\lambda\Phi_{\mathrm{NN}},\qquad 0\leq\lambda\leq1",
               .89, 1.94, 11.52, .53, size=28)
    s.txt("A dedicated sigmoid head starts near TENO5: σ(−3) ≈ 0.047.", .90, 2.63, 11.4, .30, 15.5, TEAL)
    s.txt("Observed Euler training instability", .69, 3.28, 6.16, .37, 20, bold=True)
    s.picture(instability_plot(), .6, 3.81, 6.5, 2.80,
              name="Real instability trace from models/euler_instability_trace.csv")
    s.txt("× marks a nonfinite state. Extra dissipation alone did not stabilize training.",
          .71, 6.62, 6.28, .29, 11, MUTED)
    s.txt("Physical admissibility", 7.36, 3.28, 5.16, .37, 20, bold=True)
    s.equation(r"\rho>0,\qquad p=(\gamma-1)\!\left(E-\frac{m^2}{2\rho}\right)>0",
               7.36, 3.90, 5.16, .56, size=22)
    s.txt("m = ρv; γ = 1.4 for the project benchmarks.", 7.36, 4.56, 5.05, .31, 12.5, MUTED)
    for y, label, detail in [(5.04, "Guard every substage", "Measure positivity violations before repair."),
                             (5.69, "Control optimization", "Use rollout curriculum, anchoring and clipping."),
                             (6.34, "Validate the final solver", "Mixing alone is not a positivity or TVD proof.")]:
        s.txt(label, 7.36, y, 5.10, .31, 17, bold=True)
        s.txt(detail, 7.36, y + .32, 5.12, .35, 14, MUTED)
    s.finish("Added the learned convex mixture and Euler pressure constraint; redrew the real instability trace; distinguished training guards from mathematical guarantees.")

    # 08 — Preserve WENO-NN plots but label the final TENO metrics explicitly.
    s = Slide(d, 8, "Final scalar benchmarks improve on TENO5", "Results · scalar conservation laws")
    s.txt("Earlier WENO-NN experiments", .66, 1.67, 7.56, .35, 18, bold=True)
    s.panel(.6, 2.14, 3.91, 3.00)
    s.panel(4.69, 2.14, 3.91, 3.00)
    s.picture(image(10), .74, 2.23, 3.64, 2.73, name="Original WENO-NN discontinuity width plot")
    s.picture(image(11), 4.83, 2.23, 3.64, 2.73, name="Original WENO-NN Burgers convergence plot")
    s.txt("Sharper advected discontinuities", .76, 5.28, 3.6, .35, 15, MUTED)
    s.txt("Competitive Burgers convergence", 4.86, 5.28, 3.6, .35, 15, MUTED)
    s.panel(8.89, 1.70, 3.83, 4.30, PALE_TEAL)
    s.txt("Final NN-TENO5", 9.14, 1.91, 3.32, .36, 21, bold=True)
    s.txt("Dedicated mixing head · Phase 19", 9.14, 2.37, 3.32, .34, 13, MUTED)
    s.txt("3.5%", 9.14, 2.98, 3.32, .74, 40, TEAL, bold=True)
    s.txt("lower L₂ error on advection", 9.14, 3.72, 3.30, .40, 16, INK)
    s.txt("18.6%", 9.14, 4.47, 3.32, .74, 40, TEAL, bold=True)
    s.txt("lower L₂ error on Burgers", 9.14, 5.21, 3.30, .40, 16, INK)
    s.equation(r"L_2=\frac{\|u-u_{\mathrm{ref}}\|_2}{\sqrt{N}}",
               .74, 5.88, 7.55, .60, size=23)
    s.txt("Advection: 9.8960×10⁻² → 9.5521×10⁻²   |   Burgers: 8.1670×10⁻⁴ → 6.6493×10⁻⁴",
          .74, 6.57, 11.89, .35, 13.5, MUTED)
    s.finish("Preserved and relabeled the original WENO-NN evidence; added separate final NN-TENO5 gains, actual Phase 19 error values and the L2 metric.")

    # 09 — Equal figure columns, unambiguous per-benchmark Euler results.
    metrics = json.loads((ROOT / "models/nn_teno5euler_jax_comparison_phase18.json").read_text())
    vals = {k: v["beats_teacher"] for k, v in metrics["problems"].items()}
    sod_change = 100 * (vals["sod"]["nn_teno5euler"] / vals["sod"]["teno5euler"] - 1)
    shu_gain = 100 * (1 - vals["shu_osher"]["nn_teno5euler"] / vals["shu_osher"]["teno5euler"])
    s = Slide(d, 9, "Euler gains depend on the benchmark", "Results · compressible Euler equations")
    for x, heading in [(.6, "Sod shock tube"), (6.78, "Shu–Osher shock–entropy interaction")]:
        s.panel(x, 1.71, 5.93, 3.88)
        s.txt(heading, x + .22, 1.95, 5.49, .37, 19, bold=True)
    s.picture(image(12), .83, 2.81, 5.48, 1.73, name="Original Sod wave structure diagram")
    s.txt("Rarefaction, contact discontinuity and shock", .84, 4.85, 5.40, .32, 14, MUTED)
    s.picture(image(13), 7.00, 2.62, 5.48, 2.74,
              name="Original final Phase 18 Shu–Osher comparison against the reference")
    s.txt(f"+{sod_change:.2f}% error", .82, 5.90, 2.64, .58, 28, ORANGE, bold=True)
    s.txt("Near parity; TENO5 remains slightly better.", 3.60, 5.95, 2.77, .64, 16, MUTED)
    s.txt(f"−{shu_gain:.2f}% error", 7.00, 5.90, 2.89, .58, 28, TEAL, bold=True)
    s.txt("A modest gain with the learned mixing head.", 10.01, 5.95, 2.49, .64, 16, MUTED)
    s.txt("Metric: mean L₁ over (ρ, ρv, E). 120 cells, CFL 0.4. Sod: exact reference; Shu–Osher: 320-cell TENO5 reference.",
          .75, 6.63, 11.93, .29, 12, MUTED)
    s.finish("Aligned the two Euler figures; added the exact Phase 18 benchmark percentages and evaluation conditions; preserved the slightly unfavorable Sod result.")

    # 10 — A readable conclusion with one central take-home statement.
    s = Slide(d, 10, "Learning adds value inside a robust numerical solver", "Conclusion")
    s.panel(.6, 1.76, 12.12, 1.40, PALE_TEAL)
    s.txt("Learn the reconstruction decision.\nKeep the numerical and physical structure visible.",
          .91, 2.06, 11.48, .86, 27, INK, bold=True, line_spacing=1.10)
    columns = [
        (.6, "What I built", ["WENO/TENO reconstruction framework", "Neural stencil-decision models", "Differentiable CFD training in JAX"]),
        (4.73, "What the results show", ["Scalar gains: 3.5% and 18.6% in L₂", "Shu–Osher: 1.84% lower mean L₁", "Sod: near parity, 0.35% higher error"]),
        (8.87, "What comes next", ["Improve the Euler reconstruction model", "Validate across grids and flow cases", "Extend the framework toward 2D CFD"]),
    ]
    for x, heading, rows in columns:
        s.txt(heading, x + .06, 3.64, 3.71, .39, 22, INK, bold=True)
        s.shape(x + .06, 4.20, .50, .035, fill_color=ORANGE)
        for i, text in enumerate(rows):
            s.txt(text, x + .06, 4.51 + .69 * i, 3.72, .64, 18, MUTED)
    s.txt("Discussion", .66, 6.67, 12.0, .28, 13, TEAL, bold=True)
    s.finish("Reorganized the conclusion into a clear takeaway and three balanced columns; removed overflowing text and qualified the benchmark-dependent gains.")

    # 11 — Expanded loss definitions, kept as a Q&A slide.
    s = Slide(d, 11, "The physics loss balances accuracy and regularity", "Mathematical backup · training loss")
    panels = [(.6, 1.76, "Accuracy", "Mean square error against the evolved reference."),
              (6.78, 1.76, "Total variation", "Penalize extra variation; normalize by cell count."),
              (.6, 4.15, "Boundary variation", "Compare neighbors at the same interface."),
              (6.78, 4.15, "Anchor regularization", "Limit drift from the stable pretrained parameters.")]
    for x, y, heading, desc in panels:
        s.panel(x, y, 5.93, 2.17)
        s.txt(heading, x + .24, y + .19, 5.44, .37, 21, bold=True)
        s.txt(desc, x + .24, y + .70, 5.44, .41, 15, MUTED)
    s.equation(r"\mathcal{L}_{\mathrm{MSE}}=\mathrm{mean}\!\left[(U_{\theta}^{n+K}-U_{\mathrm{ref}}^{n+K})^2\right]",
               .85, 3.08, 5.41, .50, size=22)
    s.equation(r"\mathcal{L}_{\mathrm{TV}}=\frac{[\,TV(U^{n+K})-TV(U^n)\,]_+}{N-1}",
               7.03, 3.03, 5.40, .56, size=23)
    s.equation(r"BV_{i+1/2}=|\Phi_i(x_{i+1/2})-\Phi_{i+1}(x_{i+1/2})|",
               .85, 5.31, 5.41, .43, size=21)
    s.equation(r"\mathcal{L}_{\mathrm{BVD}}=\mathrm{mean}[\,BV_{\mathrm{NN}}-BV_{\mathrm{TENO5}}\,]_+",
               .85, 5.86, 5.41, .37, size=21)
    s.equation(r"\mathcal{L}_{\mathrm{Reg}}=\frac{\|\theta-\theta_{\mathrm{anchor}}\|_2^2}{P}",
               7.03, 5.46, 5.41, .59, size=26)
    s.txt("[z]₊ = max(z, 0); TV(U) = Σᵢ |Uᵢ₊₁ − Uᵢ|; P = parameter count. Euler positivity is measured before state repair.",
          .76, 6.64, 11.90, .30, 12.5, MUTED)
    s.finish("Expanded the mathematical backup into four aligned loss definitions with the actual TV/parameter normalization and correct neighboring-cell BVD definition.")

    # 12 — Editable end-to-end workflow with separate scalar and Euler details.
    s = Slide(d, 12, "The learned decision sits inside every solver step", "Mathematical backup · complete workflow")
    s.txt("Forward numerical path", .65, 1.69, 11.99, .37, 20, bold=True)
    nodes = ["State +\nlocal stencil", "Scalar / Euler\nsetup", "WENO / TENO\nor NN selector", "Interface\nflux", "SSPRK3\nstep", "K-step\nrollout"]
    for i, label in enumerate(nodes):
        x = .60 + 2.057 * i
        s.panel(x, 2.28, 1.82, .92, PALE_TEAL if i == 2 else LIGHT)
        s.txt(label, x + .08, 2.48, 1.66, .60, 17, INK, bold=(i == 2), align="ctr")
        if i < 5:
            s.arrow(x + 1.82, 2.53, .235)
    s.shape(4.73, 3.40, 1.76, .04, fill_color=TEAL)
    s.txt("Learned component", 4.37, 3.61, 2.60, .33, 14, TEAL, bold=True, align="ctr")
    s.panel(.6, 4.19, 5.93, 2.15)
    s.panel(6.78, 4.19, 5.93, 2.15, PALE_TEAL)
    s.txt("Euler: reconstruct characteristic fluxes", .85, 4.42, 5.45, .37, 19, bold=True)
    s.equation(r"W=L U,\qquad G^{\pm}=\frac{1}{2}L\left[F(U)\pm\alpha U\right]",
               .85, 5.04, 5.41, .51, size=22)
    s.equation(r"\hat F_{i+1/2}=R(\hat G^+_{i+1/2}+\hat G^-_{i+1/2}),\quad L=R^{-1}",
               .85, 5.68, 5.41, .55, size=21)
    s.txt("Training: backpropagate physical error", 7.03, 4.42, 5.45, .37, 19, bold=True)
    s.equation(r"\theta\leftarrow\mathrm{Adam}\!\left(\theta,\nabla_{\theta}\mathcal{L}\right)",
               7.03, 5.10, 5.40, .49, size=25)
    s.txt("Reference → loss → gradient through every step\n→ update the neural reconstruction parameters.",
          7.03, 5.79, 5.36, .47, 15, MUTED)
    s.txt("Scalar implementation: reconstruct states before Lax–Friedrichs flux evaluation. Euler: split characteristic fluxes before reconstruction.",
          .73, 6.64, 11.92, .30, 11.7, MUTED)
    s.finish("Replaced the duplicated pipeline screenshot with an editable workflow; added Roe characteristic projection and gradient-based training; clarified the distinct scalar and Euler flux paths.")

    # 13 — Full reconstruction math is available without crowding the talk.
    d.register_appendix()
    s = Slide(d, 13, "Fifth-order reconstruction: candidates and TENO sensor", "Mathematical backup · reconstruction details")
    s.panel(.6, 1.77, 6.06, 4.48)
    s.panel(6.88, 1.77, 5.85, 4.48, PALE_TEAL)
    s.txt("Three third-order candidate polynomials", .87, 2.00, 5.50, .38, 19, bold=True)
    s.equation(r"q_0=\frac{1}{3}f_{i-2}-\frac{7}{6}f_{i-1}+\frac{11}{6}f_i",
               .87, 2.83, 5.47, .66, size=24)
    s.equation(r"q_1=-\frac{1}{6}f_{i-1}+\frac{5}{6}f_i+\frac{1}{3}f_{i+1}",
               .87, 3.87, 5.47, .66, size=24)
    s.equation(r"q_2=\frac{1}{3}f_i+\frac{5}{6}f_{i+1}-\frac{1}{6}f_{i+2}",
               .87, 4.91, 5.47, .66, size=24)
    s.txt("Optimal linear weights: d = (0.1, 0.6, 0.3).", .87, 5.85, 5.45, .31, 16, MUTED)
    s.txt("TENO smoothness and scale separation", 7.15, 2.00, 5.31, .38, 19, bold=True)
    s.equation(r"\tau_5=|\beta_0-\beta_2|", 7.15, 2.75, 5.31, .43, size=25)
    s.equation(r"\gamma_k=\left(1+\frac{\tau_5}{\beta_k+\varepsilon}\right)^6",
               7.15, 3.50, 5.31, .73, size=25)
    s.equation(r"\chi_k=\gamma_k/\|\gamma\|_1,\qquad\delta_k=\mathbf{1}[\chi_k\geq C_T]",
               7.15, 4.62, 5.31, .46, size=21)
    s.equation(r"\varepsilon=10^{-40},\qquad C_T=10^{-5}",
               7.15, 5.52, 5.31, .43, size=23)
    s.txt("βₖ are the Jiang–Shu smoothness indicators. NN-TENO5 learns candidate decisions with the same polynomials.\nIf every candidate is rejected, return the center value; otherwise normalize the retained weights.",
          .74, 6.47, 11.88, .49, 14, MUTED)
    s.finish("Added one mathematical backup slide with all candidate polynomials, the TENO scale-separation sensor and the exact parameters used in the repository.")

    if extend_deck is not None:
        extend_deck(d)
    d.save(output)
    review = review_dir or ROOT / "Documentation/presentation_review"
    review.mkdir(parents=True, exist_ok=True)
    (review / "equations.json").write_text(json.dumps(d.equations, indent=2) + "\n")
    (review / "slide_changes.json").write_text(json.dumps(d.changes, indent=2) + "\n")
    print(f"Saved {output}")
    print(f"{SLIDE_COUNT} slides; {len(d.equations)} vector equations with PNG fallbacks")
    print("Smallest displayed equation:", min(v["effective_point_size"] for v in d.equations))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Choose a separate output path to preserve the source deck.")
    make_deck(args.source, args.output)
