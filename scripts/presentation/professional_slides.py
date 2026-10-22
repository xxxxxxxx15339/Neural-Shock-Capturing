"""Build the restrained Arial edition with actual LaTeX-rendered equations.

Prepare the equation source, compile it with Tectonic, then build the deck:
  python professional_slides.py --prepare --work-dir /tmp/WORK --source SOURCE
  tectonic --outdir /tmp/WORK /tmp/WORK/equations.tex
  python professional_slides.py --work-dir /tmp/WORK --source SOURCE

PowerPoint text and diagrams remain editable. Equation SVGs have transparent
PNG fallbacks, and the complete LaTeX source is retained alongside the review.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess

from lxml import etree as ET
from PIL import Image

import refine_slides as base
import research_refinements


ROOT = base.ROOT
REVIEW = ROOT / "Documentation/presentation_professional"
OUTPUT = ROOT / "Scientific_Internship_WENO_NN_UM6P_Professional_LaTeX.pptx"
ORDER = [1, 2, 3, 14, 4, 5, 19, 6, 15, 7, 8, 9, 16, 10, 18, 17]
DISPLAY_NUMBER = {number: index for index, number in enumerate(ORDER, 1)}
SLIDE_COUNT = len(ORDER)
INK, MUTED, LIGHT, LINE = "242424", "595959", "F5F5F5", "DEDEDE"
GOOD, BAD = "15803D", "B91C1C"
COLOR_MAP = {
    base.INK: INK,
    base.MUTED: MUTED,
    base.ORANGE: INK,
    base.TEAL: INK,
    base.LIGHT: LIGHT,
    base.BORDER: LINE,
    base.PALE_ORANGE: LIGHT,
    base.PALE_TEAL: LIGHT,
    "638E5B": MUTED,
    "D8E5EF": "E8E8E8",
    "849DAF": "777777",
    "C89644": "999999",
    "A85162": "BBBBBB",
}
ORIGINAL_EQUATION = base.Slide.equation


def neutral(color):
    return COLOR_MAP.get(color, color)


def scientific(value):
    mantissa, exponent = f"{value:.4e}".split("e")
    raised = str(int(exponent)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mantissa} × 10{raised}"


def professional_equation(self, value, x, y, w, h, size=23, color=base.INK, align="l"):
    # Keep only equations useful in the main defense narrative.
    if self.number in {11, 12, 13}:
        return None
    if self.number == 4:
        if value.startswith(r"\alpha_k="):
            return self.txt("Smooth region:\nspread trust across the candidates.", x, 5.08, w, .88, 16, base.MUTED)
        if value.startswith(r"\delta_k="):
            return self.txt("Near a shock:\nreject a candidate crossing the jump.", x, 5.08, w, .88, 16, base.MUTED)
        if value.startswith(r"\omega_k=") or value.endswith("q_{k^*}"):
            return None
        if value.startswith(r"\hat f_{i+1/2}=\omega_0"):
            value = r"\hat f_{i+1/2}=\omega_0q_0+\omega_1q_1+\omega_2q_2"
            w, size = 8.9, 26
    if self.number == 5:
        if value.startswith("x_j'"):
            return self.txt("Scale the local values\nbefore classification.", x, 3.05, w, .63, 16, base.MUTED)
        if value.startswith(r"s_k=\mathbf"):
            return self.txt("High probability → reject the candidate.", x, y, w, .38, 16, base.MUTED)
    if self.number == 6:
        if value.startswith(r"\mathcal{L}_{\mathrm{imit}}"):
            return self.txt("Target: classical TENO5 decisions.", x, y, w, .41, 17, base.MUTED)
        if value.startswith(r"U_{\theta}^{n+K}"):
            value = r"U_{\mathrm{pred}}=\mathcal{S}_{\theta}^{K}(U_{\mathrm{start}})"
        elif value.startswith(r"\mathcal{L}_{\mathrm{MSE}}"):
            return self.txt("Compare the predicted solution\nwith the physical reference.", x, y, w, .58, 17, base.MUTED)
        elif value.startswith(r"\mathcal{L}=\lambda"):
            value = r"\mathcal{L}=\mathrm{prediction\ error}+\mathrm{stability\ penalties}"
            size = 27
    if self.number == 7 and value.startswith(r"\rho>0"):
        value, y, h, size = r"\rho>0,\qquad p>0", 3.89, .54, 28
    if self.number == 8 and value.startswith("L_2="):
        return self.txt("L₂ measures solution error against a reference. Lower is better.", x, y, w, .56, 16, base.MUTED)
    if self.number == 15 and value.startswith("g="):
        return self.txt("Clip very large gradients\nbefore updating the network with Adam.", x, 5.02, w, .67, 18, base.MUTED)
    # Give retained display-style LaTeX fractions enough vertical space.
    if self.number == 2 and value.startswith("f(u)=a u"):
        y, h = 3.08, .68
    elif self.number == 3 and value.startswith(r"\Delta t\leq"):
        y, h = 6.27, .68
    elif self.number == 5 and value.startswith("x_j'"):
        y, h = 2.94, .78
    elif self.number == 7 and value.startswith(r"\rho>0"):
        y, h = 3.70, .76
    elif self.number == 11 and value.startswith(r"\mathcal{L}_{\mathrm{TV}}"):
        y, h = 2.97, .70
    elif self.number == 12 and value.startswith("W=L U"):
        y, h = 5.00, .60
    elif self.number == 4 and value == r"i+\frac{1}{2}":
        value, y, h = r"i+\tfrac{1}{2}", 1.92, .27
    return ORIGINAL_EQUATION(self, value, x, y, w, h, size=size, color=color, align=align)


def add_slide(deck, number):
    deck.parts[f"ppt/slides/slide{number}.xml"] = deck.parts["ppt/slides/slide12.xml"]
    deck.parts[f"ppt/slides/_rels/slide{number}.xml.rels"] = deck.parts["ppt/slides/_rels/slide12.xml.rels"]
    root = ET.fromstring(deck.parts["ppt/presentation.xml"])
    ids = root.find("p:sldIdLst", base.NS)
    next_id = max(int(child.get("id")) for child in ids) + 1
    entry = base.e("p:sldId", ids, id=next_id)
    rid = f"rIdProfessional{number}"
    entry.set(f"{{{base.NS['r']}}}id", rid)
    deck.parts["ppt/presentation.xml"] = base.xml(root)
    rels = ET.fromstring(deck.parts["ppt/_rels/presentation.xml.rels"])
    ET.SubElement(rels, f"{{{base.REL}}}Relationship", Id=rid,
                  Type=base.NS["r"] + "/slide", Target=f"slides/slide{number}.xml")
    deck.parts["ppt/_rels/presentation.xml.rels"] = base.xml(rels)
    types = ET.fromstring(deck.parts["[Content_Types].xml"])
    ET.SubElement(types, f"{{{base.CT}}}Override", PartName=f"/ppt/slides/slide{number}.xml",
                  ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
    deck.parts["[Content_Types].xml"] = base.xml(types)


def extend(deck):
    # The user's six-block workflow stays in the main solver slide.
    deck.equations = [item for item in deck.equations if item["slide"] != 3]
    deck.changes = [item for item in deck.changes if item["slide"] != 3]
    s = base.Slide(deck, 3, "The solver follows a conservative numerical path", "Numerical solver · complete workflow")
    s.txt("Forward numerical path", .65, 1.69, 11.99, .37, 20, bold=True)
    nodes = ["State +\nlocal stencil", "Scalar / Euler\nsetup", "WENO / TENO\nor NN selector", "Interface\nflux", "SSPRK3\nstep", "K-step\nrollout"]
    for index, label in enumerate(nodes):
        x = .60 + 2.057 * index
        s.panel(x, 2.15, 1.82, .84)
        s.txt(label, x + .08, 2.33, 1.66, .60, 17, bold=(index == 2), align="ctr")
        if index < 5:
            s.arrow(x + 1.82, 2.37, .235)
    s.shape(4.73, 3.17, 1.76, .04, fill_color=INK)
    s.txt("Learned component", 4.37, 3.34, 2.60, .33, 14, INK, bold=True, align="ctr")
    s.panel(.6, 3.94, 5.93, 2.39)
    s.panel(6.78, 3.94, 5.93, 2.39)
    s.txt("Conservation through shared fluxes", .85, 4.21, 5.45, .37, 20, bold=True)
    s.equation(r"\frac{d\bar u_i}{dt}=-\frac{\hat f_{i+1/2}-\hat f_{i-1/2}}{\Delta x}",
               .85, 4.84, 5.41, .74, size=26)
    s.txt("Each cell changes by its incoming and outgoing flux.", .85, 5.87, 5.41, .36, 15, base.MUTED)
    s.txt("Follow the path in order", 7.03, 4.21, 5.45, .37, 20, bold=True)
    s.txt("Set up the physical problem.\nReconstruct the interface flux.\nAdvance with SSPRK3, then repeat K steps.",
          7.03, 4.91, 5.36, 1.16, 18, base.MUTED, line_spacing=1.30)
    s.insight("Detailed slides follow this path: physical setup → reconstruction → fluxes → time stepping.", y=6.56)
    s.finish("Preserved the exact six-block workflow and its order; kept this slide as an overview and moved detailed flux-splitting / RK3 equations after the reconstruction slides and before rollout training.")

    # Keep the Euler model accessible: quantities and physical admissibility.
    add_slide(deck, 14)
    s = base.Slide(deck, 14, "Euler equations describe compressible flow", "Physical model · Euler system")
    s.txt("The system tracks three conserved quantities throughout the flow.", .65, 1.68, 12.0, .39, 18, base.MUTED)
    for index, (label, detail) in enumerate([
        ("Density", "How mass is distributed in the flow."),
        ("Momentum", "How the motion of the fluid is transported."),
        ("Total energy", "Internal and kinetic energy together."),
    ]):
        x = .6 + 4.13 * index
        s.panel(x, 2.31, 3.86, 1.45)
        s.txt(label, x + .23, 2.53, 3.4, .37, 22, bold=True)
        s.txt(detail, x + .23, 3.07, 3.4, .58, 17, base.MUTED)
    s.panel(.6, 4.10, 5.93, 2.15)
    s.panel(6.78, 4.10, 5.93, 2.15)
    s.txt("State used by the solver", .85, 4.34, 5.45, .37, 20, bold=True)
    s.equation(r"U=(\rho,\,\rho v,\,E)^{T}", .85, 5.02, 5.42, .59, size=30)
    s.txt("ρ: density   v: velocity   E: total energy", .85, 5.82, 5.42, .31, 15, base.MUTED)
    s.txt("Physical states must remain valid", 7.03, 4.34, 5.45, .37, 20, bold=True)
    s.equation(r"\rho>0,\qquad p>0", 7.03, 5.02, 5.42, .59, size=30)
    s.txt("Density and pressure must remain positive.", 7.03, 5.82, 5.42, .31, 15, base.MUTED)
    s.insight("Sod and Shu–Osher test the learned reconstruction in compressible Euler flows.", y=6.53)
    s.finish("Simplified the Euler slide to density, momentum, energy and positive density/pressure; removed flux matrices, the pressure closure and characteristic-speed algebra.")

    add_slide(deck, 15)
    s = base.Slide(deck, 15, "Stable training follows a staged procedure", "Training procedure")
    s.txt("Start from a reliable reconstruction, then increase the physical training horizon.",
          .64, 1.66, 12.0, .39, 18, base.MUTED)
    steps = [
        ("01  Pretrain", "Learn TENO5 decisions\nbefore physical training."),
        ("02  Curriculum", "Extend the rollout\nfrom the previous stage."),
        ("03  Guard states", "Check physical states\nat every SSPRK3 stage."),
        ("04  Validate", "Select the best model\non the full benchmarks."),
    ]
    for index, (label, detail) in enumerate(steps):
        x = .6 + 3.08 * index
        s.panel(x, 2.35, 2.88, 1.55)
        s.txt(label, x + .19, 2.57, 2.50, .34, 19, bold=True)
        s.txt(detail, x + .19, 3.12, 2.50, .66, 15.5, base.MUTED)
    s.panel(.6, 4.19, 5.93, 2.12)
    s.panel(6.78, 4.19, 5.93, 2.12)
    s.txt("Euler rollout curriculum", .85, 4.41, 5.42, .37, 20, bold=True)
    s.equation(r"K:\quad 2\longrightarrow4\longrightarrow8\longrightarrow15",
               .85, 5.08, 5.42, .48, size=25)
    s.txt("Each stage starts from the previous stage's weights.", .85, 5.80, 5.42, .33, 15, base.MUTED)
    s.txt("Bound the optimizer's gradient", 7.03, 4.41, 5.43, .37, 20, bold=True)
    s.equation(r"g=\nabla_\theta\mathcal{L},\qquad g_{\mathrm{clip}}=\frac{g}{\max(1,\|g\|_2)}",
               7.03, 4.99, 5.42, .69, size=24)
    s.txt("This prevents a single large update from disrupting training.", 7.03, 5.80, 5.41, .42, 14, base.MUTED)
    s.insight("Short-horizon training must be checked again at the full benchmark horizon.", y=6.52)
    s.finish("Added the reproducible pretraining, curriculum, positivity-guard and validation procedure, including the actual gradient-clipping threshold.")

    add_slide(deck, 16)
    metrics = json.loads((ROOT / "models/nn_teno5euler_jax_comparison_phase18.json").read_text())
    sod = metrics["problems"]["sod"]["beats_teacher"]
    shu = metrics["problems"]["shu_osher"]["beats_teacher"]
    rows = [
        ("Advection", "Scalar L₂", 9.8960e-2, 9.5521e-2),
        ("Burgers", "Scalar L₂", 8.1670e-4, 6.6493e-4),
        ("Sod", "Mean Euler L₁", sod["teno5euler"], sod["nn_teno5euler"]),
        ("Shu–Osher", "Mean Euler L₁", shu["teno5euler"], shu["nn_teno5euler"]),
    ]
    s = base.Slide(deck, 16, "Canonical benchmarks quantify the final gains", "Results · benchmark summary")
    s.txt("Learned convex mixing compared with classical TENO5 on the same benchmark configuration.",
          .63, 1.68, 12.00, .41, 18, base.MUTED)
    xs = [.60, 2.82, 5.01, 7.03, 9.10, 10.42]
    widths = [2.11, 2.09, 1.91, 1.96, 1.20, 2.29]
    headings = ["BENCHMARK", "ERROR METRIC", "TENO5", "NN-TENO5", "RATIO R", "CHANGE IN ERROR"]
    s.panel(.6, 2.32, 12.12, .56)
    for x, w, heading in zip(xs, widths, headings):
        s.txt(heading, x + .13, 2.48, w - .20, .29, 12.5, base.MUTED, bold=True)
    for index, (label, metric, teacher, network) in enumerate(rows):
        y = 2.98 + .73 * index
        if index % 2 == 0:
            s.panel(.6, y, 12.12, .65)
        ratio = network / teacher
        change = 100 * (ratio - 1)
        values = [label, metric, scientific(teacher), scientific(network), f"{ratio:.4f}", f"{change:+.2f}%"]
        for column, (x, w, value) in enumerate(zip(xs, widths, values)):
            s.txt(value, x + .13, y + .17, w - .20, .36,
                  17 if column in (0, 5) else 15.5,
                  (GOOD if change < 0 else BAD) if column >= 3 else INK,
                  bold=(column in (0, 5)))
        s.shape(.6, y + .69, 12.12, .011, fill_color=LINE)
    s.equation(r"R=\frac{\mathrm{error}_{\mathrm{NN}}}{\mathrm{error}_{\mathrm{TENO5}}},\qquad \mathrm{gain}(\%)=100(1-R)",
               .76, 6.07, 11.89, .55, size=24)
    s.txt("Green: lower error. Red: higher error. Euler uses mean L₁ over (ρ, ρv, E); scalar results use L₂.",
          .76, 6.74, 11.88, .27, 12, base.MUTED)
    s.finish("Added an editable table of final scalar and Euler errors, relative error ratios and signed percentage changes, with the metrics explicitly distinguished.")

    add_slide(deck, 17)
    s = base.Slide(deck, 17)
    s.txt("Thank you", .65, 2.66, 12.03, .93, 48, INK, bold=True, align="ctr")
    s.shape(6.11, 3.83, 1.11, .027, fill_color=LINE)
    s.txt("Questions & discussion", .65, 4.14, 12.03, .48, 22, MUTED, align="ctr")
    s.finish("Added a final Thank you slide with centered Arial typography and the presentation's neutral styling.")

    add_slide(deck, 18)
    add_slide(deck, 19)
    research_refinements.apply(deck)
    s = base.Slide(deck, 19, "Build the interface flux, then advance with RK3", "Numerical implementation · ordered solver steps")
    s.txt("Flux splitting → WENO / TENO or NN reconstruction → interface flux → SSPRK3 → K-step rollout.",
          .63, 1.69, 12.0, .39, 18, base.MUTED)
    s.panel(.6, 2.27, 5.93, 3.96)
    s.panel(6.78, 2.27, 5.93, 3.96)
    s.txt("01  Flux splitting and interface flux", .85, 2.52, 5.45, .37, 20, bold=True)
    s.equation(r"F^{\pm}(U)=\tfrac{1}{2}\left[F(U)\pm aU\right]",
               .85, 3.10, 5.41, .55, size=24)
    s.txt("WENO / TENO or NN reconstructs each split flux.", .85, 3.88, 5.41, .36, 15, base.MUTED)
    s.equation(r"\hat F_{i+1/2}=\hat F^{+}_{i+1/2}+\hat F^{-}_{i+1/2}",
               .85, 4.51, 5.41, .50, size=24)
    s.equation(r"L_i(U)=-\frac{\hat F_{i+1/2}-\hat F_{i-1/2}}{\Delta x}",
               .85, 5.34, 5.41, .66, size=24)
    s.arrow(6.54, 4.18, .23)
    s.txt("02  SSPRK3 · three stages", 7.03, 2.52, 5.45, .37, 20, bold=True)
    s.equation(r"U^{(1)}=U^n+\Delta t\,L(U^n)", 7.03, 3.10, 5.41, .55, size=24)
    s.txt("Recompute reconstruction and fluxes at each stage.", 7.03, 3.88, 5.41, .36, 15, base.MUTED)
    s.equation(r"U^{(2)}=\tfrac{3}{4}U^n+\tfrac{1}{4}\left[U^{(1)}+\Delta t L(U^{(1)})\right]",
               7.03, 4.51, 5.41, .50, size=22)
    s.equation(r"U^{n+1}=\tfrac{1}{3}U^n+\tfrac{2}{3}\left[U^{(2)}+\Delta t L(U^{(2)})\right]",
               7.03, 5.34, 5.41, .66, size=22)
    s.txt("L(U): spatial flux balance. a: wave-speed bound. Each RK stage recomputes reconstruction and fluxes.",
          .73, 6.40, 11.89, .29, 12.5, base.MUTED)
    s.txt("03  K-step rollout: repeat the complete SSPRK3 update K times.",
          .73, 6.78, 11.89, .23, 13, INK, bold=True)
    s.finish("Placed numerical details directly after final reconstruction and before rollout training. Ordered the formulas as splitting → reconstructed interface flux → flux-balance operator → RK3 stages → K-step rollout, matching the unchanged overview.")

    # Remove the backup slides from the package, not only from the slide order.
    removed = {11, 12, 13}
    for number in removed:
        deck.parts.pop(f"ppt/slides/slide{number}.xml", None)
        deck.parts.pop(f"ppt/slides/_rels/slide{number}.xml.rels", None)
    rels = ET.fromstring(deck.parts["ppt/_rels/presentation.xml.rels"])
    for rel in list(rels):
        match = re.fullmatch(r"slides/slide(\d+)\.xml", rel.get("Target", ""))
        if match and int(match.group(1)) in removed:
            rels.remove(rel)
    deck.parts["ppt/_rels/presentation.xml.rels"] = base.xml(rels)
    types = ET.fromstring(deck.parts["[Content_Types].xml"])
    for part in list(types):
        match = re.fullmatch(r"/ppt/slides/slide(\d+)\.xml", part.get("PartName", ""))
        if match and int(match.group(1)) in removed:
            types.remove(part)
    deck.parts["[Content_Types].xml"] = base.xml(types)

    # Use neutral styling, with green/red reserved for measured result values.
    for name, data in list(deck.parts.items()):
        if not name.endswith(".xml"):
            continue
        root = ET.fromstring(data)
        for node in root.xpath("//a:srgbClr", namespaces=base.NS):
            node.set("val", neutral(node.get("val")))
        for node in root.xpath("//a:latin | //a:ea | //a:cs", namespaces=base.NS):
            node.set("typeface", "Arial")
        if re.fullmatch(r"ppt/slides/slide\d+\.xml", name):
            physical = int(re.search(r"slide(\d+)", name).group(1))
            for node in root.xpath("//a:rPr[@sz='2900']", namespaces=base.NS):
                node.set("sz", "2700")
            for node in root.xpath("//a:t", namespaces=base.NS):
                value = node.text or ""
                if re.fullmatch(r"\d{2} / \d+", value):
                    node.text = f"{DISPLAY_NUMBER[physical]:02d} / {SLIDE_COUNT}"
                if physical == 4:
                    edits = {
                        "Select using divided differences.": "Keep one reliable candidate.",
                        "β: smoothness   χ: TENO sensor": "Combine with local weights.",
                    }
                    if value in edits:
                        node.text = edits[value]
                if physical == 6 and value == "Combined training objective":
                    node.text = "Training objective, in simple terms"
                if physical == 5:
                    if value == "Constant stencils use a guarded denominator.":
                        node.text = "Constant stencils receive a safe, fixed input."
                    elif value == "Optional hard inference: threshold decisions":
                        node.text = "Inference: keep or reject a candidate"
                if physical == 7:
                    if value.startswith("A dedicated sigmoid head starts"):
                        node.text = "The mixing head learns how much to trust the neural correction."
                    elif value.startswith("m = ρv;"):
                        node.text = "Density and pressure must remain positive."
                if physical == 8 and value.startswith("Advection: 9.8960"):
                    node.text = "Final gains come from the dedicated learned mixing head (Phase 19)."
                result_color = None
                if physical == 8 and value in {"3.5%", "18.6%"}:
                    result_color = GOOD
                elif physical == 9 and value.startswith("+0.35% error"):
                    result_color = BAD
                elif physical == 9 and value.startswith("−1.84% error"):
                    result_color = GOOD
                elif physical == 10 and value.startswith(("Scalar gains:", "Shu–Osher: 1.84%")):
                    result_color = GOOD
                elif physical == 10 and value.startswith("Sod: near parity"):
                    result_color = BAD
                if result_color:
                    props = node.getparent().find("a:rPr", base.NS)
                    props.find("a:solidFill/a:srgbClr", base.NS).set("val", result_color)
        if name.startswith("ppt/theme/"):
            # The deck explicitly styles its components, but keep fallback themes neutral.
            for slot in root.xpath("//a:clrScheme/*[starts-with(local-name(),'accent')]", namespaces=base.NS):
                for child in list(slot):
                    slot.remove(child)
                base.e("a:srgbClr", slot, val=INK)
            for slot in root.xpath("//a:clrScheme/a:hlink | //a:clrScheme/a:folHlink", namespaces=base.NS):
                for child in list(slot):
                    slot.remove(child)
                base.e("a:srgbClr", slot, val=MUTED)
        deck.parts[name] = base.xml(root)

    root = ET.fromstring(deck.parts["ppt/presentation.xml"])
    rels = ET.fromstring(deck.parts["ppt/_rels/presentation.xml.rels"])
    physical_by_rid = {
        rel.get("Id"): int(re.search(r"slide(\d+)\.xml", rel.get("Target")).group(1))
        for rel in rels if rel.get("Type").endswith("/slide")
    }
    ids = root.find("p:sldIdLst", base.NS)
    entries = {physical_by_rid[entry.get(f"{{{base.NS['r']}}}id")]: entry for entry in ids
               if entry.get(f"{{{base.NS['r']}}}id") in physical_by_rid}
    for entry in list(ids):
        ids.remove(entry)
    for physical in ORDER:
        ids.append(entries[physical])
    deck.parts["ppt/presentation.xml"] = base.xml(root)
    deck.equations = [item for item in deck.equations if item["slide"] in DISPLAY_NUMBER]
    deck.changes = [item for item in deck.changes if item["slide"] in DISPLAY_NUMBER]
    for item in deck.equations:
        item["slide"] = DISPLAY_NUMBER[item["slide"]]
        item["renderer"] = "LaTeX / Tectonic / Latin Modern"
    deck.equations.sort(key=lambda item: item["slide"])
    updates = {
        4: "Preserved the five-point stencil and ENO/WENO/TENO comparison; kept only the weighted-candidate formula and replaced sensor/normalization algebra with plain-language explanations.",
    }
    for item in deck.changes:
        item["change"] = updates.get(item["slide"], item["change"])
        item["slide"] = DISPLAY_NUMBER[item["slide"]]
    deck.changes.sort(key=lambda item: item["slide"])


def monochrome_instability():
    import csv
    import matplotlib.pyplot as plt
    with (ROOT / "models/euler_instability_trace.csv").open() as source:
        rows = list(csv.DictReader(source))
    fig, ax = plt.subplots(figsize=(6, 3))
    styles = ["-", "--", "-.", ":", (0, (4, 1, 1, 1, 1, 1))]
    shades = ["242424", "454545", "656565", "858585", "A5A5A5"]
    for margin, style, shade in zip(sorted({float(r["alpha_margin"]) for r in rows}), styles, shades):
        group = [r for r in rows if float(r["alpha_margin"]) == margin]
        finite = [r for r in group if r["finite"] == "True"]
        ax.plot([int(r["step"]) for r in finite], [float(r["max_density"]) for r in finite],
                color="#" + shade, linestyle=style, linewidth=1.9, marker="o", markersize=2.7,
                label=f"{margin:g}× dissipation")
        if finite and any(r["finite"] != "True" for r in group):
            ax.scatter([int(finite[-1]["step"]) + 1], [float(finite[-1]["max_density"])],
                       color="#" + shade, marker="x", s=30)
    base.styled_axis(ax)
    ax.set_yscale("log")
    ax.set_xlabel("SSPRK3 step", fontsize=11, color="#" + MUTED)
    ax.set_ylabel("Maximum density (log scale)", fontsize=11, color="#" + MUTED)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(pad=.9)
    return base.plot_bytes(fig)


def job_key(expression, fontsize, color):
    return json.dumps([expression, fontsize, neutral(color)])


def latex_source(jobs):
    lines = [
        "% Actual LaTeX source for all formulas in the professional slide edition.",
        r"\documentclass{article}",
        r"\usepackage{amsmath,amssymb,lmodern,xcolor}",
        r"\usepackage[active,tightpage]{preview}",
        r"\setlength\PreviewBorder{1pt}",
        r"\begin{document}",
    ]
    for index, job in enumerate(jobs, 1):
        expression = job["latex"]
        lines.extend([
            f"% Equation {index:03d}",
            r"\begin{preview}",
            r"\hbox{",
            f"\\color[HTML]{{{job['color']}}}",
            f"\\fontsize{{{job['fontsize']}}}{{{job['fontsize'] * 1.2:.2f}}}\\selectfont",
            r"$\displaystyle " + expression + "$",
            "}",
            r"\end{preview}",
        ])
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def convert_equations(work, jobs):
    pdf = work / "equations.pdf"
    if not pdf.exists():
        raise RuntimeError(f"Compile {work / 'equations.tex'} with LaTeX before building.")
    subprocess.run(["pdftocairo", "-png", "-transp", "-r", "270", str(pdf), str(work / "equation")], check=True)
    # Poppler's PNG suffix uses the number of digits required for the page count.
    pngs = sorted(work.glob("equation-*.png"))
    if len(pngs) != len(jobs):
        raise RuntimeError(f"Expected {len(jobs)} equation pages, found {len(pngs)}.")

    def convert(index):
        target = work / f"equation-{index:03d}.svg"
        subprocess.run(["pdftocairo", "-svg", "-f", str(index), "-l", str(index), str(pdf), str(target)], check=True)
        root = ET.parse(str(target)).getroot()
        return target.read_bytes(), float(root.get("width").removesuffix("pt")) / 72, float(root.get("height").removesuffix("pt")) / 72

    with ThreadPoolExecutor(max_workers=4) as pool:
        vectors = list(pool.map(convert, range(1, len(jobs) + 1)))
    arts = {}
    for job, png, (svg, width, height) in zip(jobs, pngs, vectors):
        arts[job_key(job["latex"], job["fontsize"], job["color"])] = (png.read_bytes(), svg, width, height)
    return arts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    base.FONT = "Arial"
    base.plt.rcParams["font.family"] = "Liberation Sans"
    base.SLIDE_COUNT = SLIDE_COUNT
    base.Slide.equation = professional_equation
    base.instability_plot = monochrome_instability
    if args.prepare:
        jobs, seen = [], set()
        blank = BytesIO()
        Image.new("RGBA", (100, 20), (0, 0, 0, 0)).save(blank, format="PNG")

        def collect(expression, fontsize=24, color=base.INK):
            key = job_key(expression, fontsize, color)
            if key not in seen:
                jobs.append({"latex": expression, "fontsize": fontsize, "color": neutral(color)})
                seen.add(key)
            return blank.getvalue(), None, 1.0, .2

        base.math_art = collect
        base.make_deck(args.source, args.work_dir / "layout_staging.pptx",
                       review_dir=args.work_dir / "staging_review", extend_deck=extend)
        source = latex_source(jobs)
        (args.work_dir / "equations.tex").write_text(source)
        (REVIEW / "equations.tex").write_text(source)
        (args.work_dir / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
        print(f"Prepared {len(jobs)} actual LaTeX equation pages.")
    else:
        jobs = json.loads((args.work_dir / "jobs.json").read_text())
        arts = convert_equations(args.work_dir, jobs)

        def rendered(expression, fontsize=24, color=base.INK):
            return arts[job_key(expression, fontsize, color)]

        base.math_art = rendered
        base.make_deck(args.source, args.output, review_dir=REVIEW, extend_deck=extend)
        (REVIEW / "equations.tex").write_text((args.work_dir / "equations.tex").read_text())
        minimum = min(item["effective_point_size"] for item in json.loads((REVIEW / "equations.json").read_text())
                      if not (item["slide"] == 5 and item["effective_point_size"] < 18))
        print(f"{SLIDE_COUNT}-slide defense edition complete. Smallest main equation: {minimum} pt.")


if __name__ == "__main__":
    main()
