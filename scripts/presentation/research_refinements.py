"""Research-story refinements, using only the essential defense equations.

The physical slide numbers refer to the source package. The parent builder
renumbers them into the final presentation order after applying this module.
"""

from __future__ import annotations

import json
from lxml import etree as ET

import refine_slides as base

ROOT = base.ROOT
FIGURES = ROOT / "Documentation/presentation_professional/figures"
INK, MUTED, LIGHT, LINE = "242424", "595959", "F5F5F5", "DEDEDE"
GOOD, BAD = "15803D", "B91C1C"


def redraw(deck, number, title, section):
    deck.equations = [item for item in deck.equations if item["slide"] != number]
    deck.changes = [item for item in deck.changes if item["slide"] != number]
    return base.Slide(deck, number, title, section)


def down(s, x, y):
    s.txt("↓", x - .18, y, .36, .37, 21, MUTED, align="ctr")


def scientific(value):
    mantissa, exponent = f"{value:.4e}".split("e")
    raised = str(int(exponent)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mantissa} × 10{raised}"


def hyperlink(s, shape, url):
    s.next_rid += 1
    rid = f"rId{s.next_rid}"
    ET.SubElement(s.rels, f"{{{base.REL}}}Relationship", Id=rid,
                  Type=base.NS["r"] + "/hyperlink", Target=url, TargetMode="External")
    for props in shape.xpath(".//a:rPr", namespaces=base.NS):
        link = base.e("a:hlinkClick", props)
        link.set(f"{{{base.NS['r']}}}id", rid)


def motivation(deck):
    s = redraw(deck, 2, "Can learned reconstruction improve TENO5?", "Motivation · research question")
    s.picture(deck.parts["ppt/media/image2.png"], .6, 1.70, 6.00, 4.43,
              name="Original shock-wave photograph")
    s.txt("Sharp fronts require both accuracy and control of oscillations.", .63, 6.19, 5.97, .35, 13, MUTED)
    s.panel(6.95, 1.70, 5.77, 2.04)
    s.txt("Research question", 7.20, 1.94, 5.25, .38, 21, INK, bold=True)
    s.txt("Can a small learned correction reduce\nsolution error relative to TENO5,\nwhile keeping Euler rollouts stable?",
          7.20, 2.52, 5.25, 1.00, 20, INK, line_spacing=1.18)
    for y, heading, detail in [
        (4.04, "What changes?", "Candidate decisions and a learned mixing coefficient."),
        (4.93, "What stays?", "The conservative flux update and SSPRK3 solver."),
        (5.82, "How is it tested?", "Four 1D benchmarks against physical references."),
    ]:
        s.txt(heading, 7.02, y, 5.6, .34, 18, INK, bold=True)
        s.txt(detail, 7.02, y + .40, 5.63, .48, 16, MUTED)
    s.insight("Aim: improve reconstruction inside the solver—not replace the physical equations.", y=6.57)
    s.finish("Made the research question explicit, identifying the learned decisions, preserved solver and four-test evaluation without adding another equation.")


def architecture(deck):
    s = redraw(deck, 5, "The final model learns decisions and their influence", "Contribution · final NN-TENO5 architecture")
    s.txt("Two reconstruction paths share the same local data; a small head controls their blend.",
          .63, 1.67, 12.00, .40, 18, MUTED)
    s.panel(.6, 2.20, 8.00, .54)
    s.txt("Local five-point stencil", .85, 2.32, 7.5, .31, 19, INK, bold=True, align="ctr")
    down(s, 2.53, 2.80)
    down(s, 6.66, 2.80)
    s.txt("Only 4 trainable parameters", 8.88, 2.36, 3.82, .32, 15, MUTED, align="ctr")
    for x in (.6, 4.73, 8.86):
        s.panel(x, 3.18, 3.86, 1.60)
    s.txt("Classical TENO5", .83, 3.40, 3.40, .35, 20, INK, bold=True)
    s.txt("Baseline reconstruction", .83, 3.92, 3.40, .29, 15.5, MUTED)
    s.equation(r"\Phi_{\mathrm{TENO5}}", .83, 4.29, 3.40, .34, size=25)
    s.txt("Neural reconstruction", 4.96, 3.40, 3.40, .35, 20, INK, bold=True)
    s.txt("Classifier: 5 → 64 → 32 → 16 → 8 → 3", 4.96, 3.95, 3.40, .30, 13, MUTED)
    s.equation(r"\Phi_{\mathrm{NN}}", 4.96, 4.29, 1.23, .34, size=25)
    s.txt("Outputs: p₀, p₁, p₂", 6.35, 4.35, 2.02, .26, 13.5, MUTED)
    s.arrow(8.59, 3.99, .27)
    s.txt("Dedicated mixing head", 9.09, 3.40, 3.40, .35, 20, INK, bold=True)
    s.txt("Three probabilities → sigmoid → λ", 9.09, 3.95, 3.40, .30, 13.5, MUTED)
    s.equation(r"0\leq\lambda\leq1", 9.09, 4.29, 3.40, .34, size=25)
    for x in (2.53, 6.66, 10.79):
        down(s, x, 4.85)
    s.panel(.6, 5.25, 12.12, .81)
    s.equation(r"\Phi_{\mathrm{final}}=(1-\lambda)\Phi_{\mathrm{TENO5}}+\lambda\Phi_{\mathrm{NN}}",
               .88, 5.34, 11.55, .61, size=30, align="ctr")
    down(s, 6.66, 6.10)
    s.txt("Conservative flux update + SSPRK3", .8, 6.51, 11.75, .35, 20, INK, bold=True, align="ctr")
    s.finish("Replaced the classifier-only diagram with both reconstruction branches, the actual 3-to-1 sigmoid head receiving classifier probabilities, the convex blend and the unchanged conservative time update.")


def physics_loss(deck):
    s = redraw(deck, 6, "Train on the solution after real solver steps", "Method · from imitation to solver-in-the-loop")
    s.txt("Imitation learns TENO5 labels. Physics training instead optimizes the evolved solution.",
          .63, 1.68, 12.00, .40, 18, MUTED)
    boxes = [(.6, 3.25, "Start from a physical state"),
             (4.53, 3.95, "Run K complete SSPRK3 steps"),
             (9.16, 3.56, "Compare with the reference")]
    for x, w, label in boxes:
        s.panel(x, 2.29, w, .72)
        s.txt(label, x + .14, 2.50, w - .28, .31, 17, INK, bold=True, align="ctr")
    s.arrow(3.95, 2.49, .42)
    s.arrow(8.60, 2.49, .42)
    s.txt("The training gradient passes through reconstruction, fluxes and every time-integration stage.",
          .70, 3.19, 11.92, .36, 15.5, MUTED, align="ctr")
    s.equation(r"\boxed{\mathcal{L}_{\mathrm{total}}=\alpha_1\mathcal{L}_{\mathrm{MSE}}+\alpha_2\mathcal{L}_{\mathrm{TV}}+\alpha_3\mathcal{L}_{\mathrm{BVD}}+\alpha_4\mathcal{L}_{\mathrm{Reg}}}",
               .74, 3.79, 11.87, .65, size=30, align="ctr")
    s.txt("The α weights balance the loss terms; λ is reserved for the learned reconstruction mix.",
          .70, 4.60, 11.92, .30, 14, MUTED, align="ctr")
    cards = [
        ("MSE · accuracy", "Fit the reference\nsolution after K steps."),
        ("TV · oscillations", "Penalize excess variation\nrelative to the start."),
        ("BVD · interfaces", "Penalize extra interface\njumps beyond TENO5."),
        ("Reg · anchoring", "Keep weights close to\nthe pretrained model."),
    ]
    for index, (label, detail) in enumerate(cards):
        x = .6 + 3.08 * index
        s.panel(x, 5.12, 2.88, 1.32)
        s.txt(label, x + .19, 5.32, 2.50, .32, 18, INK, bold=True)
        s.txt(detail, x + .19, 5.80, 2.50, .47, 14.5, MUTED)
    s.txt("Euler also adds a positivity penalty, measured before state repair; TV and Reg are normalized.",
          .70, 6.62, 11.95, .31, 13.5, MUTED)
    s.finish("Restored the essential four-term physics loss with plain-language meanings, a K-step solver workflow and an explicit note that Euler additionally penalizes pre-repair positivity violations. No term derivations were added.")


def instability(deck):
    s = redraw(deck, 15, "Euler instability shaped the training procedure", "Research progression · diagnose, then repair")
    s.txt("What failed", .67, 1.71, 5.9, .38, 22, INK, bold=True)
    s.txt("Historical training trace—not the final benchmark runs.", .67, 2.23, 5.9, .35, 14, MUTED)
    s.picture(base.instability_plot(), .6, 2.74, 6.03, 3.18,
              name="Historical Euler instability trace from models/euler_instability_trace.csv")
    s.txt("Invalid density or pressure destabilizes the\nRoe characteristic reconstruction.",
          .75, 5.92, 5.8, .57, 15.5, INK)
    s.txt("What changed", 7.03, 1.71, 5.44, .38, 22, INK, bold=True)
    steps = [
        ("01  Detect the real violation", "Penalize invalid states before repair—not afterward."),
        ("02  Repair every SSPRK3 stage", "Treat isolated violations; guard nonfinite states."),
        ("03  Normalize and bound training", "Scale TV / Reg terms and clip large gradients."),
        ("04  Extend the horizon gradually", "Warm-start the Euler curriculum: K = 2 → 4 → 8 → 15."),
    ]
    for index, (heading, detail) in enumerate(steps):
        y = 2.31 + .97 * index
        s.panel(6.87, y, 5.85, .82)
        s.txt(heading, 7.05, y + .13, 5.45, .30, 17.5, INK, bold=True)
        s.txt(detail, 7.05, y + .49, 5.45, .29, 14, MUTED)
    s.insight("Stable training enables evaluation; it does not by itself prove better accuracy.", y=6.57)
    s.finish("Made the real Euler failure trace prominent and separated diagnosis from the four concrete repairs: pre-repair penalties, substage guards, normalized losses / clipped gradients, and the K=2-to-15 curriculum.")


def mixing_lessons(deck):
    s = redraw(deck, 7, "What mixing solved—and what it did not", "Method · lessons and safeguards")
    s.txt("A small neural correction keeps the final reconstruction close to the classical baseline.",
          .63, 1.68, 12.00, .39, 18, MUTED)
    stages = ["Imitation", "Physics training", "Euler repairs", "Learned mixing"]
    for index, stage in enumerate(stages):
        x = .6 + 3.08 * index
        s.panel(x, 2.25, 2.88, .58)
        s.txt(stage, x + .10, 2.40, 2.68, .29, 17, INK, bold=True, align="ctr")
        if index < 3:
            s.arrow(x + 2.89, 2.39, .19)
    cards = [
        ("Separate two decisions", "Classifier: which candidates\nto reject.\n\nMixing head: how much of the\nneural reconstruction to use."),
        ("Begin close to TENO5", "The head starts near a\n5% neural contribution.\n\nA small scalar blend already\nhelps; training improves it further."),
        ("Keep guarantees separate", "Convex mixing is not a\npositivity or TVD proof.\n\nState repairs are safeguards,\nnot mathematical guarantees."),
    ]
    for index, (heading, detail) in enumerate(cards):
        x = .6 + 4.13 * index
        s.panel(x, 3.24, 3.86, 2.49)
        s.txt(heading, x + .23, 3.49, 3.40, .36, 19, INK, bold=True)
        s.txt(detail, x + .23, 4.09, 3.40, 1.40, 16, MUTED, line_spacing=1.10)
    s.txt("Final Sod and Shu–Osher runs remain finite and positive without invoking state repairs.",
          .71, 6.02, 11.94, .36, 16, INK)
    s.insight("Sod still has slightly higher error: better mixing alone has not removed the backbone's limit.",
              y=6.61, color=BAD)
    s.finish("Linked the research stages and explained separate decision / mixing parameters, conservative initialization and the contribution of training. Distinguished finite final runs from formal positivity / TVD guarantees and retained Sod's limitation.")


def final_results(deck):
    configs = [
        (8, "Final scalar models improve both benchmarks", "Results · final scalar checkpoint",
         [("advection", "Advection", "3.5% lower L₂ error", "64 cells · CFL 0.5 · t = 1.0 · exact reference", GOOD),
          ("burgers", "Burgers", "18.6% lower L₂ error", "64 cells · CFL 0.5 · t = 0.5 · exact reference", GOOD)],
         "Burgers is evaluated before shock formation. Each lower panel shows a separate local-error zoom."),
        (9, "Euler gains depend on the benchmark", "Results · final Euler checkpoint",
         [("sod", "Sod", "+0.35% error · slightly worse", "120 cells · t = 0.2 · exact Riemann reference", BAD),
          ("shu_osher", "Shu–Osher", "−1.84% error · modest improvement", "120 cells · t = 1.8 · 320-cell TENO5 reference", GOOD)],
         "CFL 0.4. Top: density. Local errors and percentages use mean L₁ over density, momentum and energy."),
    ]
    for physical, title, section, cases, footer in configs:
        s = redraw(deck, physical, title, section)
        for index, (name, label, result, condition, color) in enumerate(cases):
            x = .6 + 6.18 * index
            s.txt(label, x + .07, 1.67, 5.81, .39, 22, INK, bold=True)
            s.txt(result, x + .07, 2.14, 5.81, .32, 18, color, bold=True)
            s.txt(condition, x + .07, 2.60, 5.81, .28, 12.5, MUTED)
            s.picture((FIGURES / f"{name}_final.png").read_bytes(), x, 2.96, 5.95, 3.59,
                      name=f"Final TENO5 and NN-TENO5 versus reference: {label}; solution and local error")
        s.txt(footer, .70, 6.73, 11.96, .24, 12, MUTED)
        s.finish("Replaced historical / explanatory figures with final saved-checkpoint TENO5 and NN-TENO5 solution overlays and non-overlapping local-error panels; stated grid, time, reference and metric, with favorable results green and Sod red.")


def summary_table(deck):
    records = json.loads((FIGURES / "final_comparison_metrics.json").read_text())
    s = redraw(deck, 16, "Canonical benchmarks quantify the final gains", "Results · benchmark summary")
    s.txt("Learned convex mixing compared with classical TENO5 on the same benchmark configuration.",
          .63, 1.68, 12.00, .41, 18, MUTED)
    xs = [.60, 2.82, 5.01, 7.03, 9.10, 10.42]
    widths = [2.11, 2.09, 1.91, 1.96, 1.20, 2.29]
    headings = ["BENCHMARK", "ERROR METRIC", "TENO5", "NN-TENO5", "RATIO R", "CHANGE IN ERROR"]
    s.panel(.6, 2.32, 12.12, .56)
    for x, w, heading in zip(xs, widths, headings):
        s.txt(heading, x + .13, 2.48, w - .20, .29, 12.5, MUTED, bold=True)
    for index, (name, label) in enumerate([("advection", "Advection"), ("burgers", "Burgers"),
                                           ("sod", "Sod"), ("shu_osher", "Shu–Osher")]):
        row = records[name]
        y = 2.98 + .73 * index
        if index % 2 == 0:
            s.panel(.6, y, 12.12, .65)
        color = GOOD if row["change_percent"] < 0 else BAD
        values = [label, "Scalar L₂" if index < 2 else "Mean Euler L₁", scientific(row["teno"]),
                  scientific(row["nn"]), f"{row['ratio']:.4f}", f"{row['change_percent']:+.2f}%"]
        for column, (x, w, value) in enumerate(zip(xs, widths, values)):
            s.txt(value, x + .13, y + .17, w - .20, .36,
                  17 if column in (0, 5) else 15.5,
                  color if column >= 3 else INK, bold=column in (0, 5))
        s.shape(.6, y + .69, 12.12, .011, fill_color=LINE)
    s.equation(r"R=\frac{\mathrm{error}_{\mathrm{NN}}}{\mathrm{error}_{\mathrm{TENO5}}},\qquad \mathrm{gain}(\%)=100(1-R)",
               .76, 6.00, 11.89, .54, size=24)
    s.txt("Green: lower error. Red: higher error. Euler: mean L₁ over (ρ, ρv, E). Scalar: L₂.",
          .76, 6.59, 11.88, .23, 12, MUTED)
    s.txt("Scope: four 1D cases at fixed grids—not a convergence study or a controlled runtime comparison.",
          .76, 6.85, 11.88, .20, 11, MUTED)
    s.finish("Preserved the requested six-column results table and green/red values, recomputed ratios from unrounded final checkpoint errors, and added fixed-grid / convergence / runtime validation limits.")


def conclusion(deck):
    s = redraw(deck, 10, "A solver-integrated correction with measured limits", "Conclusion · scientific contribution")
    s.txt("The contribution is the full tested workflow—not a classifier in isolation.",
          .63, 1.68, 12.0, .39, 18, MUTED)
    cards = [
        ("Differentiable reconstruction", "TENO5 inside a differentiable\nconservative solver.\n\nTrain on evolved states using\nthe four-term physics loss."),
        ("Euler diagnosis and repairs", "Identify invalid-state failures.\n\nCombine pre-repair penalties,\nsubstage safeguards and\na rollout curriculum."),
        ("Dedicated learned mixing", "Separate candidate decisions\nfrom their final influence.\n\nA small mixing head retains\nthe classical baseline."),
    ]
    for index, (heading, detail) in enumerate(cards):
        x = .6 + 4.13 * index
        s.panel(x, 2.36, 3.86, 2.69)
        s.txt(heading, x + .23, 2.62, 3.40, .38, 17.5, INK, bold=True)
        s.txt(detail, x + .23, 3.23, 3.40, 1.52, 16.5, MUTED, line_spacing=1.12)
    s.panel(.6, 5.37, 12.12, .71)
    s.txt("Measured evidence", .85, 5.59, 3.00, .31, 18, INK, bold=True)
    s.txt("3 of 4 benchmarks improve", 4.03, 5.59, 4.1, .31, 18, GOOD, bold=True)
    s.txt("Sod: +0.35% error", 8.73, 5.59, 3.7, .31, 18, BAD, bold=True)
    s.txt("Next: more grids and cases, backbone improvement, and a controlled accuracy–cost study.",
          .70, 6.36, 11.97, .33, 16, INK)
    s.txt("No established convergence result, general accuracy guarantee, or formal positivity / TVD proof.",
          .70, 6.79, 11.97, .22, 12.5, MUTED)
    s.finish("Made the differentiable solver / physics loss, Euler diagnosis and dedicated mixing head the specific contributions; retained the 3-of-4 outcome and limitations, and removed the isolated Discussion label.")


def references(deck):
    s = redraw(deck, 18, "Selected references", "Academic grounding")
    papers = [
        ("Stevens & Colonius · 2020", "Enhancement of shock-capturing methods via machine learning.",
         "Theoretical and Computational Fluid Dynamics, 34, 483–496.",
         "DOI: 10.1007/s00162-020-00531-1", "https://doi.org/10.1007/s00162-020-00531-1"),
        ("Fu, Hu & Adams · 2016", "A family of high-order targeted ENO schemes for compressible-fluid simulations.",
         "Journal of Computational Physics, 305, 333–359.",
         "DOI: 10.1016/j.jcp.2015.10.037", "https://doi.org/10.1016/j.jcp.2015.10.037"),
        ("Li, Fu & Adams · 2022", "A six-point neuron-based ENO (NENO6) scheme for compressible fluid dynamics.",
         "arXiv preprint 2207.08500.",
         "arXiv: 2207.08500", "https://arxiv.org/abs/2207.08500"),
    ]
    for index, (authors, title, venue, link_label, url) in enumerate(papers):
        y = 1.80 + 1.52 * index
        s.panel(.6, y, 12.12, 1.33)
        s.txt(authors, .86, y + .17, 11.55, .37, 21, INK, bold=True)
        s.txt(title, .86, y + .65, 11.55, .33, 17, INK)
        s.txt(venue, .86, y + 1.05, 6.27, .24, 12, MUTED)
        shape = s.txt(link_label, 7.40, y + 1.05, 5.0, .24, 12, MUTED, align="r")
        hyperlink(s, shape, url)
    s.txt("Related foundations: NENO6 uses six stencil points; the model presented here uses five.",
          .71, 6.55, 11.94, .34, 14, MUTED)
    s.finish("Added the three requested literature references, checked against primary sources, with clickable DOI / arXiv links. Kept this slide before the final Thank you.")


def apply(deck):
    motivation(deck)
    architecture(deck)
    physics_loss(deck)
    instability(deck)
    mixing_lessons(deck)
    final_results(deck)
    summary_table(deck)
    conclusion(deck)
    references(deck)
