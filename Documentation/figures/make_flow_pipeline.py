"""Figure « pipeline visuel » : de l'avion reel jusqu'a l'avancement en temps.

Chaque panneau porte des donnees reelles :
  (a) photographie schlieren NASA d'un T-38, avec la coupe 1D
  (b) signal d'intensite effectivement extrait des pixels le long de la coupe
  (c) discretisation en cellules + stencil a 5 points centre sur un choc
  (d) les trois candidats TENO5 et la decision garder/ecarter
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from PIL import Image

HERE = __file__.rsplit("/", 1)[0]

import os
LANG = os.environ.get("FIGLANG", "fr")
T = {
 "fr": dict(cut="coupe 1D", shock="choc", kept="gardé", disc="écarté",
            decision="décision garder / écarter\n"
                     r"$\longrightarrow$ remplacée par le réseau",
            t_a="(a)  Écoulement réel\nchocs autour d'un T-38 (NASA)",
            t_b="(b)  Coupe 1D\nsignal $u(x)$",
            t_c="(c)  Discrétisation\nstencil à 5 points",
            t_d="(d)  Reconstruction\nWENO5 / TENO5 / NN-TENO5",
            flux=r"Flux $\hat f_{i+1/2}$  —  séparation de Lax–Friedrichs",
            rk=r"Intégration en temps : SSPRK3 $\;\to\; u^{n+1}$",
            nextstep="pas de temps suivant"),
 "en": dict(cut="1D cut", shock="shock", kept="kept", disc="discarded",
            decision="keep / discard decision\n"
                     r"$\longrightarrow$ replaced by the network",
            t_a="(a)  Real flow\nshock waves around a T-38 (NASA)",
            t_b="(b)  1D cut\nsignal $u(x)$",
            t_c="(c)  Discretisation\n5-point stencil",
            t_d="(d)  Reconstruction\nWENO5 / TENO5 / NN-TENO5",
            flux=r"Flux $\hat f_{i+1/2}$  —  Lax–Friedrichs splitting",
            rk=r"Time integration: SSPRK3 $\;\to\; u^{n+1}$",
            nextstep="next time step"),
}[LANG]
SUF = "" if LANG == "fr" else "_en"

ACC = "#117078"     # teal
ORA = "#e04a2a"     # orange UM6P
GREY = "#5a5a5a"

# ----------------------------------------------------------------------
# Donnees : signal reel extrait de la photo schlieren
# ----------------------------------------------------------------------
full = np.asarray(Image.open(f"{HERE}/t38_schlieren.jpg").convert("L"), float)
img = full[int(0.10 * full.shape[0]):int(0.94 * full.shape[0]),
           int(0.18 * full.shape[1]):int(0.80 * full.shape[1])]
H, W = img.shape
row = int(0.60 * H)

band = img[row - 45:row + 46, :].mean(axis=0)
k = 41
band = np.convolve(band, np.ones(k) / k, mode="valid")
band = (band - band.min()) / (band.max() - band.min())
xs = np.linspace(0.0, 1.0, band.size)

# Front de choc le plus net -> centre du stencil
grad = np.gradient(band)
i_shock = int(np.argmin(grad))
x_shock = xs[i_shock]

# ----------------------------------------------------------------------
# Mise en page : 4 panneaux en ligne + fleches
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(11.4, 9.3))
fig.patch.set_facecolor("white")

# Grille 2x2 : (a) (b) / (c) (d), puis un bandeau aval en bas
PW, PH = 0.355, 0.235
XL, XR = 0.075, 0.545
YT, YB2 = 0.705, 0.320
axA = fig.add_axes([XL, YT, PW, PH])
axB = fig.add_axes([XR, YT, PW, PH])
axC = fig.add_axes([XL, YB2, PW, PH])
axD = fig.add_axes([XR, YB2, PW, PH])
axes = [axA, axB, axC, axD]

# ---------- (a) l'avion et ses chocs ----------
axA.imshow(img, cmap="gray", aspect="auto")
axA.axhline(row, color=ORA, lw=2.0, ls="--")
axA.set_xlim(0, W); axA.set_ylim(H, 0)
axA.set_xticks([]); axA.set_yticks([])
axA.text(0.025, row / H - 0.035, T["cut"], color=ORA, fontsize=10,
         fontweight="bold", transform=axA.transAxes, va="bottom")
for s in axA.spines.values():
    s.set_edgecolor(GREY); s.set_linewidth(0.8)

# ---------- (b) le signal 1D ----------
axB.plot(xs, band, color=ACC, lw=1.7)
axB.axvline(x_shock, color=ORA, lw=1.2, ls=":")
axB.annotate(T["shock"], xy=(x_shock, band[i_shock]),
             xytext=(x_shock + 0.22, 0.93), color=ORA, fontsize=9.5,
             fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=ORA, lw=1.2))
axB.set_xlim(0, 1); axB.set_ylim(-0.05, 1.12)
axB.set_xlabel("$x$", fontsize=10)
axB.set_ylabel("$u(x)$", fontsize=10)
axB.set_yticks([]); axB.set_xticks([0, 0.5, 1])
axB.grid(alpha=0.2, lw=0.5)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)

# ---------- (c) discretisation + stencil ----------
n_cells = 26
edges = np.linspace(0, 1, n_cells + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
cell_val = np.array([band[(xs >= a) & (xs < b)].mean()
                     if ((xs >= a) & (xs < b)).any() else np.nan
                     for a, b in zip(edges[:-1], edges[1:])])
cell_val = np.where(np.isnan(cell_val), np.interp(
    centers, centers[~np.isnan(cell_val)], cell_val[~np.isnan(cell_val)]),
    cell_val)

i_c = int(np.argmin(np.abs(centers - x_shock)))
i_c = min(max(i_c, 2), n_cells - 3)
sten = list(range(i_c - 2, i_c + 3))

axC.plot(xs, band, color="#b8b8b8", lw=1.0, zorder=1)
for j, (c, v) in enumerate(zip(centers, cell_val)):
    w = edges[1] - edges[0]
    inside = j in sten
    axC.add_patch(Rectangle(
        (c - w / 2, 0), w, v,
        facecolor=(ORA if inside else ACC), alpha=(0.55 if inside else 0.18),
        edgecolor=(ORA if inside else "#9fc4c7"), lw=(1.0 if inside else 0.5),
        zorder=2))
axC.plot(centers[sten], cell_val[sten], "o", color=ORA, ms=5, zorder=4)

# etiquette du stencil, placee au-dessus pour rester lisible
xl, xr = centers[sten[0]], centers[sten[-1]]
axC.annotate("", xy=(xl, 1.06), xytext=(xr, 1.06),
             arrowprops=dict(arrowstyle="|-|", color=ORA, lw=1.1,
                             mutation_scale=3))
axC.text((xl + xr) / 2, 1.11, "$u_{i-2}\\dots u_{i+2}$",
         ha="center", va="bottom", fontsize=9, color=ORA,
         fontweight="bold")
axC.set_xlim(0, 1); axC.set_ylim(0, 1.20)
axC.set_ylabel("$\\bar u_i$", fontsize=10, labelpad=2)
axC.set_yticks([]); axC.set_xticks([])
for s in ("top", "right"):
    axC.spines[s].set_visible(False)

# ---------- (d) les trois candidats et la decision ----------
axD.set_xlim(0, 1); axD.set_ylim(0, 1.20)
axD.axis("off")

cand_y = [0.92, 0.62, 0.32]
cand_lab = ["$S_0$", "$S_1$", "$S_2$"]
keep = [False, True, True]          # S0 traverse par le choc -> ecarte
for y, lab, kp in zip(cand_y, cand_lab, keep):
    col = ACC if kp else "#c0c0c0"
    axD.plot([0.10, 0.52], [y, y], color=col, lw=2.4,
             solid_capstyle="round")
    for xo in np.linspace(0.10, 0.52, 3):
        axD.plot([xo], [y], "o", color=col, ms=5)
    axD.text(0.03, y, lab, fontsize=10, color=col, va="center",
             fontweight="bold")
    axD.text(0.58, y, (T["kept"] if kp else T["disc"]), fontsize=9.5,
             color=(ACC if kp else ORA), va="center",
             fontweight="bold")
    axD.text(0.78, y, ("$\\delta_k=1$" if kp else "$\\delta_k=0$"),
             fontsize=9, color=GREY, va="center")

axD.text(0.5, 0.13, T["decision"],
         ha="center", fontsize=9.5, color=ORA, fontweight="bold")

# ---------- titres des panneaux ----------
titles = [T["t_a"], T["t_b"], T["t_c"], T["t_d"]]
for ax, t in zip(axes, titles):
    ax.set_title(t, fontsize=11, pad=8, color="#222222")

# ---------- fleches entre panneaux (serpentin 2x2) ----------
GREYA = "#555555"
# (a) -> (b)
fig.add_artist(FancyArrowPatch(
    (XL + PW + 0.008, YT + PH / 2), (XR - 0.008, YT + PH / 2),
    transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18,
    lw=1.8, color=GREYA))
# (b) -> (c) : coude explicite (descente, gauche, descente)
from matplotlib.lines import Line2D as _L2D
y_elbow = YT - 0.063
# descente decalee a gauche pour ne pas traverser le titre du panneau (c)
x_desc = XL + 0.045
fig.add_artist(_L2D(
    [XR + PW / 2, XR + PW / 2, x_desc, x_desc],
    [YT - 0.012, y_elbow, y_elbow, YB2 + PH + 0.028],
    transform=fig.transFigure, color=GREYA, lw=1.8,
    solid_capstyle="round", solid_joinstyle="round"))
fig.add_artist(FancyArrowPatch(
    (x_desc, YB2 + PH + 0.028), (x_desc, YB2 + PH + 0.006),
    transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18,
    lw=1.8, color=GREYA))
# (c) -> (d)
fig.add_artist(FancyArrowPatch(
    (XL + PW + 0.008, YB2 + PH / 2), (XR - 0.008, YB2 + PH / 2),
    transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18,
    lw=1.8, color=GREYA))

# ---------- bandeau aval : flux -> RK, de droite a gauche ----------
from matplotlib.lines import Line2D

yb = 0.115
box_w, box_h = 0.315, 0.105
x_flux = XR + PW - box_w          # aligne a droite, sous (d)
x_rk = XL                          # aligne a gauche, sous (c)
for xb, lab in [
    (x_flux, T["flux"]),
    (x_rk, T["rk"]),
]:
    fig.add_artist(FancyBboxPatch(
        (xb, yb), box_w, box_h, transform=fig.transFigure,
        boxstyle="round,pad=0.008", facecolor="#f6ddd5", edgecolor=ORA,
        lw=1.1))
    fig.text(xb + box_w / 2, yb + box_h / 2, lab, ha="center", va="center",
             fontsize=10)

# (d) -> flux : descente verticale
fig.add_artist(FancyArrowPatch(
    (x_flux + box_w / 2, YB2 - 0.012),
    (x_flux + box_w / 2, yb + box_h + 0.010),
    transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18,
    lw=1.8, color=ORA))

# flux -> RK : vers la gauche
fig.add_artist(FancyArrowPatch(
    (x_flux - 0.008, yb + box_h / 2),
    (x_rk + box_w + 0.008, yb + box_h / 2),
    transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18,
    lw=1.7, color=ORA))

# RK -> retour vers (c) : coude a gauche, en pointilles
x_back = XL + 0.030
y_mid = yb + box_h / 2
fig.add_artist(Line2D(
    [x_rk - 0.008, XL - 0.035, XL - 0.035],
    [y_mid, y_mid, YB2 + PH / 2],
    transform=fig.transFigure, color=ORA, lw=1.5, linestyle="--",
    solid_capstyle="round"))
fig.add_artist(FancyArrowPatch(
    (XL - 0.035, YB2 + PH / 2), (XL - 0.008, YB2 + PH / 2),
    transform=fig.transFigure, arrowstyle="-|>", mutation_scale=16,
    lw=1.5, color=ORA))
fig.text(XL - 0.043, (y_mid + YB2 + PH / 2) / 2, T["nextstep"],
         ha="center", va="center", fontsize=9.5, color=ORA, style="italic",
         rotation=90)

fig.savefig(f"{HERE}/pipeline_visuel{SUF}.png", dpi=175,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"ecrit pipeline_visuel{SUF}.png")
