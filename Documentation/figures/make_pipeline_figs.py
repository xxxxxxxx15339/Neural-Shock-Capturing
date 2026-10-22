"""Genere les figures d'illustration du chapitre Pipeline.

1. schlieren_cut.png : photo schlieren NASA du T-38 + coupe 1D, avec le profil
   d'intensite REELLEMENT extrait des pixels le long de la coupe.
2. sod_structure.png : structure d'onde du tube a choc de Sod (schema).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

HERE = __file__.rsplit("/", 1)[0]

import os
LANG = os.environ.get("FIGLANG", "fr")
T = {
 "fr": dict(init="(a)  Etat initial ($t = 0$)", memb="membrane",
            struct="(b)  Structure a $t > 0$ : trois ondes distinctes",
            rare="detente", cont="contact", shock="choc",
            dens=r"densite $\rho$",
            sL="$\\rho_L = 1{,}0$\n$p_L = 1{,}0$\n$u_L = 0$",
            sR="$\\rho_R = 0{,}125$\n$p_R = 0{,}1$\n$u_R = 0$"),
 "en": dict(init="(a)  Initial state ($t = 0$)", memb="membrane",
            struct="(b)  Structure at $t > 0$: three distinct waves",
            rare="rarefaction", cont="contact", shock="shock",
            dens=r"density $\rho$",
            sL="$\\rho_L = 1.0$\n$p_L = 1.0$\n$u_L = 0$",
            sR="$\\rho_R = 0.125$\n$p_R = 0.1$\n$u_R = 0$"),
}[LANG]
SUF = "" if LANG == "fr" else "_en"

# ----------------------------------------------------------------------
# 1. Schlieren + coupe 1D
# ----------------------------------------------------------------------
full = np.asarray(Image.open(f"{HERE}/t38_schlieren.jpg").convert("L"), dtype=float)
# Recadrage sur la zone ou les fronts de choc sont nets
img = full[int(0.10 * full.shape[0]):int(0.94 * full.shape[0]),
           int(0.18 * full.shape[1]):int(0.80 * full.shape[1])]
H, W = img.shape

row = int(0.60 * H)
x0, x1 = 0, W

# Le cliche de 1993 est argentique : on moyenne une bande epaisse puis on
# applique une moyenne glissante courte pour attenuer le grain sans toucher
# aux fronts eux-memes.
band = img[row - 45:row + 46, x0:x1].mean(axis=0)
k = 41
band = np.convolve(band, np.ones(k) / k, mode="valid")
band = (band - band.min()) / (band.max() - band.min())
xs = np.linspace(0.0, 1.0, band.size)

fig = plt.figure(figsize=(10.5, 7.6))
gs = GridSpec(2, 1, height_ratios=[1.75, 1.0], hspace=0.30)

ax0 = fig.add_subplot(gs[0])
ax0.imshow(img, cmap="gray", aspect="auto")
ax0.axhline(row, color="#e04a2a", lw=2.2, ls="--")
ax0.set_xlim(0, W)
ax0.set_ylim(H, 0)
ax0.axis("off")
ax0.annotate(
    "coupe 1D", xy=(0.03 * W, row), xytext=(0.03 * W, row - 0.22 * H),
    color="#e04a2a", fontsize=12, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#e04a2a", lw=1.6),
)
ax0.set_title(
    "(a)  Ondes de choc reelles autour d'un T-38 a Mach 1,1\n"
    "(photographie schlieren, NASA)",
    fontsize=11.5, pad=10,
)

ax1 = fig.add_subplot(gs[1])
ax1.plot(xs, band, color="#117078", lw=1.8)
# Fronts : on repere automatiquement les descentes les plus raides du signal
grad = np.gradient(band)
cand = np.argsort(grad)[:400]          # pentes descendantes les plus fortes
fronts = []
for idx in cand:                        # un seul marqueur par front
    if all(abs(idx - j) > 0.04 * band.size for j in fronts):
        fronts.append(idx)
    if len(fronts) == 4:
        break
fronts = sorted(xs[f] for f in fronts)

for xc in fronts:
    ax1.axvline(xc, color="#e04a2a", lw=1.1, ls=":", alpha=0.9)
    ax1.plot([xc], [band[np.argmin(abs(xs - xc))]], "o", color="#e04a2a",
             ms=5, zorder=5)
ax1.text(np.mean(fronts), 1.10,
         "fronts de choc detectes par la pente du signal",
         ha="center", color="#e04a2a", fontsize=10, fontweight="bold")
ax1.set_xlim(0, 1); ax1.set_ylim(-0.05, 1.22)
ax1.set_xlabel("$x$  (abscisse le long de la coupe)", fontsize=11)
ax1.set_ylabel("intensite schlieren\n(normalisee)", fontsize=10)
ax1.set_title(
    "(b)  Signal 1D correspondant : chaque front de choc devient une "
    "transition raide",
    fontsize=11.5, pad=8,
)
ax1.grid(alpha=0.25, lw=0.6)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

fig.savefig(f"{HERE}/schlieren_cut.png", dpi=170, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("ecrit schlieren_cut.png")

# ----------------------------------------------------------------------
# 2. Structure d'onde du tube a choc de Sod
# ----------------------------------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 3.5))

# --- Etat initial ---
axL.axvspan(0, 0.5, color="#117078", alpha=0.20)
axL.axvspan(0.5, 1.0, color="#d0d0d0", alpha=0.55)
axL.axvline(0.5, color="#e04a2a", lw=2.4)
axL.text(0.25, 0.66, T["sL"], ha="center", va="center", fontsize=11)
axL.text(0.75, 0.66, T["sR"], ha="center", va="center", fontsize=11)
axL.text(0.5, 0.12, T["memb"], ha="center", color="#e04a2a",
         fontsize=10.5, fontweight="bold")
axL.set_xlim(0, 1); axL.set_ylim(0, 1)
axL.set_yticks([]); axL.set_xticks([0, 0.5, 1])
axL.set_xlabel("$x$", fontsize=11)
axL.set_title(T["init"], fontsize=11.5)

# --- Structure a t > 0 ---
x = np.linspace(0, 1, 600)
rho = np.piecewise(
    x,
    [x < 0.26,
     (x >= 0.26) & (x < 0.49),
     (x >= 0.49) & (x < 0.68),
     (x >= 0.68) & (x < 0.85),
     x >= 0.85],
    [1.0,
     lambda t: 1.0 - 0.575 * (t - 0.26) / 0.23,
     0.426,
     0.266,
     0.125],
)
axR.plot(x, rho, color="#117078", lw=2.0)
for xc, lab, col in [(0.375, T["rare"], "#2c7fb8"),
                     (0.68, T["cont"], "#7b5aa6"),
                     (0.85, T["shock"], "#e04a2a")]:
    axR.axvline(xc, color=col, lw=1.3, ls="--", alpha=0.85)
    axR.text(xc, 1.14, lab, ha="center", color=col, fontsize=10.5,
             fontweight="bold")
axR.set_xlim(0, 1); axR.set_ylim(0, 1.30)
axR.set_xlabel("$x$", fontsize=11)
axR.set_ylabel(T["dens"], fontsize=11)
axR.set_title(T["struct"], fontsize=11.5)
axR.grid(alpha=0.25, lw=0.6)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(f"{HERE}/sod_structure{SUF}.png", dpi=170, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(f"ecrit sod_structure{SUF}.png")
