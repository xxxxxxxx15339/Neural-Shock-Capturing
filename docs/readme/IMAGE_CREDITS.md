# README image credits

The method pipeline and exact layer architecture in the README use ASCII.
The numerical plots in the results sections are measured experiment outputs.

## Conventional neural-network illustration

- Original file: `classic_neural_network.svg` (reproduced unchanged).
- README display: `classic_neural_network.png`, rasterized on a white background
  for visibility in both light and dark themes. No neurons, connections or labels
  were edited. The PNG is also distributed under CC BY-SA 3.0.
- Title: *Artificial neural network*.
- Author: Cburnett; original illustration created with Inkscape, 27 December 2006.
- Source: [Wikimedia Commons file and revision history](https://commons.wikimedia.org/wiki/File:Artificial_neural_network.svg).
- Download: [original SVG](https://upload.wikimedia.org/wikipedia/commons/e/e4/Artificial_neural_network.svg).
- License selected for reuse: [Creative Commons Attribution–ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/).
- SHA-256: `077c0f9983131f20dbe4f450555a5b0a4b80f20e728052f9f219231d378b2f0e`.

This is a generic feed-forward network illustration. Its neuron and layer counts
are not a depiction of the project's two 5–64–32–16–8–3 networks. The README's
ASCII schematic and formulas specify the implemented networks exactly.

To reproduce the display PNG with librsvg:

```bash
rsvg-convert --background-color white --width 1120 --height 1000 \
  docs/readme/classic_neural_network.svg \
  --output docs/readme/classic_neural_network.png
```

## Jet photograph

- Existing file: `Documentation/figures/t38_schlieren.jpg`.
- Title: *Schlieren photograph of T-38 shock waves*.
- Photographer: Leonard Weinstein / NASA, 13 December 1993, photo EC94-42528-1.
- Source and public-domain status: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Schlieren_photograph_of_T-38_shock_waves.jpg).
- SHA-1: `c8fccad863f1378634d684f0ca9b0fdc2f33808b`.

The photograph motivates shock capturing; it is not a project simulation result
or an input to training. These third-party images retain their stated licenses;
the repository's MIT license does not replace them.
