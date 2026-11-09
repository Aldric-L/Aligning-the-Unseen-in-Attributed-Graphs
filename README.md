# Aligning the Unseen in Attributed Graphs

Code for *Aligning the Unseen in Attributed Graphs: Interplay between Graph Geometry and Node Attributes Manifold*
(Aldric Labarthe, Roland Bouffanais, Julien Randon-Furling).

A network whose nodes carry attributes has two geometries on the same nodes: one set by what the nodes are, one by
how they connect. This repository measures where and how they disagree.

* **Phase 1.** A variational autoencoder learns the attribute manifold. The pullback metric of its decoder, $G_1$,
  is the attribute geometry.
* **Phase 2 (the twist).** The encoder is frozen, so no node can move, and the decoder is retrained until a link
  probability of latent geodesic distances fits the adjacency. The new pullback metric $G_2$ is the graph geometry
  on the same coordinates.
* **Read-outs.** The strain $\hat R = G_1^{-1} G_2$ (where, and in which direction, the geometries part) and the pair
  distortion $\delta_{ij} = \log d_2 - \log d_1$ (how the graph shortens or lengthens each path).

## Layout

```
aligning/      the method: VAE and pullback metric, geodesics, Phase 1/2 training, strain, synthetic generators
experiments/   runners, analyses, figures and tables of the paper
tests/         correctness tests (geometry, gradients, strain, first-variation formulas)
```

## Installation

```bash
conda create -n aligning python=3.11
conda activate aligning
pip install -r requirements.txt
python -m pytest -q tests/          # < 10 s
```

## Reproducing the paper

See [REPRODUCING.md](REPRODUCING.md) for the commands, run times and the map from each figure and table to the script
that produces it. Every number in the figures and tables is read from `results/`; none is typed by hand.

Data, run outputs and generated figures are not versioned: `data/`, `results/` and `outputs/` are created locally.
To keep them elsewhere, set `ALIGNING_WORKDIR` to another directory.

## Earlier version

The code of the first preprint version (`framework/` and notebooks) is kept at the tag
[`v1-preprint`](../../tree/v1-preprint).
