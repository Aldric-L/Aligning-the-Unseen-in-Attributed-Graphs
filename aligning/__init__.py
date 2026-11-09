"""
aligning: measuring where and how the geometry of a graph disagrees with the geometry of its node attributes.

Code of "Aligning the Unseen in Attributed Graphs: Interplay between Graph Geometry and Node Attributes Manifold"
(Labarthe, Bouffanais, Randon-Furling).

Phase 1: a variational autoencoder learns the attribute manifold; the pullback metric of its decoder, G1, is the
attribute geometry. Phase 2 (the twist): the encoder is frozen and the decoder retrained until a link probability
of latent geodesic distances fits the adjacency; the new pullback metric G2 is the graph geometry on the same
coordinates. Read-outs: the strain G1^{-1} G2 and the pair distortion log d2 - log d1.

Modules
-------
models      VAE (MLP encoder, variational MLP decoder) with analytic Jacobians.
metric      Pullback metric field on a latent grid (+ multilinear interpolation).
geodesics   Differentiable geodesic estimators (grid graph / straight line),
            reference solver (discrete path energy), Gaussian curvature.
heat        Graph-Laplacian heat times, link-probability kernels and losses.
train       Phase 1 / Phase 2 training loops (seeded, logged).
twist       Synthetic attributed graphs with a planted, known disagreement (strain field).
strain      Reading the twist: learned strain field, references, link-moment competitor.
scores      Node-level distortion scores of the pair distortion.
synthetic   Deterministic synthetic manifold + graph generator + node relabelling.
baselines   Graph-only, attribute-only, Phase-1-only and learned baselines.
evaluation  Thresholds, AUROC/AP/F1, confidence intervals, paired tests.
"""

__version__ = "1.0.0"
