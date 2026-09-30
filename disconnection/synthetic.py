"""Synthetic stroke patients with known structure.

The generator builds a toy brain in which lesion load and disconnection are
related but not identical, just as in real data:

- "voxels" are points in a 3-D box, each assigned to the nearest of R region
  centres (a parcellation, like the 116-region AAL atlas);
- a structural connectome links region pairs, with weights that fall off with
  distance; each connection is a straight "streamline" sampled at points;
- each patient has a spherical lesion, placed in one hemisphere;
- regional lesion load  = fraction of a region's voxels inside the lesion
  (binary lesion) or the mean lesion probability (fuzzy lesion);
- edge disconnection    = fraction of an edge's streamline inside the lesion;
- regional disconnection = weighted fraction of a region's connections that
  are disrupted (the ChaCo score of Kuceyeski et al., as in Chaco116_MNI.mat).

Behaviour (several tasks) is then generated from demographics plus either
lesion load of some critical regions (truth="load"), disconnection of some
critical edges (truth="disconnection"), both, or nothing ("none").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEMOGRAPHICS = ["age", "months_post_stroke", "lesion_volume", "sex", "handedness"]


@dataclass
class Brain:
    points: np.ndarray            # (V, 3) voxel coordinates
    region: np.ndarray            # (V,) region index of each voxel
    centres: np.ndarray           # (R, 3)
    edges: np.ndarray             # (E, 2) region pairs
    weights: np.ndarray           # (E,) connection strength
    paths: np.ndarray             # (E, S, 3) streamline sample points

    @property
    def n_regions(self):
        return len(self.centres)


def make_brain(n_regions: int = 40, n_voxels: int = 4000, n_edges: int = 200, path_samples: int = 25,
               seed: int = 0) -> Brain:
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-1, 1, size=(n_regions, 3))
    points = rng.uniform(-1, 1, size=(n_voxels, 3))
    d = ((points[:, None, :] - centres[None]) ** 2).sum(-1)
    region = d.argmin(1)
    # connections: prefer short ones (probability ~ exp(-distance / 0.6))
    iu = np.array(np.triu_indices(n_regions, 1)).T
    dist = np.linalg.norm(centres[iu[:, 0]] - centres[iu[:, 1]], axis=1)
    p = np.exp(-dist / 0.6)
    p /= p.sum()
    pick = rng.choice(len(iu), size=min(n_edges, len(iu)), replace=False, p=p)
    edges = iu[pick]
    weights = rng.gamma(2.0, 1.0, size=len(edges))
    t = np.linspace(0, 1, path_samples)[None, :, None]
    a, b = centres[edges[:, 0]][:, None, :], centres[edges[:, 1]][:, None, :]
    bend = rng.normal(scale=0.15, size=(len(edges), 1, 3)) * np.sin(np.pi * t)   # curved streamlines
    paths = a + t * (b - a) + bend
    return Brain(points, region, centres, edges, weights, paths)


@dataclass
class Patients:
    demographics: np.ndarray          # (N, 5) see DEMOGRAPHICS
    lesion_load: np.ndarray           # (N, R) binary-lesion regional load
    lesion_load_fuzzy: np.ndarray     # (N, R) fuzzy-lesion regional load
    disconnection: np.ndarray         # (N, R) regional (ChaCo-like) disconnection
    edge_disconnection: np.ndarray    # (N, E) disconnection of each connection
    behaviour: np.ndarray             # (N, T) task scores (higher = better)
    impaired: np.ndarray              # (N, T) score below the 20th percentile of controls-like cut-off
    truth: dict = field(default_factory=dict)

    def feature_sets(self) -> dict:
        """The named predictor blocks used throughout the package."""
        return {"dem": self.demographics, "ll": self.lesion_load, "ll_fuzzy": self.lesion_load_fuzzy,
                "disc": self.disconnection, "conn": self.edge_disconnection}


def lesion_measures(brain: Brain, centre, radius, softness=0.08):
    """Binary and fuzzy regional load, edge and regional disconnection for one lesion."""
    dv = np.linalg.norm(brain.points - centre, axis=1)
    binary = dv <= radius
    fuzzy = 1 / (1 + np.exp((dv - radius) / softness))
    R = brain.n_regions
    counts = np.bincount(brain.region, minlength=R).astype(float)
    counts[counts == 0] = 1
    ll = np.bincount(brain.region, weights=binary, minlength=R) / counts
    llf = np.bincount(brain.region, weights=fuzzy, minlength=R) / counts
    dp = np.linalg.norm(brain.paths - centre, axis=2)
    edge = (dp <= radius).mean(1)
    lost = brain.weights * (edge > 0)
    tot = np.bincount(brain.edges[:, 0], weights=brain.weights, minlength=R) + \
        np.bincount(brain.edges[:, 1], weights=brain.weights, minlength=R)
    dis = np.bincount(brain.edges[:, 0], weights=lost, minlength=R) + \
        np.bincount(brain.edges[:, 1], weights=lost, minlength=R)
    disc = np.divide(dis, tot, out=np.zeros(R), where=tot > 0)
    return ll, llf, edge, disc, binary.mean()


def simulate(n: int = 200, n_tasks: int = 3, truth: str = "disconnection", effect: float = 0.6,
             n_critical: int = 3, noise: float = 1.0, brain: Brain | None = None, seed: int = 0) -> Patients:
    """Simulate n patients.

    truth: "load" (behaviour depends on lesion load of n_critical regions),
    "disconnection" (on disconnection of n_critical edges), "both", or "none".
    effect: standard-deviation units of behaviour per unit of the (standardised)
    brain signal. Each task uses its own random mix of the critical features.
    """
    rng = np.random.default_rng(seed)
    brain = brain or make_brain(seed=seed + 1000)
    R, E = brain.n_regions, len(brain.edges)
    ll = np.zeros((n, R)); llf = np.zeros((n, R)); disc = np.zeros((n, R)); edge = np.zeros((n, E))
    vol = np.zeros(n)
    for i in range(n):
        c = rng.uniform(-0.8, 0.8, size=3)
        c[0] = -abs(c[0])                                  # "left hemisphere" strokes
        r = rng.uniform(0.15, 0.5)
        ll[i], llf[i], edge[i], disc[i], vol[i] = lesion_measures(brain, c, r)
    age = rng.normal(62, 12, n)
    months = rng.gamma(2.0, 12.0, n)
    sex = rng.integers(0, 2, n)
    hand = (rng.random(n) < 0.1).astype(float)
    dem = np.column_stack([age, months, vol, sex, hand])

    # critical features: the most variable regions / edges, so the signal is estimable
    crit_regions = np.argsort(-ll.std(0))[:n_critical]
    varied = np.flatnonzero(edge.std(0) > 0)
    crit_edges = varied[np.argsort(-edge[:, varied].std(0))[:n_critical]] if len(varied) else np.arange(n_critical)

    def z(a):
        s = a.std(0)
        return (a - a.mean(0)) / np.where(s > 0, s, 1)

    Y = np.zeros((n, n_tasks))
    for t in range(n_tasks):
        w_load = rng.uniform(0.5, 1.0, n_critical)
        w_disc = rng.uniform(0.5, 1.0, n_critical)
        sig = np.zeros(n)
        if truth in ("load", "both"):
            sig += z(ll[:, crit_regions]) @ w_load / np.sqrt(n_critical)
        if truth in ("disconnection", "both"):
            sig += z(edge[:, crit_edges]) @ w_disc / np.sqrt(n_critical)
        if truth not in ("load", "disconnection", "both", "none"):
            raise ValueError("truth must be load, disconnection, both or none")
        demo = -0.3 * z(age[:, None])[:, 0] + 0.2 * z(np.log1p(months)[:, None])[:, 0]
        Y[:, t] = 50 - 5 * (effect * sig + demo + noise * rng.normal(size=n))
    cut = np.percentile(Y, 40, axis=0)
    return Patients(dem, ll, llf, disc, edge, Y, (Y < cut).astype(int),
                    truth={"truth": truth, "critical_regions": crit_regions, "critical_edges": crit_edges,
                           "brain": brain})
