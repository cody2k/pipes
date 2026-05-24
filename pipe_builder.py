"""
pipe_builder.py
---------------
Build, export, and visualize 6-sided (hexagonal cross-section) pipes defined
by 3-D waypoints.  Bends are smoothed with Catmull-Rom splines.

Public API
----------
generate_pipe(waypoints, radius, n_rings, tension, return_frames) -> rings | (rings, frames)
save_pipe_csv(rings, filename)          – points only (legacy)
save_pipe_td(rings, frames, basename)   – all three TouchDesigner POP CSVs
  save_points_csv(rings, filename)
  save_vertices_csv(rings, frames, filename)
  save_primitives_csv(rings, filename)
visualize_pipes({label: rings, ...}, title, save_path)

Built-in examples (run as __main__):
  example_straight()  – matches example_pipe.csv geometry
  example_elbow()     – smooth L-bend
  example_spiral()    – spirals outward/downward from (0,1,0) to (0,-5,0)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from typing import List, Dict, Tuple


# ── Cross-section geometry ────────────────────────────────────────────────────

def _perpendicular_frame(tangent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return orthonormal (u, v) spanning the plane perpendicular to *tangent*.

    Orientation matches example_pipe.csv: for a pipe along +X, u points +Y
    and v = cross(u, t) points -Z, placing vertex 0 at the top of the hex.
    """
    t = tangent / np.linalg.norm(tangent)
    ref = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(t, ref)) > 0.99:       # tangent nearly parallel to world-Y
        ref = np.array([0.0, 0.0, 1.0])
    u = ref - np.dot(ref, t) * t         # project ref onto plane perp to t
    u /= np.linalg.norm(u)
    v = np.cross(u, t)                   # right-hand frame; matches CSV sign
    return u, v


def hex_ring(center: np.ndarray, tangent: np.ndarray,
             radius: float = 0.125) -> np.ndarray:
    """
    Return (6, 3) array of hex-ring vertices perpendicular to *tangent*.

    Vertices are ordered counter-clockwise when viewed from the +tangent side,
    starting from the "top" (projected world-up) direction.
    """
    u, v = _perpendicular_frame(tangent)
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.array([center + radius * (np.cos(a) * u + np.sin(a) * v)
                     for a in angles])


# ── Catmull-Rom spline ────────────────────────────────────────────────────────

def _cr_segment(p0, p1, p2, p3,
                t: np.ndarray, tension: float) -> np.ndarray:
    """Vectorised Catmull-Rom cubic for a single segment (t ∈ [0, 1])."""
    alpha = (1.0 - tension) / 2.0
    m1 = alpha * (p2 - p0)
    m2 = alpha * (p3 - p1)
    t2, t3 = t ** 2, t ** 3
    return (np.outer(2*t3 - 3*t2 + 1,  p1)
          + np.outer(t3  - 2*t2 + t,   m1)
          + np.outer(-2*t3 + 3*t2,     p2)
          + np.outer(t3  - t2,         m2))


def _spline_dense(waypoints, n_dense: int, tension: float) -> np.ndarray:
    """
    Dense point cloud along a Catmull-Rom spline through *waypoints*.
    Phantom end-points are reflected copies of the first/last real points.
    """
    pts = [np.asarray(p, float) for p in waypoints]
    padded = [2*pts[0] - pts[1]] + pts + [2*pts[-1] - pts[-2]]
    n_seg  = len(pts) - 1
    per    = max(4, n_dense // n_seg)
    chunks = []
    for i in range(n_seg):
        endpoint = (i == n_seg - 1)
        t = np.linspace(0.0, 1.0, per, endpoint=endpoint)
        chunks.append(_cr_segment(padded[i], padded[i+1],
                                  padded[i+2], padded[i+3], t, tension))
    return np.vstack(chunks)


def spline_sample(waypoints, n_samples: int,
                  tension: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample *n_samples* positions and tangents evenly by arc-length along
    a Catmull-Rom spline through *waypoints*.

    tension:  0.0 = standard smooth curve  |  1.0 = piecewise linear

    Returns
    -------
    positions : (n_samples, 3)
    tangents  : (n_samples, 3)  unit vectors
    """
    dense = _spline_dense(waypoints,
                          n_dense=max(4000, n_samples * 30),
                          tension=tension)

    # Arc-length cumulative sum
    diffs      = np.diff(dense, axis=0)
    seg_lens   = np.linalg.norm(diffs, axis=1)
    cum        = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total      = cum[-1]
    targets    = np.linspace(0.0, total, n_samples)

    # Interpolate positions
    positions = np.empty((n_samples, 3))
    for i, tl in enumerate(targets):
        idx = np.clip(np.searchsorted(cum, tl, side='right') - 1,
                      0, len(dense) - 2)
        sl  = seg_lens[idx] if idx < len(seg_lens) else 1.0
        frac = (tl - cum[idx]) / sl if sl > 1e-12 else 0.0
        positions[i] = dense[idx] + frac * diffs[idx]

    # Tangents via central finite differences
    tangents = np.empty_like(positions)
    tangents[1:-1] = positions[2:] - positions[:-2]
    tangents[0]    = positions[1]  - positions[0]
    tangents[-1]   = positions[-1] - positions[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    tangents /= norms

    return positions, tangents


# ── Main pipe builder ─────────────────────────────────────────────────────────

def generate_pipe(waypoints,
                  radius:        float = 0.125,
                  n_rings:       int   = 10,
                  tension:       float = 0.5,
                  return_frames: bool  = False):
    """
    Build a hexagonal pipe along a smooth Catmull-Rom path.

    Parameters
    ----------
    waypoints     : sequence of (x, y, z) control points
    radius        : circumradius of each hex cross-section
    n_rings       : number of cross-section rings
    tension       : 0.0 = smooth Catmull-Rom, 1.0 = piecewise linear
    return_frames : if True, return (rings, frames) where frames is a list of
                    (u, v) ndarray pairs — the cross-section basis for each ring

    Returns
    -------
    rings          : List of (6, 3) ndarrays
    (rings, frames): when return_frames=True
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints")
    centers, tangents = spline_sample(waypoints, n_rings, tension)
    rings = [hex_ring(c, t, radius) for c, t in zip(centers, tangents)]
    if return_frames:
        frames = [_perpendicular_frame(t) for t in tangents]
        return rings, frames
    return rings


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def rings_to_dataframe(rings: List[np.ndarray]) -> pd.DataFrame:
    """Convert rings to a DataFrame matching example_pipe.csv column layout."""
    rows = [{'P(0)': v[0], 'P(1)': v[1], 'P(2)': v[2]}
            for ring in rings for v in ring]
    df = pd.DataFrame(rows)
    df.index.name = 'index'
    return df.reset_index()          # index | P(0) | P(1) | P(2)


def save_pipe_csv(rings: List[np.ndarray], filename: str) -> None:
    """Save pipe geometry to CSV (format matches example_pipe.csv)."""
    df = rings_to_dataframe(rings)
    df.to_csv(filename, index=False)
    print(f"  Saved {len(rings)} rings / {len(rings)*6} vertices -> {filename}")


def save_points_csv(rings: List[np.ndarray], filename: str) -> None:
    """Save points CSV matching example_points.csv (index, P(0), P(1), P(2))."""
    save_pipe_csv(rings, filename)


def save_primitives_csv(rings: List[np.ndarray], filename: str) -> None:
    """
    Save primitives CSV matching example_primitives.csv.

    One quad per lateral face: (n_rings - 1) * 6 rows.
    Vertex winding: bottom-left, bottom-right, top-right, top-left
    when viewed from outside the pipe.
    """
    rows = []
    n_rings = len(rings)
    f = 0
    for i in range(n_rings - 1):
        for j in range(6):
            j2 = (j + 1) % 6
            v0 = i * 6 + j
            v1 = i * 6 + j2
            v2 = (i + 1) * 6 + j2
            v3 = (i + 1) * 6 + j
            rows.append({'index': f, 'type': 'quad',
                         'vertices': f"{v0} {v1} {v2} {v3}"})
            f += 1
    pd.DataFrame(rows).to_csv(filename, index=False)
    print(f"  Saved {f} primitives -> {filename}")


def save_vertices_csv(rings: List[np.ndarray],
                      frames: List[Tuple[np.ndarray, np.ndarray]],
                      filename: str) -> None:
    """
    Save vertices CSV matching example_vertices.csv.

    Columns: index, prim:vindex, pindex, N(0..2), Tex(0..2)

    Normals are outward radial in the cross-section plane:
        N = cos(j * π/3) * u  +  sin(j * π/3) * v
    UVs wrap the hex circumference (Tex0) and pipe length (Tex1).
    """
    n_rings = len(rings)
    n_ring_gaps = n_rings - 1
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    rows = []
    row_idx = 0
    for i in range(n_ring_gaps):
        for j in range(6):
            j2 = (j + 1) % 6
            face_idx = i * 6 + j
            # 4 corners: (ring i, vtx j), (ring i, vtx j2),
            #            (ring i+1, vtx j2), (ring i+1, vtx j)
            # Tex(0) uses j+1 (not j2) so the last face wraps to 1.0 not 0.0
            for local, ring_k, vtx_k, tex_u in (
                (0, i,     j,  j / 6),
                (1, i,     j2, (j + 1) / 6),
                (2, i + 1, j2, (j + 1) / 6),
                (3, i + 1, j,  j / 6),
            ):
                u, v = frames[ring_k]
                a = angles[vtx_k]
                normal = np.cos(a) * u + np.sin(a) * v
                tex_v = ring_k / n_ring_gaps
                rows.append({
                    'index':      row_idx,
                    'prim:vindex': f"{face_idx}:{local}",
                    'pindex':     ring_k * 6 + vtx_k,
                    'N(0)':       normal[0],
                    'N(1)':       normal[1],
                    'N(2)':       normal[2],
                    'Tex(0)':     tex_u,
                    'Tex(1)':     tex_v,
                    'Tex(2)':     0.0,
                })
                row_idx += 1
    pd.DataFrame(rows).to_csv(filename, index=False)
    print(f"  Saved {row_idx} vertex entries -> {filename}")


def save_pipe_td(rings: List[np.ndarray],
                 frames: List[Tuple[np.ndarray, np.ndarray]],
                 basename: str) -> None:
    """
    Save all three TouchDesigner POP CSVs for a pipe.

    Writes:
        {basename}_points.csv
        {basename}_vertices.csv
        {basename}_primitives.csv
    """
    save_points_csv(rings,              basename + '_points.csv')
    save_vertices_csv(rings, frames,    basename + '_vertices.csv')
    save_primitives_csv(rings,          basename + '_primitives.csv')


# ── Visualisation ─────────────────────────────────────────────────────────────

_PALETTE = ['#1565C0', '#C62828', '#2E7D32', '#E65100', '#6A1B9A', '#00838F']


def visualize_pipes(pipes: Dict[str, List[np.ndarray]],
                    title:     str = "Hexagonal Pipes",
                    save_path: str | None = None):
    """
    3-D visualisation of one or more hexagonal pipes.

    Parameters
    ----------
    pipes     : mapping of label → rings list
    title     : figure title
    save_path : if given, saves PNG to this path before plt.show()
    """
    fig = plt.figure(figsize=(16, 11))
    ax  = fig.add_subplot(111, projection='3d')

    all_verts: List[np.ndarray] = []

    for k, (label, rings) in enumerate(pipes.items()):
        color = _PALETTE[k % len(_PALETTE)]
        verts = np.vstack(rings)
        all_verts.append(verts)

        # ── Pipe surface ──────────────────────────────────────────────────────
        faces = []
        for i in range(len(rings) - 1):
            r1, r2 = rings[i], rings[i + 1]
            for j in range(6):
                j2 = (j + 1) % 6
                faces.append([r1[j], r1[j2], r2[j2], r2[j]])

        poly = Poly3DCollection(faces, alpha=0.55, linewidth=0)
        poly.set_facecolor(color)
        ax.add_collection3d(poly)

        # ── Ring outlines (subsampled for clarity) ────────────────────────────
        step = max(1, len(rings) // 25)
        for ring in rings[::step]:
            rc = np.vstack([ring, ring[0]])
            ax.plot(rc[:, 0], rc[:, 1], rc[:, 2],
                    color=color, lw=0.5, alpha=0.7)

        # ── Centreline ────────────────────────────────────────────────────────
        ctrs = np.array([r.mean(0) for r in rings])
        ax.plot(ctrs[:, 0], ctrs[:, 1], ctrs[:, 2],
                '--', color=color, lw=1.5, alpha=0.9, label=label)

    # Equal-aspect bounding box
    all_pts = np.vstack(all_verts)
    span    = np.ptp(all_pts, axis=0).max() / 2 * 1.08
    mid     = (all_pts.max(0) + all_pts.min(0)) / 2
    ax.set_xlim(mid[0]-span, mid[0]+span)
    ax.set_ylim(mid[1]-span, mid[1]+span)
    ax.set_zlim(mid[2]-span, mid[2]+span)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title, fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved figure -> {save_path}")

    return fig, ax


# ── Built-in example pipes ────────────────────────────────────────────────────

def example_straight(return_frames: bool = False):
    """
    Straight pipe from (-1, 0, 0) to (1, 0, 0).
    Reproduces the geometry in example_pipe.csv (scaled to unit length).
    """
    return generate_pipe([(-1, -1, 0), (1, -1, 0)],
                         radius=0.25, n_rings=10, tension=0.5,
                         return_frames=return_frames)


def example_elbow(return_frames: bool = False):
    """
    Smooth L-shaped elbow: (-1, 0, 0) → (1, 0, 0) → (1, 0, 2).
    Tension 0.4 gives a rounded corner without a kink.
    """
    return generate_pipe([(-1, 0, 0), (1, 0, 0), (1, 0, 2)],
                         radius=0.125, n_rings=30, tension=0.4,
                         return_frames=return_frames)


def example_spiral(return_frames: bool = False):
    """
    Pipe that starts at (0, 1, 0), spirals outward then inward while
    descending along Y, and terminates at (0, -5, 0).

    Path is defined analytically (120 waypoints) so the spline simply
    smooths the discrete steps into a continuous tube.
    """
    n  = 120
    t  = np.linspace(0.0, 1.0, n)

    y     = 1.0 - 6.0 * t                    # descend Y: 1 → -5
    r     = 1.8 * np.sin(np.pi * t)          # radius envelope: 0 → peak → 0
    angle = 2.0 * np.pi * 4.0 * t            # 4 full rotations in X-Z plane
    x     = r * np.cos(angle)
    z     = r * np.sin(angle)

    waypoints = list(zip(x.tolist(), y.tolist(), z.tolist()))
    return generate_pipe(waypoints, radius=0.08, n_rings=250, tension=0.0,
                         return_frames=return_frames)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building pipes…")

    pipe_defs = [
        ("Straight  (-1,0,0) → (1,0,0)",         example_straight, "pipe_straight"),
        ("Elbow  (-1,0,0) → (1,0,0) → (1,0,2)", example_elbow,    "pipe_elbow"),
        ("Spiral  (0,1,0) → (0,-5,0)",           example_spiral,   "pipe_spiral"),
    ]

    pipes: Dict[str, List[np.ndarray]] = {}
    for label, fn, basename in pipe_defs:
        rings, frames = fn(return_frames=True)
        pipes[label] = rings
        print(f"\n{basename}")
        save_pipe_td(rings, frames, basename)

    visualize_pipes(pipes,
                    title="Pipe Examples",
                    save_path="pipes_visualization.png")
    plt.show()
