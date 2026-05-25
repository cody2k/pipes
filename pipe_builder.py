"""
pipe_builder.py
---------------
Build, export, and visualize N-sided polygon cross-section pipes defined
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
import os
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


def poly_ring(center: np.ndarray, tangent: np.ndarray,
              radius: float = 0.125, n_sides: int = 16) -> np.ndarray:
    """
    Return (n_sides, 3) array of polygon-ring vertices perpendicular to *tangent*.

    Vertices are ordered counter-clockwise when viewed from the +tangent side,
    starting from the "top" (projected world-up) direction.
    """
    u, v = _perpendicular_frame(tangent)
    angles = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    return np.array([center + radius * (np.cos(a) * u + np.sin(a) * v)
                     for a in angles])


def hex_ring(center: np.ndarray, tangent: np.ndarray,
             radius: float = 0.125) -> np.ndarray:
    """Backward-compatible alias for poly_ring with n_sides=6."""
    return poly_ring(center, tangent, radius, n_sides=6)


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
                  return_frames: bool  = False,
                  n_sides:       int   = 16):
    """
    Build a polygon-cross-section pipe along a smooth Catmull-Rom path.

    Parameters
    ----------
    waypoints     : sequence of (x, y, z) control points
    radius        : circumradius of each cross-section polygon
    n_rings       : number of cross-section rings
    tension       : 0.0 = smooth Catmull-Rom, 1.0 = piecewise linear
    return_frames : if True, return (rings, frames) where frames is a list of
                    (u, v) ndarray pairs — the cross-section basis for each ring
    n_sides       : number of polygon sides (default 16)

    Returns
    -------
    rings          : List of (n_sides, 3) ndarrays
    (rings, frames): when return_frames=True
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints")
    centers, tangents = spline_sample(waypoints, n_rings, tension)
    rings = [poly_ring(c, t, radius, n_sides) for c, t in zip(centers, tangents)]
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
    n_sides = len(rings[0]) if rings else 0
    print(f"  Saved {len(rings)} rings / {len(rings)*n_sides} vertices -> {filename}")


def save_points_csv(rings: List[np.ndarray], filename: str) -> None:
    """Save points CSV matching example_points.csv (index, P(0), P(1), P(2))."""
    save_pipe_csv(rings, filename)


def save_primitives_csv(rings: List[np.ndarray], filename: str) -> None:
    """
    Save primitives CSV matching example_primitives.csv.

    One quad per lateral face: (n_rings - 1) * n_sides rows.
    Vertex winding: bottom-left, bottom-right, top-right, top-left
    when viewed from outside the pipe.
    """
    n_sides = len(rings[0])
    rows = []
    n_rings = len(rings)
    f = 0
    for i in range(n_rings - 1):
        for j in range(n_sides):
            j2 = (j + 1) % n_sides
            v0 = i * n_sides + j
            v1 = i * n_sides + j2
            v2 = (i + 1) * n_sides + j2
            v3 = (i + 1) * n_sides + j
            rows.append({'index': f, 'type': 'quad',
                         'vertices': f"{v0} {v1} {v2} {v3}"})
            f += 1
    pd.DataFrame(rows).to_csv(filename, index=False)
    print(f"  Saved {f} primitives -> {filename}")


def save_vertices_csv(rings: List[np.ndarray],
                      frames: List[Tuple[np.ndarray, np.ndarray]],
                      filename: str) -> None:
    """
    Save vertices CSV — one entry per point (vertex_index == point_index).

    Columns: index, pindex, N(0..2), Tex(0..2)

    Normals are outward radial in the cross-section plane:
        N = cos(j * π/3) * u  +  sin(j * π/3) * v
    UVs wrap the hex circumference (Tex0) and pipe length (Tex1).
    """
    n_rings  = len(rings)
    n_sides  = len(rings[0])
    n_gaps   = max(n_rings - 1, 1)
    angles   = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    rows = []
    for k, (ring, frame) in enumerate(zip(rings, frames)):
        u, bv = frame
        tex_v = k / n_gaps
        for j in range(n_sides):
            a = angles[j]
            normal = np.cos(a) * u + np.sin(a) * bv
            rows.append({
                'index':  k * n_sides + j,
                'pindex': k * n_sides + j,
                'N(0)':   float(normal[0]),
                'N(1)':   float(normal[1]),
                'N(2)':   float(normal[2]),
                'Tex(0)': j / n_sides,
                'Tex(1)': tex_v,
                'Tex(2)': 0.0,
            })
    pd.DataFrame(rows).to_csv(filename, index=False)
    print(f"  Saved {len(rows)} vertex entries -> {filename}")


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


# ── Internal geometry row builders ───────────────────────────────────────────

def _rotation_z_to(target: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that R @ [0,0,1] == target (normalised)."""
    target = target / np.linalg.norm(target)
    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z, target)
    s = float(np.linalg.norm(cross))
    if s < 1e-10:
        return np.eye(3) if float(np.dot(z, target)) > 0 else np.diag([1.0, -1.0, -1.0])
    axis = cross / s
    c = float(np.clip(np.dot(z, target), -1.0, 1.0))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _pipe_to_rows(rings: List[np.ndarray],
                  frames: List[Tuple[np.ndarray, np.ndarray]],
                  point_offset: int = 0,
                  prim_offset:  int = 0,
                  vert_offset:  int = 0,
                  ) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Points / primitives / vertices row-lists for one pipe with global offsets.

    Vertex model: one entry per point (vertex_index == point_index).
    Adjacent faces sharing an edge share the same vertex indices, giving TD
    explicit connectivity information.
    """
    n_rings  = len(rings)
    n_sides  = len(rings[0])
    n_gaps   = max(n_rings - 1, 1)
    angles   = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)

    pts_rows:   List[dict] = []
    verts_rows: List[dict] = []
    for k, (ring, frame) in enumerate(zip(rings, frames)):
        u, bv = frame
        tex_v = k / n_gaps
        for j in range(n_sides):
            a      = angles[j]
            normal = np.cos(a) * u + np.sin(a) * bv
            pts_rows.append({
                'index': point_offset + k * n_sides + j,
                'P(0)':  float(ring[j][0]),
                'P(1)':  float(ring[j][1]),
                'P(2)':  float(ring[j][2]),
            })
            verts_rows.append({
                'index':  vert_offset + k * n_sides + j,
                'pindex': point_offset + k * n_sides + j,
                'N(0)':   float(normal[0]),
                'N(1)':   float(normal[1]),
                'N(2)':   float(normal[2]),
                'Tex(0)': j / n_sides,
                'Tex(1)': tex_v,
                'Tex(2)': 0.0,
            })

    prims_rows: List[dict] = []
    for i in range(n_rings - 1):
        for j in range(n_sides):
            j2       = (j + 1) % n_sides
            face_idx = prim_offset + i * n_sides + j
            v0 = point_offset + i * n_sides + j
            v1 = point_offset + i * n_sides + j2
            v2 = point_offset + (i + 1) * n_sides + j2
            v3 = point_offset + (i + 1) * n_sides + j
            prims_rows.append({'index': face_idx, 'type': 'quad',
                               'vertices': f"{v0} {v1} {v2} {v3}"})

    return pts_rows, prims_rows, verts_rows


def _sphere_to_rows(pipe_end:     np.ndarray,
                    radius:       float,
                    tangent:      np.ndarray,
                    n_lat:        int = 24,
                    n_lon:        int = 36,
                    point_offset: int = 0,
                    prim_offset:  int = 0,
                    vert_offset:  int = 0,
                    ) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Points / primitives / vertices row-lists for a UV sphere.

    The sphere is positioned so its south pole sits flush at *pipe_end*;
    the north pole extends outward along *tangent*.  Sphere centre lands at
    pipe_end + radius * tangent_normalised.
    """
    t = np.asarray(tangent, float)
    t /= np.linalg.norm(t)
    center = np.asarray(pipe_end, float) + radius * t
    R = _rotation_z_to(t)   # local +Z → world tangent; local −Z → pipe end

    n_mid      = n_lat - 1  # middle latitude rings (excluding poles)
    lat_angles = np.linspace(0.0, np.pi, n_lat + 1)[1:-1]   # n_mid values
    lon_angles = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)

    # ── Vertex positions ──────────────────────────────────────────────────
    pts_local: List[np.ndarray] = [np.array([0.0, 0.0, radius])]   # north pole
    for la in lat_angles:
        for lo in lon_angles:
            pts_local.append(np.array([
                radius * np.sin(la) * np.cos(lo),
                radius * np.sin(la) * np.sin(lo),
                radius * np.cos(la),
            ]))
    pts_local.append(np.array([0.0, 0.0, -radius]))                # south pole

    south_idx = len(pts_local) - 1
    pts_world = [center + R @ p for p in pts_local]

    # Per-point rows (vertex_index == point_index)
    pts_rows:   List[dict] = []
    verts_rows: List[dict] = []
    for i, (pl, pw) in enumerate(zip(pts_local, pts_world)):
        pts_rows.append({
            'index': point_offset + i,
            'P(0)':  float(pw[0]), 'P(1)': float(pw[1]), 'P(2)': float(pw[2]),
        })
        normal = (np.asarray(pw) - center) / radius
        if i == 0:
            tu, tv = 0.5, 0.0
        elif i == south_idx:
            tu, tv = 0.5, 1.0
        else:
            ring_i = (i - 1) // n_lon
            lon_j  = (i - 1) % n_lon
            tu = lon_j / n_lon
            tv = (ring_i + 1) / n_lat
        verts_rows.append({
            'index':  vert_offset + i,
            'pindex': point_offset + i,
            'N(0)':   float(normal[0]), 'N(1)': float(normal[1]),
            'N(2)':   float(normal[2]),
            'Tex(0)': tu, 'Tex(1)': tv, 'Tex(2)': 0.0,
        })

    prims_rows: List[dict] = []
    p_idx = prim_offset

    # North cap triangles (pole -> first ring)
    for j in range(n_lon):
        j2      = (j + 1) % n_lon
        a0, a1, a2 = 0, 1 + j, 1 + j2
        prims_rows.append({'index': p_idx, 'type': 'tri',
                           'vertices': f"{point_offset+a0} {point_offset+a1} "
                                       f"{point_offset+a2}"})
        p_idx += 1

    # Middle quads
    for i in range(n_mid - 1):
        ring_s = 1 + i * n_lon
        next_s = ring_s + n_lon
        for j in range(n_lon):
            j2      = (j + 1) % n_lon
            a0, a1  = ring_s + j,  ring_s + j2
            a2, a3  = next_s + j2, next_s + j
            prims_rows.append({'index': p_idx, 'type': 'quad',
                               'vertices': f"{point_offset+a0} {point_offset+a1} "
                                           f"{point_offset+a2} {point_offset+a3}"})
            p_idx += 1

    # South cap triangles (last ring -> south pole)
    last_ring_s = 1 + (n_mid - 1) * n_lon
    for j in range(n_lon):
        j2      = (j + 1) % n_lon
        a0, a1  = last_ring_s + j, last_ring_s + j2
        prims_rows.append({'index': p_idx, 'type': 'tri',
                           'vertices': f"{point_offset+a0} {point_offset+a1} "
                                       f"{point_offset+south_idx}"})
        p_idx += 1

    return pts_rows, prims_rows, verts_rows


# ── Combined multi-pipe / reservoir export ────────────────────────────────────

def save_combined_td(pipes_data: List[Tuple[List[np.ndarray],
                                            List[Tuple[np.ndarray, np.ndarray]]]],
                     basename:         str,
                     reservoir_radius: float | None = None,
                     reservoir_n_lat:  int           = 24,
                     reservoir_n_lon:  int           = 36) -> None:
    """
    Save joined TouchDesigner POP CSVs for one or more pipe segments,
    with an optional spherical reservoir attached to the final pipe end.

    Parameters
    ----------
    pipes_data       : list of (rings, frames) tuples — one per pipe segment
    basename         : output path prefix (e.g. 'output/my_network')
    reservoir_radius : if given, append a sphere of this radius to the last
                       pipe's end; set to None to omit the reservoir
    reservoir_n_lat  : latitude bands on the sphere — higher = smoother (default 24)
    reservoir_n_lon  : longitude slices on the sphere — higher = smoother (default 36)

    Writes
    ------
    {basename}_points.csv
    {basename}_primitives.csv
    {basename}_vertices.csv
    """
    all_pts:   List[dict] = []
    all_prims: List[dict] = []
    all_verts: List[dict] = []
    pt_off = pr_off = vt_off = 0

    for rings, frames in pipes_data:
        n_sides = len(rings[0])
        pts, prims, verts = _pipe_to_rows(rings, frames, pt_off, pr_off, vt_off)
        all_pts.extend(pts)
        all_prims.extend(prims)
        all_verts.extend(verts)
        pt_off += len(rings) * n_sides
        pr_off += (len(rings) - 1) * n_sides
        vt_off += len(verts)

    if reservoir_radius is not None:
        last_rings = pipes_data[-1][0]
        pipe_end   = last_rings[-1].mean(axis=0)
        if len(last_rings) >= 2:
            t_end = last_rings[-1].mean(0) - last_rings[-2].mean(0)
        else:
            t_end = np.array([0.0, 0.0, 1.0])
        t_end = t_end / np.linalg.norm(t_end)

        pts, prims, verts = _sphere_to_rows(
            pipe_end, reservoir_radius, t_end,
            reservoir_n_lat, reservoir_n_lon,
            pt_off, pr_off, vt_off,
        )
        all_pts.extend(pts)
        all_prims.extend(prims)
        all_verts.extend(verts)

    pd.DataFrame(all_pts).to_csv(  basename + '_points.csv',     index=False)
    pd.DataFrame(all_prims).to_csv(basename + '_primitives.csv', index=False)
    pd.DataFrame(all_verts).to_csv(basename + '_vertices.csv',   index=False)
    n_segs = len(pipes_data)
    suffix = " + reservoir" if reservoir_radius is not None else ""
    print(f"  {n_segs} pipe(s){suffix}: {len(all_pts)} pts / "
          f"{len(all_prims)} prims / {len(all_verts)} verts -> {basename}_*.csv")


# ── Pipe series — smooth connected joints ─────────────────────────────────────

def generate_pipe_series(
        segment_waypoints: List[List],
        radius:            float = 0.25,
        rings_per_length:  float = 30.0,
        tension:           float = 0.55,
        n_sides:           int   = 16,
) -> List[Tuple[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]]:
    """
    Generate a series of smoothly joined pipe segments from a single spline.

    Adjacent segment lists must share their boundary waypoint:
        [[A, B], [B, C, D], [D, E]]

    One Catmull-Rom spline is generated through all waypoints so tangents are
    continuous at every junction.  Rings are then partitioned at the ring
    nearest each boundary waypoint; that ring is included in both adjacent
    segments so save_series_td can reference it without duplication.

    Parameters
    ----------
    segment_waypoints : list of waypoint lists, adjacent lists share boundary
    radius            : circumradius of the cross-section polygon (default 0.25)
    rings_per_length  : rings per world-unit of arc length (default 30)
    tension           : 0.0 = smooth Catmull-Rom, 1.0 = linear (default 0.55)
    n_sides           : polygon sides per ring (default 16)

    Returns
    -------
    list of (rings, frames) — one tuple per segment
    """
    # Flatten to one waypoint list, record junction points
    all_wps: List[np.ndarray] = [np.array(p, float) for p in segment_waypoints[0]]
    junction_wps: List[np.ndarray] = []
    for seg in segment_waypoints[1:]:
        junction_wps.append(np.array(seg[0], float))
        all_wps.extend(np.array(p, float) for p in seg[1:])

    seg_lens  = [float(np.linalg.norm(all_wps[i + 1] - all_wps[i]))
                 for i in range(len(all_wps) - 1)]
    total_len = sum(seg_lens)
    n_rings   = max(20, int(total_len * rings_per_length))

    rings, frames = generate_pipe(all_wps, radius, n_rings, tension,
                                  return_frames=True, n_sides=n_sides)

    # For each junction waypoint find the nearest ring center
    centers = np.array([r.mean(0) for r in rings])
    split_indices = [0]
    for junc in junction_wps:
        idx = int(np.argmin(np.linalg.norm(centers - junc, axis=1)))
        split_indices.append(idx)
    split_indices.append(len(rings) - 1)

    return [(rings[split_indices[i]:split_indices[i + 1] + 1],
             frames[split_indices[i]:split_indices[i + 1] + 1])
            for i in range(len(segment_waypoints))]


def _sphere_hole_to_rows(
        pipe_end:      np.ndarray,
        pipe_radius:   float,
        pipe_frame:    Tuple[np.ndarray, np.ndarray],
        t_end:         np.ndarray,
        sphere_radius: float,
        n_lat:         int   = 20,
        n_lon:         int   = 16,
        inset:         float = 0.0125,
        point_offset:  int   = 0,
        prim_offset:   int   = 0,
        vert_offset:   int   = 0,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Points / primitives / vertices for a spherical reservoir with an n-gon hole.

    n_lon longitude slices are aligned with the pipe's own cross-section frame
    so the opening exactly matches the pipe.  The rim sits *inset* units inside
    the pipe end (default 0.0125).  No south-cap faces are generated — the
    polygon boundary is the open mouth of the reservoir.

    Geometry
    --------
    sphere centre  = pipe_end + (d - inset) * t_end
    d              = sqrt(sphere_radius^2 - pipe_radius^2)
    theta_rim      = arccos(-d / sphere_radius)   [co-latitude of rim]
    """
    t = np.asarray(t_end, float)
    t /= np.linalg.norm(t)
    u, v = pipe_frame

    if sphere_radius <= pipe_radius:
        raise ValueError("sphere_radius must exceed pipe_radius")

    d             = float(np.sqrt(sphere_radius ** 2 - pipe_radius ** 2))
    sphere_center = np.asarray(pipe_end, float) + (d - inset) * t
    theta_rim     = float(np.arccos(-d / sphere_radius))   # ∈ (π/2, π)

    # Latitude grid from just past north pole down to rim (n_lat rings)
    lat_angles = np.linspace(0.0, theta_rim, n_lat + 1)[1:]   # last = theta_rim
    lon_angles = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)

    # Vertex positions: north pole + n_lat rings of n_lon
    north_pole = sphere_center + sphere_radius * t
    pts_world: List[np.ndarray] = [north_pole]
    for la in lat_angles:
        r_lat = sphere_radius * np.sin(la)
        z_lat = sphere_radius * np.cos(la)
        for lo in lon_angles:
            pts_world.append(
                sphere_center + z_lat * t + r_lat * (np.cos(lo) * u + np.sin(lo) * v)
            )

    # Per-point rows (vertex_index == point_index)
    pts_rows:   List[dict] = []
    verts_rows: List[dict] = []
    for i, pw in enumerate(pts_world):
        pts_rows.append({
            'index': point_offset + i,
            'P(0)':  float(pw[0]), 'P(1)': float(pw[1]), 'P(2)': float(pw[2]),
        })
        normal = (np.asarray(pw) - sphere_center) / sphere_radius
        if i == 0:
            tu, tv = 0.5, 0.0
        else:
            ring_i = (i - 1) // n_lon
            lon_j  = (i - 1) % n_lon
            tu = lon_j / n_lon
            tv = lat_angles[ring_i] / theta_rim
        verts_rows.append({
            'index':  vert_offset + i,
            'pindex': point_offset + i,
            'N(0)':   float(normal[0]), 'N(1)': float(normal[1]),
            'N(2)':   float(normal[2]),
            'Tex(0)': tu, 'Tex(1)': tv, 'Tex(2)': 0.0,
        })

    prims_rows: List[dict] = []
    p_idx = prim_offset

    # North cap: n_lon triangles
    for k in range(n_lon):
        k2 = (k + 1) % n_lon
        prims_rows.append({'index': p_idx, 'type': 'tri',
                           'vertices': f"{point_offset} "
                                       f"{point_offset+1+k} {point_offset+1+k2}"})
        p_idx += 1

    # Middle quads: (n_lat - 1) bands
    for i in range(n_lat - 1):
        rs = 1 + i * n_lon
        ns = rs + n_lon
        for k in range(n_lon):
            k2      = (k + 1) % n_lon
            a0, a1  = rs + k,  rs + k2
            a2, a3  = ns + k2, ns + k
            prims_rows.append({'index': p_idx, 'type': 'quad',
                               'vertices': f"{point_offset+a0} {point_offset+a1} "
                                           f"{point_offset+a2} {point_offset+a3}"})
            p_idx += 1

    # No south-cap faces — polygon rim is the open boundary
    return pts_rows, prims_rows, verts_rows


def save_series_td(
        segments:         List[Tuple[List[np.ndarray],
                                     List[Tuple[np.ndarray, np.ndarray]]]],
        basename:         str,
        reservoir_radius: float | None = None,
        reservoir_n_lat:  int           = 20,
        reservoir_inset:  float         = 0.0125,
) -> None:
    """
    Save joined TD POP CSVs for a pipe series from generate_pipe_series().

    Junction rings shared between adjacent segments are exported once only,
    eliminating duplicate vertices and any positional gap at every joint.

    Parameters
    ----------
    segments         : list of (rings, frames) from generate_pipe_series()
    basename         : output path prefix (e.g. 'output/my_series')
    reservoir_radius : if given, attach a hex-hole sphere of this radius at the end
    reservoir_n_lat  : latitude bands on the reservoir sphere — higher = smoother
                       (default 20; n_lon is derived from the pipe's n_sides)
    reservoir_inset  : how far the sphere overlaps into the pipe (default 0.0125)
    """
    all_pts:   List[dict] = []
    all_prims: List[dict] = []
    all_verts: List[dict] = []
    pt_off = pr_off = vt_off = 0
    n_sides = len(segments[0][0][0])   # sides derived from first ring

    for seg_idx, (rings, frames) in enumerate(segments):
        if seg_idx == 0:
            pts, prims, verts = _pipe_to_rows(rings, frames, pt_off, pr_off, vt_off)
            all_pts.extend(pts)
            all_verts.extend(verts)
            pt_off += len(rings) * n_sides
            vt_off += len(rings) * n_sides
        else:
            # Junction ring == last ring of previous segment; reuse its indices
            pt_off -= n_sides
            vt_off -= n_sides
            pts, prims, verts = _pipe_to_rows(rings, frames, pt_off, pr_off, vt_off)
            all_pts.extend(pts[n_sides:])      # skip first ring (already exported)
            all_verts.extend(verts[n_sides:])  # skip first ring's vertices (same)
            pt_off += len(rings) * n_sides     # net advance: (n-1)*n_sides
            vt_off += len(rings) * n_sides

        all_prims.extend(prims)
        pr_off += (len(rings) - 1) * n_sides

    if reservoir_radius is not None:
        last_rings, last_frames = segments[-1]
        pipe_end    = last_rings[-1].mean(axis=0)
        pipe_radius = float(np.linalg.norm(last_rings[-1][0] - pipe_end))
        t_end = last_rings[-1].mean(0) - last_rings[-2].mean(0)
        t_end /= np.linalg.norm(t_end)

        pts, prims, verts = _sphere_hole_to_rows(
            pipe_end, pipe_radius, last_frames[-1], t_end,
            reservoir_radius, reservoir_n_lat, n_sides, reservoir_inset,
            pt_off, pr_off, vt_off,
        )
        all_pts.extend(pts)
        all_prims.extend(prims)
        all_verts.extend(verts)

    pd.DataFrame(all_pts).to_csv(  basename + '_points.csv',     index=False)
    pd.DataFrame(all_prims).to_csv(basename + '_primitives.csv', index=False)
    pd.DataFrame(all_verts).to_csv(basename + '_vertices.csv',   index=False)
    suffix = " + reservoir" if reservoir_radius is not None else ""
    print(f"  {len(segments)} seg(s){suffix}: {len(all_pts)} pts / "
          f"{len(all_prims)} prims / {len(all_verts)} verts -> {basename}_*.csv")


# ── Visualisation ─────────────────────────────────────────────────────────────

_PALETTE = ['#1565C0', '#C62828', '#2E7D32', '#E65100', '#6A1B9A', '#00838F']


def _sphere_polys(pipe_end:  np.ndarray,
                  radius:    float,
                  tangent:   np.ndarray,
                  n_lat:     int = 12,
                  n_lon:     int = 18) -> List[np.ndarray]:
    """Return polygon vertex arrays (for Poly3DCollection) for a reservoir sphere."""
    t = np.asarray(tangent, float)
    t /= np.linalg.norm(t)
    center = np.asarray(pipe_end, float) + radius * t
    R = _rotation_z_to(t)

    n_mid      = n_lat - 1
    lat_angles = np.linspace(0.0, np.pi, n_lat + 1)[1:-1]
    lon_angles = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)

    pts_local: List[np.ndarray] = [np.array([0.0, 0.0, radius])]
    for la in lat_angles:
        for lo in lon_angles:
            pts_local.append(np.array([
                radius * np.sin(la) * np.cos(lo),
                radius * np.sin(la) * np.sin(lo),
                radius * np.cos(la),
            ]))
    pts_local.append(np.array([0.0, 0.0, -radius]))

    pts = np.array([center + R @ p for p in pts_local])
    south_idx = len(pts) - 1
    faces: List[np.ndarray] = []

    for j in range(n_lon):
        j2 = (j + 1) % n_lon
        faces.append(pts[[0, 1 + j, 1 + j2]])

    for i in range(n_mid - 1):
        rs, ns = 1 + i * n_lon, 1 + (i + 1) * n_lon
        for j in range(n_lon):
            j2 = (j + 1) % n_lon
            faces.append(pts[[rs + j, rs + j2, ns + j2, ns + j]])

    last_s = 1 + (n_mid - 1) * n_lon
    for j in range(n_lon):
        j2 = (j + 1) % n_lon
        faces.append(pts[[last_s + j, last_s + j2, south_idx]])

    return faces


def _sphere_hole_polys(
        pipe_end:      np.ndarray,
        pipe_radius:   float,
        pipe_frame:    Tuple[np.ndarray, np.ndarray],
        t_end:         np.ndarray,
        sphere_radius: float,
        n_lat:         int   = 20,
        n_lon:         int   = 16,
        inset:         float = 0.0125,
) -> List[np.ndarray]:
    """Polygon vertex arrays (for Poly3DCollection) for a polygon-hole reservoir sphere."""
    t = np.asarray(t_end, float);  t /= np.linalg.norm(t)
    u, bv = pipe_frame
    d             = float(np.sqrt(sphere_radius ** 2 - pipe_radius ** 2))
    sphere_center = np.asarray(pipe_end, float) + (d - inset) * t
    theta_rim     = float(np.arccos(-d / sphere_radius))
    lat_angles    = np.linspace(0.0, theta_rim, n_lat + 1)[1:]
    lon_angles    = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)

    north_pole = sphere_center + sphere_radius * t
    pts = [north_pole]
    for la in lat_angles:
        r_lat, z_lat = sphere_radius * np.sin(la), sphere_radius * np.cos(la)
        for lo in lon_angles:
            pts.append(sphere_center + z_lat * t + r_lat * (np.cos(lo) * u + np.sin(lo) * bv))
    pts = np.array(pts)

    faces: List[np.ndarray] = []
    for k in range(n_lon):
        k2 = (k + 1) % n_lon
        faces.append(pts[[0, 1 + k, 1 + k2]])
    for i in range(n_lat - 1):
        rs, ns = 1 + i * n_lon, 1 + (i + 1) * n_lon
        for k in range(n_lon):
            k2 = (k + 1) % n_lon
            faces.append(pts[[rs + k, rs + k2, ns + k2, ns + k]])
    return faces


def visualize_pipes(
        pipes:            Dict[str, List[np.ndarray]],
        title:            str                           = "Hexagonal Pipes",
        save_path:        str | None                    = None,
        reservoirs:       List | None                   = None,
        reservoir_holes:  List | None                   = None,
):
    """
    3-D visualisation of hexagonal pipes with optional reservoirs.

    Parameters
    ----------
    pipes           : mapping of label -> rings list
    title           : figure title
    save_path       : if given, saves PNG here before plt.show()
    reservoirs      : list of (pipe_end, radius, tangent) — full UV spheres
    reservoir_holes : list of (pipe_end, pipe_radius, pipe_frame, t_end,
                      sphere_radius) — hex-hole reservoir spheres
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
            ns = len(r1)
            for j in range(ns):
                j2 = (j + 1) % ns
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

    # Full-sphere reservoirs
    if reservoirs:
        for k, (pe, rad, tang) in enumerate(reservoirs):
            color = _PALETTE[(len(pipes) + k) % len(_PALETTE)]
            faces = _sphere_polys(pe, rad, tang)
            poly  = Poly3DCollection(faces, alpha=0.45, linewidth=0)
            poly.set_facecolor(color)
            ax.add_collection3d(poly)
            t_n = np.asarray(tang, float) / np.linalg.norm(tang)
            sc  = np.asarray(pe, float) + rad * t_n
            all_verts.append(np.array([sc - rad, sc + rad]))
            ax.scatter(*sc, color=color, s=40, alpha=0.9,
                       label=f"Reservoir {k + 1}")

    # Polygon-hole reservoir spheres
    if reservoir_holes:
        offset = len(reservoirs) if reservoirs else 0
        for k, args in enumerate(reservoir_holes):
            color = _PALETTE[(len(pipes) + offset + k) % len(_PALETTE)]
            pe, p_rad, p_frame, t_e, s_rad = args[:5]
            n_s = args[5] if len(args) > 5 else 16
            faces = _sphere_hole_polys(pe, p_rad, p_frame, t_e, s_rad, n_lon=n_s)
            poly  = Poly3DCollection(faces, alpha=0.50, linewidth=0)
            poly.set_facecolor(color)
            ax.add_collection3d(poly)
            t_n = np.asarray(t_e, float) / np.linalg.norm(t_e)
            d   = float(np.sqrt(s_rad ** 2 - p_rad ** 2))
            sc  = np.asarray(pe, float) + (d - 0.0125) * t_n
            all_verts.append(np.array([sc - s_rad, sc + s_rad]))
            ax.scatter(*sc, color=color, s=40, alpha=0.9,
                       label=f"Reservoir {offset + k + 1}")

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


def example_joined(return_frames: bool = False):
    """
    Two connected pipe segments forming an L-then-straight run.

    Segment 1: straight along +X from (0,0,0) to (2,0,0).
    Segment 2: elbow from (2,0,0) up to (2,0,1.5) then along +X to (4,0,1.5).

    Returns a list of (rings, frames) tuples — one per segment.
    """
    seg1 = generate_pipe([(0, 0, 0), (2, 0, 0)],
                         radius=0.25, n_rings=15, tension=0.5,
                         return_frames=True)
    seg2 = generate_pipe([(2, 0, 0), (2, 0, 1.5), (4, 0, 1.5)],
                         radius=0.25, n_rings=25, tension=0.4,
                         return_frames=True)
    if return_frames:
        return [seg1, seg2]
    return [seg1[0], seg2[0]]


def example_pipe_series():
    """
    Three-segment vertical pipe series using a single Catmull-Rom spline.

    Open end (inlet) at (-1, 2, -1); reservoir end at (-1, -1, 0).
    X is constant throughout; Y descends while Z drifts gradually from -1 to 0.
    Every junction is vertical-to-vertical (tangent dominantly along -Y):

        Seg 1  (-1,  2.00, -1.00) -> (-1,  0.75, -1.00)   straight down
        Seg 2  (-1,  0.75, -1.00) -> (-1, -0.25, -0.50)   curves toward Z=0
        Seg 3  (-1, -0.25, -0.50) -> (-1, -1.00,  0.00)   arrives at reservoir

    generate_pipe_series() produces a single spline so there are no kinks or
    gaps at the junctions.  A hex-hole reservoir attaches at (-1, -1, 0).
    """
    return generate_pipe_series(
        segment_waypoints=[
            [(-0.5,  3.00, -1.00), (-1,  2.00, -1.00)],
            [(-1,  2.00, -1.00), (-1,  0.75, -1.00)],
            [(-1,  0.75, -1.00), (-1, -0.25, -0.50)],
            [(-1, -0.25, -0.50), (-1, -2.00,  0.00)],
        ],
        radius=0.25,
        rings_per_length=20,
        tension=0.5,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)
    print("Building pipes…")
    """
    # ── Individual examples (unchanged geometry) ──────────────────────────
    pipe_defs = [
        ("Straight  (-1,0,0) → (1,0,0)",         example_straight, "output/pipe_straight"),
        ("Elbow  (-1,0,0) → (1,0,0) → (1,0,2)", example_elbow,    "output/pipe_elbow"),
        ("Spiral  (0,1,0) → (0,-5,0)",           example_spiral,   "output/pipe_spiral"),
    ]

    pipes: Dict[str, List[np.ndarray]] = {}
    for label, fn, basename in pipe_defs:
        rings, frames = fn(return_frames=True)
        pipes[label] = rings
        print(f"\n{basename}")
        save_pipe_td(rings, frames, basename)

    visualize_pipes(pipes,
                    title="Pipe Examples",
                    save_path="output/pipes_visualization.png")
    """
    # ── Joined pipes (no reservoir) ───────────────────────────────────────
    print("\n--- Joined pipes (legacy, separate splines) ---")
    segs = example_joined(return_frames=True)
    joined_rings = {f"Seg {i+1}": s[0] for i, s in enumerate(segs)}
    save_combined_td(segs, "output/pipe_joined")

    # ── Smooth pipe series + hex-hole reservoir ───────────────────────────
    print("\n--- Smooth pipe series ---")
    series = example_pipe_series()
    series_rings = {f"Seg {i+1}": s[0] for i, s in enumerate(series)}

    # Without reservoir
    save_series_td(series, "output/pipe_series")

    # With hex-hole reservoir
    last_rings_s, last_frames_s = series[-1]
    pipe_end_s    = last_rings_s[-1].mean(0)
    pipe_radius_s = float(np.linalg.norm(last_rings_s[-1][0] - pipe_end_s))
    t_end_s       = last_rings_s[-1].mean(0) - last_rings_s[-2].mean(0)
    t_end_s      /= np.linalg.norm(t_end_s)
    res_radius    = 0.5

    save_series_td(series, "output/pipe_series_network",
                   reservoir_radius=res_radius)

    n_sides = len(series[0][0][0])
    visualize_pipes(
        series_rings,
        title="Smooth Pipe Series with Polygon-Hole Reservoir",
        save_path="output/pipe_series_visualization.png",
        reservoir_holes=[(pipe_end_s, pipe_radius_s,
                          last_frames_s[-1], t_end_s, res_radius, n_sides)],
    )

    plt.show()
