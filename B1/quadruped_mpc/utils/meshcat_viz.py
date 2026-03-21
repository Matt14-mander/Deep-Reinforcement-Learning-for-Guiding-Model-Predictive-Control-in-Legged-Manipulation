# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Meshcat geometry helpers for quadruped MPC visualization.

Key design notes
----------------
Cylinder axis bug
    meshcat-python wraps three.js ``CylinderGeometry``, which is oriented along
    the **Y-axis** in local space (not Z).  The old code rotated Z→d, giving
    wrong orientations for mostly-vertical GRF vectors (the "horizontal line"
    artefact).  Every cylinder function here rotates **Y→d** correctly.

Friction cone
    A friction cone of coefficient μ has half-angle θ = arctan(μ) from the
    contact normal (vertical for flat ground).  The GRF vector must stay inside
    the cone for the contact to be physically valid (no slip).
    We visualize this as:
      • N wire spokes from the contact point to the cone rim
      • A closed rim circle at the cone height
      • The GRF force arrow, coloured GREEN (inside cone) or RED (violation)

Public API
----------
    draw_contact_viz(viewer, path, foot_pos, grf, mu, scale, n_spokes)
        One-shot: draws friction cone + GRF arrow at a given foot.

    draw_grf_arrow(viewer, path, origin, grf_world, mu, scale)
        Just the arrow (no cone).

    draw_friction_cone(viewer, path, apex, mu, cone_height, n_spokes,
                       color_hex, opacity)
        Just the cone wireframe.

    mc_line / mc_sphere / mc_box / mc_cylinder
        Low-level helpers, all with correct Y-axis cylinder orientation.
"""

from typing import Optional

import numpy as np

try:
    import meshcat.geometry as mg
    import meshcat.transformations as mt
    HAS_MESHCAT = True
except ImportError:
    HAS_MESHCAT = False


# ── colour helpers ────────────────────────────────────────────────────────────

def hex_to_int(h: str) -> int:
    """Convert "#rrggbb" → integer 0xRRGGBB."""
    return int(h.lstrip("#"), 16)


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """(0-1 floats) → "#rrggbb"."""
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


# ── rotation from Y-axis to an arbitrary direction ────────────────────────────

def _rot_y_to(d: np.ndarray) -> np.ndarray:
    """4×4 rotation matrix that maps the Y-axis onto unit vector *d*.

    meshcat Cylinder geometry is oriented along local Y, so every mc_cylinder
    call must rotate Y→d (not Z→d as in the original poster_demo code).
    """
    d = np.asarray(d, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.eye(4)
    d = d / n

    y = np.array([0.0, 1.0, 0.0])
    dot = np.clip(np.dot(y, d), -1.0, 1.0)
    angle = np.arccos(dot)

    axis = np.cross(y, d)
    axis_norm = np.linalg.norm(axis)

    T = np.eye(4)
    if axis_norm < 1e-8:
        # already aligned (dot≈1) or anti-aligned (dot≈-1)
        if dot < 0:
            # 180° rotation around X
            T[:3, :3] = np.array([[1, 0, 0],
                                   [0, -1, 0],
                                   [0, 0, -1]], dtype=float)
    else:
        T[:3, :3] = mt.rotation_matrix(angle, axis / axis_norm)[:3, :3]
    return T


# ── low-level primitives ──────────────────────────────────────────────────────

def mc_sphere(viewer, path: str, pos, radius: float,
              color_hex: str = "#ffffff", opacity: float = 1.0):
    """Add / update a sphere in Meshcat."""
    if not HAS_MESHCAT:
        return
    mat = mg.MeshLambertMaterial(color=hex_to_int(color_hex), opacity=opacity)
    viewer[path].set_object(mg.Sphere(radius), mat)
    T = np.eye(4)
    T[:3, 3] = np.asarray(pos, dtype=float)
    viewer[path].set_transform(T)


def mc_line(viewer, path: str, pts, color_hex: str = "#ffffff", lw: int = 2):
    """Draw a polyline through a list of 3-D points."""
    if not HAS_MESHCAT:
        return
    arr = np.array(pts, dtype=np.float32).T   # (3, N)
    geom = mg.PointsGeometry(arr)
    mat  = mg.LineBasicMaterial(color=hex_to_int(color_hex), linewidth=lw)
    viewer[path].set_object(mg.Line(geom, mat))


def mc_cylinder(viewer, path: str,
                origin,
                direction,
                length: float,
                radius: float,
                color_hex: str = "#ffffff",
                opacity: float = 1.0):
    """Draw a cylinder oriented along *direction*, centred between origin and
    origin + direction*length.

    Correctly accounts for Meshcat/three.js CylinderGeometry being Y-aligned.
    """
    if not HAS_MESHCAT:
        return
    length = max(float(length), 1e-6)
    d = np.asarray(direction, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return
    d = d / n

    # Cylinder centre = midpoint along the arrow
    centre = np.asarray(origin, dtype=float) + d * length / 2

    # Rotation Y → d
    T = _rot_y_to(d)
    T[:3, 3] = centre

    mat = mg.MeshLambertMaterial(color=hex_to_int(color_hex), opacity=opacity)
    viewer[path].set_object(mg.Cylinder(length, radius), mat)
    viewer[path].set_transform(T)


def mc_cone(viewer, path: str,
            apex,
            direction,
            height: float,
            base_radius: float,
            color_hex: str = "#ffaa00",
            opacity: float = 0.25):
    """Draw a solid cone with apex at *apex*, opening along *direction*.

    Uses a cylinder with one end capped to zero radius (three.js cone trick:
    top_radius=0).  Correctly Y-aligned.
    """
    if not HAS_MESHCAT:
        return
    height = max(float(height), 1e-6)
    d = np.asarray(direction, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return
    d = d / n

    # The meshcat Cylinder with radiusTop=0 behaves as a cone.
    # We want the apex at *apex*, so centre = apex + d*height/2
    centre = np.asarray(apex, dtype=float) + d * height / 2

    T = _rot_y_to(d)
    T[:3, 3] = centre

    mat = mg.MeshLambertMaterial(color=hex_to_int(color_hex), opacity=opacity,
                                  side=mg.MeshLambertMaterial.SideDouble
                                  if hasattr(mg.MeshLambertMaterial, "SideDouble")
                                  else None)
    if mat.side is None:
        mat = mg.MeshLambertMaterial(color=hex_to_int(color_hex), opacity=opacity)

    # Cylinder(height, radius, radiusTop, radiusBottom) — meshcat 0.3+
    try:
        cone_geom = mg.Cylinder(height, base_radius, 0.0, base_radius)
    except TypeError:
        # Older meshcat: no radiusTop/radiusBottom → draw as normal cylinder
        cone_geom = mg.Cylinder(height, base_radius)

    viewer[path].set_object(cone_geom, mat)
    viewer[path].set_transform(T)


def mc_delete(viewer, path: str):
    """Remove an object from the Meshcat scene (best-effort)."""
    if not HAS_MESHCAT:
        return
    try:
        viewer[path].delete()
    except Exception:
        pass


# ── friction cone ─────────────────────────────────────────────────────────────

def draw_friction_cone(viewer,
                       path: str,
                       apex,
                       mu: float = 0.7,
                       cone_height: float = 0.18,
                       n_spokes: int = 8,
                       color_hex: str = "#f39c12",
                       opacity: float = 0.55):
    """Draw a friction cone wireframe at a contact point.

    The cone represents the set of GRF directions that satisfy the Coulomb
    friction constraint  |F_xy| ≤ μ · F_z  (no-slip condition).

    Geometry:
      • apex at *apex* (the contact point on the ground)
      • opening upward along +Z (contact normal for flat ground)
      • half-angle θ = arctan(μ) from +Z
      • *n_spokes* line segments from apex to the rim circle
      • one closed rim circle

    Args:
        viewer:     Meshcat viewer handle (e.g. ``viz.viewer``).
        path:       Meshcat path prefix, e.g. ``"cone/LF"``.
        apex:       Contact point in world frame, shape (3,).
        mu:         Friction coefficient (default 0.7).
        cone_height: Visual height of the cone in metres.
        n_spokes:   Number of wireframe spokes.
        color_hex:  Spoke/rim colour (hex string).
        opacity:    Fill cone opacity (0 = wireframe only).
    """
    if not HAS_MESHCAT:
        return

    apex = np.asarray(apex, dtype=float)
    theta = np.arctan(mu)           # half-angle from vertical
    r_rim = cone_height * np.tan(theta)   # rim radius at cone_height

    # ── semi-transparent filled cone ─────────────────────────────────────────
    # Apex at foot, opening upward
    mc_cone(viewer, f"{path}/fill",
            apex=apex,
            direction=[0, 0, 1],
            height=cone_height,
            base_radius=r_rim,
            color_hex=color_hex,
            opacity=opacity * 0.45)

    # ── wire spokes ──────────────────────────────────────────────────────────
    rim_pts = []
    for i in range(n_spokes):
        phi = 2 * np.pi * i / n_spokes
        rim_pt = apex + np.array([
            r_rim * np.cos(phi),
            r_rim * np.sin(phi),
            cone_height,
        ])
        rim_pts.append(rim_pt.tolist())
        mc_line(viewer, f"{path}/spoke_{i}",
                [apex.tolist(), rim_pt.tolist()],
                color_hex, lw=1)

    # ── rim circle (closed) ──────────────────────────────────────────────────
    rim_pts_closed = rim_pts + [rim_pts[0]]   # close the loop
    mc_line(viewer, f"{path}/rim", rim_pts_closed, color_hex, lw=2)


# ── GRF arrow ─────────────────────────────────────────────────────────────────

def draw_grf_arrow(viewer,
                   path: str,
                   foot_pos,
                   grf_world,
                   mu: float = 0.7,
                   scale: float = 0.0020,
                   shaft_radius: float = 0.012,
                   head_radius:  float = 0.022,
                   head_frac:    float = 0.22,
                   color_ok: str  = "#2ecc71",
                   color_bad: str = "#e74c3c"):
    """Draw a GRF force arrow from *foot_pos* along *grf_world*.

    The arrow is coloured:
      • GREEN  (``color_ok``)  if the GRF is inside the friction cone
        (|F_xy| ≤ μ · F_z) — physically valid, no slip
      • RED    (``color_bad``) if the GRF violates the friction cone

    Geometry: cylindrical shaft + smaller cone tip (arrowhead).

    Args:
        viewer:       Meshcat viewer.
        path:         Path prefix for this foot, e.g. ``"grf/LF"``.
        foot_pos:     Foot position in world frame, shape (3,).
        grf_world:    GRF vector in world frame [Fx, Fy, Fz], shape (3,).
        mu:           Friction coefficient.
        scale:        Newton → metre visual scale (default 0.002 → 100 N = 0.2 m).
        shaft_radius: Cylinder radius (metres).
        head_radius:  Arrowhead cone base radius (metres).
        head_frac:    Fraction of total length used for the arrowhead.
        color_ok:     Hex colour when inside friction cone.
        color_bad:    Hex colour when friction cone violated.
    """
    if not HAS_MESHCAT:
        return

    foot_pos  = np.asarray(foot_pos,  dtype=float)
    grf_world = np.asarray(grf_world, dtype=float)
    fz        = grf_world[2]

    # Hide arrow during swing (no contact force)
    if fz < 0.5:
        mc_delete(viewer, f"{path}/shaft")
        mc_delete(viewer, f"{path}/head")
        return

    # Friction cone check
    f_xy    = np.linalg.norm(grf_world[:2])
    in_cone = f_xy <= mu * fz + 1e-6
    color   = color_ok if in_cone else color_bad

    total_len = np.linalg.norm(grf_world) * scale
    total_len = max(total_len, 0.02)

    shaft_len = total_len * (1.0 - head_frac)
    head_len  = total_len * head_frac

    d = grf_world / np.linalg.norm(grf_world)

    # Shaft
    mc_cylinder(viewer, f"{path}/shaft",
                origin=foot_pos,
                direction=d,
                length=shaft_len,
                radius=shaft_radius,
                color_hex=color, opacity=0.92)

    # Arrowhead cone (apex at shaft tip, opening along d)
    shaft_tip = foot_pos + d * shaft_len
    mc_cone(viewer, f"{path}/head",
            apex=shaft_tip,
            direction=d,
            height=head_len,
            base_radius=head_radius,
            color_hex=color, opacity=0.92)


# ── combined contact visualizer ───────────────────────────────────────────────

def draw_contact_viz(viewer,
                     path: str,
                     foot_pos,
                     grf_world,
                     mu: float = 0.7,
                     grf_scale: float = 0.002,
                     cone_height: float = 0.18,
                     n_spokes: int = 8,
                     cone_color: str  = "#f39c12",
                     cone_opacity: float = 0.50):
    """Draw friction cone + GRF arrow for one foot contact.

    Combines ``draw_friction_cone`` and ``draw_grf_arrow`` under a single path.

    During swing (Fz < 0.5 N) the GRF arrow is hidden; the cone remains visible
    as a reminder of the physical constraint.

    Args:
        viewer:       Meshcat viewer.
        path:         Path prefix, e.g. ``"contact/LF"``.
        foot_pos:     Foot contact point in world frame (3,).
        grf_world:    GRF vector in world frame [Fx, Fy, Fz] (3,).
        mu:           Friction coefficient.
        grf_scale:    Newton → metre scale for the arrow.
        cone_height:  Visual height of the friction cone.
        n_spokes:     Wireframe spokes on the cone.
        cone_color:   Cone wireframe colour.
        cone_opacity: Cone fill transparency.
    """
    draw_friction_cone(
        viewer, f"{path}/cone",
        apex=foot_pos, mu=mu,
        cone_height=cone_height, n_spokes=n_spokes,
        color_hex=cone_color, opacity=cone_opacity,
    )
    draw_grf_arrow(
        viewer, f"{path}/force",
        foot_pos=foot_pos, grf_world=grf_world,
        mu=mu, scale=grf_scale,
    )
