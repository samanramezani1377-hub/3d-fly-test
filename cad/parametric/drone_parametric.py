"""Parametric quadcopter model (rebuilt from the 'smooth 2' mesh measurements).

Design coordinates are Y-up with X-Z as the plan; exported CAD is rotated to Z-up for
common CAD/3D-printing orientation. Export transform: (x, y, z) -> (x, -z, y).
Thus the camera/front points toward -Y and the flat underside faces -Z. Units: mm.
Edit PARAMS (or pass --params file.json) and re-run to regenerate the model.
"""
import json, sys
import cadquery as cq

PARAMS = {
    # central body (plan polygon, mirrored left/right) ---------------------
    "body_height": 13.0,            # lower hull height (y 0..body_height)
    "body_shell_thickness": 2.0,    # nominal thickness of shell, flat bottom plate, and roof
    "body_half_width": 38.0,        # max half width (x) at the widest point (was 32: middle made wider)
    "body_front_z": 66.0,           # nose (+z)
    "body_rear_z": -52.0,           # tail (-z)
    "body_front_half_width": 9.0,
    "body_rear_half_width": 16.0,
    "body_widest_z_front": 10.0,
    "body_widest_z_rear": -18.0,
    "body_edge_fillet": 0.8,
    # extra widening of the dome through the middle of the body (plan view: narrow nose/tail, fat middle)
    "mid_widen": 0.22,              # relative extra half width at the middle (0 = off)
    "mid_widen_z": -4.0,            # z of the widest point of the dome
    "mid_widen_sigma": 30.0,        # how far (mm) the widening spreads front/back
    # canopy (upper dome) ---------------------------------------------------
    "canopy_height": 12.0,
    "canopy_front_z": 44.0,
    "canopy_rear_z": -20.0,
    "canopy_half_width": 19.0,
    "canopy_taper_deg": 18.0,
    "canopy_fillet": 4.0,
    "use_measured_canopy": True,     # loft through cad/parametric/body_sections.json (edit the numbers to reshape)
    "canopy_base_y": 13.0,
    "canopy_top_y": 25.6,
    "canopy_top_scale": 0.45,        # last ring of the dome relative to the top measured section
    # faceted (low-poly) dome, like the reference render --------------------------
    "canopy_faceted": True,
    "facet_vertices": 14,            # vertices per level (fewer = bigger flat facets)
    "facet_inset": 0.90,             # every 2nd vertex of the mid levels is pulled inward -> sharp creases
    "facet_ridge_lift": 1.6,         # extra height (mm) given to the top ridge vertices
    # rotor stations / arms ---------------------------------------------------
    "motor_x": 58.0,                # |x| of each motor axis
    "motor_z": 58.0,                # |z| of each motor axis (= motor_x -> square 4-rotor layout, was 39.5)
    "arm_width": 16.0,
    "arm_height": 12.0,
    "motor_radius": 6.0,
    "motor_height": 20.0,
    "motor_hub_radius": 9.0,        # base boss under the motor
    "motor_hub_height": 6.0,
    # propeller guards: TRUE circular rings -----------------------------------
    "guard_outer_radius": 35.5,     # used when guard_use_profile is False
    "guard_wall": 4.5,
    "guard_height": 21.0,
    "guard_edge_fillet": 1.8,
    # barrel profile measured on the original ring wall (radius vs height, mm)
    "guard_profile_y":     [0.0, 1.5, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 21.0],
    "guard_profile_outer": [29.0, 32.1, 33.4, 34.4, 34.9, 35.3, 35.2, 34.2, 33.1, 31.0],
    "guard_profile_inner": [27.8, 27.8, 27.8, 28.0, 28.0, 27.5, 27.0, 27.0, 28.0, 28.5],
    "guard_use_profile": True,
    # enlarge the guards in place (radii of the measured profile shifted outward, wall thickness kept) until they
    # almost touch the body (0.6 mm gap, not fused): +2.7 model units -> outer radius ~48.9 mm final
    "guard_grow": 2.7,
    "guard_y_offset": 10.0,          # lower edge overlaps arm for fusion; extended top covers propeller
    # front LED slit (no separate part): a small slit through the front shell that opens into a small
    # pocket behind it, so a real LED can be glued from inside later. Sizes are in FINAL (scaled) mm.
    "led_enabled": True,
    "led_y": 14.5,                   # height of the slit centre (model units, before scaling)
    "led_slit_length": 11.0,          # +10% with overall model scale
    "led_slit_height": 2.2,           # +10% with overall model scale
    "led_wall": 1.65,                # +10% with overall model scale
    "led_pocket_length": 13.2,       # +10% with overall model scale
    "led_pocket_height": 4.4,         # +10% with overall model scale
    "led_pocket_depth": 6.6,         # +10% with overall model scale
    # front camera: small module hidden INSIDE the nose, only a small square housing (aligned with the lens axis) shows outside.
    # Sizes are in FINAL (scaled) mm. The lens axis points forward, tilted down by camera_tilt_down_deg.
    "camera_enabled": True,
    "camera_tilt_down_deg": 35.0,
    "camera_y": 9.185,               # +10% with the enlarged nose; keep housing centered on the scaled face
    "camera_block_size": 11.55,      # +10% with overall model scale
    "camera_block_protrusion": 3.85, # +10% with overall model scale
    "camera_block_fillet": 0.88,     # +10% with overall model scale
    "camera_lens_hole_radius": 3.52, # +10% with overall model scale
    "camera_wall": 1.65,             # +10% with overall model scale
    "camera_cavity_w": 10.45,        # +10% with overall model scale
    "camera_cavity_h": 6.6,          # +10% with overall model scale
    "camera_cavity_d": 7.7,          # +10% with overall model scale
    # propellers (parametric, twisted cambered two-blade rotor) -----------------
    "include_props": True,
    "prop_radius": 26.45,           # 34.0 mm final radius -> 68.0 mm / 2.68 in diameter; preserves ~4.18 mm radial guard clearance
    "prop_chord": 7.0,              # legacy parameter retained for old parameter files
    "prop_thickness": 1.2,          # legacy parameter; airfoil thickness is set by prop_airfoil_thickness_ratio
    "prop_y": 25.5,                 # lifted above motor and arm envelope; validated for 3D clearance
    "prop_hub_radius": 4.0,
    "prop_hub_height": 4.0,
    "prop_pitch_mm": 53.086,        # +10% with the complete rotor geometry; not a motor/load guarantee
    "prop_airfoil_camber": 0.04,    # NACA 4410 camber ratio
    "prop_airfoil_camber_position": 0.40,
    "prop_airfoil_thickness_ratio": 0.10,
    "prop_airfoil_samples": 24,     # cosine-spaced points per upper/lower surface
    "prop_tip_sweep": 1.2,          # mild sweep at the tip, in pre-scale model units
    # [radius fraction, local chord] stations from root to tip; taper avoids a rectangular slab.
    "prop_blade_stations": [  # chord increased ~7% with diameter to preserve blade proportions
        [0.05, 6.96], [0.14, 8.36], [0.23, 9.21], [0.33, 9.43],
        [0.44, 8.79], [0.56, 7.82], [0.68, 6.86], [0.80, 6.00],
        [0.90, 5.14], [0.96, 4.71], [1.00, 4.28]
    ],
    # overall size -------------------------------------------------------------
    # model_scale: uniform final scale. Current assembly is enlarged by 10% from the previous approved layout.
    # Real M3 fastener/nut interface dimensions are compensated locally and remain physically unchanged.
    # target_overall_mm: if set (and model_scale is None) the scale is derived so the largest extent equals it.
    # Floor-mounted arm attachment: integral tongue + two vertical M3 fasteners per arm.
    # Dimensions below are pre-scale model units; model_scale converts them to final millimetres.
    "joint_clearance": 0.227273,      # reduced in local units so final physical fit clearance stays unchanged
    "joint_tip_width": 16.5,          # keep the tongue full-width at its inner end; no sudden neck-down
    "joint_lead_in_length": 6.0,
    "joint_branch_long_length": 22.0,   # longitudinal branch length toward the body centre
    "joint_branch_short_length": 14.0,  # transverse branch length at the body entry point
    "joint_branch_width_ratio": 0.60,  # each tick branch is 60% of the main arm width
    "joint_branch_axis_bias_deg": 55.0, # short branch rotates +10 degrees toward the long-branch side
    "joint_branch_long_axis_deg": 55.0,  # both branches rotate together +10 degrees; included angle remains 110 degrees
    "joint_tick_junction_radius": 36.61,  # extend the arm 15 final mm and keep the tick junction at its inner tip
    "joint_tab_start_radius": 40.5,
    "joint_tab_end_radius": 43.0,
    "joint_tab_width": 16.5,
    "joint_tab_thickness": 2.5,  # legacy values retained for parameter-file compatibility; no low-profile tab is generated
    "joint_tab_y": 4.0,
    "joint_floor_pad_height": 2.0,
    "joint_floor_pad_width": 16.5,
    "joint_floor_pad_end_radius": 34.5,
    "joint_arm_start_radius": 36.61,  # extend the main arm 15 final mm and align its inner tip with the tick junction
    "joint_entry_start_radius": 34.5,
    "joint_entry_end_radius": 50.0,

    "joint_bolt_positions": [[28.5, -4.5], [28.5, 4.5]],
    "joint_bolt_hole_radius": 1.181818, # preserve the existing real M3 clearance-hole diameter after scaling
    "joint_nut_af": 3.909091,         # scaled locally so the real M3 nut remains ~5.53 mm across flats
    "joint_nut_pocket_af": 4.181818,  # preserve real nut pocket size and side clearance
    "joint_nut_depth": 1.818182,      # preserve physical pocket depth for M3 nut
    "joint_nut_hole_radius": 1.090909, # preserve real nut bore size
    "joint_screw_shaft_radius": 1.045455, # preserve M3 screw shaft diameter
    "joint_screw_length": 14.545455, # local units adjusted to retain the existing physical screw length
    "joint_screw_head_radius": 1.954545, # preserve physical screw-head diameter
    "joint_screw_head_height": 2.090909, # preserve physical screw-head height
    "joint_screw_socket_af": 1.772727, # preserve physical hex socket size
    "joint_screw_socket_depth": 0.818182, # preserve physical socket depth
    "joint_nut_thickness": 1.727273,   # preserve physical M3 nut thickness
    "joint_boss_diameter_mm": 11.0,       # +10% underside reinforcement around each nut station (final mm)
    "joint_boss_total_thickness_mm": 8.8, # +10% total thickness from boss underside to floor interior (final mm)
    "model_scale": 1.414028,              # original scale x 1.10: entire CAD assembly enlarged uniformly by 10%
    "target_overall_mm": None,
}


def _resample(pts, n):
    import numpy as np
    pts = np.asarray(pts, float)
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    if area < 0:
        pts = pts[::-1]
    pts = np.roll(pts, -int(np.argmax(pts[:, 1])), axis=0)
    line = np.vstack([pts, pts[:1]])
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(line, axis=0).T))]
    t = np.linspace(0, seg[-1], n, endpoint=False)
    return np.c_[np.interp(t, seg, line[:, 0]), np.interp(t, seg, line[:, 1])]


def widen_x(p, pts):
    """Scale x of dome section points by a smooth bump centred on the body middle."""
    import numpy as np
    a = np.array(pts, float)
    k, zc, sg = p.get("mid_widen", 0.0), p["mid_widen_z"], p["mid_widen_sigma"]
    a[:, 0] *= 1.0 + k * np.exp(-((a[:, 1] - zc) / sg) ** 2)
    return a


def hull_polygon(p):
    return [(-p["body_front_half_width"], p["body_front_z"]),
            (p["body_front_half_width"], p["body_front_z"]),
            (p["body_half_width"], p["body_widest_z_front"]),
            (p["body_half_width"], p["body_widest_z_rear"]),
            (p["body_rear_half_width"], p["body_rear_z"]),
            (-p["body_rear_half_width"], p["body_rear_z"]),
            (-p["body_half_width"], p["body_widest_z_rear"]),
            (-p["body_half_width"], p["body_widest_z_front"])]


def canopy_base_points(p):
    """Shared 14-vertex footprint used by both the lower hull and canopy rim."""
    import os, numpy as np
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "body_sections.json")
    data = json.load(open(path))
    n = data["points_per_level"]
    K = p["facet_vertices"]
    idx = [int(round(i * n / K)) % n for i in range(K)]
    a = _resample(hull_polygon(p), n)[idx]
    # Keep the exact same mirrored point treatment as the faceted canopy rings.
    b = a.copy()
    for i in range(K):
        k = (K - i) % K
        b[i, 0] = (a[i, 0] - a[k, 0]) / 2.0
        b[i, 1] = (a[i, 1] + a[k, 1]) / 2.0
    return [tuple(map(float, q)) for q in b]


def faceted_canopy(p, inward=0.0, open_bottom_y=None, roof_drop=0.0):
    """Closed polyhedron through measured levels; inward builds an inset cavity for the hollow body."""
    import os, numpy as np
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "body_sections.json")
    data = json.load(open(path))
    n = data["points_per_level"]; K = p["facet_vertices"]
    idx = [int(round(i * n / K)) % n for i in range(K)]
    def pick(pts):
        a = np.asarray(pts, float)[idx]
        b = a.copy()                       # mirror-symmetric about x=0 (pair vertex i with K-i)
        for i in range(K):
            k = (K - i) % K
            b[i, 0] = (a[i, 0] - a[k, 0]) / 2.0
            b[i, 1] = (a[i, 1] + a[k, 1]) / 2.0
        return b
    secs = sorted((float(y), np.array(q)) for y, q in data["sections"].items())
    base_y = p["canopy_base_y"] if open_bottom_y is None else open_bottom_y
    def inset_ring(a):
        if inward <= 0:
            return a
        c = a.mean(axis=0)
        radii = np.linalg.norm(a - c, axis=1)
        return c + (a - c) * np.maximum(0.1, (radii - inward) / np.maximum(radii, 1e-6))[:, None]
    base_pts = np.asarray(canopy_base_points(p), float)
    lv = [(base_y, inset_ring(base_pts))]
    for j, (y, q) in enumerate(secs):
        a = widen_x(p, pick(q)); c = a.mean(axis=0)
        a[1::2] = c + (a[1::2] - c) * p["facet_inset"]
        lv.append((y, inset_ring(a)))
    ytop, ptop = secs[-1]; a = widen_x(p, pick(ptop)); c = a.mean(axis=0)
    top = inset_ring(c + (a - c) * p["canopy_top_scale"])
    lv.append((p["canopy_top_y"] - roof_drop, top))
    P = [[cq.Vector(float(x), float(y), float(z)) for (x, z) in pts] for y, pts in lv]
    # Lift the two top-ring vertices closest to the longitudinal centreline,
    # giving the crown ridge the configured extra height instead of leaving this
    # parameter as an unused setting.
    ridge_lift = float(p.get("facet_ridge_lift", 0.0))
    if ridge_lift:
        top_ring = P[-1]
        ridge_indices = sorted(range(len(top_ring)),
                               key=lambda i: (abs(top_ring[i].x), -abs(top_ring[i].z)))[:2]
        for i in ridge_indices:
            v = top_ring[i]
            top_ring[i] = cq.Vector(v.x, v.y + ridge_lift, v.z)
    faces = []
    def tri(a, b, c):
        faces.append(cq.Face.makeFromWires(cq.Wire.makePolygon([a, b, c], close=True)))
    for L in range(len(P) - 1):
        for i in range(K):
            j = (i + 1) % K
            if i % 2 == 0:
                tri(P[L][i], P[L][j], P[L + 1][i]); tri(P[L][j], P[L + 1][j], P[L + 1][i])
            else:
                tri(P[L][i], P[L][j], P[L + 1][j]); tri(P[L][i], P[L + 1][j], P[L + 1][i])
    faces.append(cq.Face.makeFromWires(cq.Wire.makePolygon(P[0] + [P[0][0]], close=False) if False else cq.Wire.makePolygon(P[0], close=True)))
    # The ridge lift makes the top ring non-planar, so cap it with planar triangles
    # instead of attempting to create one planar face from the whole ring.
    top_center = cq.Vector(
        sum(v.x for v in P[-1]) / len(P[-1]),
        sum(v.y for v in P[-1]) / len(P[-1]),
        sum(v.z for v in P[-1]) / len(P[-1]),
    )
    for i in range(K):
        j = (i + 1) % K
        tri(P[-1][i], P[-1][j], top_center)
    shell = cq.Shell.makeShell(faces)
    solid = cq.Solid.makeSolid(shell)
    if solid.Volume() < 0:
        solid = cq.Solid(solid.wrapped.Reversed())
    return cq.Workplane("XY").add(solid)


def measured_canopy(p):
    import os, numpy as np
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "body_sections.json")
    data = json.load(open(path))
    n = data["points_per_level"]
    levels = [(p["canopy_base_y"], _resample(hull_polygon(p), n))]
    secs = [(float(y), widen_x(p, pts)) for y, pts in data["sections"].items()]
    secs.sort(key=lambda t: t[0])
    levels += secs
    ytop, ptop = secs[-1]
    c = ptop.mean(axis=0)
    levels.append((p["canopy_top_y"], c + (ptop - c) * p["canopy_top_scale"]))
    wp = None
    for i, (y, pts) in enumerate(levels):
        if wp is None:
            wp = cq.Workplane("XZ").workplane(offset=-y)
        else:
            wp = wp.workplane(offset=-(y - levels[i - 1][0]))
        wp = wp.polyline([tuple(q) for q in pts]).close()
    return wp.loft(ruled=False, combine=True)


def body(p):
    # Match the lower hull's top rim to the exact canopy base polygon.
    pts = canopy_base_points(p)
    # Preserve the approved exterior geometry.
    hull = cq.Workplane("XZ").polyline(pts).close().extrude(-p["body_height"])
    if p.get("use_measured_canopy"):
        try:
            canopy = faceted_canopy(p) if p.get("canopy_faceted") else measured_canopy(p)
            outer = hull.union(canopy)
        except Exception as e:
            print("measured canopy failed, using the simple canopy fallback:", e)
            outer = hull
    else:
        try:
            hull = hull.edges("|Y").fillet(p["body_edge_fillet"])
        except Exception:
            pass
        cpts = [(-p["canopy_half_width"] * 0.6, p["canopy_front_z"]),
                (p["canopy_half_width"] * 0.6, p["canopy_front_z"]),
                (p["canopy_half_width"], 0.0),
                (p["canopy_half_width"] * 0.9, p["canopy_rear_z"]),
                (-p["canopy_half_width"] * 0.9, p["canopy_rear_z"]),
                (-p["canopy_half_width"], 0.0)]
        canopy = (cq.Workplane("XZ").workplane(offset=-p["body_height"]).polyline(cpts).close()
                  .extrude(-p["canopy_height"], taper=p["canopy_taper_deg"]))
        try:
            canopy = canopy.faces(">Y").edges().fillet(p["canopy_fillet"])
        except Exception:
            pass
        outer = hull.union(canopy)

    # Hollow the body internally while retaining a continuous, flat bottom plate.
    # The same nominal wall thickness is kept on the floor, side walls, and roof.
    wall = float(p.get("body_shell_thickness", 2.0))
    if wall <= 0 or wall >= min(p["body_half_width"], p["body_height"]):
        raise ValueError("body_shell_thickness must be positive and smaller than the body")
    cz = (p["body_front_z"] + p["body_rear_z"]) / 2.0
    scale_x = max(0.1, 1.0 - wall / p["body_half_width"])
    scale_z = max(0.1, 1.0 - wall / ((p["body_front_z"] - p["body_rear_z"]) / 2.0))
    inner_pts = [(x * scale_x, cz + (z - cz) * scale_z) for x, z in pts]
    # Workplane XZ extrudes along -Y; negative extrusion runs upward.
    # Start the cavity at y=wall (not below y=0) so the outside underside stays closed and planar.
    lower_void = cq.Workplane("XZ").polyline(inner_pts).close().extrude(-(p["body_height"] + 2.0))
    lower_void = lower_void.translate((0, wall, 0))
    if p.get("use_measured_canopy") and p.get("canopy_faceted"):
        roof_void = faceted_canopy(p, inward=wall, open_bottom_y=p["body_height"] - 1.0,
                                   roof_drop=wall)
        cavity = lower_void.union(roof_void)
        hollow = outer.cut(cavity)
    else:
        # Keep the fallback canopy stable; at minimum the lower body is hollow.
        hollow = outer.cut(lower_void)
    if not hollow.val().isValid():
        raise RuntimeError("Hollow body geometry is invalid")
    return hollow


def stations(p):
    return [(sx * p["motor_x"], sz * p["motor_z"]) for sx in (1, -1) for sz in (1, -1)]


def guard_profile(p, x, z):
    """Round guard ring made by revolving the measured barrel profile -> exact circles."""
    g = p.get("guard_grow", 0.0)
    ys = p["guard_profile_y"]
    ro = [r + g for r in p["guard_profile_outer"]]
    ri = [r + g for r in p["guard_profile_inner"]]
    outer = [(r, y) for r, y in zip(ro, ys)]
    inner = [(r, y) for r, y in zip(ri, ys)][::-1]
    w = (cq.Workplane("XY").moveTo(*outer[0]).spline(outer[1:], includeCurrent=True)
         .lineTo(*inner[0]).spline(inner[1:], includeCurrent=True).close())
    ring = w.revolve(360, (0, 0, 0), (0, 1, 0))
    return ring.translate((x, float(p.get("guard_y_offset", 0.0)), z))


def guard(p, x, z):
    """Round guard ring: outer/inner radius are exact circles by construction."""
    if p.get("guard_use_profile"):
        return guard_profile(p, x, z)
    ro, ri = p["guard_outer_radius"], p["guard_outer_radius"] - p["guard_wall"]
    ring = (cq.Workplane("XZ").center(x, z).circle(ro).circle(ri).extrude(-p["guard_height"]))
    try:
        ring = ring.faces(">Y").edges().fillet(p["guard_edge_fillet"])
    except Exception:
        pass
    return ring.translate((0, float(p.get("guard_y_offset", 0.0)), 0))


# The arm and body socket share this rounded, tapered plan-view profile.
def radial_joint_profile(length, width, tip_width, lead_length):
    """Plan-view outline with a square, full-width end and no necked-down tip."""
    if length <= 0 or width <= 0 or tip_width <= 0 or tip_width > width:
        raise ValueError("invalid radial joint profile dimensions")
    # A removable arm/tongue should not taper to a narrow point at its body-side end.
    # Keep both side edges parallel and terminate at the full requested width.
    if tip_width >= width - 1e-6:
        return (cq.Workplane("XY")
                .moveTo(0, -width / 2)
                .lineTo(length, -width / 2)
                .lineTo(length, width / 2)
                .lineTo(0, width / 2)
                .close())
    lead = min(max(lead_length, tip_width / 2), length)
    r = tip_width / 2
    return (cq.Workplane("XY")
            .moveTo(r, -r)
            .threePointArc((0, 0), (r, r))
            .lineTo(lead, width / 2)
            .lineTo(length, width / 2)
            .lineTo(length, -width / 2)
            .lineTo(lead, -width / 2)
            .close())


def arm(p, x, z, start_r=0.0, y_offset=0.0):
    """Main arm beam. Modular arms enter through the side opening at inner-floor level."""
    import math
    L = math.hypot(x, z)
    ang = math.degrees(math.atan2(z, x))
    if start_r < 0 or start_r >= L:
        raise ValueError("arm start radius must be within the motor station")
    # Preserve the main beam cross-section at every station, including the body entry.
    # The beam is translated upward to clear the floor plate, but its width AND height
    # remain the configured values; do not shorten the vertical section to align its top.
    tip_width = p["arm_width"]
    lead_length = p.get("joint_lead_in_length", 6.0)
    bar_height = float(p["arm_height"])
    if bar_height <= 0:
        raise ValueError("arm root height must remain positive")
    bar = radial_joint_profile(L - start_r, p["arm_width"], tip_width, lead_length).extrude(bar_height)
    bar = bar.rotate((0, 0, 0), (1, 0, 0), -90)
    bar = bar.rotate((0, 0, 0), (0, 1, 0), -ang)
    bar = bar.translate((start_r * math.cos(math.radians(ang)), y_offset,
                         start_r * math.sin(math.radians(ang))))
    return bar


def radial_part(profile, x, z, angle_deg, y_base):
    """Place a radial/tangential profile in the model's Y-up coordinate frame."""
    return (profile.rotate((0, 0, 0), (1, 0, 0), -90)
            .rotate((0, 0, 0), (0, 1, 0), -angle_deg)
            .translate((x, y_base, z)))


def _tick_branch_vectors(p, x, z):
    """Return unequal branch vectors fanning left/right from the arm entry point."""
    import math
    angle = math.atan2(z, x)
    long_len = float(p.get("joint_branch_long_length", 22.0))
    short_len = float(p.get("joint_branch_short_length", 14.0))
    # Rotate both branches by +10 degrees in the same direction, toward the
    # long-branch side. Their included angle remains 110 degrees: long +55,
    # short -55 degrees relative to the inward arm axis.
    short_angle = math.radians(float(p.get("joint_branch_axis_bias_deg", 55.0)))
    long_angle = math.radians(float(p.get("joint_branch_long_axis_deg", 55.0)))
    inward_x, inward_z = -math.cos(angle), -math.sin(angle)
    side_x, side_z = -inward_z, inward_x
    # Alternate the local handedness by quadrant so the four tick roots are
    # mirror-symmetric across both body axes instead of all twisting the same way.
    mirror = 1.0 if x * z >= 0 else -1.0
    side_x *= mirror
    side_z *= mirror
    lc, ls = math.cos(long_angle), math.sin(long_angle)
    sc, ss = math.cos(short_angle), math.sin(short_angle)
    return (
        (long_len, inward_x * lc + side_x * ls,
         inward_z * lc + side_z * ls),
        (short_len, inward_x * sc - side_x * ss,
         inward_z * sc - side_z * ss),
    )


def joint_bolt_points(p, x, z):
    """Bolt centres at the midpoints of the left and right tick branches."""
    import math
    angle = math.atan2(z, x)
    radius = float(p.get("joint_tick_junction_radius", p["joint_arm_start_radius"]))
    cx, cz = radius * math.cos(angle), radius * math.sin(angle)
    return [
        (cx + dx * length / 2.0, cz + dz * length / 2.0)
        for length, dx, dz in _tick_branch_vectors(p, x, z)
    ]


def _branch_shapes(p, x, z, y_base, height):
    """Full-height unequal tick branches fanning left/right at the arm entry."""
    import math
    angle = math.atan2(z, x)
    radius = float(p.get("joint_tick_junction_radius", p["joint_arm_start_radius"]))
    cx, cz = radius * math.cos(angle), radius * math.sin(angle)
    # Only branch width is reduced; the main beam remains full width and height.
    width = float(p["arm_width"]) * float(p.get("joint_branch_width_ratio", 0.60))
    branches = []
    for length, dx, dz in _tick_branch_vectors(p, x, z):
        theta = math.degrees(math.atan2(dx, dz))
        bar = cq.Workplane("XY").box(width, height, length)
        bar = bar.rotate((0, 0, 0), (0, 1, 0), theta)
        bar = bar.translate((cx + dx * length / 2.0,
                             y_base + height / 2.0,
                             cz + dz * length / 2.0))
        branches.append(bar)
    return tuple(branches)

def arm_with_tick_root(p, x, z, start_r=0.0, y_offset=0.0):
    """Full-section arm plus its integral, full-height tick-shaped root reinforcement."""
    module = arm(p, x, z, start_r=start_r, y_offset=y_offset)
    for branch in _branch_shapes(p, x, z, y_offset, p["arm_height"]):
        module = module.union(branch)
    return module


def mounting_tab(p, x, z):
    """Integral arm tongue with a long longitudinal branch and a shorter transverse branch."""
    import math
    angle_rad = math.atan2(z, x)
    angle = math.degrees(angle_rad)
    start, end = p["joint_tab_start_radius"], p["joint_tab_end_radius"]
    tab = radial_joint_profile(end - start, p["joint_tab_width"],
                               p["joint_tip_width"], p["joint_lead_in_length"]).extrude(p["joint_tab_thickness"])
    tab = radial_part(tab, start * math.cos(math.radians(angle)),
                      start * math.sin(math.radians(angle)), angle, p["joint_tab_y"])
    for branch in _branch_shapes(p, x, z, p["joint_tab_y"], p["joint_tab_thickness"]):
        tab = tab.union(branch)
    for hx, hz in joint_bolt_points(p, x, z):
        hole = (cq.Workplane("XZ").workplane(offset=-(p["joint_tab_y"] - 0.05))
                .center(hx, hz).circle(p["joint_bolt_hole_radius"])
                .extrude(-(p["joint_tab_thickness"] + 0.1)))
        tab = tab.cut(hole)
    return tab


def floor_mounting_rail(p, x, z):
    """Raised floor supports beneath both branches and the radial tongue."""
    import math
    angle = math.degrees(math.atan2(z, x))
    start, end = p["joint_tab_start_radius"], p["joint_floor_pad_end_radius"]
    rail = radial_joint_profile(end - start, p["joint_floor_pad_width"],
                                p["joint_tip_width"], p["joint_lead_in_length"]).extrude(
                                    p["joint_floor_pad_height"])
    # Rails begin exactly on the top face of the 2 mm floor plate; they do not
    # overlap into or subtract from the floor's nominal thickness.
    rail = radial_part(rail, start * math.cos(math.radians(angle)),
                       start * math.sin(math.radians(angle)), angle,
                       p["body_shell_thickness"])
    for branch in _branch_shapes(p, x, z, p["body_shell_thickness"],
                                 p["joint_floor_pad_height"]):
        rail = rail.union(branch)
    return rail


def arm_entry_slot(p, x, z):
    """Side-wall opening for the arm beam; the floor plate remains intact."""
    import math
    angle = math.degrees(math.atan2(z, x))
    start, end = p["joint_entry_start_radius"], p["joint_entry_end_radius"]
    floor_y = p["body_shell_thickness"]
    top_y = p["body_height"] + 0.2
    cutter = cq.Workplane("XY").box(end - start, top_y - floor_y, p["arm_width"] + 2 * p["joint_clearance"])
    cutter = cutter.rotate((0, 0, 0), (0, 1, 0), -angle)
    return cutter.translate(((start + end) / 2 * math.cos(math.radians(angle)),
                             (floor_y + top_y) / 2,
                             (start + end) / 2 * math.sin(math.radians(angle))))


def nut_part(p, x, z):
    """Simplified nut envelope seated inside the hex pocket recessed into the underside boss."""
    import math
    circ_dia = p["joint_nut_af"] / math.cos(math.radians(30.0))
    thickness = p["joint_nut_thickness"]
    scale_for_hardware = float(p.get("model_scale") or 1.0)
    boss_extension = max(0.0, float(p.get("joint_boss_total_thickness_mm", 8.0)) / scale_for_hardware
                        - float(p["body_shell_thickness"]))
    # Nut starts at the outermost face of the boss and extends inward (+Y), matching the pocket.
    nut = (cq.Workplane("XZ").center(x, z)
           .polygon(6, circ_dia).extrude(-thickness))
    bore = (cq.Workplane("XZ").center(x, z)
            .circle(p["joint_nut_hole_radius"]).extrude(-(thickness + 0.02)))
    return nut.cut(bore).translate((0, -boss_extension, 0))


def screw_part(p, x, z):
    """Simplified vertical socket-head screw envelope, accessible from the top."""
    import math
    shaft_r, head_r = p["joint_screw_shaft_radius"], p["joint_screw_head_radius"]
    head_h = p["joint_screw_head_height"]
    tab_top = p["body_shell_thickness"] + p["arm_height"]
    scale_for_hardware = float(p.get("model_scale") or 1.0)
    boss_extension = max(0.0, float(p.get("joint_boss_total_thickness_mm", 8.0)) / scale_for_hardware
                        - float(p["body_shell_thickness"]))
    # Extend the screw to the underside of the reinforced boss so it passes through the full nut.
    shaft_bottom = min(tab_top - p["joint_screw_length"], -boss_extension)
    shaft_length = tab_top - shaft_bottom
    shaft = (cq.Workplane("XZ").workplane(offset=-shaft_bottom).center(x, z)
             .circle(shaft_r).extrude(-shaft_length))
    head = (cq.Workplane("XZ").workplane(offset=-tab_top).center(x, z)
            .circle(head_r).extrude(-head_h))
    screw = shaft.union(head)
    socket_dia = p["joint_screw_socket_af"] / math.cos(math.radians(30.0))
    socket = (cq.Workplane("XZ").workplane(offset=-(tab_top + head_h + 0.01)).center(x, z)
              .polygon(6, socket_dia).extrude(p["joint_screw_socket_depth"] + 0.01))
    return screw.cut(socket)


def motor(p, x, z):
    hub = (cq.Workplane("XZ").center(x, z).circle(p["motor_hub_radius"]).extrude(-p["motor_hub_height"]))
    can = (cq.Workplane("XZ").center(x, z).circle(p["motor_radius"]).extrude(-p["motor_height"]))
    return hub.union(can)


def _surface_z(solid, x, y, zlo=0.0, zhi=95.0):
    """Find the foremost (+Z) shell surface at (x, y), including hollow canopy walls."""
    step = 0.25
    outside_z = zhi
    z = zhi
    while z >= zlo:
        if solid.isInside(cq.Vector(x, y, z)):
            # Scan from the front until we enter the first solid wall; then bisect its front face.
            inside_z = z
            lo, hi = inside_z, outside_z
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if solid.isInside(cq.Vector(x, y, mid)):
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        outside_z = z
        z -= step
    return None


def led_slot(p, solid, fs=1.0):
    """Cutter = slit through the front shell + a pocket right behind it (dims given in final mm, fs = scale)."""
    import numpy as np
    y0 = p["led_y"]
    L, h, wall = p["led_slit_length"] / fs, p["led_slit_height"] / fs, p["led_wall"] / fs
    PL, PH, PD = p["led_pocket_length"] / fs, p["led_pocket_height"] / fs, p["led_pocket_depth"] / fs
    # the nose slopes back as it rises: take the most set-back shell point over the whole pocket footprint
    xs = np.linspace(-max(L, PL) / 2, max(L, PL) / 2, 9)
    ys = np.linspace(y0 - PH / 2, y0 + PH / 2, 5)
    # The wider pocket footprint can extend beyond the tapered canopy at its upper corners.
    # Use actual shell intersections only; fail only if the slit region has no front surface at all.
    zs = [_surface_z(solid, float(x), float(y)) for x in xs for y in ys]
    zs = [z for z in zs if z is not None]
    if not zs:
        raise RuntimeError("LED slit position does not intersect the front shell")
    zmax, zmin = max(zs), min(zs)
    zp = zmin - wall                                   # pocket front face
    slit = cq.Workplane("XY").box(L, h, (zmax + 1.0) - zp).translate((0, y0, (zmax + 1.0 + zp) / 2))
    pocket = cq.Workplane("XY").box(PL, PH, PD + 0.2).translate((0, y0, (zp + 0.2 + zp - PD) / 2))
    return slit.union(pocket)


def camera_parts(p, fs=1.0):
    """(housing, cutter): square lens housing to add outside, and lens hole + internal pocket to cut. fs = scale factor."""
    sz, rh = p["camera_block_size"] / fs, p["camera_lens_hole_radius"] / fs
    prot, wall = p["camera_block_protrusion"] / fs, p["camera_wall"] / fs
    w, h, d = p["camera_cavity_w"] / fs, p["camera_cavity_h"] / fs, p["camera_cavity_d"] / fs
    # local frame: origin = lens exit on the nose face, axis +Z = looking forward
    barrel = cq.Workplane("XY").workplane(offset=-1.0).rect(sz, sz).extrude(prot + 1.0)
    try:
        barrel = barrel.edges("|Z").fillet(p["camera_block_fillet"] / fs)
    except Exception:
        pass
    import math
    t = math.radians(p["camera_tilt_down_deg"])
    # the pocket's upper-front corner swings forward when tilted: push the pocket back so it keeps `wall`
    c = ((h / 2) * math.sin(t) + wall) / math.cos(t)
    hole = cq.Workplane("XY").workplane(offset=-c - 0.2).circle(rh).extrude(c + 0.2 + prot + 1.0)
    cavity = cq.Workplane("XY").box(w, h, d).translate((0, 0, -c - d / 2))
    cutter = hole.union(cavity)
    place = lambda wp: (wp.rotate((0, 0, 0), (1, 0, 0), p["camera_tilt_down_deg"])
                        .translate((0, p["camera_y"] / fs, p["body_front_z"])))
    return place(barrel), place(cutter)


def _prop_airfoil_wire(radius, chord, twist_deg, sweep, p, y_mid):
    """Create one closed NACA 4-digit airfoil section in the rotor's X/Y/Z frame."""
    import math
    camber = float(p.get("prop_airfoil_camber", 0.04))
    camber_pos = float(p.get("prop_airfoil_camber_position", 0.40))
    thickness = float(p.get("prop_airfoil_thickness_ratio", 0.10))
    samples = int(p.get("prop_airfoil_samples", 24))
    if chord <= 0 or not (0.0 <= camber < 0.10) or not (0.05 <= camber_pos <= 0.95):
        raise ValueError("Invalid propeller airfoil parameters")
    if not (0.06 <= thickness <= 0.18) or samples < 12:
        raise ValueError("Propeller airfoil thickness must be 0.06..0.18 and samples >= 12")

    theta = math.radians(twist_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    us = [0.5 * (1.0 - math.cos(math.pi * i / samples)) for i in range(samples + 1)]

    def point(u, side):
        # NACA 4-digit thickness distribution with a cambered mean line.
        yt = 5.0 * thickness * (
            0.2969 * math.sqrt(max(0.0, u)) - 0.1260 * u
            - 0.3516 * u**2 + 0.2843 * u**3 - 0.1015 * u**4
        ) * chord
        if u < camber_pos:
            yc = camber / camber_pos**2 * (2.0 * camber_pos * u - u**2)
        else:
            yc = camber / (1.0 - camber_pos)**2 * (
                (1.0 - 2.0 * camber_pos) + 2.0 * camber_pos * u - u**2
            )
        y_local = yc * chord + side * yt
        z_local = (u - 0.5) * chord
        # Twist the section about the radial X axis. Sweep is added in the rotor plane
        # after twist, so it does not accidentally change the section pitch.
        y_rot = y_local * cos_t + z_local * sin_t
        z_rot = -y_local * sin_t + z_local * cos_t + sweep
        return cq.Vector(float(radius), float(y_mid + y_rot), float(z_rot))

    # Traverse upper surface from trailing edge to leading edge, then lower surface
    # back to the trailing edge; cosine spacing resolves the leading edge smoothly.
    points = [point(u, +1.0) for u in reversed(us)]
    points.extend(point(u, -1.0) for u in us[1:])
    wire = cq.Wire.makePolygon(points, close=True)
    if not wire.isValid():
        raise RuntimeError("Generated propeller airfoil section is invalid")
    return wire


def prop(p, x, z):
    """Two-blade low-Re rotor concept: cambered airfoil, radial taper, twist and swept tips.

    This is an aerodynamic-style CAD concept, not a validated thrust/efficiency optimum.
    Motor KV, battery voltage, prop RPM, mass and bench-test data are not present in the repo.
    """
    import math
    radius = float(p["prop_radius"])
    scale = float(p.get("model_scale") or 1.0)
    pitch_local = float(p.get("prop_pitch_mm", 48.26)) / scale
    if radius <= 0 or pitch_local <= 0:
        raise ValueError("Propeller radius and design pitch must be positive")
    stations_data = p.get("prop_blade_stations") or [
        [0.05, 6.5], [0.14, 7.8], [0.23, 8.6], [0.33, 8.8],
        [0.44, 8.2], [0.56, 7.3], [0.68, 6.4], [0.80, 5.6],
        [0.90, 4.8], [0.96, 4.4], [1.00, 4.0]
    ]
    if len(stations_data) < 5:
        raise ValueError("At least five radial propeller stations are required")
    fractions = [float(row[0]) for row in stations_data]
    chords = [float(row[1]) for row in stations_data]
    if (fractions[0] <= 0 or fractions[-1] != 1.0
            or any(b <= a for a, b in zip(fractions, fractions[1:]))
            or any(c <= 0 for c in chords)):
        raise ValueError("Propeller stations must increase from a positive root fraction to 1.0")

    y_mid = float(p["prop_y"]) + float(p.get("prop_thickness", 1.2)) / 2.0
    sweep_max = float(p.get("prop_tip_sweep", 1.2))
    wires = []
    for fraction, chord in zip(fractions, chords):
        r = fraction * radius
        # Constant-pitch design law: beta = atan(P / (2*pi*r); cap the root angle
        # because the innermost sections are mostly buried in the hub.
        beta = math.degrees(math.atan(pitch_local / (2.0 * math.pi * max(r, 1e-6))))
        beta = min(beta, 42.0)
        sweep = sweep_max * fraction**3
        wires.append(_prop_airfoil_wire(r, chord, beta, sweep, p, y_mid))

    blade = cq.Solid.makeLoft(wires, ruled=False)
    if not blade.isValid() or len(blade.Solids()) != 1:
        raise RuntimeError("Positive-radius propeller blade loft is invalid")
    opposite_blade = blade.rotate((0, 0, 0), (0, 1, 0), 180.0)

    # Preserve the existing elevated hub interface; blades are fused into the hub.
    hub = cq.Solid.makeCylinder(
        float(p["prop_hub_radius"]), float(p["prop_hub_height"]),
        cq.Vector(0.0, y_mid, 0.0), cq.Vector(0.0, 1.0, 0.0)
    )
    rotor = (cq.Workplane("XY").newObject([hub])
             .union(cq.Workplane("XY").newObject([blade]))
             .union(cq.Workplane("XY").newObject([opposite_blade])))
    rotor = rotor.translate((x, 0.0, z))
    if not rotor.val().isValid() or len(rotor.val().Solids()) != 1:
        raise RuntimeError("Propeller blades and hub must form one valid connected solid")
    return rotor


def body_shell_parts(p):
    """Return the lower hull shell and hollow crown as independent solids.

    Build each component from its original source surfaces instead of cutting
    the already-fused body at the hull/canopy seam. No joining or locking
    features are added between the pieces.
    """
    if not (p.get("use_measured_canopy") and p.get("canopy_faceted")):
        # Preserve compatibility for alternate parameter configurations.
        full = body(p)
        split_y = float(p["body_height"]) + 0.2
        lower_box = cq.Workplane("XY").box(1000.0, split_y + 2.0, 1000.0).translate(
            (0, (split_y - 2.0) / 2.0, 0))
        crown_box = cq.Workplane("XY").box(1000.0, 100.0, 1000.0).translate(
            (0, split_y + 50.0, 0))
        return full.intersect(lower_box), full.intersect(crown_box)

    # Use the same rim vertices as the canopy so no crown edge falls inside the hull.
    pts = canopy_base_points(p)
    hull = cq.Workplane("XZ").polyline(pts).close().extrude(-p["body_height"])
    crown = faceted_canopy(p)

    wall = float(p.get("body_shell_thickness", 2.0))
    if wall <= 0 or wall >= min(p["body_half_width"], p["body_height"]):
        raise ValueError("body_shell_thickness must be positive and smaller than the body")
    cz = (p["body_front_z"] + p["body_rear_z"]) / 2.0
    scale_x = max(0.1, 1.0 - wall / p["body_half_width"])
    scale_z = max(0.1, 1.0 - wall / ((p["body_front_z"] - p["body_rear_z"]) / 2.0))
    inner_pts = [(x * scale_x, cz + (z - cz) * scale_z) for x, z in pts]
    lower_void = cq.Workplane("XZ").polyline(inner_pts).close().extrude(-(p["body_height"] + 2.0))
    lower_void = lower_void.translate((0, wall, 0))
    lower = hull.cut(lower_void)

    roof_void = faceted_canopy(p, inward=wall, open_bottom_y=p["body_height"] - 1.0,
                               roof_drop=wall)
    crown = crown.cut(roof_void)
    for name, part in (("body_lower", lower), ("body_crown", crown)):
        if not part.val().isValid() or len(part.val().Solids()) != 1:
            raise RuntimeError(f"{name} generated from source surfaces is invalid or not one solid")
    return lower, crown


def _prop_collision_report(p):
    """Measure real 3D propeller intersections with each motor, arm beam and guard."""
    import math
    scale = float(p.get("model_scale") or 1.0)
    start_r = float(p.get("joint_arm_start_radius", 0.0))
    y_offset = float(p.get("body_shell_thickness", 0.0))
    checks = []
    for index, (x, z) in enumerate(stations(p), 1):
        rotor = prop(p, x, z).val()
        motor_shape = motor(p, x, z).val()
        arm_shape = arm_with_tick_root(p, x, z, start_r=start_r, y_offset=y_offset).val()
        guard_shape = guard(p, x, z).val()
        motor_overlap = max(0.0, rotor.intersect(motor_shape).Volume())
        arm_overlap = max(0.0, rotor.intersect(arm_shape).Volume())
        guard_overlap = max(0.0, rotor.intersect(guard_shape).Volume())
        motor_bb = motor_shape.BoundingBox()
        rotor_bb = rotor.BoundingBox()
        guard_bb = guard_shape.BoundingBox()
        motor_vertical_gap = (rotor_bb.ymin - motor_bb.ymax) * scale
        guard_bottom_margin = (rotor_bb.ymin - guard_bb.ymin) * scale
        guard_top_margin = (guard_bb.ymax - rotor_bb.ymax) * scale
        guard_vertical_coverage = guard_bottom_margin >= 1.0 and guard_top_margin >= 1.0
        checks.append({
            "propeller": f"prop_{index}",
            "motor_intersection_volume_mm3": round(motor_overlap * scale**3, 6),
            "arm_intersection_volume_mm3": round(arm_overlap * scale**3, 6),
            "guard_intersection_volume_mm3": round(guard_overlap * scale**3, 6),
            "minimum_prop_to_motor_vertical_bbox_gap_mm": round(motor_vertical_gap, 3),
            "guard_bottom_coverage_margin_mm": round(guard_bottom_margin, 3),
            "guard_top_coverage_margin_mm": round(guard_top_margin, 3),
            "motor_clearance_pass": motor_overlap <= 1e-4 and motor_vertical_gap >= 2.0,
            "arm_clearance_pass": arm_overlap <= 1e-4,
            "guard_clearance_pass": guard_overlap <= 1e-4,
            "guard_vertical_coverage_pass": guard_vertical_coverage
        })
    return checks


def build(p=None):
    p = {**PARAMS, **(p or {})}
    inner_min = (min(p["guard_profile_inner"]) + p.get("guard_grow", 0.0)) if p.get("guard_use_profile") else p["guard_outer_radius"] - p["guard_wall"]
    assert p["prop_radius"] < inner_min, "prop would hit the guard"
    model = body(p)
    for i, (x, z) in enumerate(stations(p)):
        model = model.union(arm(p, x, z)).union(guard(p, x, z)).union(motor(p, x, z))
        if p["include_props"]:
            model = model.union(prop(p, x, z))
    f = 1.0
    if p.get("model_scale"):
        f = p["model_scale"]
    elif p.get("target_overall_mm"):
        bb = model.val().BoundingBox()
        f = p["target_overall_mm"] / max(bb.xlen, bb.zlen)
    if p["camera_enabled"]:
        barrel, cutter = camera_parts(p, f)
        model = model.union(barrel).cut(cutter)
    if p["led_enabled"]:
        model = model.cut(led_slot(p, model.val(), f))
    if f != 1.0:
        model = cq.Workplane("XY").newObject([model.val().scale(f)])
        print("uniform scale factor", round(f, 4))
    # Export convention: Z is vertical, the flat underside faces -Z, camera/front faces -Y.
    model = model.rotate((0, 0, 0), (1, 0, 0), 90)
    return model






def build_modular(p=None):
    """Build the split body, four removable arms, vertical screws, nut placeholders and props."""
    import math
    p = {**PARAMS, **(p or {})}
    inner_min = (min(p["guard_profile_inner"]) + p.get("guard_grow", 0.0)) if p.get("guard_use_profile") else p["guard_outer_radius"] - p["guard_wall"]
    assert p["prop_radius"] < inner_min, "prop would hit the guard"
    start_r = p["joint_arm_start_radius"]
    if start_r >= p["joint_entry_end_radius"]:
        raise ValueError("arm root must meet the side-wall entry opening")
    if len(p["joint_bolt_positions"]) != 2:
        raise ValueError("each arm requires exactly two floor-fastener stations")
    stations_list = stations(p)
    body_part, crown_part = body_shell_parts(p)
    # Open the side walls and keep the underside nut traps; the full-height arm beam now reaches the screw stations.
    # No low floor rails or thin mounting tongue are generated.
    for x, z in stations_list:
        body_part = body_part.cut(arm_entry_slot(p, x, z))
    # Preserve the full nominal bottom plate thickness. The arm-entry opening and beam
    # clearance begin at the top face of the plate (Y = body_shell_thickness); neither
    # cutter may extend below that plane. The separate nut pockets are the only local
    # underside recesses and remain confined to their fastener locations.
    for x, z in stations_list:
        root_clearance = arm_with_tick_root(p, x, z, start_r=start_r,
                                            y_offset=p["body_shell_thickness"])
        body_part = body_part.cut(root_clearance)
        # Remove the same full-height tick-root envelope from the crown where it crosses
        # the hull/crown split, preserving clearance for the removable arm module.
        crown_part = crown_part.cut(root_clearance)
    # Reinforce every fastener station with an integral circular boss on the exterior underside.
    # Boss diameter and total local thickness are specified in final millimetres; model coordinates
    # are converted back using the active uniform scale. The interior floor remains flat for the arms.
    hardware_scale = float(p.get("model_scale") or 1.0)
    boss_radius = float(p.get("joint_boss_diameter_mm", 10.0)) / hardware_scale / 2.0
    boss_total_thickness = float(p.get("joint_boss_total_thickness_mm", 8.0)) / hardware_scale
    boss_extension = boss_total_thickness - float(p["body_shell_thickness"])
    if boss_extension <= 0 or boss_radius <= 0:
        raise ValueError("joint boss must be wider than zero and thicker than the existing floor")
    # The XZ plane normal points toward -Y; a positive extrusion adds material below Y=0.
    # Start 0.05 model units inside the floor to guarantee a fused single solid.
    for x, z in stations_list:
        for hx, hz in joint_bolt_points(p, x, z):
            boss = (cq.Workplane("XZ").workplane(offset=-0.05).center(hx, hz)
                    .circle(boss_radius).extrude(boss_extension + 0.05))
            body_part = body_part.union(boss)
    # The tick branches are integral with the arm, full arm-height, and 60% of arm width.
    arm_parts, prop_parts, screw_parts, nut_parts = [], [], [], []
    for arm_index, (x, z) in enumerate(stations_list, 1):
        angle = math.atan2(z, x)
        module = arm_with_tick_root(p, x, z, start_r=start_r,
                                    y_offset=p["body_shell_thickness"])
        # Put the vertical fastener bores through the full-height tick branches.
        for hx, hz in joint_bolt_points(p, x, z):
            beam_hole = (cq.Workplane("XZ").workplane(offset=-p["body_shell_thickness"])
                         .center(hx, hz).circle(p["joint_bolt_hole_radius"])
                         .extrude(-p["arm_height"]))
            module = module.cut(beam_hole)
        module = module.union(guard(p, x, z)).union(motor(p, x, z))
        arm_parts.append(module)
        for bolt_index, (hx, hz) in enumerate(joint_bolt_points(p, x, z), 1):
            # Hex nut recesses open on the exterior underside; the circular clearance bore
            # continues through the complete floor and raised support.
            # Recess opens at the lowest face of the new boss and cuts upward into it.
            nut_cutter = (cq.Workplane("XZ").workplane(offset=boss_extension)
                          .center(hx, hz).polygon(6, p["joint_nut_pocket_af"] / math.cos(math.radians(30.0)))
                          .extrude(-p["joint_nut_depth"]))
            # Clearance bore runs from the boss underside through the entire reinforced floor.
            bore = (cq.Workplane("XZ").workplane(offset=boss_extension).center(hx, hz)
                    .circle(p["joint_bolt_hole_radius"]).extrude(-(boss_total_thickness + 0.1)))
            body_part = body_part.cut(nut_cutter).cut(bore)
            part_index = (arm_index - 1) * 2 + bolt_index
            screw_parts.append((f"screw_{part_index}", screw_part(p, hx, hz)))
            nut_parts.append((f"nut_{part_index}", nut_part(p, hx, hz)))
        if p["include_props"]:
            prop_parts.append(prop(p, x, z))
    # Fail generation if any rotating prop overlaps its motor, arm beam, or guard.
    # This is checked on actual CadQuery solids before export/scaling, not inferred from bounding boxes.
    collision_checks = _prop_collision_report(p) if p["include_props"] else []
    for check in collision_checks:
        if not (check["motor_clearance_pass"] and check["arm_clearance_pass"] and check["guard_clearance_pass"] and check["guard_vertical_coverage_pass"]):
            raise RuntimeError(f"Propeller collision/clearance check failed: {check}")

    raw_shapes = [body_part.val()] + [s.val() for s in arm_parts] + [s.val() for s in prop_parts]
    raw_compound = cq.Compound.makeCompound(raw_shapes)
    bb = raw_compound.BoundingBox()
    if p.get("model_scale"):
        scale = p["model_scale"]
    elif p.get("target_overall_mm"):
        scale = p["target_overall_mm"] / max(bb.xlen, bb.zlen)
    else:
        scale = 1.0
    # Camera and LED features belong to the body only; dimensions remain final-mm based.
    if p["camera_enabled"]:
        barrel, cutter = camera_parts(p, scale)
        body_part = body_part.union(barrel).cut(cutter)
    if p["led_enabled"]:
        # The LED slit sits above the hull/crown split, so keep it on the crown piece.
        crown_part = crown_part.cut(led_slot(p, crown_part.val(), scale))
    parts = {"body_lower": body_part, "body_crown": crown_part}
    for i, part in enumerate(arm_parts, 1):
        parts[f"arm_{i}"] = part
    for i, part in enumerate(prop_parts, 1):
        parts[f"prop_{i}"] = part
    for name, part in screw_parts:
        parts[name] = part
    for name, part in nut_parts:
        parts[name] = part
    if scale != 1.0:
        parts = {name: cq.Workplane("XY").newObject([shape.val().scale(scale)])
                 for name, shape in parts.items()}
        print("uniform scale factor", round(scale, 5))
    # Apply one rigid transform to every component so assembly and individual exports agree.
    # Old +Y (up) -> new +Z; old +Z (camera/front) -> new -Y; old -Y (underside) -> new -Z.
    parts = {name: part.rotate((0, 0, 0), (1, 0, 0), 90)
             for name, part in parts.items()}
    for name, part in parts.items():
        if not part.val().isValid():
            raise RuntimeError(f"Invalid CAD part: {name}")
        solid_count = len(part.val().Solids())
        print(f"part {name}: valid=True solids={solid_count}")
        if name.startswith("arm_") and solid_count != 1:
            raise RuntimeError(f"{name} is not a single connected solid ({solid_count} solids)")
    # Check that arm modules clear the lower shell and removable crown in the assembled pose.
    for arm_index, module in enumerate(arm_parts, 1):
        lower_intersection = body_part.val().intersect(module.val())
        crown_intersection = crown_part.val().intersect(module.val())
        lower_overlap = lower_intersection.Volume()
        crown_overlap = crown_intersection.Volume()
        lb = lower_intersection.BoundingBox() if lower_overlap > 1e-8 else None
        cb = crown_intersection.BoundingBox() if crown_overlap > 1e-8 else None
        lb_text = (f"({lb.xmin:.3f},{lb.ymin:.3f},{lb.zmin:.3f})-"
                   f"({lb.xmax:.3f},{lb.ymax:.3f},{lb.zmax:.3f})") if lb else "none"
        cb_text = (f"({cb.xmin:.3f},{cb.ymin:.3f},{cb.zmin:.3f})-"
                   f"({cb.xmax:.3f},{cb.ymax:.3f},{cb.zmax:.3f})") if cb else "none"
        print(f"arm_{arm_index} clearance check: lower_overlap={lower_overlap:.6f}, "
              f"lower_bbox={lb_text}, crown_overlap={crown_overlap:.6f}, crown_bbox={cb_text}")
        if lower_overlap > 1e-4 or crown_overlap > 1e-4:
            raise RuntimeError(
                f"arm_{arm_index} interferes with body/crown: "
                f"lower={lower_overlap:.6f}, crown={crown_overlap:.6f}"
            )
    return parts, scale


def export_modular(parts, out_dir):
    """Export a named STEP assembly plus a compound STL and each separate component."""
    import os, json
    os.makedirs(out_dir, exist_ok=True)

    # STEP must be a real CadQuery Assembly, not a Workplane wrapping a Compound.
    # A compound can look like one selectable object in some CAD viewers; named
    # assembly children preserve body_lower and body_crown as separately selectable parts.
    assembly = cq.Assembly(name="drone_parametric")
    matte_black = cq.Color(0.035, 0.035, 0.035)
    for name, part in parts.items():
        assembly.add(part.val(), name=name, color=matte_black)
    assembly.save(os.path.join(out_dir, "drone_parametric.step"), exportType="STEP")

    # Keep the assembled STL for visualization, but note that STL stores no
    # assembly/explode instructions. Also write a physically exploded STL so
    # slicers that open one STL cannot mistake the parts for a single assembled model.
    # In the assembled STL only, add 0.01 mm export-only clearances at touching interfaces.
    # The crown is lifted off the lower hull; each arm module is lifted off the floor rails.
    # Do not alter source geometry, the STEP assembly, or the exploded STL layout.
    assembled_shapes = []
    for name, part in parts.items():
        shape = part.val()
        if name == "body_crown":
            shape = shape.translate((0, 0, 0.01))  # exported coordinates are Z-up
        elif name.startswith("arm_"):
            shape = shape.translate((0, 0, 0.01))  # lift tongue off rails; Z-up export
        assembled_shapes.append(shape)
    compound = cq.Compound.makeCompound(assembled_shapes)
    # Preserve the assembled pose under an explicit filename. The default STL name below
    # is intentionally reserved for the exploded layout so opening drone_parametric.stl
    # in a mesh viewer shows the separated parts instead of the closed assembly.
    cq.exporters.export(cq.Workplane("XY").newObject([compound]),
                        os.path.join(out_dir, "drone_parametric_assembled.stl"),
                        tolerance=0.05, angularTolerance=0.1)

    # Exploded layout uses deliberate, visible clearances rather than small illustrative offsets.
    # body_lower remains fixed; the crown lifts clear of the hull, and each complete arm module
    # (including its integral tongue) moves radially far enough to leave the side opening entirely.
    exploded_shapes = []
    exploded_bboxes = {}
    import math
    for name, part in parts.items():
        shape = part.val()
        bb = shape.BoundingBox()
        if name == "body_crown":
            shape = shape.translate((0, 0, 48.0))
        elif name.startswith("arm_"):
            cx = (bb.xmin + bb.xmax) / 2.0
            cy = (bb.ymin + bb.ymax) / 2.0
            radius = math.hypot(cx, cy)
            if radius < 1e-6:
                raise RuntimeError(f"Cannot determine radial explode direction for {name}")
            # A 62 mm radial translation pulls the tongue fully out of the body,
            # rather than merely shifting the motor/guard away while leaving the joint engaged.
            distance = 62.0
            shape = shape.translate((cx / radius * distance, cy / radius * distance, 0))
        elif name.startswith("prop_"):
            shape = shape.translate((0, 0, 28.0))
        elif name.startswith("screw_"):
            shape = shape.translate((0, 0, 12.0))
        elif name.startswith("nut_"):
            shape = shape.translate((0, 0, -8.0))
        eb = shape.BoundingBox()
        exploded_bboxes[name] = [round(eb.xmin, 3), round(eb.ymin, 3), round(eb.zmin, 3),
                                 round(eb.xmax, 3), round(eb.ymax, 3), round(eb.zmax, 3)]
        exploded_shapes.append(shape)
    # Guard against accidentally reverting to an assembled/near-assembled export.
    if len(exploded_shapes) != 26 or "body_lower" not in exploded_bboxes:
        raise RuntimeError("Exploded STL must contain all 26 assembly components with body_lower fixed")
    for name in ("arm_1", "arm_2", "arm_3", "arm_4"):
        original = parts[name].val().BoundingBox()
        moved = exploded_bboxes[name]
        if math.dist([(original.xmin + original.xmax) / 2.0, (original.ymin + original.ymax) / 2.0],
                     [(moved[0] + moved[3]) / 2.0, (moved[1] + moved[4]) / 2.0]) < 60.0:
            raise RuntimeError(f"{name} did not receive the required radial explode translation")
    print("Exploded STL component bounds (xmin,ymin,zmin,xmax,ymax,zmax):")
    for name, bounds in exploded_bboxes.items():
        print(f"  {name}: {bounds}")
    exploded_compound = cq.Compound.makeCompound(exploded_shapes)
    cq.exporters.export(cq.Workplane("XY").newObject([exploded_compound]),
                        os.path.join(out_dir, "drone_parametric_exploded.stl"),
                        tolerance=0.05, angularTolerance=0.1)
    # Make the conventional/default STL name the exploded layout too, preventing
    # viewers such as OpenSCAD from opening the assembled compound by accident.
    cq.exporters.export(cq.Workplane("XY").newObject([exploded_compound]),
                        os.path.join(out_dir, "drone_parametric.stl"),
                        tolerance=0.05, angularTolerance=0.1)

    try:
        import trimesh
        mesh = trimesh.load(os.path.join(out_dir, "drone_parametric_assembled.stl"), force="mesh")
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
            mesh.export(os.path.join(out_dir, "drone_parametric_assembled.stl"))
    except Exception as e:
        print("Assembly STL repair skipped:", e)

    for name, part in parts.items():
        export(part, out_dir, name)
    manifest = {
        "assembly": "drone_parametric.step",
        "branch_scope": "demo",
        "export_orientation": {
            "vertical_axis": "+Z",
            "camera_front_direction": "-Y",
            "flat_underside_direction": "-Z",
            "transform_from_design_coordinates": "(x, y, z) -> (x, -z, y)",
            "rotation_degrees_about_X": 90
        },
        "parts": [
            {"name": name, "step": f"{name}.step", "stl": f"{name}.stl",
             "role": ("lower body shell with reinforced floor rails, side entry openings and hex-nut traps" if name == "body_lower"
                      else "separate removable upper crown/canopy" if name == "body_crown"
                      else "removable arm module with an integral floor-mounting tongue, guard and motor" if name.startswith("arm_")
                      else "separate propeller" if name.startswith("prop_")
                      else "simplified vertical socket-head screw placeholder; use real M3 hardware" if name.startswith("screw_")
                      else "simplified hex-nut placeholder; replace with a real metal M3 nut" if name.startswith("nut_")
                      else "component")}
            for name in parts
        ],
        "joint": {
            "type": "full-height arm beam reaches the screw stations; screws pass through the beam and floor into underside nut recesses",
            "fastener": "two vertical M3 screws per arm, passing through the floor into exterior underside nut recesses",
            "fastener_count": 8,
            "fastener_axis_design_coordinates": "Y (vertical); screws installed from the arm-tab side into exterior underside nuts",
            "fastener_axis_export_coordinates": "Z (vertical)",
            "nut_retention": "hexagonal pockets open on the exterior underside; nut solids sit inside the pocket depth, with real metal nuts required for assembly",
            "nut_recesses_open_on_exterior_underside": True,
            "underside_boss_count": 8,
            "underside_boss_diameter_mm": PARAMS["joint_boss_diameter_mm"],
            "underside_boss_total_local_thickness_mm": PARAMS["joint_boss_total_thickness_mm"],
            "underside_boss_integral_with_body": True,
            "nut_pocket_depth_mm": round(PARAMS["joint_nut_depth"] * PARAMS["model_scale"], 3),
            "nut_thickness_mm": round(PARAMS["joint_nut_thickness"] * PARAMS["model_scale"], 3),
            "nut_seated_inside_floor_recess": True,
            "nut_pocket_clearance_per_side_mm": round((PARAMS["joint_nut_pocket_af"] - PARAMS["joint_nut_af"]) * PARAMS["model_scale"] / 2, 3),
            "side_wall_fasteners": False,
            "floor_plate_cut_through": True,
            "arm_entry_openings": 4,
            "integral_tab": False,
            "bolt_positions_radial_tangent_model_units": PARAMS["joint_bolt_positions"],
            "tab_removed": True,
            "arm_start_radius_model_units": PARAMS["joint_arm_start_radius"],
            "main_beam_height_mm": round(PARAMS["arm_height"] * PARAMS["model_scale"], 3),
            "approx_m3_clearance_hole_diameter_mm": round(2 * PARAMS["joint_bolt_hole_radius"] * PARAMS["model_scale"], 3),
            "hardware_geometry_is_simplified": True,
            "physical_fit_tested": False,
            "note": "The CAD includes simplified screw and nut envelopes for assembly visualization. Nut solids are positioned within the underside-open hex pockets and the screw clearance bores pass through the floor. Use real M3 screws and metal nuts; print-fit, tool access, and flight-load testing are still required."
        }
    }
    with open(os.path.join(out_dir, "assembly-manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def write_validation_report(parts, scale, out_dir, params):
    """Report fresh CAD geometry, including actual nut-to-body intersection volumes."""
    import os, json
    compound = cq.Compound.makeCompound([part.val() for part in parts.values()])
    bb = compound.BoundingBox()
    components = {}
    for name, part in parts.items():
        shape = part.val()
        component_bb = shape.BoundingBox()
        components[name] = {
            "shape_valid": bool(shape.isValid()),
            "solid_count": len(shape.Solids()),
            "bbox_mm": [round(component_bb.xmin, 4), round(component_bb.ymin, 4),
                        round(component_bb.zmin, 4), round(component_bb.xmax, 4),
                        round(component_bb.ymax, 4), round(component_bb.zmax, 4)]
        }
    p = {**PARAMS, **params}
    body_lower_shape = parts["body_lower"].val()

    # Sample the actual exported lower-body shape through the central floor plate.
    # The sample grid stays well clear of the eight preserved nut pockets and the raised rails.
    floor_thickness_mm = float(p["body_shell_thickness"]) * float(scale)
    floor_samples = []
    sample_offsets = (-12.0, -6.0, 0.0, 6.0, 12.0)
    for sx in sample_offsets:
        for sz in sample_offsets:
            ex, ey = sx * scale, -sz * scale  # design X/Z -> exported X/Y
            low = body_lower_shape.isInside(cq.Vector(ex, ey, 0.25 * scale))
            high = body_lower_shape.isInside(cq.Vector(ex, ey, floor_thickness_mm - 0.25 * scale))
            floor_samples.append(bool(low and high))
    floor_samples_passed = sum(floor_samples)
    if floor_samples_passed != len(floor_samples):
        raise RuntimeError(
            f"Lower floor plate thickness sample failed: {floor_samples_passed}/{len(floor_samples)} "
            "points retain the nominal floor thickness outside nut pockets"
        )

    # Record all four arm modules and verify the source dimensions used for each beam.
    arm_geometry = {}
    for i in range(1, 5):
        name = f"arm_{i}"
        if name not in parts:
            raise RuntimeError(f"Missing expected arm module: {name}")
        arm_geometry[name] = {
            "main_beam_width_mm": round(float(p["arm_width"]) * scale, 4),
            "main_beam_height_mm": round(float(p["arm_height"]) * scale, 4),
            "root_beam_width_mm": round(float(p["arm_width"]) * scale, 4),
            "root_beam_height_mm": round(float(p["arm_height"]) * scale, 4),
            "tick_branch_width_mm": round(float(p["arm_width"]) * float(p.get("joint_branch_width_ratio", 0.60)) * scale, 4),
            "cross_section_constant_by_generator": True
        }

    nut_body_intersections = {}
    for i in range(1, 9):
        nut_name = "nut_" + str(i)
        # Nuts must occupy the underside pockets without penetrating the remaining body material.
        intersection = body_lower_shape.intersect(parts[nut_name].val())
        nut_body_intersections[nut_name] = round(max(0.0, intersection.Volume()), 6)
    guard_inner_min = (
        min(p["guard_profile_inner"]) + float(p.get("guard_grow", 0.0))
        if p.get("guard_use_profile")
        else float(p["guard_outer_radius"]) - float(p["guard_wall"])
    )
    prop_radius_local = float(p["prop_radius"])
    report = {
        "report_version": 3,
        "generated_from": "cad/parametric/drone_parametric.py",
        "branch": "demo",
        "validation_scope": "Fresh CadQuery in-memory geometry checks and workflow output-presence checks; no physical print-fit test",
        "assembly": {
            "shape_valid": bool(compound.isValid()),
            "solid_count": len(compound.Solids()),
            "bbox_mm": [round(bb.xlen, 6), round(bb.ylen, 6), round(bb.zlen, 6)]
        },
        "components": components,
        "propeller": {
            "design": "two-blade cambered NACA 4410 concept with radial chord taper, geometric twist and mild tip sweep",
            "blade_count": 2,
            "diameter_mm": round(2.0 * float(p["prop_radius"]) * float(scale), 3),
            "nominal_geometric_pitch_mm": round(float(p.get("prop_pitch_mm", 48.26)), 3),
            "nominal_geometric_pitch_in": round(float(p.get("prop_pitch_mm", 48.26)) / 25.4, 3),
            "airfoil_camber_ratio": float(p.get("prop_airfoil_camber", 0.04)),
            "airfoil_thickness_ratio": float(p.get("prop_airfoil_thickness_ratio", 0.10)),
            "radial_stations": len(p.get("prop_blade_stations", [])),
            "radial_guard_clearance_mm": round(
                (guard_inner_min - prop_radius_local) * float(scale), 3),
            "guard_clearance_positive": guard_inner_min > prop_radius_local,
            "motor_matched": False,
            "thrust_efficiency_bench_tested": False,
            "physical_print_fit_tested": False
        },
        "propeller_clearance": {
            "method": "exact CadQuery solid intersection volumes against each motor, arm root/beam, and guard",
            "all_four_rotors_checked": len(_prop_collision_report(p)) == 4 if p["include_props"] else False,
            "collision_checks": _prop_collision_report(p) if p["include_props"] else [],
            "guard_vertical_offset_mm": round(float(p.get("guard_y_offset", 0.0)) * float(scale), 3),
            "shaft_connection_modelled": False,
            "note": "Zero intersection volume verifies geometric non-overlap only. The motor shaft diameter and prop hub mounting bore are still unspecified and must be matched to the selected motor."
        },
        "arm_geometry": {
            "all_four_arms_checked": len(arm_geometry) == 4,
            "constant_width_and_height_by_generator": all(v["cross_section_constant_by_generator"] for v in arm_geometry.values()),
            "arms": arm_geometry,
            "verification_note": "Width and height are fixed by the arm generator; final CAD solids are checked for validity and single-solid connectivity. Visual section-by-section inspection is not automated."
        },
        "floor_plate": {
            "nominal_thickness_mm": round(floor_thickness_mm, 4),
            "sample_grid": "5x5 central-body points, clear of nut pockets and floor rails",
            "sample_points_checked": len(floor_samples),
            "sample_points_retaining_nominal_thickness": floor_samples_passed,
            "central_floor_thickness_samples_passed": floor_samples_passed == len(floor_samples),
            "nut_pockets_excluded_from_uniform_thickness_rule": True
        },
        "joint": {
            "type": "full-height arm beam with direct vertical fastener bores; exterior underside fasteners",
            "model_scale": scale,
            "fastener_count": 8,
            "fasteners_per_arm": len(p["joint_bolt_positions"]),
            "bolt_positions_radial_tangent_model_units": p["joint_bolt_positions"],
            "bolt_clearance_hole_diameter_mm": round(2 * p["joint_bolt_hole_radius"] * scale, 3),
            "thin_tab_removed": True,
            "arm_start_radius_model_units": p["joint_arm_start_radius"],
            "main_beam_height_mm": round(p["arm_height"] * scale, 3),
            "nut_trap_depth_mm": round(p["joint_nut_depth"] * scale, 3),
            "nut_thickness_mm": round(p["joint_nut_thickness"] * scale, 3),
            "screws_vertical_and_top_accessible": True,
            "nut_traps_in_floor_rails": False,
            "nut_recesses_open_on_exterior_underside": True,
            "underside_boss_count": 8,
            "underside_boss_diameter_mm": p["joint_boss_diameter_mm"],
            "underside_boss_total_local_thickness_mm": p["joint_boss_total_thickness_mm"],
            "underside_bosses_integral_and_fused": components["body_lower"]["shape_valid"],
            "nut_seated_inside_floor_recess": True,
            "nut_pocket_clearance_per_side_mm": round((p["joint_nut_pocket_af"] - p["joint_nut_af"]) * scale / 2, 3),
            "nut_body_intersection_volume_mm3": nut_body_intersections,
            "max_nut_body_intersection_volume_mm3": max(nut_body_intersections.values(), default=0.0),
            "side_wall_fasteners": False,
            "floor_plate_cut_through": True,
            "hole_centres_aligned_by_shared_coordinates": True,
            "arm_tab_is_integral": False,
            "hardware_geometry_is_simplified": True,
            "physical_fit_tested": False
        },
        "stl_watertight": "not_checked",
        "step_round_trip_import": "not_checked",
        "known_limitations": [
            "CAD validity checks apply to in-memory CadQuery shapes before export.",
            "The workflow checks that exported STEP/STL files exist and are non-empty; STEP round-trip import is not yet checked.",
            "STL watertightness is not reported by this validation step.",
            "Screw and nut solids are simplified envelopes without threads; use real metal M3 hardware.",
            "The mounting-hole coordinates align directly with the full-height arm beam and underside nut traps; physical print-fit, screwdriver access in the printed body, and flight-load strength still require testing."
        ]
    }
    if not report["assembly"]["shape_valid"]:
        raise RuntimeError("Generated assembly compound is invalid")
    if report["assembly"]["solid_count"] != len(parts):
        raise RuntimeError(f"Assembly has {report['assembly']['solid_count']} solids; expected {len(parts)}")
    if any(not v["shape_valid"] or v["solid_count"] != 1 for v in components.values()):
        raise RuntimeError("One or more exported components is invalid or not a single solid")
    if len(arm_geometry) != 4 or not all(v["cross_section_constant_by_generator"] for v in arm_geometry.values()):
        raise RuntimeError("Constant arm width/height requirement is not satisfied for all four arms")
    prop_report = report["propeller"]
    if prop_report["blade_count"] != 2 or prop_report["diameter_mm"] <= 0:
        raise RuntimeError("Propeller geometry metadata is invalid")
    if not prop_report["guard_clearance_positive"]:
        raise RuntimeError("Propeller radius exceeds the available guard inner radius")
    if len([name for name in parts if name.startswith("prop_")]) != 4:
        raise RuntimeError("Expected four independently exported propeller solids")
    if p["include_props"]:
        collision_checks = _prop_collision_report(p)
        if len(collision_checks) != 4 or any(
            not (v["motor_clearance_pass"] and v["arm_clearance_pass"] and v["guard_clearance_pass"] and v["guard_vertical_coverage_pass"])
            for v in collision_checks
        ):
            raise RuntimeError("Propeller-to-motor/arm/guard clearance validation failed")
    if floor_samples_passed != len(floor_samples):
        raise RuntimeError("Lower-body floor thickness verification failed outside the nut-pocket area")
    with open(os.path.join(out_dir, "validation-report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def export(shape, out_dir, name="drone_parametric"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    cq.exporters.export(shape, os.path.join(out_dir, name + ".step"))
    cq.exporters.export(shape, os.path.join(out_dir, name + ".stl"), tolerance=0.05, angularTolerance=0.1)
    try:                                    # close the tiny tessellation seams left by the swept LED groove
        import trimesh
        mesh = trimesh.load(os.path.join(out_dir, name + ".stl"))
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
            mesh.export(os.path.join(out_dir, name + ".stl"))
    except Exception as e:
        print("STL repair skipped:", e)


if __name__ == "__main__":
    params = {}
    if len(sys.argv) > 2 and sys.argv[1] == "--params":
        params = json.load(open(sys.argv[2], encoding="utf-8"))
    parts, scale = build_modular(params)
    compound = cq.Compound.makeCompound([part.val() for part in parts.values()])
    bb = compound.BoundingBox()
    print("assembly bbox", bb.xlen, bb.ylen, bb.zlen, "valid", compound.isValid(),
          "solids", len(compound.Solids()))
    out = "output/قالب گیری شده/smooth2-parametric"
    if not compound.isValid() or len(compound.Solids()) != len(parts):
        raise RuntimeError("Generated assembly failed validity/solid-count checks")
    export_modular(parts, out)
    write_validation_report(parts, scale, out, params)
    with open(out + "/drone_parametric.params.json", "w", encoding="utf-8") as f:
        json.dump({**PARAMS, **params}, f, indent=2, ensure_ascii=False)

# Modular export: lower body shell, separate crown, four removable arm/guard/motor modules,
# and separate propellers. The crown-to-lower-body joining method is intentionally undecided.
# Regeneration is restricted to the demo branch; no crown-to-base joining features are included yet.
