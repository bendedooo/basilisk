"""Step-by-step checks for lif_geolocator.py.   Run:  python test_lif_geolocator.py

Each step prints PASS, FAIL (with what was expected) or TODO (not written yet).
The tests check physical facts, not formulas, so they do not give the answers away.
"""
import importlib.util
import os
import traceback

import numpy as np

import lif_geolocator as G

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "rbk", os.path.join(HERE, "..", "..", "..", "src", "utilities", "RigidBodyKinematics.py"))
rbk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rbk)            # Basilisk's own attitude maths, used as the reference

R = G.R_MOON_KM
CAM = G.camera_from_config(np.radians(10.0), [512, 512])     # your Basilisk camera
rng = np.random.default_rng(1)


def close(a, b, tol):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol)


def rot(axis, ang):
    c, s = np.cos(ang), np.sin(ang)
    if axis == 1:
        return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    if axis == 2:
        return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def pointing_attitude(r_N):
    """Camera pointed at the Moon centre, built like scenario_OpNavAttOD (incl. its 180 deg flip)."""
    z = -r_N / np.linalg.norm(r_N)
    x = np.cross([0, 0, 1.0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.diag([-1.0, -1.0, 1.0]) @ np.vstack((x, y, z))       # [CN]


def scene(dist_km=43000.0):
    """Random but realistic geometry: spacecraft, attitude, Moon orientation, and a flash."""
    r_N = rng.normal(size=3); r_N *= dist_km * 1000.0 / np.linalg.norm(r_N)   # [m]
    CN = pointing_attitude(r_N)
    PN = rot(3, rng.uniform(0, 6.28)) @ rot(1, rng.uniform(-0.5, 0.5))       # J2000 -> Moon-fixed
    r_P = PN @ r_N / 1000.0                                                  # [km]
    while True:                                                              # a visible flash site
        X = rng.normal(size=3); X *= R / np.linalg.norm(X)
        if (X / R) @ (r_P - X) > 0.4 * np.linalg.norm(r_P - X):
            break
    d_C = CN @ PN.T @ (X - r_P)
    u, v = CAM.f * d_C[0] / d_C[2] + CAM.cx, CAM.f * d_C[1] / d_C[2] + CAM.cy
    return dict(r_N=r_N, sigma_BN=rbk.C2MRP(CN), PN=PN, r_P=r_P, X=X, u=u, v=v)


# ----------------------------------------------------------------------------- steps
def step1():
    d0 = G.pixel_to_ray(CAM.cx, CAM.cy, CAM)
    assert close(d0, [0, 0, 1], 1e-12), f"centre pixel should give [0,0,1], got {d0}"
    d = G.pixel_to_ray(CAM.cx + 50, CAM.cy, CAM)
    assert close(np.linalg.norm(d), 1, 1e-12), "direction must have length 1"
    assert close(d, [0.017085, 0, 0.999854], 1e-6), f"50 px right: expected [0.017085, 0, 0.999854], got {d}"
    d = G.pixel_to_ray(CAM.cx, CAM.cy - 50, CAM)
    assert d[1] < 0, "a pixel ABOVE the centre should have negative y (y points down)"


def step2():
    r = np.array([0, 0, 43000.0])
    d = G.pixel_to_ray(CAM.cx + 50, CAM.cy, CAM) * np.array([1, 1, -1])   # look back at the Moon
    X = G.ray_sphere(r, d)
    assert X is not None, "this line hits the Moon"
    assert close(np.linalg.norm(X), R, 1e-6), "hit point must be on the sphere"
    assert close(X, [707.7, 0, 1586.4], 1.0), f"expected about (707.7, 0, 1586.4) km, got {X}"
    assert G.ray_sphere(r, np.array([1.0, 0, 0])) is None, "a line looking sideways misses"
    assert G.ray_sphere(r, np.array([0, 0, 1.0])) is None, "a line looking AWAY from the Moon misses"


def step3():
    for X, want in [((R, 0, 0), (0, 0, 0)), ((0, R, 0), (0, 90, 0)), ((0, 0, R), (90, None, 0)),
                    ((0, -R, 0), (0, -90, 0)), ((R + 1.5, 0, 0), (0, 0, 1.5))]:
        lat, lon, h = G.xyz_to_lat_lon_h(np.array(X, float))
        assert close(lat, want[0], 1e-9) and close(h, want[2], 1e-9), f"{X}: got lat {lat}, h {h}"
        if want[1] is not None:
            assert close(lon, want[1], 1e-9), f"{X}: expected lon {want[1]}, got {lon}"
    lat, lon, h = G.xyz_to_lat_lon_h(np.array([-R, 1e-9, 0.0]))
    assert close(abs(lon), 180, 1e-6), "the far side (-X) is longitude +/-180"


def step4():
    for _ in range(20):
        s = rng.normal(size=3) * 0.4
        assert close(G.mrp_to_dcm(s), rbk.MRP2C(s), 1e-12), "does not match Basilisk's rbk.MRP2C"
    C = G.mrp_to_dcm([0.3, -0.2, 0.1])
    assert close(C @ C.T, np.eye(3), 1e-12), "a DCM must be orthonormal"


def step5():
    s = scene()
    PC = G.camera_to_moon(s["sigma_BN"], [0, 0, 0], s["PN"])
    assert close(PC @ PC.T, np.eye(3), 1e-12), "[PC] must be a rotation"
    to_moon = -s["r_P"] / np.linalg.norm(s["r_P"])
    assert close(PC @ [0, 0, 1], to_moon, 1e-9), "the boresight should point at the Moon centre"
    # camera mounted 90 deg about body y; body turned so the camera still points at the Moon
    CB = rot(2, np.pi / 2)
    BN = CB.T @ pointing_attitude(s["r_N"])
    PC = G.camera_to_moon(rbk.C2MRP(BN), rbk.C2MRP(CB), s["PN"])
    assert close(PC @ [0, 0, 1], to_moon, 1e-9), "with a mounting rotation, sigma_CB is not applied correctly"


def step6():
    for _ in range(10):
        s = scene()
        out = G.method_a(s["u"], s["v"], CAM, s["r_N"], s["sigma_BN"], [0, 0, 0], s["PN"])
        assert out is not None, "method A missed a visible flash"
        lat, lon, h, X = out
        assert close(X, s["X"], 1e-6), f"hit point off by {np.linalg.norm(X - s['X']):.3g} km"
        assert close(h, 0, 1e-6), "h should be 0 on the sphere"


def step7():
    for dist in (43000.0, 62000.0):
        r_px = CAM.f * np.tan(np.arcsin(R / dist))
        r_C, rho = G.moon_from_disk(CAM.cx, CAM.cy, r_px, CAM)
        assert close(rho, dist, 1e-6), f"range: expected {dist}, got {rho}"
        assert close(r_C, [0, 0, -dist], 1e-6), f"Moon straight ahead -> spacecraft at (0,0,-rho), got {r_C}"
    assert close(CAM.f * np.tan(np.arcsin(R / 43000.0)), 118.32, 0.01)


def step8():
    for _ in range(10):
        s = scene()
        rho = np.linalg.norm(s["r_P"])
        disk = (CAM.cx, CAM.cy, CAM.f * np.tan(np.arcsin(R / rho)))     # Moon at the image centre
        out = G.method_b(s["u"], s["v"], CAM, disk, s["sigma_BN"], [0, 0, 0], s["PN"])
        assert out is not None, "method B missed a visible flash"
        assert close(out["X_P"], s["X"], 1e-6), f"hit point off by {np.linalg.norm(out['X_P'] - s['X']):.3g} km"
        assert close(out["rho_km"], rho, 1e-6), "range from the disk is wrong"
        assert abs(out["lat_rel"]) <= 90 and abs(out["lon_rel"]) <= 90, "relative coords: visible side is within +/-90"
    # relative frame: a flash straight up from the disk centre has lon_rel 0 and lat_rel > 0
    out = G.method_b(CAM.cx, CAM.cy - 60, CAM, (CAM.cx, CAM.cy, 118.32), [0, 0, 0], [0, 0, 0], np.eye(3))
    assert close(out["lon_rel"], 0, 1e-9) and out["lat_rel"] > 0, "image-up should be relative north"
    out = G.method_b(CAM.cx + 60, CAM.cy, CAM, (CAM.cx, CAM.cy, 118.32), [0, 0, 0], [0, 0, 0], np.eye(3))
    assert close(out["lat_rel"], 0, 1e-9) and out["lon_rel"] > 0, "image-right should be relative east"


if __name__ == "__main__":
    names = ["pixel_to_ray", "ray_sphere", "xyz_to_lat_lon_h", "mrp_to_dcm",
             "camera_to_moon", "method_a", "moon_from_disk", "method_b"]
    for i, (name, fn) in enumerate(zip(names, [step1, step2, step3, step4, step5, step6, step7, step8]), 1):
        try:
            fn()
            print(f"Step {i} {name:18s} PASS")
        except NotImplementedError:
            print(f"Step {i} {name:18s} TODO")
        except AssertionError as e:
            print(f"Step {i} {name:18s} FAIL  {e}")
        except Exception:
            print(f"Step {i} {name:18s} ERROR")
            traceback.print_exc()
