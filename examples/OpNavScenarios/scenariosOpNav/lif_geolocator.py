#
#  LIF geolocator: selenographic latitude / longitude / height of detected lunar
#  impact flashes, computed live while the Basilisk simulation runs.
#
#  Method B (the running algorithm, what the spacecraft would fly):
#      Moon disk in the image  ->  spacecraft position relative to the Moon
#      flash pixel             ->  line of sight
#      intersect with the Moon ->  attitude rotates the hit point into Moon axes
#  Method A (cross-check, simulation only):
#      the same flash pixel cast from the TRUE spacecraft position with the TRUE attitude.
#      In simulation this is the reference answer; B - A is B's error.
#
#  Part 1 is the maths (plain numpy, testable without Basilisk).
#  Part 2 is the Basilisk module that reads the messages every camera frame and calls Part 1.
#
#  Units: km inside the maths, m in Basilisk messages, pixels for image coordinates,
#  degrees for latitude / longitude.
#  Moon-fixed frame: IAU_MOON, the frame of Basilisk's J20002Pfix (pck00010.tpc).
#  Latitude is planetocentric, longitude east-positive in [-180, 180].
#
#  Height h: both methods intersect a sphere of radius R_MOON_KM, so h = |X| - R = 0
#  by construction. A real terrain height needs a lunar elevation model (LOLA); the
#  hook for it is `dem_height_km` in ray_sphere_dem() below.
#
import csv
from collections import namedtuple

import numpy as np

R_MOON_KM = 1737.4  # IAU mean lunar radius [km]

# Pinhole camera: focal length f [px] and principal point (cx, cy) [px]
Camera = namedtuple("Camera", ["f", "cx", "cy"])


def camera_from_config(field_of_view, resolution):
    r"""Pinhole camera matching Basilisk's pixelLineConverter.

    Basilisk's fieldOfView is edge to edge along the image y-axis, so half the image
    height subtends half the field of view:

        f = (H / 2) / tan(FOV / 2)                      [px]

    Pixel centres sit at integer indices 0 .. W-1, so the exact image centre is

        c_x = W / 2 - 0.5 ,   c_y = H / 2 - 0.5         [px]

    For the 10 deg, 512 x 512 camera: f = 2926.08 px, c_x = c_y = 255.5 px.

    :param field_of_view: [rad]
    :param resolution: [width, height] [px]
    """
    w, h = int(resolution[0]), int(resolution[1])
    f = (h / 2.0) / np.tan(field_of_view / 2.0)
    return Camera(f=f, cx=w / 2.0 - 0.5, cy=h / 2.0 - 0.5)


# =============================================================================
# PART 1 - THE MATHS
# =============================================================================

def pixel_to_ray(u, v, cam):
    r"""STEP 1. Pixel -> unit direction in camera coordinates.

    Pinhole model: light from the flash passes through the pinhole and lands on the
    image plane at distance f. The pixel offset from the principal point divided by f
    is the tangent of the angle off the boresight (similar triangles):

        x = (u - c_x) / f = tan(theta_x)
        y = (v - c_y) / f = tan(theta_y)

    "For every 1 unit forward, go x right and y down":

        d_C = [x, y, 1] / sqrt(x^2 + y^2 + 1)

    Camera axes (Basilisk convention): +z boresight, +x right (columns), +y down (rows).

    :param u, v: pixel coordinates (column, row) [px], may be sub-pixel
    :return: (3,) unit vector in camera axes
    """
    x = (u - cam.cx) / cam.f                 # tan of the angle in the x-z plane
    y = (v - cam.cy) / cam.f                 # tan of the angle in the y-z plane
    d = np.array([x, y, 1.0])                # point on the image plane at unit distance
    return d / np.linalg.norm(d)             # unit length: only the direction matters


def ray_sphere(r, d, R=R_MOON_KM):
    r"""STEP 2. First point where the line r + t d (t > 0) meets the sphere |X| = R.

    A point on the line is P(t) = r + t d, t = distance travelled [km] (d is unit).
    On the surface |P(t)|^2 = R^2. Expanding with d.d = 1:

        t^2 + 2 b t + c = 0 ,    b = r.d ,    c = |r|^2 - R^2

        t = -b -/+ sqrt(b^2 - c)

    Geometry: -b is the distance to the point of closest approach to the centre, and
    sqrt(b^2 - c) is half the chord inside the sphere.
      b^2 - c < 0  -> the line misses the Moon
      t1 = -b - sqrt(b^2 - c)  -> entry point on the side facing us  (the flash)
      t2 = -b + sqrt(b^2 - c)  -> exit on the far side, not visible (ignored)
      t1 <= 0      -> the Moon is behind the camera

    :param r: start point (spacecraft) relative to the sphere centre, (3,) [km]
    :param d: unit direction, (3,)
    :return: hit point X = r + t1 d, (3,) [km], or None if there is no visible hit
    """
    r = np.asarray(r, float)
    d = np.asarray(d, float)
    b = r @ d                                # projection of r on the line
    c = r @ r - R * R                        # > 0 whenever we are outside the Moon
    disc = b * b - c                         # (half chord)^2
    if disc < 0.0:
        return None                          # miss
    t1 = -b - np.sqrt(disc)                  # nearer root = entry point
    if t1 <= 0.0:
        return None                          # Moon behind the camera
    return r + t1 * d


def ray_sphere_dem(r, d, dem_height_km, iterations=8):
    r"""Optional terrain refinement for STEP 2 (gives a real h).

    Fixed-point iteration on the local radius: intersect the sphere R + h, look up
    the terrain height h at that latitude / longitude, repeat.

        R_k+1 = R_MOON + h(lat(X_k), lon(X_k))

    :param dem_height_km: function (lat_deg, lon_deg) -> height above R_MOON_KM [km],
                          e.g. sampled from the LOLA LDEM
    """
    radius = R_MOON_KM
    X = None
    for _ in range(iterations):
        X = ray_sphere(r, d, radius)
        if X is None:
            return None
        lat, lon, _ = xyz_to_lat_lon_h(X)
        radius = R_MOON_KM + dem_height_km(lat, lon)
    return ray_sphere(r, d, radius)


def xyz_to_lat_lon_h(X, R=R_MOON_KM):
    r"""STEP 3. Moon-fixed position -> latitude, longitude, height.

        lat = asin(Z / |X|)          (height above the equator over the radius)
        lon = atan2(Y, X)            (angle around from +X, east-positive)
        h   = |X| - R                (height above the reference sphere)

    atan2 uses the signs of Y and X separately, so it returns the full -180..180 range.

    :param X: point in the Moon-fixed frame, (3,) [km]
    :return: (lat [deg], lon [deg], h [km])
    """
    X = np.asarray(X, float)
    rad = np.linalg.norm(X)
    lat = np.degrees(np.arcsin(X[2] / rad))
    lon = np.degrees(np.arctan2(X[1], X[0]))
    h = float(rad - R)
    if abs(h) < 1e-9:          # round-off on the sphere (~1e-12 km) -> exactly 0
        h = 0.0
    return float(lat), float(lon), h


def mrp_to_dcm(sigma):
    r"""STEP 4. Modified Rodrigues Parameters -> direction cosine matrix.

        [C] = I + ( 8 [s~]^2 - 4 (1 - s.s) [s~] ) / (1 + s.s)^2

    where [s~] is the cross-product (skew) matrix of s. Same as Basilisk's
    rbk.MRP2C: for sigma_BN it returns [BN], which converts J2000 (N) components
    into body (B) components:  v_B = [BN] v_N.

    :param sigma: MRP set, (3,)
    :return: (3, 3) DCM
    """
    s = np.asarray(sigma, float).reshape(3)
    S = np.array([[0.0, -s[2], s[1]],
                  [s[2], 0.0, -s[0]],
                  [-s[1], s[0], 0.0]])       # [s~]
    s2 = s @ s
    return np.eye(3) + (8.0 * S @ S - 4.0 * (1.0 - s2) * S) / (1.0 + s2) ** 2


def camera_to_moon(sigma_BN, sigma_CB, J20002Pfix):
    r"""STEP 5. Rotation taking camera-frame vectors into the Moon-fixed frame.

    The pixel gives a direction relative to the camera ("left of where I point").
    The Moon, the spacecraft position and lat/lon live in Moon-fixed axes, so the
    direction is re-expressed there. Three hops, each known by a different source:

        camera -> body       [BC] = [CB]^T   mounting   (sigma_CB)
        body   -> J2000      [NB] = [BN]^T   attitude   (sigma_BN, star tracker)
        J2000  -> Moon-fixed [PN]            SPICE      (J20002Pfix, the Moon turns ~13 deg/day)

        [PC] = [PN] [BN]^T [CB]^T            (read right to left)

    Basilisk's MRPs give [BN] and [CB] in the N->B and B->C sense; for a rotation
    matrix the inverse is the transpose, hence the ^T.

    :return: (3, 3) matrix [PC]
    """
    PN = np.asarray(J20002Pfix, float).reshape(3, 3)
    BN = mrp_to_dcm(sigma_BN)
    CB = mrp_to_dcm(sigma_CB)
    return PN @ BN.T @ CB.T


def method_a(u, v, cam, r_BN_N_m, sigma_BN, sigma_CB, J20002Pfix, dem_height_km=None):
    r"""STEP 6. Method A: flash location from a known spacecraft position and attitude.

        d_P = [PC] d_C                       direction to the flash, Moon-fixed axes
        r_P = [PN] r_BN_N / 1000             spacecraft relative to the Moon [km]
        X   = r_P + t1 d_P                   first hit on the sphere
        (lat, lon, h) from X

    The simulation uses zeroBase = "moon", so r_BN_N is already Moon-centred.

    :param r_BN_N_m: spacecraft position relative to the Moon centre, J2000 axes [m]
    :return: (lat, lon, h, X_P) with X_P [km] in the Moon-fixed frame, or None on a miss
    """
    PN = np.asarray(J20002Pfix, float).reshape(3, 3)
    d_P = camera_to_moon(sigma_BN, sigma_CB, PN) @ pixel_to_ray(u, v, cam)
    r_P = PN @ np.asarray(r_BN_N_m, float) / 1000.0
    X_P = ray_sphere_dem(r_P, d_P, dem_height_km) if dem_height_km else ray_sphere(r_P, d_P)
    if X_P is None:
        return None
    lat, lon, h = xyz_to_lat_lon_h(X_P)
    return lat, lon, h, X_P


def moon_from_disk(xc, yc, r_px, cam):
    r"""STEP 7. Spacecraft position relative to the Moon, from the Moon disk.

    Direction: the disk centre is a pixel like any other,
        m_C = pixel_to_ray(xc, yc)                      camera -> Moon centre

    Distance: the disk edge is where lines of sight graze the Moon. A grazing line
    is perpendicular to the radius, so (camera, Moon centre, edge point) is a right
    triangle with hypotenuse rho and side R opposite the angular radius alpha:

        alpha = atan(r_px / f) ,    sin(alpha) = R / rho  ->  rho = R / sin(alpha)

    Position of the spacecraft relative to the Moon centre, in camera axes:
        r_C = -rho m_C

    (alpha = atan(r_px / f) is exact for a disk at the image centre; for a disk a few
    degrees off-centre the projected edge is very slightly elliptical.)

    :param xc, yc: disk centre [px];  r_px: disk radius [px]
    :return: (r_C (3,) [km], rho [km])
    """
    m_C = pixel_to_ray(xc, yc, cam)
    alpha = np.arctan(r_px / cam.f)
    rho = R_MOON_KM / np.sin(alpha)
    return -rho * m_C, float(rho)


def method_b(u, v, cam, disk, sigma_BN, sigma_CB, J20002Pfix, dem_height_km=None):
    r"""STEP 8. Method B: flash location from the Moon disk + attitude (no navigation position).

    1. r_C, rho from the disk                                   (STEP 7)
    2. d_C from the flash pixel                                 (STEP 1)
    3. X_C = r_C + t1 d_C, all in camera axes: both vectors came from the same image,
       so no rotation is needed before intersecting            (STEP 2)
    4. Relative coordinates, axes built from the image alone:
           s = r_C / rho                 Moon centre -> camera   (relative lat 0, lon 0)
           n ~ -y_C + (y_C . s) s        image "up", made perpendicular to s (relative north)
           e = n x s                     image "right"                        (relative east)
           lat_rel = asin(X_C . n / R) ,  lon_rel = atan2(X_C . e, X_C . s)
       This is the thesis's frame, solved in exact perspective instead of orthographically.
    5. Absolute coordinates: one rotation with the attitude, then STEP 3
           X_P = [PC] X_C

    :param disk: (xc, yc, r_px) of the Moon in the same image [px]
    :return: dict(lat, lon, h, lat_rel, lon_rel, rho_km, X_P, r_P_km) or None on a miss
    """
    r_C, rho = moon_from_disk(disk[0], disk[1], disk[2], cam)
    d_C = pixel_to_ray(u, v, cam)
    PC = camera_to_moon(sigma_BN, sigma_CB, J20002Pfix)
    if dem_height_km:
        # terrain lookup needs Moon-fixed lat/lon, so intersect in Moon-fixed axes
        X_P = ray_sphere_dem(PC @ r_C, PC @ d_C, dem_height_km)
        X_C = None if X_P is None else PC.T @ X_P
    else:
        X_C = ray_sphere(r_C, d_C)
        X_P = None if X_C is None else PC @ X_C
    if X_C is None:
        return None

    s = r_C / rho                                   # toward the camera
    y_C = np.array([0.0, 1.0, 0.0])                 # image down
    n = -y_C + (y_C @ s) * s                        # image up, perpendicular to s
    n /= np.linalg.norm(n)
    e = np.cross(n, s)                              # image right
    lat_rel = np.degrees(np.arcsin(np.clip(X_C @ n / np.linalg.norm(X_C), -1.0, 1.0)))
    lon_rel = np.degrees(np.arctan2(X_C @ e, X_C @ s))

    lat, lon, h = xyz_to_lat_lon_h(X_P)
    return dict(lat=lat, lon=lon, h=h, lat_rel=float(lat_rel), lon_rel=float(lon_rel),
                rho_km=rho, X_P=X_P, r_P_km=PC @ r_C)


# =============================================================================
# PART 2 - BASILISK MODULE
# =============================================================================

def _great_circle_km(a, b):
    """Surface distance between two points on the Moon [km]."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    return float(R_MOON_KM * np.arccos(np.clip(a @ b, -1.0, 1.0)))


try:
    from Basilisk.architecture import sysModel, messaging
    _SysModel = sysModel.SysModel
except ImportError:  # lets Part 1 be tested without Basilisk installed
    sysModel = messaging = None
    _SysModel = object


class LifGeolocator(_SysModel):
    r"""Live LIF geolocation. Add to "flashDetectionTask" with a priority LOWER than
    impactFlashDetection (13), e.g. 12, so it runs right after the detector.

    Input messages (subscribe each in the scenario):
      flashInMsg        OpNavCirclesMsg      impactFlashDetection.opnavCirclesOutMsg
      moonCircleInMsg   OpNavCirclesMsg      FSW opnavCirclesMsg (Hough circle of the Moon)
      cameraConfigInMsg CameraConfigMsg      cameraMod.cameraConfigOutMsg
      attNavInMsg       NavAttMsg            SimpleNavObject.attOutMsg     (method B attitude)
      scTruthInMsg      SCStatesMsg          scObject.scStateOutMsg        (method A, truth)
      moonStateInMsg    SpicePlanetStateMsg  spiceObject.planetStateOutMsgs[moon]

    Output: one printed line per flash, self.rows (list of dicts) and write_csv(path).

    Settings:
      maxDiskAge_s    oldest Moon circle method B will use [s] (Hough runs every 0.5 s)
      dem_height_km   optional terrain function (lat, lon) -> h [km]; None = sphere, h = 0
      verbose         print each flash
    """

    def __init__(self):
        super().__init__()
        self.ModelTag = "lifGeolocator"
        self.flashInMsg = messaging.OpNavCirclesMsgReader()
        self.moonCircleInMsg = messaging.OpNavCirclesMsgReader()
        self.cameraConfigInMsg = messaging.CameraConfigMsgReader()
        self.attNavInMsg = messaging.NavAttMsgReader()
        self.scTruthInMsg = messaging.SCStatesMsgReader()
        self.moonStateInMsg = messaging.SpicePlanetStateMsgReader()
        self.maxDiskAge_s = 0.6
        self.dem_height_km = None
        self.verbose = True
        self.rows = []
        self._lastFlashTime = None

    def Reset(self, CurrentSimNanos):
        self.rows = []
        self._lastFlashTime = None

    def UpdateState(self, CurrentSimNanos):
        # ---- 1. a new, valid flash detection? ----------------------------------
        if not self.flashInMsg.isWritten():
            return
        flash = self.flashInMsg()
        if not flash.valid or flash.timeTag == self._lastFlashTime:
            return
        self._lastFlashTime = flash.timeTag

        # ---- 2. camera, attitude, truth, Moon orientation ----------------------
        camCfg = self.cameraConfigInMsg()
        cam = camera_from_config(camCfg.fieldOfView, camCfg.resolution)
        sigma_CB = list(camCfg.sigma_CB)
        truth = self.scTruthInMsg()
        sigma_BN_nav = list(self.attNavInMsg().sigma_BN)
        pfix = np.array(self.moonStateInMsg().J20002Pfix, float).reshape(3, 3)

        # ---- 3. most recent Moon disk (Hough circle), if recent enough ----------
        disk, disk_age = None, None
        if self.moonCircleInMsg.isWritten():
            mc = self.moonCircleInMsg()
            disk_age = (CurrentSimNanos - mc.timeTag) * 1e-9
            if mc.valid and mc.circlesRadii[0] > 0 and disk_age <= self.maxDiskAge_s:
                disk = (mc.circlesCenters[0], mc.circlesCenters[1], mc.circlesRadii[0])

        r_true_km = float(np.linalg.norm(truth.r_BN_N)) / 1000.0
        dt_image = (CurrentSimNanos - flash.timeTag) * 1e-9

        # ---- 4. every flash candidate in this frame ----------------------------
        for k, rad in enumerate(flash.circlesRadii):
            if rad <= 0:
                continue                      # unused slot
            u, v = flash.circlesCenters[2 * k], flash.circlesCenters[2 * k + 1]
            row = dict(time_s=flash.timeTag * 1e-9, candidate=k, u_px=u, v_px=v,
                       image_age_s=dt_image, disk_age_s=disk_age, range_true_km=r_true_km)

            A = method_a(u, v, cam, truth.r_BN_N, truth.sigma_BN, sigma_CB, pfix,
                         self.dem_height_km)
            if A is not None:
                row.update(A_lat_deg=A[0], A_lon_deg=A[1], A_h_km=A[2])
            else:
                row["A_status"] = "miss"

            B = None
            if disk is not None:
                B = method_b(u, v, cam, disk, sigma_BN_nav, sigma_CB, pfix, self.dem_height_km)
                if B is not None:
                    row.update(B_lat_deg=B["lat"], B_lon_deg=B["lon"], B_h_km=B["h"],
                               B_lat_rel_deg=B["lat_rel"], B_lon_rel_deg=B["lon_rel"],
                               B_range_km=B["rho_km"], range_err_km=B["rho_km"] - r_true_km)
                else:
                    row["B_status"] = "miss"
            else:
                row["B_status"] = "no recent Moon disk"

            if A is not None and B is not None:
                row["B_minus_A_km"] = _great_circle_km(B["X_P"], A[3])
            self.rows.append(row)
            if self.verbose:
                print(self._format(row))

    @staticmethod
    def _format(r):
        def fmt(p):
            if f"{p}_lat_deg" in r:
                return (f"({r[f'{p}_lat_deg']:+7.2f}, {r[f'{p}_lon_deg']:+8.2f}, "
                        f"h {r[f'{p}_h_km']:.2f} km)")
            return r.get(f"{p}_status", "-")
        msg = (f"LIF t={r['time_s']:8.2f} s #{r['candidate']} px=({r['u_px']:6.1f},{r['v_px']:6.1f})"
               f"  B: {fmt('B')}  A(truth): {fmt('A')}")
        if "B_minus_A_km" in r:
            msg += f"  B-A={r['B_minus_A_km']:.1f} km  range err={r['range_err_km']:+.0f} km"
        return msg

    def write_csv(self, path):
        """Save every located flash candidate (one row each) to a CSV file."""
        keys = []
        for r in self.rows:
            keys += [k for k in r if k not in keys]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
