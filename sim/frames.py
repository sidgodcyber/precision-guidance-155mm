"""
Quaternion algebra, direction cosine matrices and Euler conversions.

CONVENTIONS (SIXDOFSPEC.md section 1) -- do not deviate:
  q = [w, x, y, z], unit norm, maps BODY -> EARTH:
        v_earth = R(q) @ v_body
        v_body  = R(q).T @ v_earth
  Euler sequence 3-2-1: yaw psi (about Z, down), then pitch theta
  (about Y, right), then roll phi (about X, forward).
  Because Z is down, POSITIVE PITCH IS NOSE-UP. Quadrant elevation
  therefore maps directly onto theta at launch.

Everything here is pure: no I/O, no globals, no hidden state.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "quat_normalize",
    "quat_multiply",
    "quat_conjugate",
    "dcm_from_quat",
    "quat_from_euler",
    "euler_from_quat",
    "quat_derivative",
    "rotate_body_to_earth",
    "rotate_earth_to_body",
]


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return q scaled to unit norm. Raises on a zero quaternion."""
    n = np.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n == 0.0:
        raise ValueError("cannot normalise a zero quaternion")
    return q / n


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a (x) b, both [w, x, y, z]."""
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate; for a unit quaternion this is the inverse rotation."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def dcm_from_quat(q: np.ndarray) -> np.ndarray:
    """
    Direction cosine matrix R such that v_earth = R @ v_body.

    R is orthonormal with det(+1) whenever q is a unit quaternion.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )


def quat_from_euler(psi: float, theta: float, phi: float) -> np.ndarray:
    """
    Body->earth quaternion from a 3-2-1 (yaw, pitch, roll) Euler triple.

    psi   yaw about +Z (down), radians, measured from the frame X axis
    theta pitch about +Y (right), radians, POSITIVE NOSE-UP
    phi   roll about +X (forward), radians, positive right-wing-down

    Sanity: with psi = phi = 0 and theta > 0, R(q) @ [1,0,0] is
    [cos(theta), 0, -sin(theta)] -- the nose points up (negative Z).
    """
    cpsi, spsi = np.cos(0.5 * psi), np.sin(0.5 * psi)
    cth, sth = np.cos(0.5 * theta), np.sin(0.5 * theta)
    cphi, sphi = np.cos(0.5 * phi), np.sin(0.5 * phi)
    return np.array(
        [
            cpsi * cth * cphi + spsi * sth * sphi,
            cpsi * cth * sphi - spsi * sth * cphi,
            cpsi * sth * cphi + spsi * cth * sphi,
            spsi * cth * cphi - cpsi * sth * sphi,
        ]
    )


def euler_from_quat(q: np.ndarray) -> tuple[float, float, float]:
    """
    Recover (psi, yaw), (theta, pitch), (phi, roll) from a body->earth
    quaternion. theta is clamped at +-90 deg; near that singularity the
    yaw/roll split is ill-conditioned. This is a diagnostic/reporting
    routine only -- the integrator never touches Euler angles.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    # theta from R[2,0] = -sin(theta)
    s = -2.0 * (x * z - w * y)
    s = min(1.0, max(-1.0, s))
    theta = np.arcsin(s)
    psi = np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
    phi = np.arctan2(2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))
    return float(psi), float(theta), float(phi)


def quat_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """
    qdot = 0.5 * q (x) [0, omega_body].

    omega_body = [p, q, r] are body-frame angular rates in rad/s.
    """
    p, qq, r = omega_body[0], omega_body[1], omega_body[2]
    w, x, y, z = q[0], q[1], q[2], q[3]
    return 0.5 * np.array(
        [
            -x * p - y * qq - z * r,
            w * p + y * r - z * qq,
            w * qq - x * r + z * p,
            w * r + x * qq - y * p,
        ]
    )


def rotate_body_to_earth(q: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    return dcm_from_quat(q) @ v_body


def rotate_earth_to_body(q: np.ndarray, v_earth: np.ndarray) -> np.ndarray:
    return dcm_from_quat(q).T @ v_earth
