"""
Step 5, Task A: what the sensors are exposed to, and what that decides.

Produces every number in docs/SENSOR-MODELS.md and the JSON the other
navigation modules read, docs/nav_sensors.json.

WHAT THIS MODULE MEASURES RATHER THAN ASSUMES
---------------------------------------------
1. SATURATION. The body spin and the despun nose rate through the whole
   flight, against the full scale of three gyro classes. The result decides
   the sensor layout and it is a measurement, not a design preference.
2. THE CENTRIPETAL TERM, body-mounted against despun, at the transverse
   offset a fuze-well kit can actually hold.
3. THE LEVER-ARM TERMS at the despun station, which turn out to be dominated
   by the servo's OWN brake torque acting on a 1.8 g m^2 nose.
4. ROLL OBSERVABILITY from the magnetometer, swept over firing azimuth. This
   is the one that varies across the envelope and the one nobody would guess:
   the field-to-axis angle collapses in the second half of some trajectories
   and the roll signal collapses with it.
5. THE EULER IDENTITY that licenses the despun mounting, verified numerically
   on a flown trajectory rather than only in a docstring.

Run:  python -m analysis.nav_sensors
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from sim import dynamics as dyn, frames
from gnc import sensors as sn
from analysis import guidance_cep as gc, nav_common as nc, roll_servo as rs

#: Gyro full scales to test the layout against, deg/s. The first two are the
#: range the brief names as commodity; 8000 is included so the trend is a
#: curve rather than two points.
FULL_SCALES = (125.0, 250.0, 400.0, 1000.0, 2000.0, 4000.0, 8000.0)

#: Transverse offsets of the IMU from the spin axis, m.
OFFSETS = (0.003, 0.010, 0.025, 0.050)

#: Firing azimuths for the roll-observability sweep, deg east of true north.
AZIMUTHS = tuple(range(0, 360, 30))


# ===========================================================================
def saturation(traj: dict) -> dict:
    """
    When does each gyro class come out of saturation, and how much of the
    flight does it spend pinned?

    Reported for the BODY (which no MEMS part survives) and for the DESPUN
    nose (which is the whole argument).
    """
    t = traj["t"]
    t_dep = float(t[0])
    p_body = traj["omega"][:, 0]
    p_nose = p_body + traj["nose_rate"]
    out = {"t_dep": t_dep, "duration_s": float(t[-1] - t[0]),
           "body_spin_deg_s": {"at_deploy": float(math.degrees(p_body[0])),
                               "at_impact": float(math.degrees(p_body[-1])),
                               "max": float(np.degrees(np.abs(p_body)).max())},
           "nose_rate_deg_s": {"at_deploy": float(math.degrees(p_nose[0])),
                               "max": float(np.degrees(np.abs(p_nose)).max())},
           "classes": {}}
    for fs in FULL_SCALES:
        row = {}
        for name, w in (("body", p_body), ("despun", p_nose)):
            over = np.abs(np.degrees(w)) >= fs
            idx = np.where(over)[0]
            row[name] = {
                "fraction_saturated": float(over.mean()),
                "last_exceeded_s_after_deploy":
                    (float(t[idx[-1]] - t_dep) if idx.size else None),
                "clear_after_s": (float(t[idx[-1]] - t_dep) if idx.size else 0.0),
            }
        # ... and what it costs after the despin transient, which is the
        # regime a part is really chosen for.
        m = t >= t_dep + 2.2
        row["despun"]["fraction_saturated_after_despin"] = float(
            (np.abs(np.degrees(p_nose[m])) >= fs).mean())
        out["classes"][str(fs)] = row
    m = t >= t_dep + 2.2
    pn = np.abs(np.degrees(p_nose[m]))
    out["nose_rate_after_despin_deg_s"] = {
        "max": float(pn.max()), "p99": float(np.percentile(pn, 99)),
        "p95": float(np.percentile(pn, 95)), "median": float(np.median(pn)),
        "mean": float(pn.mean())}
    tr_rate = np.degrees(np.abs(traj["omega"][:, 1:3]))
    out["transverse_body_rate_deg_s_max"] = float(tr_rate.max())
    return out


def lever_terms(traj: dict, mapdata: dict) -> dict:
    """
    The two transport accelerations at the IMU station, body-mounted and
    despun, at several transverse offsets.

    THE CENTRIPETAL ONE is the layout argument and it is enormous on the body.
    THE TANGENTIAL ONE is the surprise: the nose's moment of inertia is
    1.8 g m^2 and the brake applies up to 0.85 N m, so the SERVO'S OWN
    ACTUATOR swings the nose hard enough to write metres per second squared
    onto the accelerometers.
    """
    from dataclasses import replace as dc_replace
    from sim import canards as cn
    from analysis import guidance_authority as ga

    plant = rs.make_plant(brake_max=ga.BRAKE)
    nose = dc_replace(plant.nose, deploy_time=float(traj["t"][0]),
                      brake_command=lambda tt: 0.0, steering_gate=lambda tt: True)
    model = cn.guided_model(rs.base_model(), rs.GEOMETRY, nose)

    t = traj["t"]
    y = np.zeros(dyn.STATE_SIZE)
    i0 = int(np.argmax(t >= t[0] + 2.2))
    out = {"offsets": {}, "nose_inertia_kg_m2": float(plant.nose.inertia)}
    rows = {r: [] for r in OFFSETS}
    f_cg, alpha_x, spin = [], [], []
    for i in range(i0, len(t), 7):
        y[0:3] = traj["position"][i]; y[3:6] = traj["velocity"][i]
        y[6:10] = traj["quaternion"][i]; y[10:13] = traj["omega"][i]
        y[13] = traj["nose_angle"][i]; y[14] = traj["nose_rate"][i]
        yd = dyn.derivative(float(t[i]), y, model)
        for off in OFFSETS:
            lever = np.array([0.399, off, 0.0])
            tr = sn.truth_at(float(t[i]), y, model, sn.NAGPUR, lever,
                             (0.439, 0.010, 0.0), ydot=yd)
            om_n, al_n = tr.omega_nose, tr.alpha_nose
            tang = np.cross(al_n, lever)
            cent = np.cross(om_n, np.cross(om_n, lever))
            rows[off].append((np.linalg.norm(tang), np.linalg.norm(cent),
                              sn.body_mounted_centripetal(tr.spin, off)))
            if off == 0.025:
                f_cg.append(np.linalg.norm(tr.specific_force_nose - tang - cent))
                alpha_x.append(abs(float(al_n[0])))
                spin.append(abs(float(tr.spin)))
    for off, v in rows.items():
        a = np.array(v)
        out["offsets"][f"{off:.3f}"] = {
            "tangential_rms": float(np.sqrt((a[:, 0] ** 2).mean())),
            "tangential_max": float(a[:, 0].max()),
            "centripetal_despun_rms": float(np.sqrt((a[:, 1] ** 2).mean())),
            "centripetal_despun_max": float(a[:, 1].max()),
            "centripetal_body_max_g": float(a[:, 2].max() / sn.G0),
            "centripetal_body_mean_g": float(a[:, 2].mean() / sn.G0),
        }
    out["flight_specific_force_mean"] = float(np.mean(f_cg))
    out["flight_specific_force_max_g"] = float(np.max(f_cg) / sn.G0)
    out["nose_angular_accel_rad_s2"] = {
        "mean": float(np.mean(alpha_x)), "max": float(np.max(alpha_x))}
    #: The offset at which a body-mounted accelerometer would stay inside a
    #: 40 g part. Solving w^2 r = 40 g at the deployment spin.
    out["body_mounted_offset_for_40g_mm"] = float(
        1000.0 * 40.0 * sn.G0 / max(np.max(spin) ** 2, 1e-9))
    return out


def observability(traj: dict) -> dict:
    """
    Roll observability from the magnetometer, swept over firing azimuth.

    Only the field component TRANSVERSE to the body axis is modulated by roll,
    so the roll signal is |B| sin(field-to-axis angle) and a measurement error
    transverse to the field costs 1/sin of it in roll. This sweep is the
    reason `NavConfig.mag_min_transverse` exists: on some azimuths the angle
    collapses in the second half of the flight and the magnetometer stops
    being a roll sensor at all.

    The trajectory is not re-flown per azimuth. Rotating the gun line and
    rotating the FIELD are the same operation on this geometry, and the
    trajectory is symmetric about its own X axis to the drift, so sweeping
    `GeomagneticField.azimuth` sweeps the engagement.
    """
    t = traj["t"]
    quat, pos = traj["quaternion"], traj["position"]
    idx = range(0, len(t), 50)
    out = {"azimuths_deg": list(AZIMUTHS), "rows": []}
    for az in AZIMUTHS:
        field = sn.GeomagneticField(intensity=sn.NAGPUR.intensity,
                                    declination=sn.NAGPUR.declination,
                                    inclination=sn.NAGPUR.inclination,
                                    azimuth=math.radians(az))
        ang = []
        for i in idx:
            x_e = frames.dcm_from_quat(quat[i])[:, 0]
            ang.append(field.axis_angle(pos[i], x_e))
        ang = np.array(ang)
        s = np.sin(ang)
        # The roll error amplification is 1/sin, and its average over the
        # flight is what actually matters -- but the WORST is what decides
        # whether the update has to be gated.
        out["rows"].append({
            "azimuth_deg": az,
            "angle_deg": {"min": float(np.degrees(ang.min())),
                          "median": float(np.degrees(np.median(ang))),
                          "max": float(np.degrees(ang.max()))},
            "sin_min": float(s.min()), "sin_median": float(np.median(s)),
            "amplification_median": float(np.median(1.0 / np.maximum(s, 1e-6))),
            "fraction_below_15deg": float((s < math.sin(math.radians(15.0))).mean()),
            "roll_sigma_median_deg": float(
                math.degrees(0.015 / max(np.median(s), 1e-6))),
        })
    return out


def euler_identity(traj: dict) -> dict:
    """
    Verify, on a flown trajectory, that the despun IMU's own 3-2-1 Euler roll
    IS `phi_body + phi_rel` and its pitch and yaw ARE the body's.

    The whole architecture rests on this, so it is measured here as well as
    asserted in `tests/test_navigation.py`.
    """
    quat, na = traj["quaternion"], traj["nose_angle"]
    e_roll, e_pitch, e_yaw = [], [], []
    for i in range(0, len(traj["t"]), 200):
        C = frames.dcm_from_quat(quat[i]) @ sn.nose_from_body(na[i])
        th = -math.asin(max(-1.0, min(1.0, C[2, 0])))
        ph = math.atan2(C[2, 1], C[2, 2])
        ps = math.atan2(C[1, 0], C[0, 0])
        psi_b, th_b, ph_b = frames.euler_from_quat(quat[i])
        d = ph - (ph_b + na[i])
        e_roll.append(abs(math.atan2(math.sin(d), math.cos(d))))
        e_pitch.append(abs(th - th_b))
        e_yaw.append(abs(math.atan2(math.sin(ps - psi_b), math.cos(ps - psi_b))))
    return {"max_roll_error_rad": float(max(e_roll)),
            "max_pitch_error_rad": float(max(e_pitch)),
            "max_yaw_error_rad": float(max(e_yaw)),
            "samples": len(e_roll)}


def alignment_reference(traj: dict) -> dict:
    """
    How good is the velocity vector as an attitude reference?

    The magnetometer fixes two attitude degrees of freedom and leaves the
    rotation about the field free. The velocity vector is the second
    reference, and its error is the angle between the body axis and the
    air-relative velocity -- which step 1's own 6-DOF supplies, so this is
    measured rather than assumed. The decomposition matters: a MEAN offset
    would be an attitude bias the filter cannot see, while a zero-mean swing
    is only noise.
    """
    quat, vel = traj["quaternion"], traj["velocity"]
    ys, zs, tot = [], [], []
    for i in range(0, len(traj["t"]), 5):
        C = frames.dcm_from_quat(quat[i])
        u = vel[i] / np.linalg.norm(vel[i])
        b = C.T @ u
        ys.append(b[1]); zs.append(b[2])
        tot.append(math.acos(max(-1.0, min(1.0, float(b[0])))))
    ys, zs, tot = np.array(ys), np.array(zs), np.array(tot)
    return {
        "total_aoa_deg": {"mean": float(np.degrees(tot).mean()),
                          "rms": float(math.degrees(math.sqrt((tot ** 2).mean()))),
                          "p95": float(np.degrees(np.percentile(tot, 95))),
                          "max": float(np.degrees(tot).max())},
        "yaw_component_deg": {"mean": float(np.degrees(ys).mean()),
                              "rms": float(np.degrees(np.sqrt((ys ** 2).mean())))},
        "pitch_component_deg": {"mean": float(np.degrees(zs).mean()),
                                "rms": float(np.degrees(np.sqrt((zs ** 2).mean())))},
    }


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/nav_sensors.json")
    ap.add_argument("--engagement", default="long")
    args = ap.parse_args(argv)

    sn.warn_low_confidence()
    t0 = time.time()
    md = gc.load_maps()
    print(f"flying the reference trajectory ({args.engagement}) ...", flush=True)
    traj = nc.fly_truth(args.engagement, md)
    print(f"  {traj['t'].size} samples at the IMU rate, "
          f"{time.time() - t0:.1f} s", flush=True)

    out = {"engagement": args.engagement,
           "config": {"full_scales_deg_s": list(FULL_SCALES),
                      "offsets_m": list(OFFSETS),
                      "azimuths_deg": list(AZIMUTHS),
                      "field": {"intensity_nT": sn.NAGPUR.intensity / sn.NT,
                                "declination_deg": math.degrees(sn.NAGPUR.declination),
                                "inclination_deg": math.degrees(sn.NAGPUR.inclination),
                                "site": "Nagpur 21.15 N 79.09 E"}},
           "confidence": {k: list(v) for k, v in sn.SENSOR_CONFIDENCE.items()},
           "low_confidence": sn.low_confidence_keys()}

    print("  saturation ...", flush=True)
    out["saturation"] = saturation(traj)
    print("  lever-arm terms ...", flush=True)
    out["lever"] = lever_terms(traj, md)
    print("  roll observability ...", flush=True)
    out["observability"] = observability(traj)
    print("  the Euler identity ...", flush=True)
    out["euler_identity"] = euler_identity(traj)
    print("  the velocity attitude reference ...", flush=True)
    out["alignment"] = alignment_reference(traj)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out} in {time.time() - t0:.1f} s")

    s = out["saturation"]
    print(f"\n  body spin at deploy   {s['body_spin_deg_s']['at_deploy']:10.0f} deg/s")
    print(f"  body spin at impact   {s['body_spin_deg_s']['at_impact']:10.0f} deg/s")
    for fs in ("2000.0", "4000.0"):
        c = s["classes"][fs]
        print(f"  +-{float(fs):.0f} deg/s: body pinned "
              f"{100 * c['body']['fraction_saturated']:.0f} % of flight; "
              f"despun clear after t_dep+{c['despun']['clear_after_s']:.2f} s")
    l = out["lever"]["offsets"]["0.025"]
    print(f"  centripetal at 25 mm: body {l['centripetal_body_max_g']:.0f} g, "
          f"despun {l['centripetal_despun_rms'] / sn.G0:.3f} g")
    print(f"  Euler identity holds to {out['euler_identity']['max_roll_error_rad']:.2e} rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
