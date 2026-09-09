"""Mach number at the deployment point for every sweep case. Small helper."""
from __future__ import annotations

import json
import os

import numpy as np

from analysis.authority import FIRING_TABLE, REPRESENTATIVE, _baselines, _deploy_index


def main():
    idx = list(range(len(FIRING_TABLE)))
    BL = _baselines(idx)
    out = {"apogee": {}, "deploy": {}}
    for k, b in BL.items():
        i, t = _deploy_index(b, 0.0)
        tag = "c%d_qe%g" % (b["charge"], b["qe_mils"])
        out["apogee"][tag] = {
            "t": t, "mach": b["log_mach"][i],
            "range": b["range"], "tof": b["tof"],
            "apogee_fraction_of_tof": t / b["tof"],
        }
    for k in REPRESENTATIVE:
        b = BL[k]
        tag = "c%d_qe%g" % (b["charge"], b["qe_mils"])
        for f in (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5):
            i, t = _deploy_index(b, f)
            out["deploy"]["%s_f%+.2f" % (tag, f)] = {
                "t": t, "mach": b["log_mach"][i]}
    with open(os.path.join("docs", "deploy_mach.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for tag, v in out["apogee"].items():
        print("%-14s t=%5.1f  M=%.3f  range=%7.0f  apogee/tof=%.2f"
              % (tag, v["t"], v["mach"], v["range"], v["apogee_fraction_of_tof"]))
    print()
    for tag, v in out["deploy"].items():
        print("%-22s t=%5.1f  M=%.3f" % (tag, v["t"], v["mach"]))


if __name__ == "__main__":
    main()
