import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open("results/step8a_sens.json"))
T = np.array(S["T"])
VN = np.array(S["vnom"])
DOS = np.array(S["dv_dvos"])
DR = np.array(S["dv_dratio"])
RHO0 = S["rho0"]
K27 = int(np.argmin(np.abs(T - 27)))
VT = 8.617333e-5 * (T + 273.15)
TARGET = VN[K27]
N = 10000
STEP = 0.015
AVT_LIST = [3.0, 5.0, 8.0]
GM_LOAD = 0.5
PAIR_WL, LOAD_WL = 50 * 2, 10 * 2
SIG_IS = 0.13 * 0.01662
RSIG_UNIT = 0.0472 / np.sqrt(2.85) * 131.17
R1U, R2U = 1536.7, 1481.3
E_R1 = RSIG_UNIT / R1U / np.sqrt(7)
E_R2 = RSIG_UNIT / R2U / np.sqrt(65)


def tc(v):
    return (v.max(axis=1) - v.min(axis=1)) / (v[:, K27] * (T.max() - T.min())) * 1e6


def model_tc(shift):
    v = (VN + DR * shift)[None, :]
    return float(tc(v)[0])


def draw(seed=1):
    r = np.random.default_rng(seed)
    return {k: r.standard_normal(N) for k in
            ("pair", "load", "q1", "q2", "r1", "r2a", "r2b")}


def build(z, avt):
    vos = avt * 1e-3 * (np.sqrt(2.0 / PAIR_WL) * z["pair"]
                        + GM_LOAD * np.sqrt(2.0 / LOAD_WL) * z["load"])
    dis = SIG_IS * (z["q1"] - z["q2"] / np.sqrt(8))
    e1, e2a, e2b = E_R1 * z["r1"], E_R2 * z["r2a"], E_R2 * z["r2b"]
    eq = vos[:, None] + VT[None, :] * (dis + e2a - e2b)[:, None]
    d = RHO0 * ((e2a + e2b) / 2 - e1)
    v = VN[None, :] + DOS[None, :] * eq + DR[None, :] * d[:, None]
    return v


def trim(v, bits):
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    raw = np.round((TARGET - v[:, K27]) / (DR[K27] * STEP))
    c = np.clip(raw, lo, hi)
    sat = float(np.mean(raw != c))
    return v + DR[None, :] * (c * STEP)[:, None], sat


def stats(v):
    v27 = v[:, K27]
    t = tc(v)
    return {"v27_mean": float(v27.mean()), "v27_sigma_mV": float(v27.std() * 1e3),
            "v27_maxdev_mV": float(np.abs(v27 - TARGET).max() * 1e3),
            "tc_mean": float(t.mean()), "tc_sigma": float(t.std()),
            "tc_p99": float(np.percentile(t, 99)), "tc_max": float(t.max())}


if __name__ == "__main__":
    print("Model check vs step 7d (tt, ratio offset: model TC vs simulated TC)")
    for off, sim in ((-0.2, 35.91), (-0.1, 24.75), (0.0, 15.45),
                     (0.1, 19.42), (0.2, 28.80)):
        print(f"  {off:+.2f}: model {model_tc(off):6.2f}   sim {sim:6.2f}")
    z = draw()
    res = {}
    print(f"\nN={N}, tt corner, mismatch only (no process corners); "
          f"trim target {TARGET:.4f} V at 27 C, step {STEP}")
    hdr = (f"{'AVT':>4s}{'case':>10s}{'Vref sig mV':>12s}{'max dev mV':>11s}"
           f"{'TC mean':>9s}{'TC sig':>8s}{'TC p99':>8s}{'TC max':>8s}{'sat %':>7s}")
    print(hdr)
    for avt in AVT_LIST:
        v = build(z, avt)
        cases = [("untrimmed", v, 0.0)]
        for b in (4, 6):
            vt, sat = trim(v, b)
            cases.append((f"{b}-bit", vt, sat))
        for name, vv, sat in cases:
            s = stats(vv)
            res[f"{avt}_{name}"] = dict(s, sat=sat)
            print(f"{avt:4.1f}{name:>10s}{s['v27_sigma_mV']:12.2f}{s['v27_maxdev_mV']:11.2f}"
                  f"{s['tc_mean']:9.2f}{s['tc_sigma']:8.2f}{s['tc_p99']:8.2f}"
                  f"{s['tc_max']:8.2f}{sat*100:7.1f}")
    # source breakdown at AVT = 5 (Vref27 sigma, mV, untrimmed)
    parts = {}
    zero = {k: np.zeros(N) for k in z}
    for k in z:
        zz = dict(zero)
        zz[k] = z[k]
        parts[k] = build(zz, 5.0)[:, K27].std() * 1e3
    print("\nVref27 sigma by source at AVT=5 (mV):",
          {k: round(float(x), 3) for k, x in parts.items()})
    json.dump(res, open("results/step8b_montecarlo.json", "w"), indent=1)
    v = build(z, 5.0)
    v4, _ = trim(v, 4)
    v6, _ = trim(v, 6)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for vv, lab in ((v, "untrimmed"), (v4, "4-bit"), (v6, "6-bit")):
        ax[0].hist(vv[:, K27] * 1e3, bins=60, alpha=0.5, label=lab)
        ax[1].hist(tc(vv), bins=60, alpha=0.5, label=lab)
    ax[0].set_xlabel("Vref at 27 C (mV)")
    ax[1].set_xlabel("TC (ppm/C)")
    ax[0].legend()
    ax[0].set_title("Sensitivity MC, AVT=5 mV*um (assumed)")
    plt.tight_layout()
    plt.savefig("results/step8b_montecarlo.png", dpi=120)
    print("done")
