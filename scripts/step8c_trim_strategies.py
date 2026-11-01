import json
import numpy as np
import step8b_montecarlo as m

CODES = np.arange(-32, 32)
OSTEP = 1e-4


def tc_trim(v):
    best = None
    tcs = np.array([m.tc(v + m.DR[None, :] * (c * m.STEP)) for c in CODES])
    k = np.argmin(tcs, axis=0)
    c = CODES[k]
    return v + m.DR[None, :] * (c * m.STEP)[:, None]


def off_trim(v):
    raw = np.round((m.TARGET - v[:, m.K27]) / (m.DOS[m.K27] * OSTEP))
    c = np.clip(raw, -32, 31)
    return v + m.DOS[None, :] * (c * OSTEP)[:, None], float(np.mean(raw != c))


if __name__ == "__main__":
    z = m.draw()
    res = {}
    print(f"N={m.N}, tt, mismatch only; ratio step {m.STEP}, offset step {OSTEP*1e3:.1f} mV, 6-bit")
    print(f"{'AVT':>4s}{'strategy':>16s}{'Vref sig mV':>12s}{'max dev mV':>11s}"
          f"{'TC mean':>9s}{'TC sig':>8s}{'TC p99':>8s}{'TC max':>8s}")
    for avt in m.AVT_LIST:
        v = m.build(z, avt)
        vo, sat = off_trim(v)
        cases = [("untrimmed", v),
                 ("a: Vref ratio", m.trim(v, 6)[0]),
                 ("b: TC ratio", tc_trim(v)),
                 (f"c: off+TC ({sat*100:.1f}%sat)", tc_trim(vo))]
        for name, vv in cases:
            s = m.stats(vv)
            res[f"{avt}_{name}"] = s
            print(f"{avt:4.1f}{name:>16s}{s['v27_sigma_mV']:12.2f}{s['v27_maxdev_mV']:11.2f}"
                  f"{s['tc_mean']:9.2f}{s['tc_sigma']:8.2f}{s['tc_p99']:8.2f}{s['tc_max']:8.2f}")
    json.dump(res, open("results/step8c_trim_strategies.json", "w"), indent=1)
    print("done")
