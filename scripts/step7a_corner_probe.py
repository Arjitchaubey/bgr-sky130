import os, json, time, subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.expanduser("~/analog/bgr-sky130")
BUILD = os.path.join(ROOT, "build")
RES = os.path.join(ROOT, "results")
os.makedirs(BUILD, exist_ok=True)
os.makedirs(RES, exist_ok=True)

CORNERS = ["tt", "ss", "ff", "sf", "fs", "ll", "hh",
           "ss_ll", "ff_hh", "ss_hh", "ff_ll"]
TEMPS = [-40, 27, 125]
L1, L2, N1, N2 = 2.003, 1.924, 7, 65
VR, VDD, NV = 0.01, 1.8, 5


def make_cir(corner, dat):
    t = f"""* step7a corner probe {corner}
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice {corner}
.options reltol=1e-6 vntol=1e-9 abstol=1e-14
Vdd vdd 0 {VDD}
XQ1 0 0 a sky130_fd_pr__pnp_05v5_W3p40L3p40
I1 0 a dc 5u
Vr1 n1 0 dc {VR}
X1 n1 0 0 sky130_fd_pr__res_xhigh_po_2p85 L={L1}
Vr2 n2 0 dc {VR}
X2 n2 0 0 sky130_fd_pr__res_xhigh_po_2p85 L={L2}
XMn dn dn 0 0 sky130_fd_pr__nfet_01v8 W=10 L=2
I2 0 dn dc 1u
XMp dp dp vdd vdd sky130_fd_pr__pfet_01v8 W=10 L=2
I3 dp 0 dc 1u
.control
dc temp -40 125 1
wrdata {dat} v(a) v(dn) v(dp) i(Vr1) i(Vr2)
.endc
.end
"""
    return os.path.expandvars(t)


def run(corner):
    cir = os.path.join(BUILD, f"s7a_{corner}.cir")
    dat = os.path.join(BUILD, f"s7a_{corner}.dat")
    log = os.path.join(BUILD, f"s7a_{corner}.log")
    if os.path.exists(dat):
        os.remove(dat)
    with open(cir, "w") as f:
        f.write(make_cir(corner, dat))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    p = subprocess.run(["ngspice", "-b", cir], capture_output=True,
                       text=True, env=env)
    with open(log, "w") as f:
        f.write(p.stdout + p.stderr)
    try:
        d = np.loadtxt(dat)
        nc = d.shape[1]
        if nc == 2 * NV:
            cols = [d[:, 2 * k + 1] for k in range(NV)]
        elif nc == NV + 1:
            cols = [d[:, k + 1] for k in range(NV)]
        else:
            return corner, {"error": f"unexpected column count {nc}"}
        temps = d[:, 0]
        va, vdn, vdp, i1, i2 = cols
        out = {}
        for T in TEMPS:
            k = int(np.argmin(np.abs(temps - T)))
            if abs(temps[k] - T) > 0.01:
                return corner, {"error": f"temp {T} not on grid"}
            r1 = VR / abs(i1[k])
            r2 = VR / abs(i2[k])
            out[str(T)] = {"vbe": float(va[k]), "r1u": float(r1),
                           "r2u": float(r2),
                           "ratio": float(N2 * r2 / (N1 * r1)),
                           "vgsn": float(vdn[k]),
                           "vsgp": float(VDD - vdp[k])}
        return corner, out
    except Exception as e:
        return corner, {"error": repr(e)}


if __name__ == "__main__":
    t0 = time.time()
    print(f"Running {len(CORNERS)} simulations")
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = dict(ex.map(run, CORNERS))
    with open(os.path.join(RES, "step7a_corner_probe.json"), "w") as f:
        json.dump(res, f, indent=1)
    ref = res.get("tt", {})
    print(f"{'corner':7s}{'T':>5s}{'Vbe mV':>9s}{'dVbe':>7s}{'R1u Ohm':>10s}"
          f"{'dR1 %':>8s}{'ratio':>8s}{'Vgsn':>8s}{'Vsgp':>8s}")
    for c in CORNERS:
        r = res[c]
        if "error" in r:
            print(c, "ERROR", r["error"])
            continue
        for T in TEMPS:
            x = r[str(T)]
            dv = dr = float("nan")
            if "error" not in ref and ref:
                dv = (x["vbe"] - ref[str(T)]["vbe"]) * 1e3
                dr = (x["r1u"] / ref[str(T)]["r1u"] - 1) * 100
            print(f"{c:7s}{T:5d}{x['vbe']*1e3:9.2f}{dv:7.2f}{x['r1u']:10.2f}"
                  f"{dr:8.2f}{x['ratio']:8.4f}{x['vgsn']:8.4f}{x['vsgp']:8.4f}")
    print(f"done in {time.time()-t0:.0f} s")
