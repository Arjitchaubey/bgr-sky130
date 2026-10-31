import os, subprocess
import numpy as np

W, L = 10, 2
TEMPS = [-40, 27, 125]
TARGETS = [1e-6, 10e-6]
LIB = ".lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice tt"

def make(kind, T, out):
    if kind == "pfet":
        dev = f"XM d g s s sky130_fd_pr__pfet_01v8 W={W} L={L}"
        src = "VS s 0 1.8\nVG g 0 0\nVD d 0 0.9"
        sweep = "dc VG 0.3 1.8 0.01"
    else:
        dev = f"XM d g 0 0 sky130_fd_pr__nfet_01v8 W={W} L={L}"
        src = "VG g 0 0\nVD d 0 0.9"
        sweep = "dc VG 0.2 1.2 0.01"
    return f"""* {kind} Vgs at fixed Id
{LIB}
.temp {T}
{dev}
{src}
.control
{sweep}
let id = i(VD)
wrdata {out} id
quit
.endc
.end
"""

os.makedirs("build", exist_ok=True)
print(f"Vgs (nfet) / Vsg (pfet) in volts, W/L = {W}/{L}\n")
print(f"{'device':>6} {'Id':>6} " + " ".join(f"{T:>7}C" for T in TEMPS))
for kind in ("pfet", "nfet"):
    res = {}
    for T in TEMPS:
        out = f"build/vth_{kind}_{T}.csv"
        if os.path.exists(out):
            os.remove(out)
        path = f"build/vth_{kind}_{T}.spice"
        open(path, "w").write(os.path.expandvars(make(kind, T, out)))
        subprocess.run(["ngspice", "-b", path], capture_output=True, text=True)
        if not os.path.exists(out):
            print(f"ngspice failed for {kind} at {T}C"); raise SystemExit(1)
        d = np.loadtxt(out)
        vg, i = d[:, 0], np.abs(d[:, 1])
        vgs = (1.8 - vg) if kind == "pfet" else vg
        o = np.argsort(i)
        for tgt in TARGETS:
            res[(tgt, T)] = float(np.interp(tgt, i[o], vgs[o]))
    for tgt in TARGETS:
        print(f"{kind:>6} {tgt*1e6:4.0f}uA " + " ".join(f"{res[(tgt, T)]:8.3f}" for T in TEMPS))
