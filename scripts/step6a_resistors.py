import os, re, glob, json, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np

des = json.load(open("results/design.json"))
UNIT = des["R1"] / 7.0
LIBDIR = os.path.expandvars("$PDK_ROOT/sky130A/libs.ref/sky130_fd_pr/spice")
MODELS = ["sky130_fd_pr__res_high_po_1p41", "sky130_fd_pr__res_xhigh_po_1p41",
          "sky130_fd_pr__res_high_po_2p85", "sky130_fd_pr__res_xhigh_po_2p85"]
L1, L2, VT = 10.0, 40.0, 0.1
N_UNITS = 137

def find_subckt(name):
    for f in glob.glob(f"{LIBDIR}/*res*.spice"):
        txt = re.sub(r"\n\s*\+", " ", open(f, errors="ignore").read())
        m = re.search(rf"^\s*\.subckt\s+{name}\s+(.*)$", txt, re.I | re.M)
        if m:
            toks = re.sub(r"\s*=\s*", "=", m.group(1)).split()
            return [t for t in toks if "=" not in t], [t for t in toks if "=" in t]
    return None, None

def build(name, pins, out):
    tail = " ".join(["0"] * (len(pins) - 1))
    return f"""* resistor characterization (two lengths)
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice tt
.options reltol=1e-6 vntol=1e-9 abstol=1e-14
Vs1 n1 0 DC {VT}
X1 n1 {tail} {name} L={L1}
Vs2 n2 0 DC {VT}
X2 n2 {tail} {name} L={L2}
.control
dc temp -40 125 5
wrdata {out} i(Vs1) i(Vs2)
quit
.endc
.end
"""

os.makedirs("build", exist_ok=True)
jobs = []
for m in MODELS:
    pins, params = find_subckt(m)
    print(f"{m}\n    pins={pins}  params={params}")
    if pins is None:
        print("    NOT FOUND in the PDK spice files, skipping"); continue
    out = f"build/s6a_{m}.csv"
    if os.path.exists(out):
        os.remove(out)
    jobs.append((m, pins, out))

def go(job):
    m, pins, out = job
    path = f"build/s6a_{m}.spice"
    open(path, "w").write(os.path.expandvars(build(m, pins, out)))
    r = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True)
    return job, os.path.exists(out), r.stdout[-800:]

print(f"\nRunning {len(jobs)} simulations, 3 at a time ...")
with ThreadPoolExecutor(max_workers=3) as ex:
    res = list(ex.map(go, jobs))

summary = {}
print(f"\n{'model':<34} {'Rsheet':>7} {'Rend':>6} {'TC1':>8} {'TC2':>9} {'dR -40..125':>11} {'L unit':>7} {'area 137u':>10}")
print(f"{'':<34} {'ohm/sq':>7} {'ohm':>6} {'ppm/C':>8} {'ppm/C^2':>9} {'% of R27':>11} {'um':>7} {'um^2':>10}")
for (m, pins, out), ok, tail in res:
    if not ok:
        print(f"{m}: FAILED\n{tail}"); continue
    d = np.loadtxt(out)
    T, i1, i2 = d[:, 0], np.abs(d[:, 1]), np.abs(d[:, 3])
    Ra, Rb = VT / i1, VT / i2
    slope = (Rb - Ra) / (L2 - L1)
    r0 = Ra - L1 * slope
    s27, r027 = np.interp(27, T, slope), np.interp(27, T, r0)
    Lu = (UNIT - r027) / s27
    a, b = re.search(r"_(\d+)p(\d+)$", m).groups()
    W = float(f"{a}.{b}")
    R = r0 + slope * Lu
    R27 = np.interp(27, T, R)
    c2, c1, _ = np.polyfit(T - 27, R / R27 - 1, 2)
    dr = (R[-1] - R[0]) / R27 * 100
    area = N_UNITS * Lu * W
    print(f"{m:<34} {s27*W:7.1f} {r027:6.1f} {c1*1e6:8.0f} {c2*1e6:9.3f} {dr:11.2f} {Lu:7.2f} {area:10.0f}")
    summary[m] = dict(W=W, sheet=s27 * W, rend=r027, tc1=c1 * 1e6, tc2=c2 * 1e6,
                      dR_pct=dr, L_unit=Lu, area_137=area)
json.dump(summary, open("results/step6a_resistors.json", "w"), indent=2)
