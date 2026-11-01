import os, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

des = json.load(open("results/design.json"))
s6c = json.load(open("results/step6c_summary.json"))
M = s6c["model"]
L1, L2 = s6c["L_R1_unit"], s6c["L_R2_unit"]
N1, N2 = s6c["n_R1"], s6c["n_R2"]
R1, VBE0, I0 = des["R1"], des["vbe0"], des["I"]
VG0 = s6c.get("vref27", 1.211)
TEMPS = (-40, 27, 125)

Q = "sky130_fd_pr__pnp_05v5_W3p40L3p40"
PF = "sky130_fd_pr__pfet_01v8"
NF = "sky130_fd_pr__nfet_01v8"
q2 = "\n".join(f"XQ2{i} 0 0 e2 {Q}" for i in "abcdefgh")


def chain(tag, na, nb, n, L):
    nodes = [na] + [f"{tag}_{k}" for k in range(1, n)] + [nb]
    return "\n".join(f"X{tag}_{k} {nodes[k]} {nodes[k+1]} 0 {M} L={L:.4f}"
                     for k in range(n))


def netlist(T, out):
    ns = (f".nodeset v(vref)={VG0} v(a)={VBE0} v(b)={VBE0} "
          f"v(e2)={VBE0 - I0*R1} v(g)=0.72 v(n1)=0.73 v(tail)=0.2 "
          "v(vbn)=0.55 v(sn)=0 v(s2)=0")
    return f"""* step9a noise {T} C
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice tt
.options reltol=1e-6 vntol=1e-9 abstol=1e-14
.temp {T}
VDD vdd 0 DC 1.8 AC 1
XQ1 0 0 a {Q}
{q2}
{chain("r2a", "vref", "a", N2, L2)}
{chain("r2b", "vref", "b", N2, L2)}
{chain("r1", "b", "e2", N1, L1)}
XMP vref g vdd vdd {PF} W=100 L=2
XMT tail vbn 0 0 {NF} W=20 L=2
XM1 n1 b tail 0 {NF} W=50 L=2
XM2 g a tail 0 {NF} W=50 L=2
XM3 n1 n1 vdd vdd {PF} W=10 L=2
XM4 g n1 vdd vdd {PF} W=10 L=2
XMB vbn vbn 0 0 {NF} W=10 L=2
XMBP vbn g vdd vdd {PF} W=10 L=2
C1 g vdd 20p
CL vref 0 1p
XMsp sn vref vdd vdd {PF} W=1 L=8
XMsn sn vref 0 0 {NF} W=2 L=1
XMst g sn s2 0 {NF} W=1 L=1
XMd s2 s2 0 0 {NF} W=2 L=1
{ns}
.control
noise v(vref) VDD dec 20 0.1 1e8
setplot noise1
wrdata {out} onoise_spectrum
quit
.endc
.end
"""


def go(T):
    out = f"build/s9a_{T}.dat"
    if os.path.exists(out):
        os.remove(out)
    path = f"build/s9a_{T}.spice"
    with open(path, "w") as f:
        f.write(os.path.expandvars(netlist(T, out)))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    p = subprocess.run(["ngspice", "-b", path], capture_output=True,
                       text=True, env=env)
    with open(path.replace(".spice", ".log"), "w") as f:
        f.write(p.stdout + p.stderr)
    return T


def integ(f, psd, f1, f2):
    m = (f >= f1) & (f <= f2)
    return float(np.sqrt(np.trapz(psd[m], f[m])))


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs("build", exist_ok=True)
    print(f"Running {len(TEMPS)} simulations")
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(go, TEMPS))
    res = {}
    print(f"{'T':>5s}{'1 Hz nV/rtHz':>14s}{'1 kHz nV/rtHz':>15s}"
          f"{'0.1-10Hz uVrms':>16s}{'10Hz-100k uVrms':>17s}")
    for T in TEMPS:
        try:
            d = np.loadtxt(f"build/s9a_{T}.dat")
            f, v = d[:, 0], d[:, 1]
        except Exception as e:
            print(f"{T:5d} ERROR {e!r} (see build/s9a_{T}.log)")
            continue
        psd = v ** 2
        r = {"spot_1Hz_nV": float(np.interp(1.0, f, v) * 1e9),
             "spot_1kHz_nV": float(np.interp(1e3, f, v) * 1e9),
             "int_0p1_10_uV": integ(f, psd, 0.1, 10) * 1e6,
             "int_10_100k_uV": integ(f, psd, 10, 1e5) * 1e6}
        res[str(T)] = r
        print(f"{T:5d}{r['spot_1Hz_nV']:14.1f}{r['spot_1kHz_nV']:15.1f}"
              f"{r['int_0p1_10_uV']:16.2f}{r['int_10_100k_uV']:17.2f}")
    json.dump(res, open("results/step9a_noise.json", "w"), indent=1)
    print(f"done in {time.time()-t0:.0f} s")
