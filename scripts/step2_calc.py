import json
import numpy as np

k, q = 1.380649e-23, 1.602176634e-19
T0 = 27.0
N = 8            # emitter-area ratio
I = 5e-6         # branch current (A)

d = np.loadtxt("results/pnp_vbe_vs_temp.csv")
T, V = d[:, 0], d[:, 1]
vbe0 = float(np.interp(T0, T, V))
m = (T >= 5) & (T <= 50)
ctat = float(np.polyfit(T[m], V[m], 1)[0])      # V/C, local slope near 27C

VT = k * (T0 + 273.15) / q
dvbe = VT * np.log(N)
ptat_slope = (k / q) * np.log(N)                # V/C
ratio = -ctat / ptat_slope                      # R2/R1 for zero first-order drift
R1 = dvbe / I
R2 = ratio * R1
vref = vbe0 + ratio * dvbe

print(f"VBE(27C)            = {vbe0:.4f} V")
print(f"CTAT slope (near 27)= {ctat*1e3:.3f} mV/C")
print(f"VT(27C)             = {VT*1e3:.3f} mV")
print(f"dVBE = VT*ln({N})     = {dvbe*1e3:.2f} mV")
print(f"PTAT slope          = {ptat_slope*1e3:.4f} mV/C")
print(f"R2/R1               = {ratio:.2f}")
print(f"R1                  = {R1/1e3:.2f} kOhm")
print(f"R2                  = {R2/1e3:.2f} kOhm")
print(f"Predicted Vref(27C) = {vref:.4f} V")
print(f"Min supply estimate = {vref + 0.25 + 0.0:.2f} V (Vref + ~0.25 V PMOS headroom)")

json.dump(dict(N=N, I=I, R1=R1, R2=R2, vbe0=vbe0, vref_pred=vref),
          open("results/design.json", "w"), indent=2)
