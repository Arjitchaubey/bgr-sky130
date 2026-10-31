import numpy as np
d = np.loadtxt("results/pnp_vbe_vs_temp.csv")
t, v = d[:, 0], d[:, 1]
slope = np.polyfit(t, v, 1)[0] * 1e3
print(f"VBE(-40C) = {np.interp(-40, t, v):.4f} V")
print(f"VBE( 27C) = {np.interp( 27, t, v):.4f} V")
print(f"VBE(125C) = {np.interp(125, t, v):.4f} V")
print(f"Slope     = {slope:.3f} mV/C")
