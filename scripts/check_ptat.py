import numpy as np
d = np.loadtxt("results/ptat_dvbe_vs_temp.csv")
t, v = d[:, 0], d[:, 1]
slope = np.polyfit(t, v, 1)[0] * 1e3
print(f"dVBE(-40C) = {np.interp(-40, t, v)*1e3:.2f} mV")
print(f"dVBE( 27C) = {np.interp( 27, t, v)*1e3:.2f} mV   (ideal 53.7 mV for N=8)")
print(f"dVBE(125C) = {np.interp(125, t, v)*1e3:.2f} mV")
print(f"Slope      = {slope:.4f} mV/C   (ideal 0.1792 mV/C)")
