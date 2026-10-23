"""
Quantum-well laser for lambda = 947 nm (numpy + matplotlib only)

Well      : Ga0.47In0.53As(y)P(1-y), lattice-matched to InP
Barriers  : InP (not used in the energy calculation: infinite well model)
Substrate : InP

Parameters (300 K), source: matprop.ru (Ioffe), GaInAsP lattice-matched to InP:
  Eg  = 1.344 - 0.738y + 0.138y^2   [eV]
  me  = 0.08 - 0.039y               [m0]
  mhh = 0.46                        [m0]

Assumptions: infinite square well, heavy holes only, e1-hh1 transition,
~45 meV left for the confinement energy (e1 + hh1).
Lattice-matched well -> no strain, critical thickness practically infinite.
"""
import numpy as np
import matplotlib.pyplot as plt

hbar, m0, q = 1.054571817e-34, 9.1093837e-31, 1.602176634e-19  # SI
hc = 1239.84                                                     # eV nm

# 1) Target transition energy
lam = 947.0                      # nm
E_ph = hc / lam                  # eV (~1.309 eV)
print(f"lambda = {lam:.0f} nm -> target E = {E_ph:.4f} eV")

# 2) As fraction y from the target band gap
#    0.138 y^2 - 0.738 y + (1.344 - Eg_target) = 0
Eg_target = E_ph - 0.045
y = (0.738 - np.sqrt(0.738**2 - 4 * 0.138 * (1.344 - Eg_target))) / 0.276
Eg = 1.344 - 0.738 * y + 0.138 * y**2
me, mhh = 0.08 - 0.039 * y, 0.46

print(f"Well: Ga0.47In0.53As{y:.3f}P{1-y:.3f}")
print(f"      Eg = {Eg:.4f} eV, me = {me:.4f} m0, mhh = {mhh:.2f} m0")


# 3) Bound states of the infinite well (Schroedinger equation already solved)
#    psi_n(z) = sqrt(2/L) sin(n pi z / L)
#    E_n      = hbar^2 pi^2 n^2 / (2 m L^2)
def E_n(n, m, L):
    """Energy [eV] of level n for mass m [m0] in a well of width L [nm]."""
    return hbar**2 * np.pi**2 * n**2 / (2 * m * m0 * (L * 1e-9)**2) / q


# 4) Well width from the transition condition E_ph = Eg + E1_e + E1_hh
#    -> L = pi hbar sqrt((1/me + 1/mhh) / (2 m0 (E_ph - Eg)))
dE = (E_ph - Eg) * q             # confinement energy [J]
L = np.pi * hbar * np.sqrt((1 / me + 1 / mhh) / (2 * m0 * dE)) * 1e9  # nm

E1e, E1h = E_n(1, me, L), E_n(1, mhh, L)
E_tr = Eg + E1e + E1h
print(f"\nWell width L = {L:.2f} nm")
print(f"e1 = {E1e*1e3:.1f} meV, hh1 = {E1h*1e3:.1f} meV")
print(f"e1-hh1 transition = {E_tr:.4f} eV -> lambda = {hc/E_tr:.1f} nm")

# 5) Band diagram (zero = valence band edge of the well)
#    InP barriers drawn for reference only (2/3 - 1/3 band offset rule)
dEc, dEv = 2 / 3 * (1.344 - Eg), 1 / 3 * (1.344 - Eg)
print(f"InP barriers: dEc = {dEc*1e3:.0f} meV, dEv = {dEv*1e3:.0f} meV")
zz = [-20, 0, 0, L, L, L + 20]

plt.figure(figsize=(8, 6))
plt.plot(zz, Eg + np.array([dEc, dEc, 0, 0, dEc, dEc]), "k", lw=2,
         label="Conduction band")
plt.plot(zz, -np.array([dEv, dEv, 0, 0, dEv, dEv]), "gray", lw=2,
         label="Valence band")
plt.plot([0, L], [Eg + E1e] * 2, "b--", label=f"e1 = {E1e*1e3:.1f} meV")
plt.plot([0, L], [-E1h] * 2, "r--", label=f"hh1 = {E1h*1e3:.1f} meV")
plt.plot([L / 2] * 2, [-E1h, Eg + E1e], "g", lw=2,
         label=f"Transition {E_tr:.3f} eV = {hc/E_tr:.0f} nm")
plt.xlabel("z [nm]")
plt.ylabel("Energy [eV]")
plt.title(f"Ga0.47In0.53As{y:.3f}P{1-y:.3f} / InP QW, L = {L:.1f} nm")
plt.legend(loc="center right", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("band_structure_final.png", dpi=150)
plt.show()



