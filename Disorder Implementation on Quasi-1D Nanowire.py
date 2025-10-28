
"""
Purna Paudel
Generate smooth random disorder potential (Vdis) and its projection (Vxd) in a quasi-1D nanowire.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ======================================================================
# 1. Physical constants
# ======================================================================
hbar = 6.58211899e-16       # eV·s
m0 = 9.10938215e-31          # kg
e0 = 1.602176487e-19         # C
eta_m = hbar**2 * e0 * 1e20 / m0   # ħ²/m₀ in eV·Å²
mu_B = 5.7883818066e-2       # meV/T
meV_per_K = 8.6173325e-2     # meV/K

# ======================================================================
# 2. System parameters
# ======================================================================
Nx = 500         # wire length (Lx = 2 µm)
Ny = 26          # wire width (Ly = 124 nm)
ax = 40.0        # unit cell (Å)
ay = 40.0        # unit cell (Å)
ms = 0.03        # effective mass (m*/m₀)

# hopping (meV)
tx = 1000 * eta_m / (2 * ax**2 * ms)
ty = 1000 * eta_m / (2 * ay**2 * ms)

# Rashba spin–orbit coupling (meV)
alpha_x = 125.0 / ax
alpha_y = 125.0 / ay

# superconducting parameters
Delta_00 = 0.25   # parent SC gap (meV)
Gamma = 0.75      # SM–SC coupling

# ======================================================================
# 3. Generate uniform random SM impurities
# ======================================================================
Nimp = 52
Rimp = np.column_stack((
    np.random.uniform(0, Nx + 1, Nimp),
    np.random.uniform(0, Ny + 1, Nimp)
))

#print(Rimp)

# plot impurity positions
plt.figure(figsize=(10, 2))
plt.scatter(Rimp[:, 0], Rimp[:, 1], color='tab:blue', s=40)
plt.xlabel("x (lattice sites)")
plt.ylabel("y (lattice sites)")
plt.title("Random Impurity Positions")
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()

# ======================================================================
# 4. Disorder potential
# ======================================================================
Lambda_dis = 180.0  # decay length (Å)

def fVd(x):
    """Exponential decay function."""
    return np.exp(-x / Lambda_dis)

def generate_Vdis(Nx, Ny, Rimp, ax, ay):
    """Generate smooth random disorder potential."""
    Nimp = len(Rimp)
    vtp = np.zeros((Nx, Ny), dtype=float)

    x_grid = np.arange(1, Nx + 1)[:, None]
    y_grid = np.arange(1, Ny + 1)[None, :]

    for kk in range(Nimp):
        sign = (-1)**(kk + 1)
        dd = np.sqrt((x_grid - Rimp[kk, 0])**2 * ax**2 +
                     (y_grid - Rimp[kk, 1])**2 * ay**2)
        vtp += sign * fVd(dd)

    # subtract mean and normalize variance
    vtp -= np.mean(vtp)
    vtp /= np.sqrt(np.mean(vtp**2))
    return vtp

Vdis = generate_Vdis(Nx, Ny, Rimp, ax, ay)

# Construct Vdis4 (2Ny × Nx)
v2 = np.zeros((Nx, 2 * Ny))
for ii in range(Nx):
    jj_index = (np.arange(1, 2 * Ny + 1) // 2)
    v2[ii, :] = Vdis[ii, np.clip(jj_index, 0, Ny - 1)]
Vdis4 = np.hstack((v2, -v2))

# ======================================================================
# 5. Plot Vdis (DensityPlot)
# ======================================================================

plt.figure(figsize=(10, 2))
sns.heatmap(
    Vdis.T,
    cmap="inferno",
    cbar=True,
    cbar_kws={'label': 'V_dis (arb. units)'},
    xticklabels=False,
    yticklabels=False
    
)
plt.xlabel("x (μm)")
plt.ylabel("y (nm)")
plt.title("Disorder Potential $V_{dis}(x, y)$")
plt.gca().set_aspect(0.75)
plt.show()

# ======================================================================
# 6. 1D projection: Vxd = sum_y Vdis(x, y) * Y0(y)
# ======================================================================
Y0 = np.ones(Ny)
Vxd = np.array([np.dot(Vdis[ii, :], Y0) for ii in range(Nx)])

plt.figure(figsize=(10, 2))
plt.plot(np.arange(Nx), Vxd, color='royalblue')
plt.xlabel("x (lattice sites)")
plt.ylabel("V_xd")
plt.title("Projected Disorder Potential $V_{xd}(x)$")
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()

# mean and variance
DeltaVd = np.mean(Vxd)
VarVd = np.mean(Vxd**2)
print(f"ΔVd = {DeltaVd:.4f}")
print(f"<Vxd²> = {VarVd:.4f}")

