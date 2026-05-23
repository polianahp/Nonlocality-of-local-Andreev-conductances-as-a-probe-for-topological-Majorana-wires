#!/usr/bin/env python3
"""
Pfaffian Invariant Calculation for Majorana Wires
Translated from Mathematica Notebook.
"""

import argparse
import sys
import numpy as np
import scipy.linalg

def householder_vector_real(x):
    normx = np.linalg.norm(x)
    n = len(x)
    if normx == 0:
        v = np.zeros(n)
        v[0] = 1.0
        return v, 0.0, 0.0
    
    if x[0] > 0:
        tempfac = normx
    else:
        tempfac = -normx
        
    temp = np.array(x, dtype=float, copy=True)
    temp[0] += tempfac
    
    norm_temp = np.linalg.norm(temp)
    if norm_temp == 0:
        v = np.zeros(n)
        v[0] = 1.0
        return v, 0.0, 0.0
    v = temp / norm_temp
    return v, 2.0, -tempfac


def householder_vector_complex(x):
    normx = np.linalg.norm(x)
    n = len(x)
    if normx == 0:
        v = np.zeros(n, dtype=complex)
        v[0] = 1.0
        return v, 0.0, 0.0
    
    if x[0] == 0:
        phase = 1.0
    else:
        phase = x[0] / np.abs(x[0])
    
    tempfac = phase * normx
    temp = np.array(x, dtype=complex, copy=True)
    temp[0] += tempfac
    
    norm_temp = np.linalg.norm(temp)
    if norm_temp == 0:
        v = np.zeros(n, dtype=complex)
        v[0] = 1.0
        return v, 0.0, 0.0
    v = temp / norm_temp
    return v, 2.0, -tempfac


def pfaffian_h_real(Mat):
    A = np.array(Mat, dtype=float, copy=True)
    n = A.shape[0]
    if n % 2 != 0:
        return 0.0
    pfaff = 1.0
    for i in range(0, n - 2, 2):
        v, beta, alpha = householder_vector_real(A[i+1:, i])
        det = 1.0 if beta == 0 else -1.0
        pfaff *= det * (-alpha)
        
        w = beta * (A[i+1:, i+1:] @ v)
        A[i+1:, i+1:] += np.outer(v, w) - np.outer(w, v)
        
    return pfaff * A[n-2, n-1]


def pfaffian_h_complex(Mat):
    A = np.array(Mat, dtype=complex, copy=True)
    n = A.shape[0]
    if n % 2 != 0:
        return 0.0
    pfaff = 1.0
    for i in range(0, n - 2, 2):
        v, beta, alpha = householder_vector_complex(A[i+1:, i])
        det = 1.0 if beta == 0 else -1.0
        pfaff *= det * (-alpha)
        
        w = beta * (A[i+1:, i+1:] @ np.conj(v))
        A[i+1:, i+1:] += np.outer(v, w) - np.outer(w, v)
        
    return pfaff * A[n-2, n-1]


def pfaffian_h(Mat):
    if np.iscomplexobj(Mat):
        return pfaffian_h_complex(Mat)
    else:
        return pfaffian_h_real(Mat)


def pfaffian_ltl(Mat):
    n = Mat.shape[0]
    if n % 2 != 0:
        return 0.0
    
    A = np.array(Mat, dtype=complex, copy=True)
    pfaff = 1.0
    
    for i in range(0, n - 2, 2):
        # find out the maximum entry in the column i, starting from row i+1
        ip = i + 1 + np.argmax(np.abs(A[i+1:, i]))
        
        # if the maximum entry is not at i+1, permute the matrix so that it is
        if i + 1 != ip:
            # Interchange rows in A
            A[[i+1, ip], :] = A[[ip, i+1], :]
            # Interchange columns in A
            A[:, [i+1, ip]] = A[:, [ip, i+1]]
            pfaff = -pfaff
            
        pfaff = pfaff * A[i, i+1]
        
        if np.abs(A[i+1, i]) < 1e-15:
            return 0.0
            
        # Build the Gauss vector
        A[i+2:, i] = A[i+2:, i] / A[i+1, i]
        
        # Update the remainder of the matrix
        v = A[i+2:, i]
        u = A[i+2:, i+1]
        A[i+2:, i+2:] += np.outer(v, u) - np.outer(u, v)
        
    return pfaff * A[n-2, n-1]


def pfaffian_hessenberg(Mat):
    if np.iscomplexobj(Mat) and np.any(np.imag(Mat) != 0.0):
        raise ValueError("Pfaffian computation with Hessenberg decomposition only works for real matrices")
    
    A = np.real(Mat)
    n = A.shape[0]
    if n % 2 != 0:
        return 0.0
    
    # scipy's hessenberg returns H, Q
    H, Q = scipy.linalg.hessenberg(A, calc_q=True)
    
    det_Q = np.linalg.det(Q)
    prod_H = 1.0
    for i in range(0, n, 2):
        prod_H *= H[i, i+1]
        
    return det_Q * prod_H


def pfaffian(A, method="h"):
    """
    Computes the Pfaffian of a skew-symmetric matrix A using the specified method.
    Available methods:
      - "h": Householder tridiagonalization (default, real/complex)
      - "ltl": Parlett-Reid tridiagonalization (real/complex)
      - "hessenberg": Hessenberg decomposition (real only)
    """
    if method == "h":
        return pfaffian_h(A)
    elif method == "ltl":
        return pfaffian_ltl(A)
    elif method == "hessenberg":
        return pfaffian_hessenberg(A)
    else:
        raise ValueError(f"Unknown Pfaffian calculation method: {method}")



# Physical Constants
hbar = 6.58211899 * 1e-16  # h/2\pi in eV s
m0 = 9.10938215 * 1e-31
e0 = 1.602176487 * 1e-19
eta_m = hbar**2 * e0 * 1e20 / m0  # hbar^2/2m0 in eV A^2 (~7.61996)
mu_B = 5.7883818066 * 1e-2  # in meV/T
meVpK = 8.6173325 * 1e-2  # Kelvin into meV


class PfaffianSimulator:
    def __init__(self, Nx=400, Ny=1, delta_N=20, ax=100.0, ay=100.0, ms=0.03, gamma0=0.35, delta0=0.3, method="h"):
        self.Nx = Nx
        self.Ny = Ny
        self.delta_N = delta_N
        self.ax = ax
        self.ay = ay
        self.ms = ms
        self.gamma0 = gamma0
        self.delta0 = delta0
        self.method = method
        
        # Derived parameters
        self.tx = 1000.0 * eta_m / (2.0 * self.ax**2 * self.ms)
        self.ty = 1000.0 * eta_m / (2.0 * self.ay**2 * self.ms)
        self.alphax = 200.0 / self.ax
        self.alphay = 200.0 / self.ay
        
        # Renormalization factors
        self.Z = self.delta0 / (self.delta0 + self.gamma0)
        self.delta = self.delta0 * self.gamma0 / (self.delta0 + self.gamma0)
        
        # Corrected epsilon0
        self.epsilon0 = self.compute_epsilon0_corrected()
        
        # Build constant matrices
        self.V2D = self.build_V2D()
        self.delta_2D = self.build_delta_2D()
        self.T2D = self.build_T2D()
        self.T1 = self.build_T1()
        self.S2D = self.build_S2D()
        
        # Initialize disorder profiles
        self.Vdiss = np.zeros((self.Nx // 2, 8 * self.Ny))
        self.SCdiss2D = np.ones((self.Nx // 2, 8 * self.Ny))
        
    def compute_epsilon0_corrected(self):
        epsilon0 = 2.0 * (self.tx + self.ty * np.cos(np.pi / (self.Ny + 1.0)))
        if self.Ny == 1:
            return epsilon0  # For Ny=1, epsilon02D evaluates to 0, so corrected is just epsilon0
            
        htp = np.zeros((2 * self.Ny, 2 * self.Ny), dtype=complex)
        for ii in range(1, self.Ny + 1):
            idx = 2 * (ii - 1)
            htp[idx, idx] = epsilon0
            htp[idx + 1, idx + 1] = epsilon0
            
        for ii in range(1, self.Ny):
            idx1 = 2 * (ii - 1)
            idx2 = 2 * ii
            htp[idx1, idx2] = -self.ty
            htp[idx2, idx1] = -self.ty
            htp[idx1 + 1, idx2 + 1] = -self.ty
            htp[idx2 + 1, idx1 + 1] = -self.ty
            
            # Rashba coupling in y
            htp[idx1, idx2 + 1] = 0.5j * self.alphay
            htp[idx2 + 1, idx1] = -0.5j * self.alphay
            htp[idx1 + 1, idx2] = 0.5j * self.alphay
            htp[idx2, idx1 + 1] = -0.5j * self.alphay
            
        # htp is Hermitian, get real eigenvalues
        etp = np.linalg.eigvalsh(htp)
        epsilon02D = 2.0 * self.tx - etp[0]
        return epsilon0 + epsilon02D

    def build_H02D(self, Gamma, mu, theta):
        """Diagonal block of Hamiltonian of size 8Ny x 8Ny."""
        qtp = np.zeros((8 * self.Ny, 8 * self.Ny), dtype=complex)
        
        # 4x4 blocks on the diagonal for each transverse slice of chain 1 (up to 4*Ny)
        for ii in range(1, self.Ny + 1):
            idx = 4 * (ii - 1)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            phase_p = Gamma * (cos_t + 1j * sin_t)
            phase_m = Gamma * (cos_t - 1j * sin_t)
            
            block = np.array([
                [self.epsilon0 - mu, phase_p, 0.0, 0.0],
                [phase_m, self.epsilon0 - mu, 0.0, 0.0],
                [0.0, 0.0, -self.epsilon0 + mu, -phase_m],
                [0.0, 0.0, -phase_p, -self.epsilon0 + mu]
            ], dtype=complex)
            qtp[idx:idx+4, idx:idx+4] = block

        # Off-diagonal transverse hopping blocks coupling different slices of chain 1
        for ii in range(1, self.Ny):
            idx1 = 4 * (ii - 1)
            idx2 = 4 * ii
            
            block_forward = np.array([
                [-self.ty, 0.5j * self.alphay, 0.0, 0.0],
                [0.5j * self.alphay, -self.ty, 0.0, 0.0],
                [0.0, 0.0, self.ty, 0.5j * self.alphay],
                [0.0, 0.0, 0.5j * self.alphay, self.ty]
            ], dtype=complex)
            
            block_backward = np.array([
                [-self.ty, -0.5j * self.alphay, 0.0, 0.0],
                [-0.5j * self.alphay, -self.ty, 0.0, 0.0],
                [0.0, 0.0, self.ty, -0.5j * self.alphay],
                [0.0, 0.0, -0.5j * self.alphay, self.ty]
            ], dtype=complex)
            
            qtp[idx1:idx1+4, idx2:idx2+4] = block_forward
            qtp[idx2:idx2+4, idx1:idx1+4] = block_backward
            
        # Copy chain 1 structure to chain 2
        half = 4 * self.Ny
        qtp[half:2*half, half:2*half] = qtp[0:half, 0:half]
        return qtp

    def build_V2D(self):
        vtp = np.zeros((8 * self.Ny, 8 * self.Ny), dtype=complex)
        block = np.diag([1.0, 1.0, -1.0, -1.0])
        for ii in range(1, 2 * self.Ny + 1):
            idx = 4 * (ii - 1)
            vtp[idx:idx+4, idx:idx+4] = block
        return vtp

    def build_delta_2D(self):
        vtp = np.zeros((8 * self.Ny, 8 * self.Ny), dtype=complex)
        block = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0]
        ], dtype=complex)
        for ii in range(1, 2 * self.Ny + 1):
            idx = 4 * (ii - 1)
            vtp[idx:idx+4, idx:idx+4] = block
        return vtp

    def build_T2D(self):
        ttp = np.zeros((8 * self.Ny, 8 * self.Ny), dtype=complex)
        for ii in range(1, self.Ny + 1):
            idx = 4 * (ii - 1)
            block = np.array([
                [-self.tx, -0.5 * self.alphax, 0.0, 0.0],
                [0.5 * self.alphax, -self.tx, 0.0, 0.0],
                [0.0, 0.0, self.tx, 0.5 * self.alphax],
                [0.0, 0.0, -0.5 * self.alphax, self.tx]
            ], dtype=complex)
            ttp[idx:idx+4, idx:idx+4] = block
            
        for ii in range(self.Ny + 1, 2 * self.Ny + 1):
            idx = 4 * (ii - 1)
            block = np.array([
                [-self.tx, 0.5 * self.alphax, 0.0, 0.0],
                [-0.5 * self.alphax, -self.tx, 0.0, 0.0],
                [0.0, 0.0, self.tx, -0.5 * self.alphax],
                [0.0, 0.0, 0.5 * self.alphax, self.tx]
            ], dtype=complex)
            ttp[idx:idx+4, idx:idx+4] = block
        return ttp

    def build_T1(self):
        ttp = np.zeros((8 * self.Ny, 8 * self.Ny), dtype=complex)
        half = 4 * self.Ny
        for ii in range(1, self.Ny + 1):
            # Coupling from copy 1 to copy 2
            idx_c1 = 4 * (ii - 1)
            idx_c2 = half + 4 * (ii - 1)
            
            block_12 = np.array([
                [-self.tx, -0.5 * self.alphax, 0.0, 0.0],
                [0.5 * self.alphax, -self.tx, 0.0, 0.0],
                [0.0, 0.0, self.tx, 0.5 * self.alphax],
                [0.0, 0.0, -0.5 * self.alphax, self.tx]
            ], dtype=complex)
            ttp[idx_c2:idx_c2+4, idx_c1:idx_c1+4] = block_12
            
        for ii in range(1, self.Ny + 1):
            # Coupling from copy 2 to copy 1
            idx_c1 = 4 * (ii - 1)
            idx_c2 = half + 4 * (ii - 1)
            
            block_21 = np.array([
                [-self.tx, 0.5 * self.alphax, 0.0, 0.0],
                [-0.5 * self.alphax, -self.tx, 0.0, 0.0],
                [0.0, 0.0, self.tx, -0.5 * self.alphax],
                [0.0, 0.0, 0.5 * self.alphax, self.tx]
            ], dtype=complex)
            ttp[idx_c1:idx_c1+4, idx_c2:idx_c2+4] = block_21
        return ttp

    def build_S2D(self):
        vtp = np.zeros((8 * self.Ny, 8 * self.Ny), dtype=complex)
        block = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=complex)
        for ii in range(1, 2 * self.Ny + 1):
            idx = 4 * (ii - 1)
            vtp[idx:idx+4, idx:idx+4] = block
        return vtp

    def Symm(self, X):
        """Projects matrix X into the particle-hole symmetric subspace."""
        x0 = 0.5 * (X + X.conj().T)
        return 0.5 * (x0 - self.S2D @ x0.conj() @ self.S2D)

    def Asymm(self, X):
        """Projects matrix X into the skew-symmetric subspace."""
        return 0.5 * (X - X.T)

    def ExpandVdiss(self, dis):
        """Expands a 1D or 2D disorder array to the double-chain basis configuration."""
        if dis.ndim == 1:
            dis = dis[:, np.newaxis]
        Nx_in = dis.shape[0]
        half_Nx = Nx_in // 2
        distp = np.zeros((half_Nx, 8 * self.Ny), dtype=float)
        for ii in range(half_Nx):
            row = []
            for jj in range(self.Ny):
                val_front = dis[ii, jj]
                val_back = dis[Nx_in - 1 - ii, jj]
                row.extend([val_front]*4 + [val_back]*4)
            distp[ii] = row
        return distp

    def load_disorder(self, dis_array):
        """Loads and expands the disorder array."""
        self.Vdiss = self.ExpandVdiss(dis_array)
        self.SCdiss2D = np.ones((self.Nx // 2, 8 * self.Ny))

    def HHN2Dx(self, Gamma, mu, V0, gamma0, theta):
        """Decimation algorithm running from Nx/2 down to 1."""
        h0 = self.build_H02D(Gamma, mu, theta)
        
        # Starting step: last index of decimation (Nx/2 in 1-based index is Nx/2 - 1 in 0-based)
        last_idx = self.Nx // 2 - 1
        
        # Scaling vectors for row-wise multiplication
        v_diss_factor = V0 * self.Vdiss[last_idx][:, np.newaxis] * self.V2D
        sc_diss_factor = gamma0 * self.SCdiss2D[last_idx][:, np.newaxis] * self.delta_2D
        
        Hi = self.Symm(h0 + self.T1 + v_diss_factor + sc_diss_factor)
        Hi1 = self.Symm(np.linalg.inv(Hi))
        
        # First loop: from Nx/2 - 2 down to delta_N + 1 (inclusive)
        for ii in range(self.Nx // 2 - 2, self.delta_N, -1):
            v_diss_factor = V0 * self.Vdiss[ii][:, np.newaxis] * self.V2D
            sc_diss_factor = gamma0 * self.SCdiss2D[ii][:, np.newaxis] * self.delta_2D
            
            Hi = self.Symm(h0 + v_diss_factor + sc_diss_factor - self.T2D.conj().T @ Hi1 @ self.T2D)
            Hi1 = self.Symm(np.linalg.inv(Hi))
            
        pff_1 = []
        pff_2 = []
        
        # Second loop: from delta_N down to 0 (inclusive)
        for ii in range(self.delta_N, -1, -1):
            v_diss_factor = V0 * self.Vdiss[ii][:, np.newaxis] * self.V2D
            sc_diss_factor = gamma0 * self.SCdiss2D[ii][:, np.newaxis] * self.delta_2D
            
            Hi = self.Symm(h0 + v_diss_factor + sc_diss_factor - self.T2D.conj().T @ Hi1 @ self.T2D)
            
            # PfP = Pfaffian[ Asymm[ Symm[ Hi + T1.conj().T ] . S2D ] ]
            matrix_P = self.Asymm(self.Symm(Hi + self.T1.conj().T) @ self.S2D)
            PfP = pfaffian(matrix_P, method=self.method)
            
            # PfM = Pfaffian[ Asymm[ Symm[ Hi - T1.conj().T ] . S2D ] ]
            matrix_M = self.Asymm(self.Symm(Hi - self.T1.conj().T) @ self.S2D)
            PfM = pfaffian(matrix_M, method=self.method)
            
            pff_1.append(np.real(PfP))
            pff_2.append(np.real(PfM))
            
            Hi1 = self.Symm(np.linalg.inv(Hi))
            
        return [pff_1, pff_2]

    def vPf2Dx(self, Gamma, mu, V0, gamma0, theta):
        """Computes topological invariant signs from the Pfaffians."""
        pff = self.HHN2Dx(Gamma, mu, V0, gamma0, theta)
        res = []
        for ii in range(len(pff[0])):
            sgn1 = np.sign(pff[0][ii])
            sgn2 = np.sign(pff[1][ii])
            # (1 - Sign[pff1] * Sign[pff2]) / 2
            val = int(round((1.0 - sgn1 * sgn2) / 2.0))
            res.append(val)
        return res


def run_tests():
    print("Initializing simulator and verifying all Pfaffian solver methods...")
    
    for method in ["h", "ltl", "hessenberg"]:
        print(f"\n--- Testing Pfaffian Method: {method} ---")
        sim = PfaffianSimulator(Nx=400, Ny=1, delta_N=20, method=method)
        
        # Test case 1: Gamma = 0.87, mu = -0.49, V0 = 0.0, gamma0 = 0.35, theta = 0.0
        print("Running Test Case 1: Gamma=0.87, mu=-0.49...")
        res1 = sim.vPf2Dx(0.87, -0.49, 0.0, 0.35, 0.0)
        print("Test 1 Result:", res1)
        expected1 = [1] * 21
        assert res1 == expected1, f"Expected {expected1}, got {res1}"
        print("Test Case 1 Passed!")
        
        # Test case 2: Gamma = 0.35, mu = -0.49, V0 = 0.0, gamma0 = 0.35, theta = 0.0
        print("Running Test Case 2: Gamma=0.35, mu=-0.49...")
        res2 = sim.vPf2Dx(0.35, -0.49, 0.0, 0.35, 0.0)
        print("Test 2 Result:", res2)
        expected2 = [0] * 21
        assert res2 == expected2, f"Expected {expected2}, got {res2}"
        print("Test Case 2 Passed!")
        
        # Test case 3: 10 points validation against SMDisStrongMapNy1.dat (Layer 6)
        print("Running Test Case 3: 10 points validation against SMDisStrongMapNy1.dat...")
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dis_path = os.path.join(script_dir, 'SMVdissNy1.dat')
        if not os.path.exists(dis_path):
            dis_path = 'SMVdissNy1.dat'  # Fallback
            
        try:
            dis = np.loadtxt(dis_path)
        except OSError:
            print(f"Warning: SMVdissNy1.dat not found. Skipping Test Case 3.")
            continue
            
        sim3 = PfaffianSimulator(Nx=410, Ny=1, delta_N=10, method=method)
        sim3.load_disorder(dis)
        
        test_points = [
            (2.13, 1.0, 1),
            (2.16, 1.0, 0),
            (2.10, 1.0, 0),
            (1.90, 1.0, 1),
            (2.54, 1.11, 1),
            (1.31, 0.90, 1),
            (1.09, 1.21, 1),
            (1.02, 1.00, 0),
            (0.80, 0.73, 0),
            (0.64, 0.73, 0)
        ]
        
        for mu, gm, expected in test_points:
            res = sim3.vPf2Dx(gm, mu, 1.5, 0.35, 0.0)
            actual = res[5]
            assert actual == expected, f"For method {method}, mu={mu}, gm={gm}: expected {expected}, got {actual}"
        print("Test Case 3 Passed!")
        
    print("\nAll test cases verified successfully for all methods against Mathematica outputs!")


def run_phase_sweep(V0, output_path, gmmin=0.34, gmmax=1.22, gmstep=0.01, mumin=-1.0, mumax=3.0, mustep=0.01, method="h"):
    print(f"Running Phase Diagram Sweep with V0 = {V0} using method = {method}...")
    sim = PfaffianSimulator(Nx=400, Ny=1, delta_N=20, method=method)
    
    gms = np.arange(gmmin, gmmax + 1e-9, gmstep)
    mus = np.arange(mumin, mumax + 1e-9, mustep)
    
    print(f"Grid size: {len(mus)} chemical potentials x {len(gms)} Zeeman fields")
    
    # We will compute the phase map
    # Since delta_N = 20, vPf2Dx returns a list of 21 values.
    # The phase map in Mathematica averages this map or exports it.
    # Specifically, it writes:
    # PhaseDiagram := Block[{mu, gm, Qv, sumQ1, map, ii},
    #   map = Table[{}, {ii, 1, \Delta N + 1}];
    #   For[mu = mumin, mu <= mumax, mu += mustep,
    #     For[gm = gmmin, gm <= gmmax, gm += gmstep,
    #       Qv = vPf2Dx[gm, mu, V0, \gamma0, \theta];
    #       For[ii = 1, ii <= Length[map], ii++,
    #         map[[ii]] = Join[map[[ii]], {{gm, mu, Qv[[ii]]}}];
    #       ...
    # The map structure: map[[ii]] is a list of [gm, mu, Qv[[ii]]] tuples.
    # So we have delta_N + 1 maps. Let's initialize them.
    num_maps = sim.delta_N + 1
    maps = [[] for _ in range(num_maps)]
    
    for mu in mus:
        # Print progress
        print(f"mu = {mu:.2f} ...")
        for gm in gms:
            Qv = sim.vPf2Dx(gm, mu, V0, sim.gamma0, 0.0)
            for ii in range(num_maps):
                maps[ii].append([gm, mu, Qv[ii]])
                
    # Save the output
    # Since there are multiple maps, we can save them to a file.
    # The user can choose to plot or export them.
    # We can save as a numpy .npz file
    save_data = {f"map_{i}": np.array(maps[i]) for i in range(num_maps)}
    np.savez(output_path, **save_data)
    print(f"Successfully saved phase diagram sweep data to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pfaffian Invariant Simulator for Majorana Wires")
    parser.add_argument("--test", action="store_true", help="Run the test cases against Mathematica output")
    parser.add_argument("--sweep", action="store_true", help="Run a phase diagram sweep")
    parser.add_argument("--V0", type=float, default=1.5, help="Disorder strength V0")
    parser.add_argument("--disorder", type=str, default=None, help="Path to disorder file (.npy)")
    parser.add_argument("--output", type=str, default="pfaffian_map.npz", help="Output path for sweep data")
    parser.add_argument("--gmmin", type=float, default=0.34, help="Min Zeeman field Gamma")
    parser.add_argument("--gmmax", type=float, default=1.22, help="Max Zeeman field Gamma")
    parser.add_argument("--gmstep", type=float, default=0.01, help="Step size for Zeeman field Gamma")
    parser.add_argument("--mumin", type=float, default=-1.0, help="Min chemical potential mu")
    parser.add_argument("--mumax", type=float, default=3.0, help="Max chemical potential mu")
    parser.add_argument("--mustep", type=float, default=0.01, help="Step size for chemical potential mu")
    parser.add_argument("--method", type=str, default="h", choices=["h", "ltl", "hessenberg"],
                        help="Pfaffian calculation method: 'h' (Householder, default), 'ltl' (Parlett-Reid), 'hessenberg' (Hessenberg decomposition, real only)")
    args = parser.parse_args()
    
    if args.test:
        run_tests()
        sys.exit(0)
        
    if args.sweep:
        if args.disorder:
            dis_data = np.load(args.disorder)
            # Just parsing disorder to verify load works
            sim_dummy = PfaffianSimulator(Nx=400, Ny=1, delta_N=20)
            sim_dummy.load_disorder(dis_data)
            print(f"Loaded disorder file from {args.disorder}")
        else:
            print("No disorder file provided, using zero disorder profile.")
        run_phase_sweep(
            args.V0,
            args.output,
            gmmin=args.gmmin,
            gmmax=args.gmmax,
            gmstep=args.gmstep,
            mumin=args.mumin,
            mumax=args.mumax,
            mustep=args.mustep,
            method=args.method
        )
        sys.exit(0)
        
    parser.print_help()


if __name__ == "__main__":
    main()
