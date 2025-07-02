#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def assemble_beam(n_el, L_total, E, I):
    n_node = n_el + 1
    ndof = 2 * n_node
    K = np.zeros((ndof, ndof))
    L = L_total / n_el
    # elemental stiffness (4×4)
    ke_base = (E*I / L**3) * np.array([
        [ 12,   6*L,  -12,   6*L],
        [6*L, 4*L*L, -6*L, 2*L*L],
        [-12, -6*L,   12,  -6*L],
        [6*L, 2*L*L, -6*L, 4*L*L]
    ])
    for e in range(n_el):
        gdof = [2*e, 2*e+1, 2*e+2, 2*e+3]
        for i in range(4):
            for j in range(4):
                K[gdof[i], gdof[j]] += ke_base[i,j]
    return K, L

def uniform_load_vector(n_el, L_total, w):
    # Equivalent nodal loads for uniform load on each element:
    # [F_i, M_i, F_j, M_j] = wL/2 [1, L/6, 1, -L/6]
    n_node = n_el + 1
    ndof = 2 * n_node
    F = np.zeros(ndof)
    L = L_total / n_el
    # elemental load vector
    fe_base = (w * L / 2) * np.array([1, L/6, 1, -L/6])
    for e in range(n_el):
        gdof = [2*e, 2*e+1, 2*e+2, 2*e+3]
        for i in range(4):
            F[gdof[i]] += fe_base[i]
    return F

def apply_bc(K, F, fixed_dofs):
    free = np.setdiff1d(np.arange(K.shape[0]), fixed_dofs)
    Kff = K[np.ix_(free, free)]
    Ff  = F[free]
    Df  = np.linalg.solve(Kff, Ff)
    D   = np.zeros(K.shape[0])
    D[free] = Df
    return D

def exact_cantilever(L, w, E, I, x):
    # Analytical deflection v(x) and slope theta(x) for w uniformly loaded cantilever
    # v(x) = (w/(24EI)) * x^2 * (6L^2 - 4Lx + x^2)
    # theta(x) = dv/dx = (w/(6EI)) * x * (3L^2 - 3Lx + x^2)
    v = (w / (24*E*I)) * x**2 * (6*L**2 - 4*L*x + x**2)
    theta = (w / (6*E*I)) * x * (3*L**2 - 3*L*x + x**2)
    return v, theta

def main():
    # Parameters
    L_total = 6.0       # m
    n_el    = 8
    E       = 210e9     # Pa
    I       = 4e-4      # m^4
    w       = 5e3       # N/m  (i.e. 5 kN/m)

    # 1. Assemble
    K, Le = assemble_beam(n_el, L_total, E, I)

    # 2. Load vector
    F = uniform_load_vector(n_el, L_total, w)

    # 3. Apply BCs (cantilever fixed at left end: dof 0 & 1 fixed)
    fixed = [0, 1]
    D = apply_bc(K, F, fixed)

    # 4. Extract nodal deflections & slopes
    n_node = n_el + 1
    results = []
    for i in range(n_node):
        x = i * Le
        v_num   = D[2*i]
        th_num  = D[2*i+1]
        v_ex, th_ex = exact_cantilever(L_total, w, E, I, x)
        results.append([i, x, v_num, v_ex, th_num, th_ex])

    # 5. Print comparison table
    headers = ["Node","x (m)","v_num (m)","v_ex (m)","θ_num (rad)","θ_ex (rad)"]
    print(tabulate(results, headers=headers, floatfmt=".6e", tablefmt="github"))

    # 6. Plot deflection shapes
    xs = np.linspace(0, L_total, 200)
    vs_ex, ths_ex = exact_cantilever(L_total, w, E, I, xs)
    vs_num = []
    for x in xs:
        # find element and local ξ to interpolate D (omitted for brevity)
        vs_num.append(np.interp(x, [node[1] for node in results], [node[2] for node in results]))
    plt.plot(xs, vs_ex, 'k-', label="Exact")
    plt.plot(xs, vs_num, 'r--', label="FEM")
    plt.legend(); plt.xlabel("x [m]"); plt.ylabel("v [m]")
    plt.title("Cantilever Deflection: Exact vs. FEM")
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    main()
