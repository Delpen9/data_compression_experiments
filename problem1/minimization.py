import numpy as np

def cgsolve(A, b, tol, maxiter, verbose):
    x = np.zeros_like(b)
    r = b.copy()
    d = r.copy()
    delta = r.T @ r
    delta0 = b.T @ b
    numiter = 0
    bestx = x.copy()
    bestres = np.sqrt(delta / delta0)
    while numiter < maxiter and delta > tol**2 * delta0:
        q = A(d)
        alpha = delta / (d.T @ q)
        x += alpha * d

        if numiter % 50 == 0:
            r = b - A(x)
        else:
            r -= alpha * q

        delta_old = delta
        delta = r.T @ r
        beta = delta / delta_old
        d = r + beta * d
        numiter += 1
        res = np.sqrt(delta / delta0)
        
        if res < bestres:
            bestx = x.copy()
            bestres = res

        if verbose and numiter % 10 == 0:
            print(f"CG Iteration = {numiter}, Residual = {res:.3e}")
    return bestx, bestres, numiter

def l1eq_pd(x0, A, At, b, pdtol=1e-3, pdmaxiter=50, cgtol=1e-8, cgmaxiter=200):
    largescale = callable(A)

    N = len(x0)

    alpha = 0.01
    beta = 0.5
    mu = 10

    gradf0 = np.concatenate([np.zeros(N), np.ones(N)],  axis = 0)

    x = x0.copy()
    u = 1.01 * np.max(np.abs(x)) * np.ones(N) + 1e-2

    fu1 = x - u
    fu2 = -x - u

    lamu1 = -1 / fu1
    lamu2 = -1 / fu2
    
    if largescale:
        v = -A(lamu1 - lamu2)
        Atv = At(v)
        rpri = A(x) - b
    else:
        v = -A @ (lamu1 - lamu2)
        Atv = A.T @ v
        rpri = A @ x - b

    sdg = -(fu1.T @ lamu1 + fu2.T @ lamu2)
    tau = mu * 2 * N / sdg

    rcent = np.concatenate([-lamu1 * fu1, -lamu2 * fu2], axis = 0) - 1 / tau
    rdual = gradf0 + np.concatenate([lamu1 - lamu2, -lamu1 - lamu2],  axis = 0) + np.concatenate([Atv, np.zeros(N)],  axis = 0)
    resnorm = np.linalg.norm(np.concatenate([rdual, rcent, rpri],  axis = 0))

    pditer = 0
    done = (sdg < pdtol) or (pditer >= pdmaxiter)
    while not done:
        pditer += 1

        w1 = -1 / tau * (-1 / fu1 + 1 / fu2) - Atv
        w2 = -1 - 1 / tau * (1 / fu1 + 1 / fu2)
        w3 = -rpri

        sig1 = -lamu1 / fu1 - lamu2
