import numpy as np

def hf(g, e1, e2):
    hf = np.zeros([3, 2, 2])
    hf[0] = np.array([[0, 0], [g, 0]])
    hf[1] = np.array([[e1, 0], [0, e2]])
    hf[2] = np.array([[0, g], [0, 0]])
    return hf

def u(g, e1, e2, w, t):
    w12 = e1-e2
    k = np.sqrt(g**2 + 0.25*(w+w12)**2)
    W = w+w12

    # Coefficients of U(t) in the interaction picture
    c1a = (1/k)*np.exp(0.5j*t*W)*(k*np.cos(k*t)+(-0.5j*W)*np.sin(k*t))
    c2a = (g/(1j*k))*np.exp(-0.5j*t*W)*np.sin(k*t)
    c1b = c2a*np.exp(1j*t*W)
    c2b = np.conj(c1a)

    # Factors to transform into Schrodinger picture
    exp1 = np.exp(-1j*e1*t)
    exp2 = np.exp(-1j*e2*t)

    return np.array([[exp1*c1a, exp1*c1b], [exp2*c2a, exp2*c2b]])
