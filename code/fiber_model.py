"""Polygerinos et al., T-RO 2015 -- analytical model of a soft fiber-reinforced bending actuator.

Vectorised version of the course notebook (Assignment.ipynb): same equations, but
Gauss-Legendre quadrature (so Monte Carlo over thousands of geometries is fast) and
selectable wall materials.  Units: mm, MPa, N.  Pressures returned in kPa.

Moment balance (paper Eq. 14, 19, 21):
    M(theta) = bottom-wall term + 2 * int_0^t int_0^{pi/2} s(lambda) * (...) dphi dtau
    P        = M(theta) / kappa,   kappa = (4 a^3 + 3 pi a^2 b) / 6
with axial stretch lambda = (radius) * theta / L + 1.

Material stress s(lambda) (nominal axial stress):
    Neo-Hookean (paper form):   s = mu * (lambda - lambda^-3)
    Yeoh, plane-strain form:    s = 2 (lambda - lambda^-3) * dW/dI1,
                                dW/dI1 = C1 + 2 C2 (I1 - 3),  I1 = lambda^2 + 1 + lambda^-2
      (the lambda - lambda^-3 form of the paper is the plane-strain one; this choice
       reproduces the Quad-Chart value 191.1 kPa for C1=0.11, C2=0.02 MPa.)
"""
import numpy as np
from numpy.polynomial.legendre import leggauss

BASE = dict(a=8.0, b=2.0, t=2.0, L=160.0)      # mm, baseline actuator
MU_FIT = 0.314                                  # MPa, fitted lumped modulus in the paper
C1_YEOH, C2_YEOH = 0.11, 0.02                   # MPa, Yeoh constants (Quad Chart)

_X, _W = leggauss(32)                           # Gauss-Legendre nodes/weights on [-1, 1]


def set_quadrature(n):
    """Change the Gauss-Legendre order (default 32). Lower orders are faster for Monte Carlo."""
    global _X, _W
    _X, _W = leggauss(n)


# ---------------------------------------------------------------- materials
def neo_hookean(mu):
    return lambda lam: mu * (lam - lam ** -3)


def yeoh(c1=C1_YEOH, c2=C2_YEOH, scale=1.0):
    """Yeoh, plane-strain form; `scale` multiplies C1 and C2 together."""
    def s(lam):
        i1 = lam ** 2 + 1.0 + lam ** -2
        return 2.0 * (lam - lam ** -3) * scale * (c1 + 2.0 * c2 * (i1 - 3.0))
    return s


# ---------------------------------------------------------------- core model
def _col(x):
    return np.atleast_1d(np.asarray(x, float)).reshape(-1, 1, 1)


def _top_wall(stress, theta, a, b, L, t0, t1):
    """2 * int_{t0}^{t1} int_0^{pi/2} s(lam) ((a+tau)^2 sin(phi) + b (a+tau)) dphi dtau"""
    a, b, L, t0, t1 = map(_col, (a, b, L, t0, t1))
    tau = 0.5 * (t1 - t0) * _X.reshape(1, -1, 1) + 0.5 * (t1 + t0)
    wt = 0.5 * (t1 - t0) * _W.reshape(1, -1, 1)
    phi = (0.25 * np.pi * (_X + 1.0)).reshape(1, 1, -1)
    wp = (0.25 * np.pi * _W).reshape(1, 1, -1)
    lam = (b + np.sin(phi) * (a + tau)) * theta / L + 1.0
    f = stress(lam) * ((a + tau) ** 2 * np.sin(phi) + b * (a + tau))
    return 2.0 * np.sum(wt * wp * f, axis=(1, 2))


def _bottom_wall(stress, theta, a, b, L, t):
    """int_0^b s(lam) * 2 (a+t) beta dbeta"""
    a, b, L, t = (np.atleast_1d(np.asarray(v, float)).reshape(-1, 1) for v in (a, b, L, t))
    be = 0.5 * b * (_X.reshape(1, -1) + 1.0)
    wb = 0.5 * b * _W.reshape(1, -1)
    lam = be * theta / L + 1.0
    return np.sum(wb * stress(lam) * 2.0 * (a + t) * be, axis=1)


def pressure(theta, a=BASE["a"], b=BASE["b"], t=BASE["t"], L=BASE["L"],
             wall=None, sheath=None, sheath_t=1.0):
    """Pressure (kPa) needed for bending angle `theta` (rad).

    a, b, t, L may be scalars or equal-length arrays (broadcast together).
    wall   : stress function, default Neo-Hookean with the fitted modulus 0.314 MPa.
    sheath : optional stress function of an *added* outer layer of thickness `sheath_t`
             on the top (curved) wall, tau in [t, t + sheath_t].  Modelling choice that
             reproduces the Quad-Chart sheath numbers (204.3 / 239.7 kPa); it is an
             assumption, not something taken from the paper.
    """
    wall = wall or neo_hookean(MU_FIT)
    a, b, t, L = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, float)) for v in (a, b, t, L)))
    M = _bottom_wall(wall, theta, a, b, L, t) + _top_wall(wall, theta, a, b, L, 0.0, t)
    if sheath is not None:
        M = M + _top_wall(sheath, theta, a, b, L, t, t + sheath_t)
    kappa = (4.0 * a ** 3 + 3.0 * np.pi * a ** 2 * b) / 6.0
    return 1e3 * M / kappa


def p_per_mu(theta):
    """Pressure per unit Neo-Hookean modulus at the nominal geometry (kPa/MPa).
    Used to turn a pressure difference into an 'equivalent mu_eff shift'."""
    return float(pressure(theta, wall=neo_hookean(1.0))[0])


def equiv_mu(dP_kpa, theta):
    """Neo-Hookean modulus change (MPa) that produces the same pressure change at `theta`."""
    return np.asarray(dP_kpa) / p_per_mu(theta)


# ---------------------------------------------------------------- reference (notebook) implementation
def pressure_quad_reference(theta, a=8.0, b=2.0, t=2.0, L=160.0, mu_bar=MU_FIT):
    """Nested scipy.quad version copied from the notebook, for cross-checking only."""
    from scipy import integrate
    s = lambda lam: mu_bar * (lam - lam ** -3)
    kappa = (4 * a ** 3 + 3 * np.pi * a ** 2 * b) / 6.0
    bottom = integrate.quad(lambda be: s(be * theta / L + 1) * 2 * (a + t) * be, 0, b)[0]

    def inner(tau):
        f = lambda ph: s((b + np.sin(ph) * (a + tau)) * theta / L + 1) * \
                       ((a + tau) ** 2 * np.sin(ph) + b * (a + tau))
        return integrate.quad(f, 0, np.pi / 2)[0]

    top = 2 * integrate.quad(inner, 0, t)[0]
    return 1e3 * (bottom + top) / kappa
