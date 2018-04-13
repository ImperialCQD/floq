import numpy as np
from ..systems.ensemble import EnsembleBase
from ..systems.parametric_system import ParametricSystemBase

class SpinEnsemble(EnsembleBase):
    """A system of n non-interacting spins, where each spin is described by the
    Hamiltonian
        H(t) = w/2 s_z + 1/2 sum_k (a_k s_x + b_k s_y) sin(k omega t).

    Commonly, the values for w will be set slightly different for each spin, and
    a_k and b_k will be multiplied by some attenuation factor for each spin.
    Fidelities etc. will then be computed as ensemble averages."""
    def __init__(self, n, ncomp, omega, freqs, amps):
        """Initialise a SpinEnsemble instance with
          - n: number of spins
          - ncomp: number of components in the control pulse
           -> hf will have nc = 2*comp+1 components
          - omega: base frequency of control pulse
          - freqs: vector of n frequencies
          - amps: vector of n amplitudes."""
        self.n = n
        self.ncomp = ncomp
        self.omega = omega
        self.freqs = freqs
        self.amps = amps
        self.max_amp = np.argmax(amps)
        self.np = 2 * ncomp  # number of control parameters
        self.nc = 2 * ncomp+1
        self.nz = 3 * np.ones(self.n, dtype=int)
        self._systems = [SpinSystem(ncomp, amps[i], freqs[i], omega)\
                         for i in range(n)]
    @property
    def systems(self):
        return self._systems

class RandomisedSpinEnsemble(SpinEnsemble):
    """A system of n non-interacting spins, where each spin is described by the
    Hamiltonian
        H(t) = w/2 s_z + 1/2 sum_k (a_k s_x + b_k s_y) sin(k omega t).

    The n spins will be instantiated with randomised detunings and
    amplitudes, distributed as follows:
    - frequencies: normal distribution with given FWHM (2 sqrt(2 ln 2) sigma)
                   and mean 0,
    - amplitudes: Uniform distribution with given width around 1.0."""
    def __init__(self, n, ncomp, omega, fwhm, amp_width):
        """Initialise a SpinEnsemble instance with
            - n: number of spins
            - ncomp: number of components in the control pulse
             -> hf will have nc = 2*comp+1 components
            - omega: base frequency of control pulse
            - fwhm: full width at half-max of the Gaussian distribution of
                    detunings
            - amp_width: amplitudes will be drawn from a uniform distribution
                         around 1 with this width."""
        sigma = fwhm / 2.35482
        freqs = np.random.normal(loc=0.0, scale=sigma, size=n)
        amps = amp_width * (2 * np.random.rand(n) - 1) + np.ones(n)
        super(RandomisedSpinEnsemble, self)\
            .__init__(n, ncomp, omega, freqs, amps)

class SpinSystem(ParametricSystemBase):
    """Describes a single spin with an amplitude (amp) that attenuates the
    control pulse, and a detuning (freq), controlled by a pulse with ncomp
    components and a carrier frequency of omega.

    The Hamiltonian is
        H(t) = w/2 s_z + 1/2 sum_k (a_k s_x + b_k s_y) sin(k omega t),
    with k going over ncomp components of the pulse.

    ncomp implies 2 * ncomp = np control parameters, and np + 1 non-zero Fourier
    components in Hf."""
    def __init__(self, ncomp, amp, freq, omega):
        """Initialise a SpinSystem."""
        super(SpinSystem, self).__init__()
        self.ncomp = ncomp
        self.amp = amp
        self.freq = freq
        self.omega = omega
        self.nz = 3
        self.dhf = dhf(ncomp) # independent of controls!

    def _hf(self, controls):
        """Compute hf, given 2*ncomp control parameters, ordered as follows:
            [a1, b1, a2, b2, ... a_nc, b_nc]."""
        return hf(self.ncomp, self.freq, self.amp * controls)

    def _dhf(self, controls):
        # dHf is independent of the controls, return cached value
        return self.dhf

def hf(ncomp, freq, controls):
    """Assemble hf for one spin, given a detuning freq, the control amplitudes
    controls, and ncomp components of the control pulse."""
    nc = 2 * ncomp + 1  # number of components in hf
    hf = np.zeros((nc, 2, 2), dtype=np.complex128)
    for k in range(ncomp):
        # Controls are ordered in reverse compared to how they are placed in hf
        a = controls[-2 * k - 2]
        b = controls[-2 * k - 1]
        # The controls are placed symmetrically around
        # the centre of hf, so we can place them at the
        # same time to save us some work!
        hf[k,:,:] = np.array([[0.0, 0.25 * (1j * a + b)],
                              [0.25 * (1j * a - b), 0.0]])
        hf[-k - 1,:,:] = np.array([[0.0, -0.25 * (1j * a + b)],
                                   [0.25 * (-1j * a + b), 0.0]])
    # Set centre (with Fourier index 0)
    hf[ncomp] = np.array([[freq / 2, 0.0],
                          [0.0, -freq / 2]])
    return hf

def dhf(ncomp):
    """Assemble dhf for one spin given ncomp components in the control pulse."""
    nc = 2 * ncomp + 1
    np_ = 2 * ncomp
    dhf = np.zeros([np_, nc, 2, 2], dtype=np.complex128)
    for k in range(ncomp):
        i_a = -2 * k - 2
        i_b = -2 * k - 1
        dhf[i_a, k, :, :] = np.array([[0.0, 0.25j],
                                      [0.25j, 0.0]])
        dhf[i_a, -k-1, :, :] = np.array([[0.0, -0.25j],
                                         [-0.25j, 0.0]])
        dhf[i_b, k, :, :] = np.array([[0.0, 0.25],
                                      [-0.25, 0.0]])
        dhf[i_b, -k-1, :, :] = np.array([[0.0, -0.25],
                                         [0.25, 0.0]])
    return dhf
