import numpy as np
import floq

def hamiltonian(ncomp, freq, controls):
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
    hf[ncomp] = np.array([[0.5 * freq, 0.0],
                          [0.0, -0.5 * freq]], dtype=np.complex128)
    return hf

def dhamiltonian(ncomp):
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

def spin(n_components, amplitude, split, **kwargs):
    def _hamiltonian(controls):
        return hamiltonian(n_components, split, amplitude * controls)
    _dhamiltonian = dhamiltonian(n_components)
    return floq.System(_hamiltonian, _dhamiltonian, **kwargs)

class SpinEnsemble(floq.system.EnsembleBase):
    """
    A system of n non-interacting spins, where each spin is described by the
    Hamiltonian
        H(t) = w/2 s_z + 1/2 sum_k (a_k s_x + b_k s_y) sin(k omega t).

    Commonly, the values for w will be set slightly different for each spin, and
    a_k and b_k will be multiplied by some attenuation factor for each spin.
    Fidelities etc. will then be computed as ensemble averages."""
    def __init__(self, n_systems, n_components, omega, frequencies, amplitudes):
        """Initialise a SpinEnsemble instance with
          - n: number of spins
          - ncomp: number of components in the control pulse
           -> hf will have nc = 2*comp+1 components
          - omega: base frequency of control pulse
          - freqs: vector of n frequencies
          - amps: vector of n amplitudes."""
        self.n_systems = n_systems
        self.n_components = n_components
        self.frequency = omega
        self.frequencies = frequencies
        self.amplitudes = amplitudes
        self.__systems = [spin(n_components, a, f, frequency=omega, n_zones=31)\
                          for a, f in zip(amplitudes, frequencies)]

    @property
    def systems(self):
        return self.__systems

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
        super().__init__(n, ncomp, omega, freqs, amps)

if __name__ == '__main__':
    n_components = 3
    spin_system = spin(n_components, 1.0, 0.01, frequency=2*np.pi)
    control = np.random.rand(2 * n_components)
    print(spin_system.u(0.5, control))
    print("\n")
    print(spin_system.du_dcontrols(0.5, control))
