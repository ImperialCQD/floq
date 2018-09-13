import floq
import numpy as np

def rabi(energy_split, **kwargs):
    """
    Create a 'Rabi' Hamiltonian which has a given energy splitting between
    spin-up and spin-down.  The Hamiltonian is
        H = energy_split * sigmaz
            + controls[0] * (e^{-iwt} * sigma- + e^{iwt} * sigma+),
    where `controls` is an `array_like` with one element, which is passed to the
    `floq.System` methods after the time parameter.
    """
    _hamiltonian = np.zeros((3, 2, 2), dtype=np.complex128)
    _hamiltonian[1, 0, 0] = 0.5 * energy_split
    _hamiltonian[1, 1, 1] = -0.5 * energy_split
    def hamiltonian(controls):
        _hamiltonian[0, 1, 0] = _hamiltonian[2, 0, 1] = controls[0]
        return _hamiltonian
    dhamiltonian = np.zeros((1, 3, 2, 2), dtype=np.complex128)
    dhamiltonian[0, 0, 1, 0] = dhamiltonian[0, 2, 0, 1] = 1.0
    if 'n_zones' not in kwargs:
        kwargs['n_zones'] = 11
    return floq.System(hamiltonian, dhamiltonian, **kwargs)

if __name__ == '__main__':
    # Create the system, which then takes one control
    system = rabi(10.0, frequency=2*np.pi)

    # Calculate the time-evolution operator at half a period for a control of
    # 0.1.
    print(system.u(0.5, [1.0]))

    # Calculate the derivative of the time-evolution operator at half a period
    # with respect to the control parameter.
    print(system.du_dcontrols(0.5, [1.0]))

    # Calculate the derivative of the time-evolution operator with respect to
    # time at a certain point in time.
    print(system.du_dt(1.0, [0.2]))
