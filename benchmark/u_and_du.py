import sys
sys.path.append('..')
import numpy as np
import floq.systems.spins as spins
import timeit


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


def time_u(get_u, hf, params):
    time = min(timeit.Timer(wrapper(get_u, hf, params)).repeat(5, 20))/20
    return " U: " + str(round(time*1000, 3)) + " ms per execution"


def time_du(func, hf, dhf, params):
    time = min(timeit.Timer(wrapper(func, hf, dhf, params)).repeat(3, 1))
    return "dU: " + str(round(time, 3)) + " s per execution"


ncomp = 6
n = 1
freqs = 1.1*np.ones(n)
amps = 1.0*np.ones(n)
s = spins.SpinEnsemble(n, ncomp, 1.5, freqs, amps)

controls = 0.5*np.ones(2*ncomp)
s.set_nz(controls, 1.5)
system = s.get_systems(controls, 1.5)[0]

print "---- Current version (Numba)"
import floq.core.evolution as ev

print time_u(ev.get_u, system.hf, system.params)
print time_du(ev.get_u_and_du, system.hf, system.dhf, system.params)


print "---- Better algorithm for dU"
import floq.museum.p4.evolution as ev

print time_u(ev.get_u, system.hf, system.params)
print time_du(ev.get_u_and_du, system.hf, system.dhf, system.params)


print "---- Use less Python"
import floq.museum.p3.evolution as ev

print time_u(ev.get_u, system.hf, system.params)
print time_du(ev.get_u_and_du, system.hf, system.dhf, system.params)


print "---- Use sparse matrix library"
import floq.museum.p2.evolution as ev

print time_u(ev.get_u, system.hf, system.params)
print time_du(ev.get_u_and_du, system.hf, system.dhf, system.params)
print "---- Baseline"
import floq.museum.p1.evolution as ev

print time_u(ev.get_u, system.hf, system.params)
print time_du(ev.get_u_and_du, system.hf, system.dhf, system.params)