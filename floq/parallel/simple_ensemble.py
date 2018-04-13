# You got nothing to lose but your chains!
import multiprocessing as mp
import numpy as np
from ..optimization.fidelity import FidelityBase

def run_fid(pair):
    fid, ctrl = pair
    fid.f(ctrl)
    return fid

def run_dfid(pair):
    fid, ctrl = pair
    fid.df(ctrl)
    return fid

class ParallelEnsembleFidelity(FidelityBase):
    """With a given Ensemble, and a FidelityComputer, calculate the average
    fidelity over the whole ensemble."""
    def __init__(self, ensemble, fidelity, **params):
        super(ParallelEnsembleFidelity, self).__init__(ensemble)
        self.fidelities = [fidelity(sys, **params) for sys in ensemble.systems]
        self.pool = mp.Pool()

    def f(self, controls_and_t):
        self.dispatch_f_to_pool(controls_and_t)
        return np.mean([fid.f(controls_and_t) for fid in self.fidelities])

    def df(self, controls_and_t):
        self.dispatch_df_to_pool(controls_and_t)
        return np.mean([fid.df(controls_and_t) for fid in self.fidelities],
                       axis=0)

    def dispatch_f_to_pool(self, controls_and_t):
        items = [(fid, controls_and_t) for fid in self.fidelities]
        self.fidelities = self.pool.map(run_fid, items)

    def dispatch_df_to_pool(self, controls_and_t):
        items = [(fid, controls_and_t) for fid in self.fidelities]
        self.fidelities = self.pool.map(run_dfid, items)
