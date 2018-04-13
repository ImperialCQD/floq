import logging
import multiprocessing as mp
import numpy as np
from ..optimization.fidelity import FidelityBase

class FidelityMaster(FidelityBase):
    """With a given Ensemble, and a FidelityComputer, calculate the average
    fidelity over the whole ensemble by distributing the work onto nworker
    sub-processes, which should run in parallel if there are enough cores
    available.

    Note: After use, the FidelityMaster should be forced to kill the child
    processes by calling the kill() method. It is recommended to use a new
    FidelityMaster thereafter."""
    def __init__(self, nworker, ensemble, fidelity, **params):
        super(FidelityMaster, self).__init__(ensemble)
        self.fidelities = [fidelity(sys, **params) for sys in ensemble.systems]
        self.n = len(ensemble.systems)
        self.nworker = nworker
        self.fidelities_chunked = chunks(self.fidelities, nworker)
        self._make_workers()

    def _make_workers(self):
        """Spawn the workers, set up pipes to talk to them."""
        in_pipes = [mp.Pipe() for i in range(self.nworker)]
        out_pipes = [mp.Pipe() for i in range(self.nworker)]
        self.workers = []
        logging.info('Attempting to spawn workers')
        for i in range(self.nworker):
            worker = FidelityWorker(self.fidelities_chunked[i], in_pipes[i][1],\
                                    out_pipes[i][1])
            worker.start()
            self.workers.append(worker)
        logging.info('Successfully spawned workers')
        self.ins = [in_pipes[i][0] for i in range(self.nworker)]
        self.outs = [out_pipes[i][0] for i in range(self.nworker)]

    def _f(self, controls_and_t):
        """Compute the average fidelity of the ensemble."""
        for pipe in self.ins:
            pipe.send(['f', controls_and_t])
        return np.sum([pipe.recv() for pipe in self.outs]) / self.n

    def _df(self, controls_and_t):
        """Compute the average gradient of the fidelity of the ensemble."""
        for pipe in self.ins:
            pipe.send(['df', controls_and_t])
        return np.sum([pipe.recv() for pipe in self.outs], axis=0) / self.n

    def kill(self):
        """Terminate the workers spawned."""
        for pipe in self.ins:
            pipe.send(None) # tell workers to stop run()
        for worker in self.workers:
            worker.terminate()  # shut them down

class FidelityWorker(mp.Process):
    """Wraps a list of FidelityComputers, performing their computations in a
    separate process. Communication with the 'master' process is done via Pipes.
    F and dF will be added up locally, since only averages are used in the
    optimisation. This reduces the amount of data to be communicated between
    processes.

    Note: the start() methods needs to be run before computations are
    performed."""
    def __init__(self, fids, pipe_in, pipe_out):
        super(FidelityWorker, self).__init__()
        self.pipe_in = pipe_in
        self.pipe_out = pipe_out
        self.fids = fids
        logging.info('Worker initialised with '
                     + str(len(self.fids))
                     + ' fidelities')

    def run(self):
        """When this is run, the Worker starts listening on its in_pipe, it
        stops when None is sent through the pipe."""
        msg = self.pipe_in.recv()
        while msg is not None:
            out_msg = np.sum([fid.f(msg[1]) for fid in self.fids])\
                      if msg[0] == 'f' else\
                      np.sum([fid.df(msg[1]) for fid in self.fids], axis=0)
            self.pipe_out.send(out_msg)
            msg = self.pipe_in.recv()

def chunks(l, n):
    """Split list l into n chunks as uniformly as possible."""
    k, m = divmod(len(l), n)
    return [l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
