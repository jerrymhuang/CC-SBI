from math import prod
from tqdm.auto import tqdm


class TqdmSimulator:
    def __init__(self, base_simulator, total, desc="Simulating"):
        self.base = base_simulator
        self.pbar = tqdm(total=total, desc=desc)

    def sample(self, batch_shape=None, **kwargs):
        # forward to the real simulator
        out = self.base.sample(batch_shape, **kwargs)

        # how many did we just produce?
        produced = 1
        if isinstance(batch_shape, int):
            produced = batch_shape
        elif isinstance(batch_shape, tuple):
            produced = prod(batch_shape)
        else:
            # fall back: infer from output (dict of arrays is common in BayesFlow)
            v = next(iter(out.values())) if isinstance(out, dict) and out else None
            if hasattr(v, "shape") and len(v.shape) > 0:
                produced = v.shape[0]

        self.pbar.update(produced)
        return out

    # proxy any other attributes BayesFlow might expect (e.g., config)
    def __getattr__(self, name):
        return getattr(self.base, name)

    def close(self):
        self.pbar.close()


def simulate_with_tqdm(workflow, n, desc="Simulating"):
    wrapped = TqdmSimulator(workflow.simulator, total=n, desc=desc)
    try:
        workflow.simulator = wrapped
        return workflow.simulate(n)            # BayesFlow passes batch_shape=n
    finally:
        wrapped.close()
        workflow.simulator = wrapped.base      # restore
