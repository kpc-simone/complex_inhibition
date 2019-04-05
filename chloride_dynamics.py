import numpy as np
import nengo
from nengo.processes import WhiteSignal as white

class ControllableSynapse(object):
    def __init__(self, dimensions, timesteps):
        #dimensions of internal state variable
        self.history = np.zeros((dimensions, timesteps))
        self.filtered = np.zeros((dimensions, timesteps))

    def step(self, t, x):
        #roll buffer, update last element, return first element
        self.history = np.roll(self.history, shift=-1, axis=1)
        self.history[:,-1] = x[0]*x[1]
        return self.history[:, 0]
        
    def integrate(self, t, x):
        self.filtered = np.roll(self.filtered, shift=-1, axis=1)
        self.filtered[:, -1] = 0.99 * self.filtered[:, -1] + 0.01 * x
        return self.filtered[:, 0]

def thresh(t,x):
    a = 0.9
    return (a - x) / a

dt = 0.001
n_neurons=100

ctr_synapse = ControllableSynapse(1, timesteps = int(dt / dt))
cl_buildup = ControllableSynapse(1, timesteps = int(0.01 / dt))
inh_history = ControllableSynapse(1, timesteps = int(0.01 / dt))

model = nengo.Network()
with model:
    
    #network inputs
    inp1 = nengo.Node(0)
    inp2 = nengo.Node(0)
    
    #intermediate nodes - for calculation
    complex_proj = nengo.Node(ctr_synapse.step, size_in=2, size_out=1)
    compute_proj = nengo.Node(thresh,size_in=1, size_out=1)
    chloride = nengo.Node(cl_buildup.integrate, size_in=1, size_out=1)
    inhibition_history = nengo.Node(inh_history.integrate, size_in=1, size_out=1)
    
    #populations
    pop1 = nengo.Ensemble(n_neurons, dimensions=1)   
    pop2 = nengo.Ensemble(n_neurons, dimensions=1)
    
    #connections
    nengo.Connection(inp1, pop1)
    nengo.Connection(complex_proj,inhibition_history)
    nengo.Connection(inhibition_history,chloride)
    nengo.Connection(inp2, pop2)
    nengo.Connection(chloride, compute_proj)
    nengo.Connection(compute_proj, complex_proj[1])
    nengo.Connection(pop2, chloride)
    
    #projections
    nengo.Connection(pop1, complex_proj[0])
    nengo.Connection(complex_proj,pop2.neurons,transform=[[-1]] * n_neurons)