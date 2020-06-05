import numpy as np
from scipy.special import expit   # sigmoid

class RBM():
    def __init__(self, n_vis=28*28, n_hid=100):
        self.n_vis = n_vis
        self.n_hid = n_hid
        # Parameters
        self.W = 0.1 * np.random.randn(n_vis, n_hid)
        self.vbias = np.zeros(n_vis)
        self.hbias = -4.0 * np.ones(n_hid)
        # Gradients
        self.W_grad = np.zeros(self.W.shape)
        self.vbias_grad = np.zeros(n_vis)
        self.hbias_grad = np.zeros(n_hid)
        # Velocities - for momentum
        self.W_vel = np.zeros(self.W.shape)
        self.vbias_vel = np.zeros(n_vis)
        self.hbias_vel = np.zeros(n_hid)
    
    def h_given_v(self, v):
        '''
        input:
            - v: (batch_size, n_vis)
        output:
            - p(H|v) = sigmoid(W^Tv + hbias): (batch_size, n_hid)
            - samples from p(H|v): (batch_size, n_hid)
        '''
        p = expit(np.matmul(v, self.W) + self.hbias)
        return (p, np.random.binomial(1, p=p))
    
    def v_given_h(self, h):
        '''
        input:
            - h: (batch_size, n_hid)
        output:
            - p(V|h) = sigmoid(Wh + vbias): (batch_size, n_vis)
            - samples from p(V|h): (batch_size, n_vis)
        '''
        p = expit(np.matmul(h, self.W.T) + self.vbias)
        return (p, np.random.binomial(1, p=p))
    
    def compute_error_and_grads(self, batch, burn_in=0, num_steps=1, method="cd"):
        '''
        Function to compute the gradient of parameters and store in param_grad variables
        and reconstruction error.
        input:
            - batch: (batch_size, n_vis)
            - burn_in: Number of burn in steps for Gibbs sampling
            - num_steps: Number of steps for Gibbs sampling chain to run
            - method: Method for computing gradients. Available options:
                    - "cd": Contrastive Divergence
        output:
            - recon_error: Reconstruction error

        TODO:
        	- Implement PCD and FPCD.
        	- Use Gibbs sampling averaging, instead of taking just last value.
        '''
        b_size = batch.shape[0]
        v0 = batch.reshape(b_size, -1)
        
        # Compute gradients - Positive Phase
        ph0, h0 = self.h_given_v(v0)
        W_grad = np.matmul(v0.T, ph0)
        vbias_grad = np.sum(v0, axis=0)
        hbias_grad = np.sum(ph0, axis=0)
        
        # Compute gradients - Negative Phase
        
        # only contrastive with k = 1, i.e., method="cd"

        pv1, v1 = self.v_given_h(h0)
        ph1, h1 = self.h_given_v(pv1)
        
        W_grad -= np.matmul(pv1.T, ph1)
        vbias_grad -= np.sum(pv1, axis=0)
        hbias_grad -= np.sum(ph1, axis=0)
        
        self.W_grad = W_grad/b_size
        self.hbias_grad = hbias_grad/b_size
        self.vbias_grad = vbias_grad/b_size
        
        recon_err = np.mean(np.sum((v0 - pv1)**2, axis=1), axis=0) # sum of squared error averaged over the batch
        return recon_err
    
    def update_params(self, lr, momentum=0):
        '''
        Function to update the parameters based on the stored gradients.
        input:
            - lr: Learning rate
            - momentum
        '''
        self.W_vel *= momentum
        self.W_vel += (1.-momentum) * lr * self.W_grad
        self.W += self.W_vel
        
        self.vbias_vel *= momentum
        self.vbias_vel += (1.-momentum) * lr * self.vbias_grad
        self.vbias += self.vbias_vel
        
        self.hbias_vel *= momentum
        self.hbias_vel += (1.-momentum) * lr * self.hbias_grad
        self.hbias += self.hbias_vel
        
    def reconstruct(self, v):
        '''
        Reconstructing visible units from given v.
        v -> h0 -> v1
        input:
            - v: (batch_size, n_vis)
        output:
            - prob of reconstructed v: (batch_size, n_vis)
        '''
        ph0, h0 = self.h_given_v(v)
        pv1, v1 = self.v_given_h(ph0)
        return pv1
    
    def avg_free_energy(self, v):
        '''
        Compute the free energy of v averaged over the batch.
        input:
            - v: (batch_size, n_vis)
        output:
            - average of free energy: where free energy = - v.vbias - Sum_j (log(1 + exp(hbias + v_j*W_:,j)) )
        '''
        x = self.hbias + np.matmul(v, self.W)
        free_energy_batch = -np.matmul(v, self.vbias) - np.sum(np.log(1 + np.exp(x)), axis=1)
        return np.mean(free_energy_batch)
    
    def gen_model_sample(self, start=None, num_iters=1000):
        '''
        Generate random samples of visible unit from the model using Gibbs sampling.
        input:
            - start: Any starting value of v.
            - num_iters: Number of iterations of Gibbs sampling.
        '''
        if(start is None):
            v = np.random.randn(self.n_vis)
        else:
            v = start
        for _ in range(num_iters):
            ph, h = rbm.h_given_v(v)
            pv, v = rbm.v_given_h(h)
        return v