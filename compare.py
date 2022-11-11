# Import ica function
from ica_torch import ica as ica_gpu
from ica.ica.ica import ica as ica_cpu
#from ica.ica.ica_gpu import ica_gpu as ica_theano
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch.distributions import Uniform, SigmoidTransform, AffineTransform, TransformedDistribution
import pandas as pd


def get_logistic(a, b):
    base = Uniform(0, 1)
    transforms = [SigmoidTransform().inv, AffineTransform(loc=a, scale=b)]
    return TransformedDistribution(base, transforms)


def main():
    # Define matrix dimensions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Nobs = 1000  # Number of observation
    Nvars = 50000  # Number of variables
    Ncomp = 100  # Number of components

    # Simulated true sources
    #S_true = np.random.logistic(0,1,(Ncomp,Nvars))
    
    # Simulated true mixing
    #A_true = np.random.normal(0,1,(Nobs,Ncomp))
    REP = 10
    rows = []
    for i in range(REP):
        print("***** REP %d *****" % i)
        torch.manual_seed(314159+i)
        np.random.seed(314159+i)
        A_true = torch.randn(Nobs, Ncomp, device=device)
        S_true = get_logistic(0, 1).sample((Ncomp, Nvars)).to(device)
        # X = AS
        X = torch.matmul(A_true, S_true)
        # add some noise
        X = X + torch.randn(*X.shape, device=device)
        # apply ICA on X and ask for 2 components

        # GPU
        print("[[GPU]]")
        model_gpu = ica_gpu(Ncomp)

        start = time.time()
        A, S = model_gpu.fit(X)
        total = time.time() - start
        print('[[GPU]] total time: {}'.format(total))

        # compare if our estimates are accurate
        # correlate A with Atrue and take
        A = A.detach().cpu().numpy()
        S = S.detach().cpu().numpy()
        A_true = A_true.detach().cpu().numpy()
        S_true = S_true.detach().cpu().numpy()
        print("SHAPES ", A.shape, A_true.shape)
        #aCorr = np.abs(np.corrcoef(A.T, A_true.T)[
        #    :Ncomp, Ncomp:]).max(axis=0).mean()
        sCorr = np.abs(np.corrcoef(S, S_true)[
            :Ncomp, Ncomp:]).max(axis=0).mean()
        row_gpu = dict(time=total, mode="pytorch",
                       mixing_accuracy=sCorr)
        print("[[GPU]] Accuracy of estimated sources: %.2f" % sCorr)
        #print("[[GPU]] Accuracy of estimated mixing: %.2f" % aCorr)

        # CPU
        print("[[CPU]]")
        model_cpu = ica_cpu(Ncomp)
        X = X.detach().cpu().numpy()
        start = time.time()
        A, S = model_cpu.fit(X)
        total = time.time() - start
        print('[[CPU]] total time: {}'.format(total))

        # compare if our estimates are accurate
        # correlate A with Atrue and take
        #aCorr = np.abs(np.corrcoef(A.T, A_true.T)[
        #    :Ncomp, Ncomp:]).max(axis=0).mean()
        sCorr = np.abs(np.corrcoef(S, S_true)[
            :Ncomp, Ncomp:]).max(axis=0).mean()
        row_cpu = dict(time=total, mode="numpy",
                       mixing_accuracy=sCorr)
        print("[[CPU]] Accuracy of estimated sources: %.2f" % sCorr)
        #print("[[CPU]] Accuracy of estimated mixing: %.2f" % aCorr)

        # THEANO
        """
        print("[[THEANO]]")
        model_theano = ica_theano(Ncomp)

        start = time.time()
        A, S = model_cpu.fit(X)
        total = time.time() - start
        print('[[THEANO]] total time: {}'.format(total))

        # compare if our estimates are accurate
        # correlate A with Atrue and take
        aCorr = torch.abs(torch.corrcoef(A.T, A_true.T)[
            :Ncomp, Ncomp:]).max(axis=0).mean()
        sCorr = torch.abs(torch.corrcoef(S, S_true)[
            :Ncomp, Ncomp:]).max(axis=0).mean()
        row_theano = dict(time=total, mode="theano",
                          source_accuracy=aCorr, mixing_accuracy=sCorr)
        print("[[THEANO]] Accuracy of estimated sources: %.2f" % sCorr)
        print("[[THEANO]] Accuracy of estimated mixing: %.2f" % aCorr)
        rows.append(row_theano)
        """
        rows.append(row_gpu)
        rows.append(row_cpu)

    df = pd.DataFrame(rows)
    df.to_csv("comparison.csv")


if __name__ == "__main__":
    main()
