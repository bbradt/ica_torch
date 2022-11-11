# Import ica function
from ica_torch import ica
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch.distributions import Uniform, SigmoidTransform, AffineTransform, TransformedDistribution


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
    S_true = get_logistic(0, 1).sample((Ncomp, Nvars)).to(device)
    # Simulated true mixing
    #A_true = np.random.normal(0,1,(Nobs,Ncomp))
    A_true = torch.randn(Nobs, Ncomp, device=device)
    # X = AS
    X = torch.matmul(A_true, S_true)
    # add some noise
    X = X + torch.randn(*X.shape, device=device)
    # apply ICA on X and ask for 2 components

    model = ica(Ncomp)

    start = time.time()
    A, S = model.fit(X)
    total = time.time() - start
    print('total time: {}'.format(total))
    # compare if our estimates are accurate
    # correlate A with Atrue and take
    aCorr = torch.abs(torch.corrcoef(A.T, A_true.T)[
        :Ncomp, Ncomp:]).max(axis=0).mean()
    sCorr = torch.abs(torch.corrcoef(S, S_true)[
                      :Ncomp, Ncomp:]).max(axis=0).mean()

    print("Accuracy of estimated sources: %.2f" % sCorr)
    print("Accuracy of estimated mixing: %.2f" % aCorr)


if __name__ == "__main__":
    main()
