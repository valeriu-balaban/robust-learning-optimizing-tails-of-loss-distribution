import torch

from math import log, sqrt
from torch import Tensor
from typing import Optional, List


def generalized_cross_entropy(input: Tensor, target: Tensor, q: Optional[float]=0.7, 
                              k: Optional[float]=None, weights: Optional[Tensor]=None,
                              version:Optional[int]=1, num_classes:Optional[int]=10):
  """
  Interpolates between a linear function (q = 1) and log function (q = 0).

  Loss function introduced in "Generalized Cross Entropy Loss for Training Deep 
  Neural Networks with Noisy Labels" by Zhilu Zhang and Mert R. Sabuncu 
  
  Args:
    input: tensor of size (N, C) where N is number of samples and C is the 
        number of classes. The values of the matrix should not be normalized to
        represent probabilities, this will be done by the function.
    target: tensor of size (N) where each value is between 0 and C-1.
    q: Interpolation parameter that selects the shape of the function. Values
        close to 1 increase robustness but decrease convergence and a value 
        close to 0 produces a different behavior. Default: 0.7 as in the paper.
    k: Truncation treshold, larger values lead to tighter bounds and hence more 
        noise-robustness, too large of a threshold would precipitate too many 
        discarded samples for training. Optimal value depends on the noise level
        in the labels. The value used in the paper is 0.5.
    weights (Tensor, optional): Selects which samples to use for training. If k 
        is not none but no weights are given then they will be computed by
        thresholding the probabilities of input with k.
    version: Selects the implementation version of the method. Version 1 uses
        gather approach, version 2 uses 1 hot encoding approach. Default: 1
    num_classes: Number of classes in the target vector. Required for version 2.

  Example::

    >>> # input is of size N x C = 3 x 5
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> # each element in target has to have 0 <= value < C
    >>> target = torch.tensor([1, 0, 4])
    >>> output = generalized_cross_entropy(input, target)
    >>> output.backward()
   
  """

  prob   = input.softmax(dim=1)
  
  if version == 1:
    f_j  = prob.gather(dim=1, index=target.unsqueeze(dim=1)).flatten()
  elif version == 2:
    f_j  = (prob * torch.nn.functional.one_hot(target, num_classes)).sum(dim=1)

  losses = (1 - f_j ** q) / q

  # Should we truncate?
  if k is not None:

    # Compute weights
    if weights is None:
      weights = 1.0 * (f_j > k)

    # Use eq (12)
    losses = weights * losses + (1 - weights) * (1 - k ** q) / q 
     

  return losses.mean()


def term_transformation(losses: Tensor, t: Optional[int]=-2, version: Optional[int]=1):
  """
  Computes the tilted loss with the tilt value of t.

  Tilted minimization problem introduced in "Tilted Empirical Risk Minimization"
  by Tian Li, Ahmad Beirami, Maziar Sanjabi, Virginia Smith
  
  Args:
    losses: tensor of size (N) where N is number of samples and the entries are
        value of the loss function for each sample.
    t: Interpolation parameter that selects what objective to minimize. Positive
        values tilt towards max loss recoverd when t approaches +inf, negative 
        values tilt towards min loss recoverd when t approaches -inf, and the
        standard mean value optimization for t close to 0. Default: -2 as in the
        paper.
    version: Selects the implementation version of the method. Version 1 uses
        logsumexp method to preserve precision, version 2 is the naive 
        implementation. Default: 1

  Example::

    >>> # input is of size N x C = 3 x 5
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> # each element in target has to have 0 <= value < C
    >>> target = torch.tensor([1, 0, 4])
    >>> losses = torch.nn.functional.cross_entropy(input, target, reduction="none")
    >>> output = term_transformation(losses, t=-2)
    >>> output.backward()
   
  """

  if version == 1:
    loss = (t * losses - log(losses.shape[0])).logsumexp(dim=0) / t
  elif version == 2:
    loss = (t * losses).exp().mean().log() / t

  return loss 


def taylor_cross_entropy(input: Tensor, target: Tensor, t: Optional[int]=2,
                         version:Optional[int]=1, num_classes:Optional[int]=10):
  """
  Approximates the cross entropy using the Taylor expansion up to a term t.

  Method introduced in "Can Cross Entropy Loss Be Robust to Label Noise?" by 
  Lei Feng, Senlin Shu, Zhuoyi Lin, Fengmao Lv, Li Li, Bo An
  
  Args:
    input: tensor of size (N, C) where N is number of samples and C is the 
        number of classes. The values of the matrix should not be normalized to
        represent probabilities, this will be done by the function.
    target: tensor of size (N) where each value is between 0 and C-1.
    t: Taylor series order to be used for the approximation. Default: 2 as in the
        paper.
    version: Selects the implementation version of the method. Version 1 uses
        gather approach, version 2 uses 1 hot encoding approach. Default: 1
    num_classes: Number of classes in the target vector. Required for version 2.

  Example::

    >>> # input is of size N x C = 3 x 5
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> # each element in target has to have 0 <= value < C
    >>> target = torch.tensor([1, 0, 4])
    >>> output = taylor_cross_entropy(input, target)
    >>> output.backward()
   
  """

  prob   = input.softmax(dim=1)
  if version == 1:
    f_y  = prob.gather(dim=1, index=target.unsqueeze(dim=1)).flatten()
  elif version == 2:
    f_y  = (prob * torch.nn.functional.one_hot(target, num_classes)).sum(dim=1)

  losses = 0

  for k in range(1, t+1):
    losses += (1 - f_y)**t / t

  return losses.mean()


def normalized_cross_entropy(input: Tensor, target: Tensor, version:Optional[int]=1, 
                             num_classes:Optional[int]=10):
  """
  Normalized crosss entropy, symmetric loss.

  Method introduced in "Normalized Loss Functions for Deep Learning with Noisy 
  Labels" by Xingjun Ma, Hanxun Huang, Yisen Wang, Simone Romano, Sarah Erfani, 
  James Bailey
  
  Args:
    input: tensor of size (N, C) where N is number of samples and C is the 
        number of classes. The values of the matrix should not be normalized to
        represent probabilities, this will be done by the function.
    target: tensor of size (N) where each value is between 0 and C-1.
    version: Selects the implementation version of the method. Version 1 uses
        gather approach, version 2 uses 1 hot encoding approach. Default: 1
    num_classes: Number of classes in the target vector. Required for version 2.

  Example::

    >>> # input is of size N x C = 3 x 5
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> # each element in target has to have 0 <= value < C
    >>> target = torch.tensor([1, 0, 4])
    >>> output = normalized_cross_entropy(input, target)
    >>> output.backward()
   
  """
  log_prob = torch.nn.functional.log_softmax(input, dim=1)
  if version == 1:
    log_fy = log_prob.gather(dim=1, index=target.unsqueeze(dim=1)).flatten()
  elif version == 2:
    log_fy = (log_prob * torch.nn.functional.one_hot(target, num_classes)).sum(dim=1)

  return (log_fy / log_prob.sum(dim=1)).mean()


def reverse_cross_entropy(input: Tensor, target: Tensor, num_classes:Optional[int]=10):
  """
  Reverse crosss entropy, used to make cross entropy a symmetric loss.

  Method introduced in "Symmetric Cross Entropy for Robust Learning with Noisy 
  Labels" by Yisen Wang, Xingjun Ma, Zaiyi Chen, Yuan Luo, Jinfeng Yi, James Bailey
 
  Args:
    input: tensor of size (N, C) where N is number of samples and C is the 
        number of classes. The values of the matrix should not be normalized to
        represent probabilities, this will be done by the function.
    target: tensor of size (N) where each value is between 0 and C-1.
    num_classes: Number of classes in the target vector.

  Example::

    >>> # input is of size N x C = 3 x 5
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> # each element in target has to have 0 <= value < C
    >>> target = torch.tensor([1, 0, 4])
    >>> output = reverse_cross_entropy(input, target)
    >>> output.backward()
   
  """
  prob     = input.softmax(dim=1).clamp(min=1e-7)
  log_1hot = torch.nn.functional.one_hot(target, num_classes).float().clamp(min=1e-4).log()

  return -(prob * log_1hot).sum(dim=1).mean()


class DistributionalVariancePenalization(nn.Module):
    """PyTorch module for the batch robust loss estimator"""
    def __init__(self, lmbda, gamma=1, tol=1e-4, max_iter=1000):
        """
        Parameters
        ----------
        lmbda : float
            Variance penalization factor, i.e., lmbda * Var
        gamma : float
            Exponent of the varaince term, i.e., Var ** gamma
        tol : float, optional
            Tolerance parameter for the bisection
        max_iter : int, optional
            Number of iterations after which to break the bisection
        """
        super().__init__()
        self.lmbda    = lmbda
        self.gamma    = gamma
        self.tol      = tol
        self.max_iter = max_iter


    def findQ(self, v):
        m = v.shape[0]
        size = self.lmbda ** 2
        size *= v.var().pow(2*self.gamma-1)

        if (v.max() - v.min()) / v.max() <= MIN_REL_DIFFERENCE:
            return torch.ones_like(v) / m

        # failsafe for batch sizes small 
        if m <= 1 + 2 * size:
            out = (v == v.max()).float()
            out /= out.sum()
            return out


        if self.lmbda > 0:
          def p(eta):
              pp = torch.relu(v - eta)
              return pp / pp.sum()

          def bisection_target(eta):
              pp = p(eta)
              w = m * pp - 1
              return torch.mean(w ** 2) - size

          eta_min = -(1.0 / ((2 * size + 1).sqrt() - 1)) * v.max()
          eta_max = v.max()

        else:
          def p(eta):
              pp = torch.relu(eta - v)
              return pp / pp.sum()

          def bisection_target(eta):
              pp = p(eta)
              w = m * pp - 1
              return size - torch.mean(w ** 2)

          eta_min = v.min() + 1e-5
          eta_max = v.max()


        eta_star = bisection(
            eta_min, eta_max, bisection_target,
            tol=self.tol, max_iter=self.max_iter)

        return p(eta_star)

    def forward(self, v):
        """Value of the robust loss
        Note that the best response is computed without gradients
        Parameters
        ----------
        v : torch.Tensor
            Tensor containing the individual losses on the batch of examples
        Returns
        -------
        loss : torch.float
            Value of the robust loss on the batch of examples
        """
        self.v = v
        
        if self.lmbda == 0:
            self.prob = torch.ones_like(v) / v.shape[0]
            return v.mean()

        else:
            with torch.no_grad():
                self.prob = self.findQ(v)

            return torch.dot(self.prob, v)
