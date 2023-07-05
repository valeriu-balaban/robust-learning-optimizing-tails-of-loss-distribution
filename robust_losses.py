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


def MAE(input: Tensor, target: Tensor, num_classes:Optional[int]=10):
  prob     = input.softmax(dim=1).clamp(min=1e-7)
  C_1hot   = torch.nn.functional.one_hot(target, num_classes).float()

  return torch.nn.functional.l1_loss(prob, C_1hot, reduction='none').mean(dim=1)



def moments_penalization(losses: Tensor, lambdas:List[float]=[1.0,], 
                         convex: Optional[bool]=True, z_mean: Optional[Tensor]=None,
                         normalize: Optional[bool]=True):
  """
  Reverse crosss entropy, used to make cross entropy a symmetric loss.

  Introduced in current paper.

  Args:
    losses: tensor of size (N) where N is number of samples and the entries are
        value of the loss function for each sample.
    lambdas: list of floating numper representing the penalization factor for 
        the moments of the loss. Examples, lamdas = [1.0, -0.5] is equivalent to
        1.0*mean(losses) -0.5*var(losses). Default: [1.0,] equivalent to 
        optimizing the mean.
    convex: make negative weights be 0 to preserve the convexity of the loss 
        function at the cost of possible reduced penalization impact. Default: True
    z_mean: mean vector to use to calculate weights.

  Example::

    >>> # input is of size N x C = 3 x 5
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> # each element in target has to have 0 <= value < C
    >>> target = torch.tensor([1, 0, 4])
    >>> losses = torch.nn.functional.cross_entropy(input, target, reduction="none")
    >>> output = moments_penalization(losses, [1.0, -0.5])
    >>> output.backward()
   
  """
  z      = losses.detach()

  if z_mean is None:
    z_mean = z.mean()
    
  weights = torch.zeros_like(z)
  for k, lmbda in enumerate(lambdas):
    if lmbda:
      if k == 0:
        weights += lmbda
      else:
        weights += lmbda * (z - z_mean) ** k

  if convex:
    weights = weights.clip(min=0.0)

  if normalize:
    weights = weights / weights.mean()

  return (weights * losses).mean(), weights / weights.shape[0]



def moments_2stages(losses: Tensor, lambda_neg: float, lambda_pos: float):

  z      = losses.detach()
  z_mean = z.mean()

  weights_neg = 1 + lambda_neg * (z - z_mean)
  weights_pos = 1 + lambda_pos * (z - z_mean)
    
  weights = weights_neg.clip(min=0.0) * weights_pos.clip(min=0.0)

  return (weights * losses).mean(), weights / weights.shape[0]



def moments_penalization_cross_entropy(losses: Tensor, lambda_1=1.0, lambda_2=0.0, 
                                       convex: Optional[bool]=True):
  """
  Reverse crosss entropy, used to make cross entropy a symmetric loss.

  Introduced in current paper.

  Args:
    losses: tensor of size (N) where N is number of samples and the entries are
        value of the loss function for each sample.
    lambdas: list of floating numper representing the penalization factor for 
        the moments of the loss. Examples, lamdas = [1.0, -0.5] is equivalent to
        1.0*mean(losses) -0.5*var(losses). Default: [1.0,] equivalent to 
        optimizing the mean.
    convex: make negative weights be 0 to preserve the convexity of the loss 
        function at the cost of possible reduced penalization impact. Default: True
    z_mean: mean vector to use to calculate weights.

  Example::

    >>> # input is of size N x C = 3 x 5
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> # each element in target has to have 0 <= value < C
    >>> target = torch.tensor([1, 0, 4])
    >>> losses = torch.nn.functional.cross_entropy(input, target, reduction="none")
    >>> output = moments_penalization(losses, [1.0, -0.5])
    >>> output.backward()
   
  """
  z        = losses.detach()
  z_mean   = z.mean()
  x_values = (-z).exp()
  weights  = lambda_1 + lambda_2 * (z - z_mean)
  
  # Critical point (CP) where convexity breaks, second derivative < 0
  x_c      = (lambda_1 / (2 * lambda_2) - z_mean / 2 + 1).exp()
  
  if convex and (x_values < x_c).sum():
    l_c = -x_c.log()                              # loss value at CP
    w_c = lambda_1 + lambda_2 * (l_c - z_mean)    # weight value at CP

    # derivative of the weights*loss at CP
    d_c = -1/x_c * (l_c * lambda_2 + lambda_1 + lambda_2 * (l_c - z_mean))

    # interpolate
    idx          = x_values < x_c
    weights[idx] = (w_c * l_c + d_c * (x_values[idx] - x_c)) / z[idx]

  return (weights * losses).mean()


def HUBER(input: Tensor, target: Tensor, num_classes:Optional[int]=10, delta:Optional[float]=1.0):
  prob     = input.softmax(dim=1).clamp(min=1e-7)
  C_1hot   = torch.nn.functional.one_hot(target, num_classes).float()

  return torch.nn.functional.huber_loss(prob, C_1hot, delta=delta, reduction='none').mean(dim=1)


def robust_mean(losses: Tensor, eta=0.04, delta=0.05):
  """
  https://arxiv.org/pdf/1907.11391.pdf
  """
  z        = losses.detach()
  N        = z.shape[0]
  epsilon  = 8 * eta + 12 * torch.log(torch.tensor(4 / delta)) / N
  
  weights  = torch.ones_like(z)
  alpha    = torch.quantile(z, epsilon)
  beta     = torch.quantile(z, 1 - epsilon)

  weights[z < alpha] = alpha / z[z < alpha]
  weights[z > beta]  = beta  / z[z > beta]

  return (weights * losses).mean()


#################################################
#   Robust Losses 
# from https://github.com/daniellevy/fast-dro
#################################################

"""
PyTorch modules for computing robust losses with
for (KL-regularized) CVaR, constrained-chi^2 and penalized-chi^2
uncertainty sets.
Includes losses appropriate for our porposed batch and MLMC gradient estimators
as well as losses for the dual-SGM and primal-dual methods.
"""

import torch
import torch.nn as nn
import numpy as np

GEOMETRIES = ('cvar', 'chi-square')
MIN_REL_DIFFERENCE = 1e-5


def chi_square_value(p, v, reg):
    """Returns <p, v> - reg * chi^2(p, uniform) for Torch tensors"""
    
    if reg > 0:
      m = p.shape[0]

      with torch.no_grad():
          chi2 = (0.5 / m) * reg * (torch.norm(m * p - 1, p=2) ** 2)
    else:
      chi2 = 0

    return torch.dot(p, v) - chi2


def cvar_value(p, v, reg):
    """Returns <p, v> - reg * KL(p, uniform) for Torch tensors"""
    m = p.shape[0]

    with torch.no_grad():
        idx = torch.nonzero(p)  # where is annoyingly backwards incompatible
        kl = np.log(m) + (p[idx] * torch.log(p[idx])).sum()

    return torch.dot(p, v) - reg * kl


def fenchel_kl_cvar(v, alpha):
    """Returns the empirical mean of the Fenchel dual for KL CVaR"""
    v -= np.log(1 / alpha)
    v1 = v[torch.lt(v, 0)]
    v2 = v[torch.ge(v, 0)]
    w1 = torch.exp(v1) / alpha - 1
    w2 = (v2 + 1) * (1 / alpha) - 1
    return (w1.sum() + w2.sum()) / v.shape[0]


def bisection(eta_min, eta_max, f, tol=1e-6, max_iter=500):
    """Expects f an increasing function and return eta in [eta_min, eta_max] 
    s.t. |f(eta)| <= tol (or the best solution after max_iter iterations"""
    lower = f(eta_min)
    upper = f(eta_max)

    # until the root is between eta_min and eta_max, double the length of the 
    # interval starting at either endpoint.
    while lower > 0 or upper < 0:
        length = eta_max - eta_min
        if lower > 0:
            eta_max = eta_min
            eta_min = eta_min - 2 * length
        if upper < 0:
            eta_min = eta_max
            eta_max = eta_max + 2 * length

        lower = f(eta_min)
        upper = f(eta_max)

    for _ in range(max_iter):
        eta = 0.5 * (eta_min + eta_max)

        v = f(eta)

        if torch.abs(v) <= tol:
            return eta

        if v > 0:
            eta_max = eta
        elif v < 0:
            eta_min = eta

    # if the minimum is not reached in max_iter, returns the current value
    # logging.warning('Maximum number of iterations exceeded in bisection')
    return 0.5 * (eta_min + eta_max)


def huber_loss(x, delta=1.):
    """ Standard Huber loss of parameter delta
    https://en.wikipedia.org/wiki/Huber_loss
    returns 0.5 * x^2 if |a| <= \delta
            \delta * (|a| - 0.5 * \delta) o.w.
    """
    if torch.abs(x) <= delta:
        return 0.5 * (x ** 2)
    else:
        return delta * (torch.abs(x) - 0.5 * delta)


class RobustLoss(nn.Module):
    """PyTorch module for the batch robust loss estimator"""
    def __init__(self, size, reg, geometry, tol=1e-4,
                 max_iter=1000, debugging=False):
        """
        Parameters
        ----------
        size : float
            Size of the uncertainty set (\rho for \chi^2 and \alpha for CVaR)
            Set float('inf') for unconstrained
        reg : float
            Strength of the regularizer, entropy if geometry == 'cvar'
            $\chi^2$ divergence if geometry == 'chi-square'
        geometry : string
            Element of GEOMETRIES
        tol : float, optional
            Tolerance parameter for the bisection
        max_iter : int, optional
            Number of iterations after which to break the bisection
        """
        super().__init__()
        self.size = size
        self.reg = reg
        self.geometry = geometry
        self.tol = tol
        self.max_iter = max_iter
        self.debugging = debugging

        self.is_erm = size == 0

        if geometry not in GEOMETRIES:
            raise ValueError('Geometry %s not supported' % geometry)

        if geometry == 'cvar' and self.size > 1:
            raise ValueError(f'alpha should be < 1 for cvar, is {self.size}')

    def best_response(self, v):
        size = self.size
        reg = self.reg
        m = v.shape[0]

        if self.geometry == 'cvar':
            if self.reg > 0:
                if size == 1.0:
                    return torch.ones_like(v) / m

                def p(eta):
                    x = (v - eta) / reg
                    return torch.min(torch.exp(x),
                                     torch.Tensor([1 / size]).type(x.dtype)) / m

                def bisection_target(eta):
                    return 1.0 - p(eta).sum()

                eta_min = reg * torch.logsumexp(v / reg - np.log(m), 0)
                eta_max = v.max()

                if torch.abs(bisection_target(eta_min)) <= self.tol:
                    return p(eta_min)
            else:
                cutoff = int(size * m)
                surplus = 1.0 - cutoff / (size * m)

                p = torch.zeros_like(v)
                idx = torch.argsort(v, descending=True)
                p[idx[:cutoff]] = 1.0 / (size * m)
                if cutoff < m:
                    p[idx[cutoff]] = surplus
                return p

        if self.geometry == 'chi-square':
            if (v.max() - v.min()) / v.max() <= MIN_REL_DIFFERENCE:
                return torch.ones_like(v) / m

            if size == float('inf'):
                assert reg > 0

                def p(eta):
                    return torch.relu(v - eta) / (reg * m)

                def bisection_target(eta):
                    return 1.0 - p(eta).sum()

                eta_min = min(v.sum() - reg * m, v.min())
                eta_max = v.max()

            else:
                assert size < float('inf')

                # failsafe for batch sizes small compared to
                # uncertainty set size
                if m <= 1 + 2 * size:
                    out = (v == v.max()).float()
                    out /= out.sum()
                    return out

                if reg == 0:
                  if size > 0:
                    def p(eta):
                        pp = torch.relu(v - eta)
                        return pp / pp.sum()

                    def bisection_target(eta):
                        pp = p(eta)
                        w = m * pp - torch.ones_like(pp)
                        return 0.5 * torch.mean(w ** 2) - size

                    eta_min = -(1.0 / (np.sqrt(2 * size + 1) - 1)) * v.max()
                    eta_max = v.max()

                  else:
                    def p(eta):
                        pp = torch.relu(eta - v)
                        return pp / pp.sum()

                    def bisection_target(eta):
                        pp = p(eta)
                        w = m * pp - torch.ones_like(pp)
                        return -size - 0.5 * torch.mean(w ** 2)

                    eta_min = v.min() + 1e-5
                    eta_max = v.max()
                else:
                    def p(eta):
                        pp = torch.relu(v - eta)

                        opt_lam = max(
                            reg, torch.norm(pp) / np.sqrt(m * (1 + 2 * size))
                        )

                        return pp / (m * opt_lam)

                    def bisection_target(eta):
                        return 1 - p(eta).sum()

                    eta_min = v.min() - 1
                    eta_max = v.max()

        eta_star = bisection(
            eta_min, eta_max, bisection_target,
            tol=self.tol, max_iter=self.max_iter)

        if self.debugging:
            return p(eta_star), eta_star
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
        if self.is_erm:
            return v.mean()
        else:
            with torch.no_grad():
                self.prob = self.best_response(v)

            if self.geometry == 'cvar':
                return cvar_value(self.prob, v, self.reg)
            elif self.geometry == 'chi-square':
                return chi_square_value(self.prob, v, self.reg)


class DualRobustLoss(torch.nn.Module):
    """Dual formulation of the robust loss, contains trainable parameter eta"""

    def __init__(self, size, reg, geometry, eta_init=0.0):
        """Constructor for the dual robust loss
        Parameters
        ----------
        size : float
            Size of the uncertainty set (\rho for \chi^2 and \alpha for CVaR)
            Set float('inf') for unconstrained
        reg : float
            Strength of the regularizer, entropy if geometry == 'cvar'
            \chi^2 divergence if geometry == 'chi-square'
        geometry : string
            Element of GEOMETRIES
        eta_init : float
            Initial value for equality constraint Lagrange multiplier eta
        """
        super().__init__()
        self.eta = torch.nn.Parameter(data=torch.Tensor([eta_init]))
        self.geometry = geometry
        self.size = size
        self.reg = reg

        if geometry not in GEOMETRIES:
            raise ValueError('Geometry %s not supported' % geometry)

    def forward(self, v):
        """Value of the dual loss on the batch of examples
        Parameters
        ----------
        v : torch.Tensor
            Tensor containing the individual losses on the batch of examples
        Returns
        -------
        loss : torch.float
            Value of the dual of the robust loss on the batch of examples
        """
        n = v.shape[0]

        if self.geometry == 'cvar':
            if self.reg == 0:
                return self.eta + torch.relu(v - self.eta).mean() / self.size
            else:
                return self.eta + self.reg * fenchel_kl_cvar(
                    (v - self.eta) / self.reg, self.size)

        elif self.geometry == 'chi-square':
            w = torch.relu(v - self.eta)

            if self.size == float('inf'):
              
                # Compute prob
                with torch.no_grad():
                  self.prob = w / (self.reg * n)

                return ((0.5 / self.reg) * (w ** 2).mean()
                        + 0.5 * self.reg + self.eta)
            else:
                if self.reg == 0:

                    # Compute prob
                    with torch.no_grad():
                      self.prob = w / w.sum()

                    return self.eta + np.sqrt((1 + 2 * self.size) / n) * torch.norm(w, p=2)
                else:
                    return self.eta + 0.5 * self.reg + huber_loss(
                        torch.norm(w, p=2) / np.sqrt(n * self.reg),
                        delta=np.sqrt(self.reg * (1 + 2 * self.size)))


class DualVariance(torch.nn.Module):

  def __init__(self, rho, eta_init=0.0):
    """
    Dual formulation of the variance penalization using Chi2 divergence 
  
    Contains trainable parameter eta

    Based on Dual Robust Loss of Daniel Levy and Yair Carmon from "Large-Scale 
    Methods for Distributionally Robust Optimization"
      
    Parameters
    ----------
    rho : float
        Defines size of the uncertainty set, rho * Var(losses)
    eta_init : float
        Initial value for equality constraint Lagrange multiplier eta
    """
    super().__init__()
    
    self.eta = torch.nn.Parameter(data=torch.Tensor([eta_init]))
    self.rho = rho


  def forward(self, losses):
    """Value of the dual loss on the batch of examples
    Parameters
    ----------
    losses : torch.Tensor
        Tensor containing the individual losses on the batch of examples
    Returns
    -------
    loss : torch.float
        Value of the dual of the robust loss on the batch of examples
    """
    n = losses.shape[0]

    losses_truncated = torch.relu(losses - self.eta)

    # Compute prob and save for monitoring
    with torch.no_grad():
      self.prob = losses_truncated / losses_truncated.sum()
      
      k  = torch.sqrt((1.0 + self.rho * losses.var()) / n)


    return self.eta + k * torch.norm(losses_truncated, p=2)


#### Update Variance Penalization #####

class VariancePenalization(nn.Module):
    """PyTorch module for the batch robust loss estimator"""
    def __init__(self, lmbda, varexp=0, tol=1e-4, max_iter=1000):
        """
        Parameters
        ----------
        size : float
            Size of the uncertainty set (\rho for \chi^2)
        tol : float, optional
            Tolerance parameter for the bisection
        max_iter : int, optional
            Number of iterations after which to break the bisection
        """
        super().__init__()
        self.size     = lmbda
        self.varexp   = varexp
        self.tol      = tol
        self.max_iter = max_iter


    def best_response(self, v):
        size = self.size
        m = v.shape[0]

        # if size and self.varexp:
        size *= v.var().pow(self.varexp)

        if (v.max() - v.min()) / v.max() <= MIN_REL_DIFFERENCE:
            return torch.ones_like(v) / m

        # failsafe for batch sizes small compared to
        # uncertainty set size
        if m <= 1 + 2 * size:
            out = (v == v.max()).float()
            out /= out.sum()
            return out


        if size > 0:
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
              return -size - torch.mean(w ** 2)

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
        
        if self.size == 0:
            self.prob = torch.ones_like(v) / v.shape[0]
            return v.mean()

        else:
            with torch.no_grad():
                self.prob = self.best_response(v)

            return torch.dot(self.prob, v)



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
