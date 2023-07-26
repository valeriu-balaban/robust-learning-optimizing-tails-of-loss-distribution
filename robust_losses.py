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


class DistributionalVariancePenalization(torch.nn.Module):
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
        MIN_REL_DIFFERENCE = 1e-5

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



### DISTRIBTUTIONAL MOMENTS PENALIZATION

from scipy import optimize

def f_div(q, alpha):
  """Computes f-divergence of order alpha between q and the uniform distribution"""

  if alpha == 1:
    return (q * (q * q.shape[0] + 2 * torch.finfo(q.dtype).eps).log()).sum()
  else:
    return ((q * q.shape[0]).pow(alpha) - 1).mean() / (alpha * (alpha - 1))


def limit_dist(z, limit="max"):
  """Returns the limit distribution that maximizes/minimizes the expectation of z"""

  q = (z == z.max()) if limit == "max" else (z == z.min())
  return q / q.sum()


def delta_dist(z, delta, alpha, limit="max"):
  """Returns the distribution given an offset value rho"""

  sign = +1 if limit == "max" else -1

  if alpha == 1:
    q = (sign * (z / delta)).exp()
  else:
    q = torch.relu(sign * (z - delta)).pow(1 / (alpha - 1))

  return q / q.sum()



def distributional_moments_penalization(z, rho, alpha=2, max_iter=100, debug=False):  
  """
  Finds and returns the discrete distrbution Q_n that recovers distributional
  moments penalization. Implements FindQ procedure of the paper.
  Procedure is influenced by the RobustLoss of https://github.com/daniellevy/fast-dro
  
  Parameters
  ----------
  z : Tensor
    Values of the random variable for which to penalize the variance
  rho : float
    Radius of the neighborhood of the discrete empirical distribution to consider
  alpha : float
    Order of the alpha divergence to use when defining the neighborhood
  max_iter : int, optional
    Number of iterations after which to stop the bisection search
  debug: bool, optional
    If True debugging messages are displayed

  Returns
  ----------
  q : Tensor
    Probabilities correspoding to each entry of the tensor z
  """

  # Check for variance minimization
  assert rho >= 0, f"rho greater than 0 expected, got: {rho}"

  try:

    # Check for enough variance in z
    if (z.max() - z.min()) / z.max() <= 1e-5:
      return torch.ones_like(z) / z.shape[0], 0

    # Check if the target divergence is achievable
    q_lim = limit_dist(z, limit="min")

    if debug:
      print("Max div", f_div(q_lim, alpha))

    if f_div(q_lim, alpha) < rho:
      return q_lim, 0


    f = lambda delta: f_div(delta_dist(z, delta, alpha, limit="min"), alpha) - rho

    # Set search upper bound
    a = z.min() + 2 * torch.finfo(z.dtype).eps
    upper   = f(a)

    if debug:
      print("Upper", upper)

    b = z.max()
    lower   = f(b)

    if debug:
      print("Lower", lower)

    if debug:
      print("Initial search bounds:", (b, a), " and difference:", (lower, upper))

    # Double the interval until the root is between a and b
    while lower > 0:
      length    = b - a
      a = b
      b = b + 2 * length

      upper = lower
      lower = f(b)
      

      if debug:
        print("Updated search bounds:", (b, a), " and difference:", (lower, upper))

    r = optimize.brentq(f, a, b, full_output=True)
    
    if debug:
      print(r)

    delta = r[0]
    return delta_dist(z, delta, alpha, limit="min"), delta
  
  except Exception as e:
    print(e)
    return torch.ones_like(z) / z.shape[0], float("nan")




def compute_tail_metric(losses):
  """Computes the tail of the distribution by evaluating the cummulant at 1 and substracting the mean"""
  
  z = losses.detach()
  return torch.logsumexp(z, dim=0) - log(z.shape[0]) - z.mean()


def distributional_variance_penalization(z, lmbda=0.0, tol=1e-5, max_iter=100):
  """
  Finds and returns the discrete distrbution Q_n that recovers distributional
  variance penalization. Implements FindQ procedure of the paper.

  Procedure extends the RobustLoss of https://github.com/daniellevy/fast-dro

  Parameters
  ----------
  z : Tensor
    Values of the random variable for which to penalize the variance
  lmbda : float
    Variance penalization factor, i.e., lmbda * Var
  gamma : float
    Exponent of the varaince term, i.e., Var ** gamma
  tol : float, optional
    Tolerance parameter for the bisection
  max_iter : int, optional
    Number of iterations after which to stop the bisection search

  Returns
  ----------
  q : Tensor
    Probabilities correspoding to each entry of the tensor z
  """

  debug = False
  m = z.shape[0]
  target_D_chi2 = lmbda 

  if debug:
    print("Variance", z.var(), "Target D_chi2", target_D_chi2)

  # Check for variance minimization
  assert lmbda >= 0, f"lmbda greater than 0 expected, got: {lmbda}"

  # Check for enough variance in z
  if (z.max() - z.min()) / z.max() <= tol:
    return torch.ones_like(z) / m, 0

  # Check if the target divergence is achievable
  r     = (z == z.max()).float()
  q_max = r / r.sum()

  if (z.shape[0] * q_max - 1).square().mean()/2 < target_D_chi2:
    # Return the distribution that achieves the maximum divergence
    return q_max, 0


  def f(rho):
    """Computes Chi 2 divergence between z and q generated using rho"""
    r = torch.relu(z - rho)
    q = r / r.sum()
    current_D_chi2 = (z.shape[0] * q - 1).square().mean()/2

    return current_D_chi2 - target_D_chi2

  # Set search upper bound
  rho_max = z.max() - torch.finfo(z.dtype).eps
  upper   = f(rho_max)
  
  # Not enough precision, solution is within epsilon of z.max()
  if upper < 0:
    return q_max, 0


  # Set search lower bound
  rho_min = -(1.0 / ((2 * target_D_chi2 + 1).sqrt() - 1)) * z.max()
  lower   = f(rho_min)

  # Samples are too close
  if lower.isfinite() == False:
    return torch.ones_like(z) / m, 

  if debug:
    print("Initial search bounds:", (rho_min, rho_max), " and difference:", (lower, upper))

  # Double the interval until the root is between rho_min and rho_max
  while lower > 0:
    length  = rho_max - rho_min
    rho_max = rho_min
    rho_min = rho_min - 2 * length

    lower = f(rho_min)
    upper = f(rho_max)
    
    if debug:
      print("Updated search bounds:", (rho_min, rho_max), " and difference:", (lower, upper))

  # Search using bisection
  for _ in range(max_iter):
    rho = 0.5 * (rho_min + rho_max)

    diff = f(rho)

    if debug:
      print("Searching interval:", (rho_min, rho_max), "and half interval diff:", diff)

    if torch.abs(diff) <= tol:
      r = torch.relu(z - rho)
      q = r / r.sum()
      return q, rho

    if diff > 0:
      rho_max = rho
    elif diff < 0:
      rho_min = rho

  # Reached max_iter, return what we have
  r = torch.relu(z - 0.5 * (rho_min + rho_max) )
  q = r / r.sum()
  return q, 0.5 * (rho_min + rho_max)
