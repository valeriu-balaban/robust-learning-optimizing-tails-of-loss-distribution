import torch
import numpy as np

from torch import default_generator, randperm
from torch.utils.data import Subset, random_split
from numpy.testing import assert_array_almost_equal
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST


class MemoryIterableDataset(torch.utils.data.IterableDataset):
  """
  Saves images in contiguous memory on the device for faster transformation
  during training. If device is "cuda" the transformations will be performed on
  the GPU. Batches are not shuffled, however, randomness can be inserted by
  using random transformations such as RandomCrop or RandomHorizontalFlip.
  """
  def __init__(self, images, noisy_labels, true_labels, batch_size:int=1, 
               device:str="cpu", transformations=None, random_offset:bool=False,
               random_batch:bool=False, drop_last:bool=True):
    
    
    super(MemoryIterableDataset).__init__()
    
    self.images           = images.to(device).contiguous()
    self.noisy_labels     = noisy_labels.to(device).contiguous()
    self.true_labels      = true_labels.to(device).contiguous()

    self.batch_size       = batch_size
    self.transformations  = transformations

    self.random_offset    = random_offset
    self.random_batch     = random_batch
    
    self.N                = images.shape[0]
    self.cursor           = 0

    self.drop_last        = True


  def __iter__(self):
    self.cursor  = 0

    if self.random_offset:
      self.cursor  = torch.randint(0, self.batch_size, (1,)).item()

    return self


  def __next__(self):
    batch_size = self.batch_size

    if self.cursor >= self.N:
      raise StopIteration

    if self.cursor + batch_size > self.N:
      
      if self.drop_last:
        raise StopIteration

      else:
        batch_size = self.N - self.cursor

    if self.random_batch:
      i            = torch.randint(0, self.N - self.batch_size, (1,)).item()
      idxs         = slice(i, i + self.batch_size)
    else:
      idxs         = slice(self.cursor, self.cursor + batch_size)

    self.cursor += batch_size

    if self.transformations:
      return  self.transformations(self.images[idxs]), \
              self.noisy_labels[idxs], \
              self.true_labels[idxs],
    else:
      return  self.images[idxs], \
              self.noisy_labels[idxs], \
              self.true_labels[idxs],


#### DATASETS WITH NOISE ####

class FMNISTNOISY(FashionMNIST):

  _GROUP1 = [9, 2, 3, 5, 4]
  _GROUP2 = [1, 0, 5, 3, 7]

  def __init__(self, root, train=True,  transform=None,      target_transform=None,             download=False,   nosiy_rate=0.0, asym=False, imbalanced=False):
    super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download,)
      
    # Save targets before we shuffle
    self.original_targets = self.targets.detach().clone()
    
    self.imbalanced = imbalanced

    if asym:
      # automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
      source_class = [9, 2, 3, 5, 4]
      target_class = [1, 0, 5, 3, 7]
      for s, t in zip(source_class, target_class):
        cls_idx = np.where(np.array(self.targets) == s)[0]
        n_noisy = int(nosiy_rate * cls_idx.shape[0])
        noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
        for idx in noisy_sample_index:
            self.targets[idx] = t


    elif nosiy_rate > 0:

      n_samples = len(self.targets)
      n_noisy = int(nosiy_rate * n_samples)
      print("%d Noisy samples" % (n_noisy))
      class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
      class_noisy = int(n_noisy / 10)
      noisy_idx = []
      for d in range(10):
        noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
        noisy_idx.extend(noisy_class_index)
        print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
      for i in noisy_idx:
        self.targets[i] = self. other_class(n_classes=10, current_class=self.targets[i])
      print(len(noisy_idx))
      print("Print noisy label generation statistics:")
      for i in range(10):
        n_noisy = np.sum(np.array(self.targets) == i)
        print("Noisy class %s, has %s samples." % (i, n_noisy))


  def random_split(self, ratio: float, generator=default_generator):
    """ADD doc string"""
    
    if not self.imbalanced:

      # Randomly split data
      train_size  = int(ratio * len(self.targets))
      valid_size  = len(self.targets) - train_size
      
      return random_split(self, [train_size, valid_size], generator=generator)

    else:
      # Randomly split imbalanced classes
      indices = randperm(len(self.targets), generator=generator)
      targets_reordered = torch.tensor(self.targets)[indices]

      group1_mask = torch.zeros_like(targets_reordered).bool()

      for target_class in self._GROUP1:
        group1_mask = torch.logical_or(group1_mask, targets_reordered == target_class)

      group2_mask = torch.zeros_like(targets_reordered).bool()

      for target_class in range(targets_reordered.max() + 1):
        # skip previously processed taget classes
        if target_class in self._GROUP1:
          continue

        mask   = (targets_reordered == target_class)
        mask_1 = mask.cumsum(dim=0) <= int(ratio * mask.sum())
        mask_2 = torch.logical_not(mask_1)
        
        group1_mask = torch.logical_or(group1_mask, mask.logical_and(mask_1))
        group2_mask = torch.logical_or(group2_mask, mask.logical_and(mask_2))

      
      return Subset(self, indices[group1_mask].tolist()), \
            Subset(self, indices[group2_mask].tolist())


  def __getitem__(self, index: int):
    item = super().__getitem__(index)

    original_target = self.original_targets[index]

    return item[0], item[1], original_target, item[1] != original_target


  def other_class(self, n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
      error_str = "class_ind must be within the range (0, nb_classes - 1)"
      raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    
    return other_class



class CIFAR10NOISY(CIFAR10):

  _GROUP1 = [9, 2, 3, 5, 4]
  _GROUP2 = [1, 0, 5, 3, 7]


  def __init__(self, root, train=True,  transform=None,      target_transform=None,             download=False,   nosiy_rate=0.0, asym=False, imbalanced=False):
    super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download,)
      
    # Save targets before we shuffle
    self.original_targets = self.targets.copy()
    
    self.imbalanced = imbalanced

    if asym:
      # automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
      source_class = [9, 2, 3, 5, 4]
      target_class = [1, 0, 5, 3, 7]
      for s, t in zip(source_class, target_class):
        cls_idx = np.where(np.array(self.targets) == s)[0]
        n_noisy = int(nosiy_rate * cls_idx.shape[0])
        noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
        for idx in noisy_sample_index:
            self.targets[idx] = t


    elif nosiy_rate > 0:

      n_samples = len(self.targets)
      n_noisy = int(nosiy_rate * n_samples)
      print("%d Noisy samples" % (n_noisy))
      class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
      class_noisy = int(n_noisy / 10)
      noisy_idx = []
      for d in range(10):
        noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
        noisy_idx.extend(noisy_class_index)
        print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
      for i in noisy_idx:
        self.targets[i] = self. other_class(n_classes=10, current_class=self.targets[i])
      print(len(noisy_idx))
      print("Print noisy label generation statistics:")
      for i in range(10):
        n_noisy = np.sum(np.array(self.targets) == i)
        print("Noisy class %s, has %s samples." % (i, n_noisy))


  def random_split(self, ratio: float, generator=default_generator):
    """ADD doc string"""
    
    if not self.imbalanced:

      # Randomly split data
      train_size  = int(ratio * len(self.targets))
      valid_size  = len(self.targets) - train_size
      
      return random_split(self, [train_size, valid_size], generator=generator)

    else:
      # Randomly split imbalanced classes
      indices = randperm(len(self.targets), generator=generator)
      targets_reordered = torch.tensor(self.targets)[indices]

      group1_mask = torch.zeros_like(targets_reordered).bool()

      for target_class in self._GROUP1:
        group1_mask = torch.logical_or(group1_mask, targets_reordered == target_class)

      group2_mask = torch.zeros_like(targets_reordered).bool()

      for target_class in range(targets_reordered.max() + 1):
        # skip previously processed taget classes
        if target_class in self._GROUP1:
          continue

        mask   = (targets_reordered == target_class)
        mask_1 = mask.cumsum(dim=0) <= int(ratio * mask.sum())
        mask_2 = torch.logical_not(mask_1)
        
        group1_mask = torch.logical_or(group1_mask, mask.logical_and(mask_1))
        group2_mask = torch.logical_or(group2_mask, mask.logical_and(mask_2))

      
      return Subset(self, indices[group1_mask].tolist()), \
            Subset(self, indices[group2_mask].tolist())


  def __getitem__(self, index: int):
    item = super().__getitem__(index)

    original_target = self.original_targets[index]

    return item[0], item[1], original_target, item[1] != original_target


  def other_class(self, n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
      error_str = "class_ind must be within the range (0, nb_classes - 1)"
      raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    
    return other_class


class CIFAR100NOISY(CIFAR100):

  _GROUP1 = [
    0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 
    40, 41, 42, 43, 44, 50, 51, 52, 53, 54, 60, 61, 62, 63, 64, 70, 71, 72, 73, 
    74, 80, 81, 82, 83, 84, 90, 91, 92, 93, 94
  ]
  _GROUP2 = [
    5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29, 35, 36, 37, 38, 39, 
    45, 46, 47, 48, 49, 55, 56, 57, 58, 59, 65, 66, 67, 68, 69, 75, 76, 77, 78, 
    79, 85, 86, 87, 88, 89, 95, 96, 97, 98, 99
  ]


  def __init__(self, root, train=True, transform=None, target_transform=None, download=False, nosiy_rate=0.0, asym=False, imbalanced=False, seed=0):

    super().__init__(root, train=train, download=download, transform=transform, target_transform=target_transform)
    
    # Save targets before we shuffle
    self.original_targets = self.targets.copy()

    self.imbalanced = imbalanced

    if asym:
      """mistakes are inside the same superclass of 10 classes, e.g. 'fish'
      """
      nb_classes = 100
      P = np.eye(nb_classes)
      n = nosiy_rate
      nb_superclasses = 20
      nb_subclasses = 5

      if n > 0.0:
        for i in np.arange(nb_superclasses):
          init, end = i * nb_subclasses, (i+1) * nb_subclasses
          P[init:end, init:end] = self.build_for_cifar100(nb_subclasses, n)

          y_train_noisy = self.multiclass_noisify(np.array(self.targets), P=P, random_state=seed)
          actual_noise = (y_train_noisy != np.array(self.targets)).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        self.targets = y_train_noisy.tolist()
      return
    elif nosiy_rate > 0:
      n_samples = len(self.targets)
      n_noisy = int(nosiy_rate * n_samples)
      print("%d Noisy samples" % (n_noisy))
      class_index = [np.where(np.array(self.targets) == i)[0] for i in range(100)]
      class_noisy = int(n_noisy / 100)
      noisy_idx = []
      for d in range(100):
        noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
        noisy_idx.extend(noisy_class_index)
        print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
      for i in noisy_idx:
        self.targets[i] = self.other_class(n_classes=100, current_class=self.targets[i])
      print(len(noisy_idx))
      print("Print noisy label generation statistics:")
      for i in range(100):
        n_noisy = np.sum(np.array(self.targets) == i)
        print("Noisy class %s, has %s samples." % (i, n_noisy))
      return


  def __getitem__(self, index: int):
    item = super().__getitem__(index)

    original_target = self.original_targets[index]

    return item[0], item[1], original_target, item[1] != original_target

  
  def random_split(self, ratio: float, generator=default_generator):
    """ADD doc string"""
    
    if not self.imbalanced:

      # Randomly split data
      train_size  = int(ratio * len(self.targets))
      valid_size  = len(self.targets) - train_size
      
      return random_split(self, [train_size, valid_size], generator=generator)

    else:
      # Randomly split imbalanced classes
      indices = randperm(len(self.targets), generator=generator)
      targets_reordered = torch.tensor(self.targets)[indices]

      group1_mask = torch.zeros_like(targets_reordered).bool()

      for target_class in self._GROUP1:
        group1_mask = torch.logical_or(group1_mask, targets_reordered == target_class)

      group2_mask = torch.zeros_like(targets_reordered).bool()

      for target_class in range(targets_reordered.max() + 1):
        # skip previously processed taget classes
        if target_class in self._GROUP1:
          continue

        mask   = (targets_reordered == target_class)
        mask_1 = mask.cumsum(dim=0) <= int(ratio * mask.sum())
        mask_2 = torch.logical_not(mask_1)
        
        group1_mask = torch.logical_or(group1_mask, mask.logical_and(mask_1))
        group2_mask = torch.logical_or(group2_mask, mask.logical_and(mask_2))

      
      return Subset(self, indices[group1_mask].tolist()), \
            Subset(self, indices[group2_mask].tolist())


  def build_for_cifar100(self, size, noise):
    """ random flip between two random classes.
    """
    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
      P[i, i+1] = noise

    # adjust last row
    P[size-1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


  def multiclass_noisify(self, y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
      i = y[idx]
      # draw a vector with only an 1
      flipped = flipper.multinomial(1, P[i, :], 1)[0]
      new_y[idx] = np.where(flipped == 1)[0]

    return new_y


  def other_class(self, n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
      error_str = "class_ind must be within the range (0, nb_classes - 1)"
      raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class
