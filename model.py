import sys
import json

import wandb
import torch
import torchvision


import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List

from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from torchvision import transforms 

from torchvision.models import resnet34

import pytorch_lightning as pl

from torchmetrics import Accuracy

# Local imports
from datasets import FMNISTNOISY, CIFAR10NOISY, CIFAR100NOISY, MemoryIterableDataset

from robust_losses import \
  moments_penalization, \
  moments_2stages, \
  normalized_cross_entropy, \
  reverse_cross_entropy, \
  MAE, \
  HUBER, \
  robust_mean, \
  moments_penalization_cross_entropy, \
  term_transformation, \
  generalized_cross_entropy, \
  taylor_cross_entropy, \
  RobustLoss, \
  DualRobustLoss, \
  DualVariance, \
  VariancePenalization, \
  DistributionalVariancePenalization




class Model(pl.LightningModule):
  """
  Pytorch Lightning Implementation
  """

  def __init__(self, params):
    super().__init__()

    self.save_hyperparameters(params)

    if self.hparams["dataset"] == "CIFAR-10":
      """
      Model used in ICCV2019 paper "Symmetric Cross Entropy for Robust Learning with Noisy Labels"
      Reproduced from https://github.com/HanxunH/SCELoss-Reproduce
      """
      if not self.hparams["complex_model"]:
        self.conv_layers = nn.Sequential(
          self.conv_block(3, 64, 3),
          self.conv_block(64, 64, 3),
          nn.MaxPool2d(kernel_size=2, stride=2),
          self.conv_block(64, 128, 3),
          self.conv_block(128, 128, 3),
          nn.MaxPool2d(kernel_size=2, stride=2),
          self.conv_block(128, 196, 3),
          self.conv_block(196, 196, 3),
          nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
          nn.Linear(3136, 256),
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.Linear(256, 10)
        )
      else:
        self.resnet = resnet34(num_classes=10)


      self.acc_metric = Accuracy(average='none', num_classes=10)

      self.acc_clean_metric = Accuracy()
      self.acc_noisy_metric = Accuracy()
    
    elif self.hparams["dataset"] == "CIFAR-100":
      self.resnet = resnet34(num_classes=100)
      self.acc_metric = Accuracy(average='none', num_classes=100)

    elif self.hparams["dataset"] == "MNIST":
      self.conv_layers = nn.Sequential(
        self.conv_block(1, 32, 3),
        self.conv_block(32, 32, 3),
        nn.MaxPool2d(kernel_size=3, stride=2),
        self.conv_block(32, 64, 3),
        self.conv_block(64, 64, 3),
        nn.MaxPool2d(kernel_size=3, stride=2),
      )
        
      self.fc_layers = nn.Sequential(
        nn.Linear(2304, 1152),
        nn.ReLU(inplace=True),
        nn.Linear(1152, 576),
        nn.ReLU(inplace=True),
        nn.Linear(576, 10)
      )

      self.acc_metric = Accuracy(average='none', num_classes=10)

      self.acc_clean_metric = Accuracy()
      self.acc_noisy_metric = Accuracy()


    self.Z_bar = None

    self.max_acc = 0
    self.loss_log = []

    self.RL_const = RobustLoss(
      self.hparams["lambda_2"], 
      0, 
      'chi-square',
       max_iter=self.hparams["lambda_4"]
    )

    self.RL_reg = RobustLoss(
      float('inf'), 
      self.hparams["lambda_2"], 
      'chi-square',
       max_iter=self.hparams["lambda_4"]
    )

    self.RL_const_dual = DualRobustLoss(
      self.hparams["lambda_2"], 
      0, 
      'chi-square',
    )

    self.RL_reg_dual = DualRobustLoss(
      float('inf'), 
      self.hparams["lambda_2"], 
      'chi-square',
    )

    self.DV = DualVariance(self.hparams["lambda_2"])

    self.VarPen = VariancePenalization(self.hparams["lambda_2"], self.hparams["lambda_3"])
    self.DistVarPen = DistributionalVariancePenalization(self.hparams["lambda_2"], self.hparams["lambda_3"])


    num_classes = 100 if self.hparams["dataset"] == "CIFAR-100" else 10

    self.class_weights = torch.ones((num_classes,))

    if self.hparams["imbalanced"] and self.hparams["lambda_4"] != 0:
      # compute class weights when training with imbalanced data
      GROUP1 = [9, 2, 3, 5, 4]
      alpha  = self.hparams["lambda_4"]

      self.class_weights *= (1 - alpha)
      self.class_weights[GROUP1] *= alpha / (1 - alpha)

      # Make weights be mean 1 not to affect convergence speed
      self.class_weights *= 1 / self.class_weights.mean()

    # Metrics to compute for each validatiop dataset
    if self.hparams["imbalanced"]:
      self.ds_metrics = nn.ModuleList([
        Accuracy(),
        Accuracy(),
        Accuracy()
      ])
      self.ds_perclass_metrics = nn.ModuleList([
        Accuracy(average='none', num_classes=num_classes),
        Accuracy(average='none', num_classes=num_classes),
        Accuracy(average='none', num_classes=num_classes)
      ])
    else:
      self.ds_metrics = nn.ModuleList([
        Accuracy(),
        Accuracy()
      ])
      self.ds_perclass_metrics = nn.ModuleList([
        Accuracy(average='none', num_classes=num_classes),
        Accuracy(average='none', num_classes=num_classes)
      ])


  
  @staticmethod
  def conv_block(in_planes, out_planes, kernel_size=3):
    """Convolutional Block"""

    return nn.Sequential(
      nn.Conv2d(
        in_planes, 
        out_planes, 
        kernel_size=kernel_size, 
        padding=(kernel_size - 1) // 2
      ),
      nn.BatchNorm2d(out_planes),
      nn.ReLU(inplace=True)
    )


  @staticmethod
  def chi2_divergence(prob):
    return (prob.size()[0] * prob - 1).square().mean()


  @staticmethod
  def sample_utilization(prob, zero_threshold=0.0001):
    """Ratio of samples with a non-zero probability"""
    return (prob > zero_threshold).sum() / prob.size()[0]



  def forward(self, x):
    if self.hparams["dataset"] == "CIFAR-10":
      if not self.hparams["complex_model"]:
        return self.fc_layers(self.conv_layers(x).view(-1, 3136))
      else:
        return self.resnet(x)
    
    elif self.hparams["dataset"] == "CIFAR-100":
      return self.resnet(x)

    elif self.hparams["dataset"] == "MNIST":
      return self.fc_layers(self.conv_layers(x).view(-1, 2304))


  def training_step(self, batch, batch_idx):
    x, y_target, y_target_original = batch
    y_hat       = self(x)

    if y_hat.device != self.class_weights.device:
      self.class_weights = self.class_weights.to(y_hat.device)


    if self.hparams["loss_function"] == "moments":

      l1, l2  = self.hparams["lambda_1"], self.hparams["lambda_2"]

      # self.hparams["lambda_1"] = l1
      # self.hparams["lambda_2"] = l2
      # self.hparams["lambda_3"] = 0.0

      losses      = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
      loss, prob  = moments_penalization(losses, [l1, l2], normalize=self.hparams["normalize"])

      self.log_dict({
        "prob-chi2-divergence": self.chi2_divergence(prob / prob.sum()),
        "prob-chi2-normalization": prob.sum(),
        "prob-sample-utilization": self.sample_utilization(prob),
        "loss-variance": losses.var(),
      })

    elif self.hparams["loss_function"] == "newvarpen":

      losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none", weight=self.class_weights)
      loss    = self.VarPen(losses)

      self.log_dict({
        "prob-chi2-divergence": self.chi2_divergence(self.VarPen.prob),
        "prob-sample-utilization": self.sample_utilization(self.VarPen.prob),
        "loss-variance": losses.var(),
      })

    elif self.hparams["loss_function"] == "distributional-varpen":

      losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none", weight=self.class_weights)
      loss    = self.DistVarPen(losses)

      self.log_dict({
        "prob-chi2-divergence": self.chi2_divergence(self.DistVarPen.prob),
        "prob-sample-utilization": self.sample_utilization(self.DistVarPen.prob),
        "loss-variance": losses.var(),
      })

    elif self.hparams["loss_function"] == "focal-loss":

      losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
      weights = (1 - (-losses).exp()).pow(self.hparams["lambda_2"])

      loss = (weights * losses).mean()

      self.log_dict({
        "loss-variance": losses.var(),
      })

    elif self.hparams["loss_function"] == "moments-2stages":

      l1, l2  = self.hparams["lambda_1"], self.hparams["lambda_2"]

      losses      = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
      loss, prob  = moments_2stages(losses, self.hparams["lambda_2"], self.hparams["lambda_3"])

      self.log_dict({
        "prob-chi2-divergence": self.chi2_divergence(prob / prob.sum()),
        "prob-chi2-normalization": prob.sum(),
        "prob-sample-utilization": self.sample_utilization(prob),
        "loss-variance": losses.var(),
      })

    elif self.hparams["loss_function"] == "moments-dual":

      losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
      loss    = self.DV(losses)

      self.log_dict({
        "prob-eta": self.DV.eta,
        "prob-chi2-divergence": self.chi2_divergence(self.DV.prob),
        "prob-sample-utilization": self.sample_utilization(self.DV.prob),
        "loss-variance": losses.var(),
      })

    
    elif self.hparams["loss_function"] == "chi2-constraint":

      losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
      loss    = self.RL_const(losses)

      self.log_dict({
        "prob-chi2-divergence": self.chi2_divergence(self.RL_const.prob),
        "prob-sample-utilization": self.sample_utilization(self.RL_const.prob),
        "loss-variance": losses.var(),
      })


    elif self.hparams["loss_function"] == "chi2-regularized":

      losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
      loss    = self.RL_reg(losses)

      self.log_dict({
        "prob-chi2-divergence": self.chi2_divergence(self.RL_reg.prob),
        "prob-sample-utilization": self.sample_utilization(self.RL_reg.prob),
        "loss-variance": losses.var(),
      })

    elif self.hparams["loss_function"] == "chi2-constraint-dual":

      losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
      loss    = self.RL_const_dual(losses)

      self.log_dict({
        "prob-eta": self.RL_const_dual.eta,
        "prob-chi2-divergence": self.chi2_divergence(self.RL_const_dual.prob),
        "prob-sample-utilization": self.sample_utilization(self.RL_const_dual.prob),
        "loss-variance": losses.var(),
      })


    elif self.hparams["loss_function"] == "chi2-regularized-dual":

      losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
      loss    = self.RL_reg_dual(losses)

      self.log_dict({
        "prob-eta": self.RL_reg_dual.eta,
        "prob-chi2-normalization": self.RL_reg_dual.prob.sum(),
        "prob-chi2-divergence": self.chi2_divergence(self.RL_reg_dual.prob),
        "prob-sample-utilization": self.sample_utilization(self.RL_reg_dual.prob),
        "loss-variance": losses.var(),
      })

    elif self.hparams["loss_function"] == "direct-varpen":

      losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
      loss    = losses.mean() + self.hparams["lambda_2"] * losses.var().pow(self.hparams["lambda_3"])
      
      self.log_dict({
        "loss-variance": losses.var(),
      })

    elif self.hparams["loss_function"] == "moments-mae":

      l1, l2  = self.hparams["lambda_1"], self.hparams["lambda_2"]
      num_cls = 100 if self.hparams["dataset"] == "CIFAR-100" else 10

      losses  = MAE(y_hat, y_target, num_cls)
      loss    = moments_penalization(losses, [l1, l2], normalize=self.hparams["normalize"])

    elif self.hparams["loss_function"] == "robust-mean":

      eta, delta  = self.hparams["lambda_3"], self.hparams["lambda_3"]
      losses      = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
      loss        = robust_mean(losses, eta, delta)

    elif self.hparams["loss_function"] == "huber":

      l1, l2  = self.hparams["lambda_1"], self.hparams["lambda_2"]
      num_cls = 100 if self.hparams["dataset"] == "CIFAR-100" else 10

      losses  = HUBER(y_hat, y_target, num_cls)
      loss    = losses.mean()
      # loss    = moments_penalization(losses, [l1, l2], normalize=self.hparams["normalize"])

    elif self.hparams["loss_function"] == "moments-var":

      l1, l2  = self.hparams["lambda_1"], self.hparams["lambda_2"]

      # self.hparams["lambda_1"] = l1
      # self.hparams["lambda_2"] = l2
      # self.hparams["lambda_3"] = 0.0

      losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
      loss    = moments_penalization(losses, [l1, l2], normalize=self.hparams["normalize"])

    elif self.hparams["loss_function"] == "moments-square":

      l1, l2, l3  = self.hparams["lambda_1"], self.hparams["lambda_2"], self.hparams["lambda_3"]

      losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
      loss    = moments_penalization(losses, [l1, l2, l3], normalize=self.hparams["normalize"])

    elif self.hparams["loss_function"] == "moments-cubic":

      l1, l2, l3, l4  = self.hparams["lambda_1"], self.hparams["lambda_2"], self.hparams["lambda_3"], self.hparams["lambda_4"]

      losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
      loss    = moments_penalization(losses, [l1, l2, l3, l4], normalize=self.hparams["normalize"])
      
    elif self.hparams["loss_function"] == "moments-convex":

      l1, l2  = self.hparams["lambda_1"], self.hparams["lambda_2"]

      losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
      loss    = moments_penalization_cross_entropy(losses, l1, l2, convex=True)

    elif self.hparams["loss_function"] == "moments-skew":

      l1, l2, l3  = [1.0, -0.5, self.hparams["lambda_3"]]

      self.hparams["lambda_1"] = l1
      self.hparams["lambda_2"] = l2

      losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
      loss    = moments_penalization(losses, [l1, l2, l3])

    elif self.hparams["loss_function"] == "moments-mean":

      l1, l2  = self.hparams["lambda_1"], self.hparams["lambda_2"]

      self.hparams["lambda_1"] = l1
      self.hparams["lambda_2"] = l2
      self.hparams["lambda_3"] = 0.0

      alpha       = self.hparams["moments_alpha"]
      num_classes = 100 if self.hparams["dataset"] == "CIFAR-100" else 10

      losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
      z       = losses.detach()

      # Compute mean loss for each class
      if not hasattr(self, 'class_mean'):
        self.class_mean = torch.zeros(num_classes, device=z.device)

      z_mean = torch.zeros_like(z)
      for c in range(num_classes):
        mask               = (y_target == c)
        if mask.any():
          self.class_mean[c] = alpha * z[mask].mean() + (1.0 - alpha) * self.class_mean[c]
          z_mean[mask]       = self.class_mean[c]

      if self.hparams["wandb_log_hist"]:
        wandb.log({"class-mean": wandb.Histogram(self.class_mean.cpu().numpy())})

      loss    = moments_penalization(losses, [l1, l2], z_mean=z_mean)

    elif self.hparams["loss_function"] == "normalized-symmetric":

      loss1   = normalized_cross_entropy(y_hat, y_target, num_classes=100 if self.hparams["dataset"] == "CIFAR-100" else 10)
      loss2   = reverse_cross_entropy(y_hat, y_target, num_classes=100 if self.hparams["dataset"] == "CIFAR-100" else 10)

      if self.hparams["dataset"] == "CIFAR-100":
        loss    = 10 * loss1 + 0.1 * loss2
      else:
        loss    = 10 * loss1 + 1 * loss2
      
    elif self.hparams["loss_function"] == "term":

      losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
      loss    = term_transformation(losses, t=self.hparams["lambda_2"])

    elif self.hparams["loss_function"] == "generalized":

      loss    = generalized_cross_entropy(y_hat, y_target, q=self.hparams["lambda_2"], num_classes=100 if self.hparams["dataset"] == "CIFAR-100" else 10)

    elif self.hparams["loss_function"] == "taylor":

      if self.hparams["dataset"] == "CIFAR-100":
        loss    = taylor_cross_entropy(y_hat, y_target, t=6, num_classes=100)
      else:
        loss    = taylor_cross_entropy(y_hat, y_target, t=2, num_classes=10)

    elif self.hparams["loss_function"] == "symmetric":
      
      loss1   = nn.functional.cross_entropy(y_hat, y_target)
      loss2   = reverse_cross_entropy(y_hat, y_target, num_classes=100 if self.hparams["dataset"] == "CIFAR-100" else 10)

      if self.hparams["dataset"] == "CIFAR-100":
        loss    = self.hparams["lambda_2"] * loss1 + self.hparams["lambda_3"] * loss2
      else:
        loss    = self.hparams["lambda_2"] * loss1 + self.hparams["lambda_3"] * loss2

    elif self.hparams["loss_function"] == "classical":

      loss    = nn.functional.cross_entropy(y_hat, y_target, weight=self.class_weights)


    self.log("train_loss", loss)

    return loss


  def validation_step(self, batch, batch_idx, dataloader_idx):
    """
    Track accuracies per class and per dataset
    """
    x, y_target, y_target_original = batch
    y_hat       = self(x)
    y_pred      = torch.softmax(y_hat, dim=1).argmax(dim=1)

    self.ds_metrics[dataloader_idx](y_pred, y_target)
    self.ds_perclass_metrics[dataloader_idx](y_pred, y_target)

    # Log additiona info for the noisy validationa dataset
    if self.hparams["wandb_log_hist"] and self.hparams["noise_rate"] and dataloader_idx == 0:

      if self.hparams["loss_function"] == "moments":

        l1, l2  = self.hparams["lambda_1"], self.hparams["lambda_2"]

        losses      = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
        loss, prob  = moments_penalization(losses, [l1, l2], normalize=self.hparams["normalize"])

        return {
          "losses": losses.tolist(),
          "prob": prob.tolist(),
          "isclean": (y_target == y_target_original).tolist()
        }

      elif self.hparams["loss_function"] == "chi2-constraint-dual":

        losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
        loss    = self.RL_const_dual(losses)

        return {
          "losses": losses.tolist(),
          "prob": self.RL_const_dual.prob.tolist(),
          "isclean": (y_target == y_target_original).tolist()
        }

      elif self.hparams["loss_function"] == "distributional-varpen":

        losses  = nn.functional.cross_entropy(y_hat, y_target, reduction="none")
        loss    = self.DistVarPen(losses)

        return {
          "losses": losses.tolist(),
          "prob": self.DistVarPen.prob.tolist(),
          "isclean": (y_target == y_target_original).tolist()
        }

    return {} 


  def validation_epoch_end(self, validation_outputs):
    """
    Compute and log accuracies per class and per dataset
    """

    log_data = {}

    for idx, metric in enumerate(self.ds_metrics):
      log_data["acc-" + self.valid_ds_names[idx]] = metric.compute()

    self.log_dict(log_data)

    for idx, metric in enumerate(self.ds_perclass_metrics):
      class_acc = metric.compute()

      self.log_dict({ "class-{}-acc-{}".format(c, self.valid_ds_names[idx]): a for c, a in enumerate(class_acc)})


    validation_data = {}
    
    # Aggregate batches into one table and save
    for validation_datasets in validation_outputs:

      # skip lists of empty objects
      if not all(validation_datasets):
        continue

      for vd in validation_datasets:

        for k, v in vd.items():
          if k in validation_data:
            validation_data[k] += v
          else:
            validation_data[k]  = v

    if validation_data:
      # print(validation_data)
      table = wandb.Table(
        columns = list(validation_data.keys()),
        data = list(zip(*validation_data.values()))
      )
    
      wandb.log({"prob-tracking-noisy": table})

      
      

  def configure_optimizers(self):
    if self.hparams["dataset"] in ["CIFAR-10", "MNIST"]:
      
      if not self.hparams["complex_model"]:
        main_optim = torch.optim.SGD(
          [
            {
              "params": self.conv_layers.parameters(), 
              "weight_decay": self.hparams["conv_l2_reg"],
            },
            {
              "params": self.fc_layers.parameters(), 
              "weight_decay": self.hparams["fc_l2_reg"],
            },
            {
              "params": self.RL_const_dual.parameters(),
            },
            {
              "params": self.DV.parameters(),
            }
          ],
          lr = self.hparams["learning_rate"],
          momentum = 0.9,
          nesterov = True,
        )
      else:
        main_optim = torch.optim.SGD(
          params = self.resnet.parameters(),
          weight_decay = self.hparams["conv_l2_reg"],
          lr = self.hparams["learning_rate"],
          momentum = 0.9,
          nesterov = True,
        )
    
    elif self.hparams["dataset"] == "CIFAR-100":
      
      main_optim = torch.optim.SGD(
        [
          {
            "params": self.resnet.parameters(), 
            "weight_decay": self.hparams["conv_l2_reg"],
          },
          {
            "params": self.RL_const_dual.parameters(),
          },
          {
            "params": self.DV.parameters(),
          }
        ],
        lr = self.hparams["learning_rate"],
        momentum = 0.9,
        nesterov = True,
      )

    if self.hparams["lr_scheduler"] == "stepped":
      lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        main_optim, 
        milestones=[self.hparams["lr_milestone"], 2*self.hparams["lr_milestone"]], 
        gamma=0.1
      )
    elif self.hparams["lr_scheduler"] == "continuous":
      lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        main_optim, 
        lr_lambda = lambda epoch: 1.0 / 10 ** (epoch / self.hparams["lr_milestone"]), 
      )

    return [main_optim], [lr_scheduler]


  def prepare_data(self):
    """Downloads and splits CIFAR dataset into training, validation, and test"""

    if self.hparams["dataset"] in ["CIFAR-10", "CIFAR-100"]:
      CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
      CIFAR_STD  = [0.24703233, 0.24348505, 0.26158768]

      preprocess_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
      ])

      if self.hparams["dataset"] == "CIFAR-10":
        train_ds = CIFAR10NOISY('data/', True, preprocess_transform, download=True, 
                                    asym = self.hparams["asym_noise"], 
                                    nosiy_rate = self.hparams["noise_rate"],
                                    imbalanced=self.hparams["imbalanced"])
        valid_ds_clean = CIFAR10NOISY('data/', False, preprocess_transform, download=True,
                                    asym = False, nosiy_rate = 0)

      else:
        train_ds = CIFAR100NOISY('data/', True, preprocess_transform, download=True, 
                                    asym = self.hparams["asym_noise"], 
                                    nosiy_rate = self.hparams["noise_rate"],
                                    imbalanced=self.hparams["imbalanced"])
        valid_ds_clean = CIFAR100NOISY('data/', False, preprocess_transform, download=True,
                                    asym = False, nosiy_rate = 0)

    elif self.hparams["dataset"] == "MNIST":
      transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
      ])

      train_ds = FMNISTNOISY('data/', True, transform, download=True, 
                                  asym = self.hparams["asym_noise"], 
                                  nosiy_rate = self.hparams["noise_rate"],
                                  imbalanced=self.hparams["imbalanced"])
      valid_ds_clean = FMNISTNOISY('data/', False, transform, download=True,
                                  asym = False, nosiy_rate = 0)


    # When training with imbalanced classes we use 3 validation datasets
    if self.hparams["imbalanced"]:

      # train_ds_noisy, valid_ds_noisy = train_ds.random_split(self.hparams["train_ratio"])
      ds_imbalanced, test_ds_imbalanced = train_ds.random_split(self.hparams["train_ratio"])

      # Split the dataset into train and validation
      train_size  = int(len(ds_imbalanced) * 0.9)
      valid_size = len(ds_imbalanced) - train_size

      train_ds_imbalanced, valid_ds_imbalanced = random_split(ds_imbalanced, [train_size, valid_size])

      print("#"*20)
      print("Training dataset size:", len(train_ds_imbalanced))
      print("Validation dataset sizes:", (len(valid_ds_imbalanced), len(test_ds_imbalanced), len(valid_ds_clean)))
      print("#"*20)

      self.train_ds =  train_ds_imbalanced
      self.valid_ds = (valid_ds_imbalanced, test_ds_imbalanced, valid_ds_clean)
      self.valid_ds_names = ("valid-imbalanced", "test-imbalanced", "test-balanced")


    # When training on noisy label we use to validation datasets (clean and noisy)
    else:
      train_ds_noisy, valid_ds_noisy = train_ds.random_split(self.hparams["train_ratio"])
      
      print("#"*20)
      print("Training dataset size:", len(train_ds_noisy))
      print("Validation dataset sizes:", (len(valid_ds_noisy), len(valid_ds_clean)))
      print("#"*20)

      self.train_ds =  train_ds_noisy
      self.valid_ds = (valid_ds_noisy, valid_ds_clean)
      self.valid_ds_names = ("noisy", "clean")


  def train_dataloader(self):

    if self.hparams["dataset"] in ["CIFAR-10", "CIFAR-100"]:

      # Compile transforms as a torch script for speed
      training_transforms = torch.jit.script(torch.nn.Sequential(
        transforms.RandomCrop(32, padding=[4,]),
        transforms.RandomHorizontalFlip(),
      ))

    else:
      training_transforms = None

    imgs, noisy_labels, true_labels, _ = zip(*(self.train_ds[i] for i in range(len(self.train_ds))))

    imgs = torch.stack(imgs)
    noisy_labels = torch.tensor(noisy_labels)
    true_labels  = torch.tensor(true_labels)

    return MemoryIterableDataset(imgs, noisy_labels, true_labels, 
      self.hparams["batch_size"], self.device, transformations=training_transforms,
      random_batch=True,
    )

  def val_dataloader(self):

    val_dl_list = []

    for ds in self.valid_ds:
      imgs, noisy_labels, true_labels, _ = zip(*(ds[i] for i in range(len(ds))))

      imgs = torch.stack(imgs)
      noisy_labels = torch.tensor(noisy_labels)
      true_labels  = torch.tensor(true_labels)

      val_dl_list.append(MemoryIterableDataset(imgs, noisy_labels, true_labels, 
        self.hparams["valid_batch_size"], self.device, drop_last=False
      ))

    return val_dl_list
