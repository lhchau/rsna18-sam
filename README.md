# Experiment 

## Hyperparameter setting

Model: ResNet18, ResNet50, ResNet101, Wide ResNet-28-10

### SAM

```
batchsize: 256, epochs: 200, scheduler: CosineScheduler, rho: 0.05, lr: 0.1, weight_decay: 0.0005 or 0.0001, momentum: 0.9
```

### GSAM

```
ResNet50: 
rho_max: 0.04, rho_min: 0.02, alpha: 0.01, lr_max: 1.6, lr_min: 1.6e-2, wd: 0.3, base: SGD, scheduler: LinearScheduler, epochs: 90, warmup steps: 5k

ResNet101:
rho_max: 0.04, rho_min: 0.02, alpha: 0.01, lr_max: 1.6, lr_min: 1.6e-2, wd: 0.3, base: SGD, scheduler: LinearScheduler, epochs: 90, warmup steps: 5k
```

### SAF

```
ResNet:
batchsize: 4096, lr: 1.4, epochs: 90 epochs 
scheduler: CosineScheduler
```

### GAM

```
ResNets, WideResNet, ResNeXt, PyramidNet and Vision Transformer (ViTs)
epochs: 200 (scratch), 
basic data augmentation: horizontal flip, padding 4 pixels, and random crop
advanced data augmentation: cutout regularization, RandAugment and AutoAugment
```

## Experimental Observation

- During the first half of the training, discarding the second order terms does not impact the general direction of the training, as the cosine similarity between the first and second order updates are very close to 1. However, when the model nears convergence, the similarity between both types of updates becomes weaker