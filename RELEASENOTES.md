# v.0.0.0

## general
- update `utils/testing.py` to `utils/samplers.py`

## Model
- Rework SVDs. SVD functions for each layer are passed to `get_svds()`.
- add function `update_output` to `model_wrap`.
- Remove dry_run(). Corevectors run a dry image internally.
- Remove `add_hooks()`, not it is done in `set_target_modules()`.
- Add `set_activations()` function, which set the model to save activation in `model._acts` or to not save activations.

## SVDs
- Implement [`kernel_svd`](https://arxiv.org/pdf/2208.06894) for `Conv2d` layers. 
- Support `torch.nn.Conv2d` layers with no bias, and with groups.
- `model.svd` implements `channel_wise` SVD for Conv2D layers per default.

## Corevectors
- Add `conv2d_kernel_svd_projection`. Update names of SVD projection functions. 
- Remove the necessity of saving activations. add `parse_ds()` to get dataset information. If needed, e.g., in the DMD case, activations can be extracted using `get_activations()` as before.
- Images, Labels, Results, and Outputs were moved to the `corevector._actds` instead of `corevector._cvsds`.
- `get_corevectors()` now accepts a generic dimensionality reduction function.
- `CoreVectors` implements `ChannelWiseMean_conv()` as `average pooling` dimemsionality reduction.

## Peepholes
- Add `trim_kernel_corevectors()` 
- `ClassifierBase` receives a `CoreVectors` class as argument instead of dataloader.
- `ClassifierBase` can be saved and loaded. It receives `path` and `name` arguments for saving.
- Peepholes now accept multiple layers as `target_layers` argument.
- `Peepholes` objects now compute peepholes using a `DrillBase` class.
- `Peepholes` implements [Deep Mahalanobis Distance](https://arxiv.org/abs/1807.03888) as peephole computation.
- Framework now accepts any `torch.nn.Module` for computing `corevectors` and `peepholes`, appropriate functions need to be provided. `target_layers` now are `target_modules`.

## Attack detection
- Implement [Feature Squeezing](https://arxiv.org/abs/1704.01155) as attack detection method.
