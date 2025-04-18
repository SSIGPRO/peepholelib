# v.0.0.0

## general
- update `utils/testing.py` to `utils/samplers.py`

## SVDs
- Support `torch.nn.Conv2d` layers with no bias, and with groups
- `model.svd` implements `channel_wise` SVD for Conv2D layers per default.

## Corevectors
- Images, Labels, Results, and Outputs were moved to the `corevector._actds` instead of `corevector._cvsds`
- `get_corevectors()` now accepts a generic dimensionality reduction function
- `CoreVectors` implements `ChannelWiseMean_conv()` as `average pooling` dimemsionality reduction 

## Peepholes
- `ClassifierBase` receives a `CoreVectors` class as argument instead of dataloader
- `ClassifierBase` can be saved and loaded. It receives `path` and `name` arguments for saving.
- Peepholes now accept multiple layers as `target_layers` argument
- `Peepholes` objects now compute peepholes using a `DrillBase` class
- `Peepholes` implements [Deep Mahalanobis Distance](https://arxiv.org/abs/1807.03888) as peephole computation
- Framework now accepts any `torch.nn.Module` for computing `corevectors` and `peepholes`, appropriate functions need to be provided. `target_layers` now are `target_modules`

