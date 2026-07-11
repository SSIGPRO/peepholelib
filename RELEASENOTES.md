# DEVEL

## Datasets
- `random_subsampling` and `dist_preserving` now accept `n_samples` (int or dict) instead of a fraction.
- Add `balanced_subsampling`: samples a fixed number of examples per class using `WeightedRandomSampler`.

## Model
- `ModelWrap` now extends `nn.Module` instead of using `ABCMeta`. `forward()` replaces `__call__`, and `set_normalizer()` replaces `prepend_normalizer()`.
- `InputNormalizer` buffers registered via `register_buffer` for correct device handling.

## Peepholes
- Add `MRC` (Multi-Range Coverage) driller.

## Scores
- Add `CAM_exp_score` and `CAM_lin_score`.

## Training
- Refactor training utilities: `trainingLoops`, `validationLoops`, `savingLoops` replaced by `train_loops`, `val_loops`, `test_loops`, `save_fns`, `load_fns`.
- `Trainer` is no longer abstract; loop functions are injected via `partial` and can be swapped without subclassing.
- `Trainer` takes `datasets`, `dataloader_kwargs`, and `iterations` as dicts keyed by `dataset_key`. `DataLoader`s are built once in `__init__`.
- Rename `Trainer.validation_loop` to `Trainer.val_loop`.
- Fix `save_every=0` crash, phase-number detection for names containing digits, and resume crash when a checkpoint saved without a scheduler is loaded into a scheduler-configured run.

# v.0.0.2

## Datasets
- Implements MNIST and DTD (Textures)
- Move transforms for `parsedDatase` class. Dataset values are saved raw.
- Separate the parsed dataset and inference values into different `PTD`s. Add `parse_dataset()` and `parse_inference()` functions for each.
- Add `inference_fn` for parsing datasets.
- Implement AWA e CUB wrappers

## SVDs
- Implement `kernel` SVD reduction for convolution layers.
- move SVDs to dimentionality reduction class in `coreVectors/dimReduction`.

## Corevectors
- Normalization always apply. A normalization is saved for each `loader` and `module`, and corevectors are denormalized and renormalized.
- Model's modules are saved in different PTDs
- Wrap dimentionality reduction within classes.
- Implement dimentionality reduction base.

## Peepholes
- Add `retry_load_time` to sleep on a load fail. Usefull for tuning.
- Model's modules are saved in different PTDs
- add Flag to control wether or not call `_compute_empirical_posteriors()` in classifiers' `fit()`. 
- Classifiers now call `compute_empirical_posteriors` inside `fit` to have a common interface with `DMD`.
- drillers' `load()` returns a bool indicating if it has been fitted and saved as common interface for checking it.
- Move corevectors parsing (old `trim_corevectors`) inside respective dimentionality reduction class.
- remove the `peepholes` key from `peepholes/peepholes.Peepholes._phs[<loaders>][<layer>]`.
- remove suffix from `peepholelib/peepholes/drill_base.Drill_base`, it is supposed to be managed at experiment level.

# v.0.0.1

## general
- update `utils/testing.py` to `utils/samplers.py`

## Datasets
- Adds MNIST and DTD (Textures)
- Rework `datasets`. Now each dataset is istantiated individually, previous `coreVectors.parse_ds()` is moved to `DatasetBase`

## Model
- Rework SVDs. SVD functions for each layer are passed to `get_svds()`.
- add function `update_output` to `model_wrap`.
- Remove `dry_run()`. Corevectors run a dry image internally.
- Remove `add_hooks()`, not it is done in `set_target_modules()`.
- Add `set_activations()` function, which set the model to save activation in `model._acts` or to not save activations.

## SVDs
- Implement [`kernel_svd`](https://arxiv.org/pdf/2208.06894) for `Conv2d` layers. 
- Support `torch.nn.Conv2d` layers with no bias, and with groups.
- `model.svd` implements `channel_wise` SVD for Conv2D layers per default.

## Corevectors
- Move `parse_ds` to `datasets.DatasetBase`
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

## Evaluation
- Many scores are implemented
