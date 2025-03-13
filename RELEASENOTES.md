# v.0.0.0
- update `utils/testing.py` to `utils/samplers.py`
- Support `torch.nn.Conv2d` layers with no bias, and with groups
- Peepholes now accept multiple layers as `target_layers` argument
- Images, Labels, Results, and Outputs were moved to the `corevector._actds` instead of `corevector._cvsds`
- `get_corevectors()` now accepts a generic dimensionality reduction function
