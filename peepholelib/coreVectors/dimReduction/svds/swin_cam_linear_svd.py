# torch stuff
import torch

# Our stuff
from .vit_linear_svd import ViTLinearSVD

class SwinCAMLinearSVD(ViTLinearSVD):
    def __init__(self, **kwargs):
        '''
        SVD projection of `torch.Linear` activations in Swin transformers, reducing the tokens with the Class Activation Map (CAM, Zhou et al. 2016) of the predicted class. Swin has no class token: its head is `norm -> average pool -> head`, so `logit_y = mean_t(w_y . norm(x_t)) + b_y` and `cam_t = w_y . norm(x_t)` is the exact contribution of the final token `t` to the predicted logit `y`. The CAM is computed from the same forward pass used to get the activations, with a single hook on `model._model.norm` shared by all instances on the same model (see `remove_hook()`), and mapped to the layer's token grid (e.g. each final token covers 2x2 tokens of the previous stage).
        SVD computation, caching and `parser()` are the same as `ViTLinearSVD`.

        Args:
        - k (int): if given, the corevector is the mean of the `k` tokens with the highest CAM. If `None`, it is the mean weighted by the positive CAM values (uniform if none is positive). Defaults to `None`.
        - Other args as in `ViTLinearSVD`.
        '''
        ViTLinearSVD.__init__(self, **kwargs)
        model = kwargs['model']
        self.k = kwargs.get('k', None)

        self.head = model._model.head

        # a single hook per model, shared by all reducers, storing the last norm output on the module itself
        self.norm = model._model.norm
        if not hasattr(self.norm, '_cam_handle'):
            self.norm._cam_handle = self.norm.register_forward_hook(lambda module, inputs, output: setattr(module, '_cam_out', output.detach()))
        return

    def remove_hook(self):
        '''
        Removes the hook on `model._model.norm` and its stored output. Affects all `SwinCAMLinearSVD` sharing the same model.
        '''
        if hasattr(self.norm, '_cam_handle'):
            self.norm._cam_handle.remove()
            del self.norm._cam_handle
        if hasattr(self.norm, '_cam_out'):
            del self.norm._cam_out
        return

    def __call__(self, **kwargs):
        '''
        Applies the SVD projection to every token of `torch.Linear` activations, and reduces the tokens using the CAM of the predicted class. The output has shape `[ns, q]`, where `ns` is the number of samples in the batch, and `q` the SVD rank.

        Args:
        - act_data (torch.tensor): batched input activations, `[ns, h, w, c]`

        Returns:
        - cvs (torch.tensor) = batched projected activations
        '''
        act_data = kwargs['act_data']
        n_act, h = act_data.shape[0], act_data.shape[1]

        # CAM of the predicted class on the final tokens, [ns, h_f, w_f]
        norm_out = self.norm._cam_out
        pred = self.head(norm_out.mean(dim=(1, 2))).argmax(dim=1)
        cam = (norm_out*self.head.weight[pred][:, None, None, :]).sum(dim=-1)

        # map the CAM to the layer's token grid, [ns, n_tokens]
        f = h//cam.shape[1]
        cam = cam.repeat_interleave(f, dim=1).repeat_interleave(f, dim=2).flatten(start_dim=1)

        # project every token, [ns, n_tokens, q]
        acts = act_data.flatten(start_dim=1, end_dim=2)
        if self.use_bias:
            acts = torch.cat((acts, torch.ones_like(acts[..., :1])), dim=-1)
        cvs = acts@self.reduct_m.T

        if self.k is not None:
            idx = cam.topk(self.k, dim=1).indices
            return cvs[torch.arange(n_act, device=cvs.device)[:, None], idx].mean(dim=1)

        w = cam.clamp(min=0)
        w_sum = w.sum(dim=1, keepdim=True)
        w = torch.where(w_sum > 0, w/w_sum.clamp(min=1e-12), 1/w.shape[1])
        return (w[..., None]*cvs).sum(dim=1)
