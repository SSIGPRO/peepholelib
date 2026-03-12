import torch
import time
import math
from autoattack.autopgd_base import APGDAttack, APGDAttack_targeted
from .attack_base import AttackBase

import torch.nn as nn
import torch.nn.functional as F
from autoattack.other_utils import L0_norm, L1_norm, L2_norm
from autoattack.checks import check_zero_gradients
from autoattack.autopgd_base import L1_projection

class APGDTnew(APGDAttack_targeted):
    def attack_single_run(self, x, y, x_init=None):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L1':
            t = torch.randn(x.shape).to(self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = x + t + delta
        
        if not x_init is None:
            x_adv = x_init.clone()
            if self.norm == 'L1' and self.verbose:
                print('[custom init] L1 perturbation {:.5f}'.format(
                    (x_adv - x).abs().view(x.shape[0], -1).sum(1).max()))
            
        
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]
            ).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]
            ).to(self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        if not self.is_tf_model:
            if self.loss == 'ce':
                criterion_indiv = nn.CrossEntropyLoss(reduction='none')
            elif self.loss == 'ce-targeted-cfts':
                criterion_indiv = lambda x, y: -1. * F.cross_entropy(x, y,
                    reduction='none')
            elif self.loss == 'dlr':
                criterion_indiv = self.dlr_loss
            elif self.loss == 'dlr-targeted':
                criterion_indiv = self.dlr_loss_targeted
            elif self.loss == 'ce-targeted':
                criterion_indiv = self.ce_loss_targeted
            else:
                raise ValueError('unknowkn loss')
        else:
            if self.loss == 'ce':
                criterion_indiv = self.model.get_logits_loss_grad_xent
            elif self.loss == 'dlr':
                criterion_indiv = self.model.get_logits_loss_grad_dlr
            elif self.loss == 'dlr-targeted':
                criterion_indiv = self.model.get_logits_loss_grad_target
            else:
                raise ValueError('unknowkn loss')
        
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            if not self.is_tf_model:
                with torch.enable_grad():
                    logits = self.model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            else:
                if self.y_target is None:
                    logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y)
                else:
                    logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y,
                        self.y_target)
                grad += grad_curr
        
        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        if self.loss in ['dlr', 'dlr-targeted']:
            # check if there are zero gradients
            check_zero_gradients(grad, logger=self.logger)
        # again KAMI commented following two lines
        # acc = logits.detach().max(1)[1] == y
        # acc_steps[0] = acc + 0
        pred0 = logits.detach().max(1)[1]
        success0 = (pred0 == self.y_target)
        acc = (~success0).float()
        acc_steps[0] = acc + 0
        # up to here was changed

        loss_best = loss_indiv.detach().clone()

        alpha = 2. if self.norm in ['Linf', 'L2'] else 1. if self.norm in ['L1'] else 2e-2
        step_size = alpha * self.eps * torch.ones([x.shape[0], *(
            [1] * self.ndims)]).to(self.device).detach()
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        n_fts = math.prod(self.orig_dim)
        if self.norm == 'L1':
            k = max(int(.04 * self.n_iter), 1)
            if x_init is None:
                topk = .2 * torch.ones([x.shape[0]], device=self.device)
                sp_old =  n_fts * torch.ones_like(topk)
            else:
                topk = L0_norm(x_adv - x) / n_fts / 1.5
                sp_old = L0_norm(x_adv - x)
            #print(topk[0], sp_old[0])
            adasp_redstep = 1.5
            adasp_minstep = 10.
            #print(step_size[0].item())
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)
        n_reduced = 0

        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                        x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(
                        x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                        x - self.eps), x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * self.normalize(grad)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)

                elif self.norm == 'L1':
                    grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                    topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                    grad_topk = grad_topk[u, topk_curr].view(-1, *[1]*(len(x.shape) - 1))
                    sparsegrad = grad * (grad.abs() >= grad_topk).float()
                    x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                        L1_norm(sparsegrad.sign(), keepdim=True) + 1e-10)
                    
                    delta_u = x_adv_1 - x
                    delta_p = L1_projection(x, delta_u, self.eps)
                    x_adv_1 = x + delta_u + delta_p
                    
                    
                x_adv = x_adv_1 + 0.

            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                if not self.is_tf_model:
                    with torch.enable_grad():
                        logits = self.model(x_adv)
                        loss_indiv = criterion_indiv(logits, y)
                        loss = loss_indiv.sum()
    
                    grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                else:
                    if self.y_target is None:
                        logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y)
                    else:
                        logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y, self.y_target)
                    grad += grad_curr
            
            grad /= float(self.eot_iter)
            # KAMI commented following to enable targer comparison
            pred_label = logits.detach().max(1)[1]
            success = (pred_label == self.y_target)

            acc = torch.min(acc, (~success).float())
            acc_steps[i + 1] = acc + 0

            ind_succ = success.nonzero(as_tuple=False).view(-1)
            if ind_succ.numel() > 0:
                x_best_adv[ind_succ] = x_adv[ind_succ].detach()
            # pred_label = logits.detach().max(1)[1]
            # success = (pred_label == self.y_target)
            # new_succ = success & (acc > 0.5)            
            # acc = torch.min(acc, (~success).float())
            # acc_steps[i + 1] = acc + 0
            # ind_succ = new_succ.nonzero(as_tuple=False).view(-1)
            # if ind_succ.numel() > 0:
            #     x_best_adv[ind_succ] = x_adv[ind_succ].detach()
            # up to here was changed

            if self.verbose:
                str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                    step_size.mean(), topk.mean() * n_fts) if self.norm in ['L1'] else ''
                print('[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
                    i, loss_best.sum(), acc.float().mean(), str_stats))
                #print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))
            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1 + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0

              counter3 += 1

              if counter3 == k:
                  if self.norm in ['Linf', 'L2']:
                      fl_oscillation = self.check_oscillation(loss_steps, i, k,
                          loss_best, k3=self.thr_decr)
                      fl_reduce_no_impr = (1. - reduced_last_check) * (
                          loss_best_last_check >= loss_best).float()
                      fl_oscillation = torch.max(fl_oscillation,
                          fl_reduce_no_impr)
                      reduced_last_check = fl_oscillation.clone()
                      loss_best_last_check = loss_best.clone()
    
                      if fl_oscillation.sum() > 0:
                          ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                          step_size[ind_fl_osc] /= 2.0
                          n_reduced = fl_oscillation.sum()
    
                          x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                          grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                      k = max(k - self.size_decr, self.n_iter_min)
                  
                  elif self.norm == 'L1':
                      sp_curr = L0_norm(x_best - x)
                      fl_redtopk = (sp_curr / sp_old) < .95
                      topk = sp_curr / n_fts / 1.5
                      step_size[fl_redtopk] = alpha * self.eps
                      step_size[~fl_redtopk] /= adasp_redstep
                      step_size.clamp_(alpha * self.eps / adasp_minstep, alpha * self.eps)
                      sp_old = sp_curr.clone()
                  
                      x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                      grad[fl_redtopk] = grad_best[fl_redtopk].clone()
                  
                  counter3 = 0
                  #k = max(k - self.size_decr, self.n_iter_min)

        #
        
        return (x_best, acc, loss_best, x_best_adv)
class perturbnew(APGDTnew): # APGDAttack_targeted was OG
    def perturb(self, x, y=None, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        """

        assert self.loss in ['dlr-targeted'] #'ce-targeted'
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        if not self.is_tf_model:
            y_pred = self.model(x).max(1)[1]
        else:
            y_pred = self.model.predict(x).max(1)[1]
        if y is None:
            #y_pred = self._get_predicted_label(x)
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        # acc = y_pred == y
        acc = torch.ones_like(y_pred).float() # assume none has hit target in the begining
        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(
                self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        startt = time.time()
        
        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
                .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1])
            if self.verbose:
                print('using schedule [{}x{}]'.format('+'.join([str(c
                    ) for c in epss]), '+'.join([str(c) for c in iters])))
        
        for counter in range(self.n_restarts):
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0:
                ind_to_fool = ind_to_fool.unsqueeze(0)

            if ind_to_fool.numel() != 0:
                x_to_fool = x[ind_to_fool].clone()
                y_to_fool = y[ind_to_fool].clone()

                full_target = self.y_target # slice to match the ones unfooled, otherwise dimensions disagree
                self.y_target = full_target[ind_to_fool].detach().clone() 

                best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(
                    x_to_fool, y_to_fool, x_init=x_init
                )
                
                # trial, what was I even trialing?
                with torch.no_grad():
                    logits = self.model(adv_curr)
                    pred = logits.argmax(1)
                    success = (pred == self.y_target)
                    self.y_target = full_target
                ind_success = success.nonzero().view(-1)
                adv[ind_to_fool[ind_success]] = adv_curr[ind_success]
                acc[ind_to_fool[ind_success]] = 0
        return adv

class myAPGD(AttackBase):
    def __init__(self, **kwargs):
        """
        AutoPGD
        https://arxiv.org/abs/2003.01690

        :param predict:       forward pass function
        :param norm:          Lp-norm of the attack ('Linf', 'L2', 'L0' supported)
        :param n_restarts:    number of random restarts
        :param n_iter:        number of iterations
        :param eps:           bound on the norm of perturbations
        :param seed:          random seed for the starting point
        :param loss:          loss to optimize ('ce', 'dlr' supported)
        :param eot_iter:      iterations for Expectation over Trasformation
        :param rho:           parameter for decreasing the step size
        """
        AttackBase.__init__(self, **kwargs)

        self.norm = kwargs.get('norm', 'Linf')
        self.eps = kwargs.get('eps', 8 / 255) # 0.5/255
        self.steps = kwargs.get('steps', 300) # 100
        self.n_restarts = kwargs.get('n_restarts', 20) #1
        self.loss = kwargs.get('loss', 'dlr') #ce
        self.verbose = kwargs.get('verbose', False)
        self.rho = kwargs.get('rho', 0.75)
        self.eot_iter = kwargs.get('eot_iter', 1)

        self.targeted = kwargs.get('targeted', False)
        self.target_mode = kwargs.get('target_mode', None)
        self.target_class = kwargs.get('target_class', None)

        if self.targeted:
            self.atk = perturbnew(
                self.model._model,
                norm=self.norm,
                eps=self.eps,
                n_iter=self.steps,
                n_restarts=self.n_restarts,
                eot_iter=self.eot_iter,
                rho=self.rho,
                verbose=self.verbose
            )
        else:
            self.atk = APGDAttack(
                self.model._model,
                norm=self.norm,
                eps=self.eps,
                n_iter=self.steps,
                n_restarts=self.n_restarts,
                loss=self.loss,
                eot_iter=self.eot_iter,
                rho=self.rho,
                verbose=self.verbose
            )

    def __call__(self, images, labels):

        # Untargeted
        if not self.targeted:
            return self.atk.perturb(images, labels)

        # ---- force deterministic logits for target selection ---- I should remove this part in future, wasnt needed I am sure
        was_training = self.model._model.training
        self.model._model.eval()
        with torch.no_grad():
            logits = self.model._model(images)
        if was_training:
            self.model._model.train()
        # ----------------------------------------------------------

        num_classes = logits.shape[1]

        # targeted

        if self.target_mode == "least_likely":
            sorted_idx = logits.sort(dim=1)[1]
            y_target = sorted_idx[:, 0]

            # avoid true label
            mask = y_target == labels
            if mask.any():
                y_target[mask] = sorted_idx[mask, 1]

        elif self.target_mode == "random":
            y_target = torch.randint(
                0, num_classes, labels.shape, device=labels.device
            )

            mask = y_target == labels
            while mask.any():
                y_target[mask] = torch.randint(
                    0, num_classes, y_target[mask].shape, device=labels.device
                )
                mask = y_target == labels

        elif self.target_mode == "fixed":
            y_target = torch.full_like(labels, self.target_class)

        self.atk.y_target = y_target

        return self.atk.perturb(images, labels)