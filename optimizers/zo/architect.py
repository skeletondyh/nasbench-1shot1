import numpy as np
import torch
import copy
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.arch_lr = args.arch_learning_rate
        self.explore = args.explore
        #self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        #                                  lr=args.arch_learning_rate, betas=(0.5, 0.999),
        #                                  weight_decay=args.arch_weight_decay)

    def _train_loss(self, model, input, target):
        return model._loss(input, target)

    def _val_loss(self, model, input, target):
        return model._loss(input, target)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self._train_loss(model=self.model, input=input, target=target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(moment + dtheta, alpha=eta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        '''self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()'''
        loss_val = self._val_loss(self.model, input_valid, target_valid)
        grads_arch = torch.autograd.grad(loss_val, self.model.arch_parameters())
        sum_square = sum(torch.sum(torch.square(each)) for each in grads_arch)
        grads_arch = [each / torch.sqrt(sum_square) for each in grads_arch]
        params = copy.deepcopy(self.model.state_dict())

        ## one direction
        for a, g in zip(self.model.arch_parameters(), grads_arch):
            a.data.add_(g, alpha=self.explore)
        
        loss_train = self._train_loss(self.model, input_train, target_train)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss_train, self.model.parameters())).data + self.network_weight_decay * theta
        update = moment + dtheta

        offset = 0
        for p in self.model.parameters():
            length = np.prod(p.size())
            p.data.sub_(update[offset:offset + length].view(p.size()), alpha=eta)
            offset += length
        
        with torch.no_grad():
            eval_along = self._val_loss(self.model, input_valid, target_valid)
        
        self.model.load_state_dict(params)

        ## another
        for a, g in zip(self.model.arch_parameters(), grads_arch):
            a.data.sub_(g, alpha=2*self.explore)
        
        loss_train = self._train_loss(self.model, input_train, target_train)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss_train, self.model.parameters())).data + self.network_weight_decay * theta
        update = moment + dtheta

        offset = 0
        for p in self.model.parameters():
            length = np.prod(p.size())
            p.data.sub_(update[offset:offset + length].view(p.size()), alpha=eta)
            offset += length
        
        with torch.no_grad():
            eval_oppsite = self._val_loss(self.model, input_valid, target_valid)
        
        self.model.load_state_dict(params)

        ## go back
        diff = (eval_along - eval_oppsite).item()

        for a, g in zip(self.model.arch_parameters(), grads_arch):
            a.data.add_(g, alpha=self.explore)
            a.data.sub_(g, alpha=self.arch_lr * diff / (2 * self.explore))

    def _backward_step(self, input_valid, target_valid):
        loss = self._val_loss(self.model, input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = self._val_loss(model=unrolled_model, input=input_valid, target=target_valid)

        # Compute backwards pass with respect to the unrolled model parameters
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]

        # Compute expression (8) from paper
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        # Compute expression (7) from paper
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(ig.data, alpha=eta)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)
        loss = self._train_loss(self.model, input=input, target=target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(v, alpha=2*R)
        loss = self._train_loss(self.model, input=input, target=target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
