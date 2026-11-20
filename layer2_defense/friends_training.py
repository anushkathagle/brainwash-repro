"""
friends_training.py

Layer-2 (FRIENDS) versions of the victim's training for EWC, RWALK and AFEC.

Each class below inherits everything from the original approaches/<method>.py
learner and replaces only the functions that run the training of a task:

  train(...)        copy of the original, plus ONE marked block at the top of the
                    epoch loop:  at epoch == warm-up length, compute the friendly
                    noise eta from the current model and train on x + eta from then on
                    (and switch the random noise on at its start epoch, if held back),
                    plus ONE marked block at the end of the epoch loop: optional
                    per-epoch accuracy log
  train_epoch(...)  copy of the original, plus ONE marked line after a batch of
                    images is taken: fresh random noise r on every step; and one marked
                    block after net.train(): the BatchNorm control / head-only epochs
                    (AFEC also train_emp_epoch for its second network)

The added lines are the ones between the markers:

    # >>> LAYER2: ...
    # <<< LAYER2

Everything else is the original code (only commented-out dead code and unused
timing variables were left out). approaches/*.py and the attacker's code
(main_brainwash.py) are not touched, evaluation is untouched, and with no
FriendsDefense attached these trainers behave exactly like the originals --
validate_friends.py V4 checks that by training both and comparing the weights.

The friendly noise is computed from the model after the warm-up epochs, so the
model has already trained on `warmup_epochs` epochs of the (poisoned) data when
eta is computed -- that is the paper's Algorithm 2 (5 warm-up epochs of 80).

NOTE: importing this module imports approaches/*, which parse sys.argv at
import time (they call get_args()); set sys.argv first, as main_baselines.py
style scripts do.
"""

import time
from copy import deepcopy

import numpy as np
import torch

from approaches import ewc, rwalk, afec_ewc


def _no_noise(images):
    return images


class EWC(ewc.Appr):
    random_noise = staticmethod(_no_noise)
    friends = None  # a FriendsDefense (friends.py), attached by FriendsDefense.attach()

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):
        best_loss = np.inf
        best_model = ewc.utils.get_model(self.model)
        lr = self.lr
        self.optimizer = self._get_optimizer(lr)

        # Loop epochs
        for e in range(self.nepochs):
            # >>> LAYER2: FRIENDS: friendly noise eta once after the warm-up epochs; random noise from its start epoch
            if self.friends is not None:
                xtrain = self.friends.on_epoch_start(self, xtrain, t, data, e)
            # <<< LAYER2

            # Train
            self.train_epoch(t, xtrain, ytrain, e)

            # Valid
            valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
            print('task: {} epoch {} Valid: loss={:.3f}, acc={:5.1f}% |'.format(t, e, valid_loss, 100 * valid_acc), end='')
            print(' lr : {:.6f}'.format(self.optimizer.param_groups[0]['lr']))
            print()
            best_model = ewc.utils.get_model(self.model)

            # >>> LAYER2: optional per-epoch old-task accuracy log
            if self.friends is not None:
                self.friends.on_epoch_end(self, t, data, e)
            # <<< LAYER2

        # Restore best
        ewc.utils.set_model_(self.model, best_model)

        # Update old
        self.model_old = deepcopy(self.model)
        self.model_old.train()
        ewc.utils.freeze_model(self.model_old) # Freeze the weights

        # Fisher ops
        if t > 0:
            fisher_old = {}
            for n, _ in self.model.named_parameters():
                fisher_old[n] = self.fisher[n].clone()
        self.fisher = ewc.utils.fisher_matrix_diag(t, xtrain, ytrain, self.model, self.criterion)
        if t > 0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            for n, _ in self.model.named_parameters():
                self.fisher[n] = (self.fisher[n] + fisher_old[n] * t) / (t + 1)       # Checked: it is better than the other option

        for p_idx, (n, p) in enumerate(self.model.named_parameters()):
            self.model.register_buffer('{}_fisher'.format(n.replace('.', '_')), self.fisher[n])

        return


    def train_epoch(self, t, x, y, epoch):
        self.model.train()
        # >>> LAYER2: BatchNorm control (and BatchNorm kept fixed during head-only epochs)
        if self.friends is not None:
            self.friends.set_bn_mode(self.model)
        # <<< LAYER2

        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r)

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r): b = r[i:i + self.sbatch]
            else: b = r[i:]
            images = x[b].to(ewc.device)
            targets = y[b].to(ewc.device)

            # >>> LAYER2: FRIENDS random noise r, fresh every training step
            images = self.random_noise(images)
            # <<< LAYER2

            # Forward current model
            outputs = self.model.forward(images)[t]
            loss = self.criterion(t, outputs, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        return


class RWALK(rwalk.Appr):
    random_noise = staticmethod(_no_noise)
    friends = None  # a FriendsDefense (friends.py), attached by FriendsDefense.attach()

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):
        best_loss = np.inf
        best_model = rwalk.utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)

        # Loop epochs
        for e in range(self.nepochs):
            # >>> LAYER2: FRIENDS: friendly noise eta once after the warm-up epochs; random noise from its start epoch
            if self.friends is not None:
                xtrain = self.friends.on_epoch_start(self, xtrain, t, data, e)
            # <<< LAYER2

            # Train
            self.train_epoch(t, xtrain, ytrain)

            # Valid
            valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')
            print()

            best_model = rwalk.utils.get_model(self.model)
            print()

            # >>> LAYER2: optional per-epoch old-task accuracy log
            if self.friends is not None:
                self.friends.on_epoch_end(self, t, data, e)
            # <<< LAYER2

        # Restore best
        rwalk.utils.set_model_(self.model, best_model)

        # Update old
        self.model_old = deepcopy(self.model)
        rwalk.utils.freeze_model(self.model_old) # Freeze the weights

        # Update fisher & s
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    self.fisher[n] = self.fisher_running[n].clone()
                    self.s[n] = (1/2) * self.s_running[n].clone()
                    self.s_running[n] = self.s[n].clone()

        self.register_fisher_into_buffer()
        self.rigister_s_into_buffer()

        return


    def train_epoch(self, t, x, y):
        self.model.train()
        # >>> LAYER2: BatchNorm control (and BatchNorm kept fixed during head-only epochs)
        if self.friends is not None:
            self.friends.set_bn_mode(self.model)
        # <<< LAYER2

        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r)

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r): b = r[i:i + self.sbatch]
            else: b = r[i:]
            images = x[b]
            targets = y[b]

            images, targets = images.cuda(), targets.cuda()

            # >>> LAYER2: FRIENDS random noise r, fresh every training step
            images = self.random_noise(images)
            # <<< LAYER2

            # Forward current model
            outputs = self.model.forward(images)[t]
            loss = self.criterion(t, outputs, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            # Compute Fisher & s
            self.update_fisher_and_s()

        return


class AFEC(afec_ewc.Appr):
    random_noise = staticmethod(_no_noise)      # main network
    random_noise_emp = staticmethod(_no_noise)  # AFEC's second ("empty") network
    friends = None  # a FriendsDefense (friends.py), attached by FriendsDefense.attach()

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):
        best_loss = np.inf
        best_model = afec_ewc.utils.get_model(self.model)
        lr = self.lr
        self.optimizer = self._get_optimizer(lr)
        self.optimizer_emp = self._get_optimizer_emp(lr)
        self.add_emp = 0

        if t == 0:
            self.model_emp = deepcopy(self.model) #use the same initialization
            self.model_emp_tmp = deepcopy(self.model)

        # Loop epochs
        for e in range(self.nepochs):
            # >>> LAYER2: FRIENDS: friendly noise eta once after the warm-up epochs; random noise from its start epoch
            if self.friends is not None:
                xtrain = self.friends.on_epoch_start(self, xtrain, t, data, e)
            # <<< LAYER2

            # Train
            clock0 = time.time()

            num_batch = xtrain.size(0)

            #train the empty net and measure fim
            if t > self.add_emp-1:

                self.train_emp_epoch(t, xtrain, ytrain, e)
                # freeze the empty net
                self.model_emp_tmp = deepcopy(self.model_emp)
                self.model_emp_tmp.train()
                afec_ewc.utils.freeze_model(self.model_emp_tmp)

                # Fisher ops
                self.fisher_emp, _ = afec_ewc.utils.fisher_matrix_diag_emp(t, xtrain, ytrain, self.model_emp, self.criterion)

            self.train_epoch(t, xtrain, ytrain, e)

            clock1 = time.time()
            train_loss, train_acc = self.eval(t, xtrain, ytrain)
            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1, 1000*self.sbatch*(clock1-clock0)/num_batch,
                1000*self.sbatch*(clock2-clock1)/num_batch, train_loss, 100*train_acc), end='')
            # Valid
            valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100*valid_acc), end='')
            print(' lr : {:.6f}'.format(self.optimizer.param_groups[0]['lr']))

            best_model = afec_ewc.utils.get_model(self.model)

            # >>> LAYER2: optional per-epoch old-task accuracy log
            if self.friends is not None:
                self.friends.on_epoch_end(self, t, data, e)
            # <<< LAYER2

            # after pretrain in task 0, copy the PT model as empty
            if t == 0:
                self.model_pt = deepcopy(self.model)

        # Restore best
        afec_ewc.utils.set_model_(self.model, best_model)

        # Update old
        self.model_old = deepcopy(self.model)
        self.model_old.train()
        afec_ewc.utils.freeze_model(self.model_old) # Freeze the weights

        # Fisher ops
        if t > 0:
            fisher_old = {}
            for n, _ in self.model.named_parameters():
                if 'heads' not in n:
                    fisher_old[n] = self.fisher[n].clone()
        self.fisher = afec_ewc.utils.fisher_matrix_diag(t, xtrain, ytrain, self.model, self.criterion)
        if t > 0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            for n, _ in self.model.named_parameters():
                if 'heads' not in n:
                    self.fisher[n] = (self.fisher[n] + fisher_old[n] * t) / (t + 1)       # Checked: it is better than the other option

        self.register_fisher_into_buffer()

        return


    def train_epoch(self, t, x, y, epoch):
        self.model.train()
        # >>> LAYER2: BatchNorm control (and BatchNorm kept fixed during head-only epochs)
        if self.friends is not None:
            self.friends.set_bn_mode(self.model)
        # <<< LAYER2

        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r)

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r): b = r[i:i + self.sbatch]
            else: b = r[i:]
            images = x[b].to(afec_ewc.device)
            targets = y[b].to(afec_ewc.device)

            # >>> LAYER2: FRIENDS random noise r, fresh every training step
            images = self.random_noise(images)
            # <<< LAYER2

            # Forward current model
            outputs = self.model.forward(images)[t]
            loss = self.ce(outputs, targets)

            if t > self.add_emp:
                loss_fg = self.criterion_fg(t)
                loss += loss_fg

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            del loss
            del images, targets, outputs

        return

    def train_emp_epoch(self, t, x, y, epoch):
        self.model_emp.train()
        # >>> LAYER2: BatchNorm control (and BatchNorm kept fixed during head-only epochs)
        if self.friends is not None:
            self.friends.set_bn_mode(self.model_emp)
        # <<< LAYER2

        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r)

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r): b = r[i:i + self.sbatch]
            else: b = r[i:]
            images = x[b].to(afec_ewc.device)
            targets = y[b].to(afec_ewc.device)

            # >>> LAYER2: FRIENDS random noise r, fresh every training step
            images = self.random_noise_emp(images)
            # <<< LAYER2

            # train empty net
            # Forward current model
            outputs = self.model_emp.forward(images)[t]
            loss = self.ce(outputs, targets)

            # Backward
            self.optimizer_emp.zero_grad()
            loss.backward()
            self.optimizer_emp.step()

        return


TRAINERS = {'ewc': EWC, 'rwalk': RWALK, 'afec_ewc': AFEC}