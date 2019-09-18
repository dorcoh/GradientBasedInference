import torch
from allennlp.models import Model
from copy import deepcopy
from warnings import warn

from gbi.instances_store import InstanceStore
from gbi.constraints_function import g


class GradientBasedInference:
    def __init__(self,
                 model: Model,
                 learning_rate: float = 1e-3,
                 # regularization parameter
                 alpha: float = 0,
                 # store instances groups
                 store: bool = True,
                 enable_cuda: bool = False,
                 ) -> None:

        self.alpha = alpha
        self.enable_cuda = enable_cuda
        self.model_copy = deepcopy(model)
        self.model = None
        self.original_weights = [p for p in self.model_copy.parameters()]
        self.learning_rate = learning_rate
        self.optimizer = None
        # stats for inference loop
        self.stats = {
            # true positive by initial model
            'tp': 0,
            # g zero
            'gzero_start': 0,
            'gzero_next': 0,
            # fixed through gradient inference loop
            'fixed': 0,
            # failed (passed iterations limit)
            'fix_failed': 0,
            'total': 0,
        }

        # store pickled list of instances groups
        self.instances_store = InstanceStore(disabled=not store)

    def _infer(self, x):
        # Set to evaluate mode
        self.model.eval()

        # forward
        output_dict = self.model(**x)

        # infer
        y_hat = self.model.decode(output_dict)

        likelihood = output_dict['class_probabilities']

        return y_hat, likelihood

    def _compute_regularization(self):
        """regularization: normalized difference between original and new weights"""
        reg = torch.tensor(0.0)
        norm = torch.tensor(0.0)

        if self.alpha == 0:
            return reg

        for orig_param, new_param in zip(self.original_weights, self.model.parameters()):
            orig_flat = orig_param.reshape(-1)
            new_flat = new_param.reshape(-1)
            diff = torch.abs(orig_flat-new_flat)
            reg += torch.sum(diff)
            norm += torch.sum(diff**2)

        if torch.eq(reg, 0.0):
            return reg

        reg = reg / torch.sqrt(norm)

        return reg

    def _parameters_update(self, likelihood, g_result):
        """
        perform loss computation and parameters update
        """
        # Set the model to train mode
        self.model.train()
        self.optimizer.zero_grad()

        unbatched_likelihood = likelihood.view(likelihood.shape[2], likelihood.shape[1])
        # scale max likelihood with G and compute constraint loss
        constraint_loss = torch.sum(unbatched_likelihood * g_result)

        reg = self._compute_regularization()
        reg_loss = self.alpha * reg

        loss = constraint_loss + reg_loss

        print("Constraint loss: ", constraint_loss.item(), ",Reg loss: ", reg_loss.item(),
              ", Total loss: ", loss.item())

        loss.backward()

        self.optimizer.step()

    def gradient_inference(self, x, iterations, num_samples, verbose=False):
        self.stats['total'] += 1
        i = 0
        # revert to original model and init optimizer
        self.model = deepcopy(self.model_copy)
        if self.enable_cuda and torch.cuda.is_available():
            self.model = self.model.cuda(0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        y_hat = None
        while i < iterations:
            # infer
            y_hat, likelihood = self._infer(x)
            if torch.any(torch.isnan(likelihood)):
                warn("loss has NaN values")
                return None

            # skip true positive
            predicted_tags = y_hat['tags'][0]
            true_tags = x['metadata'][0]['gold_tags']
            if predicted_tags == true_tags:
                if i == 0:
                    self.stats['tp'] += 1
                    print("Exit loop due to true positive")
                else:
                    self.stats['fixed'] += 1
                    self.instances_store.append('fixed', x)
                    print("Exit loop to due to fix")
                    self._loop_stats(i, predicted_tags, true_tags)
                break

            # compute g (or skip zero g)
            g_result = g(x, y_hat, verbose)
            if torch.all(g_result == 0):
                if i == 0:
                    self.stats['gzero_start'] += 1
                    self.instances_store.append('gzero', x)
                else:
                    self.stats['gzero_next'] += 1
                    self.instances_store.append('failed', x)
                print("Exit loop due to zero g function")
                self._loop_stats(i, predicted_tags, true_tags)
                break

            self._loop_stats(i, predicted_tags, true_tags)

            # compute loss and update parameters
            self._parameters_update(likelihood, g_result)
            i += 1

        if i == iterations:
            self.stats['fix_failed'] += 1
            self.instances_store.append('failed', x)

        if self.stats['total'] % 10 == 0 or self.stats['total'] == num_samples:
            self.instances_store.update_instances_store()

        return y_hat

    def print_stats(self):
        print("Statistics:", self.stats)

    @staticmethod
    def _loop_stats(i, predicted_tags, true_tags):
        print("Iteration: ", i + 1)
        print("Pred: ", predicted_tags)
        print("True: ", true_tags)

    def append_stats(self, args):
        """append stats to disk"""
        with open('stats.txt', 'a') as handle:
            handle.write("%s, %s\n" % (str(args), str(self.stats)))
