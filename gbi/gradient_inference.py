import torch
from allennlp.data import DataIterator, Instance
from allennlp.models import Model
from copy import deepcopy
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

from gbi.instances_store import InstanceStore


def extract_bio_span_indices(predicted_tags):
    if predicted_tags[0]:
        # workaround for batch with size 1
        predicted_tags = predicted_tags[0]
    spans_indices = list()
    current_span_indices = []
    length = len(predicted_tags) - 1
    for i, tag in enumerate(predicted_tags):
        if i != length:
            next_tag = predicted_tags[i + 1]
            # first
            if tag[0] == 'B' and next_tag[0] == 'I':
                current_span_indices.append(i)
                continue
            # others
            elif tag[0] == 'I' and next_tag[0] == 'I':
                current_span_indices.append(i)
                continue
            # collect
            elif tag[0] == 'I' and next_tag[0] != 'I':
                current_span_indices.append(i)
                spans_indices.append(tuple(current_span_indices))
                current_span_indices = []

    return spans_indices


def extract_spans_bio(instance, predicted_tags):
    """extract and return bio spans as tuples in a set, e.g.,
    ('the', 'dog') if relative predicted tags were B-***, I-***"""
    spans_indices = extract_bio_span_indices(predicted_tags)
    sentence = instance['metadata'][0]['words']
    spans = set()
    for indices in spans_indices:
        current_span = [sentence[i] for i in indices]
        spans.add(tuple(current_span))
    return spans


def collect_spans(x, y_hat):
    """get both kind of spans: from bio tag and parse tree"""
    spans_bio = set()
    spans_parse_tree = set()
    if 'spans' in x['metadata'][0]:
        spans_parse_tree = x['metadata'][0]['spans']
        spans_bio = extract_spans_bio(x, y_hat['tags'])
    return spans_bio, spans_parse_tree


def check_consecutive(span, words_list):
    """checks whether text span exists in a sentence, e.g.,
    words_list: ['Blue', 'sky' , 'and' , 'yellow', 'sun'],
     span: ('and', 'yellow'), returns: [2,3] """
    text_span = list(span)
    span_len = len(text_span)
    if span_len == 0:
        return None
    indices = []
    curr_idx = 0
    for word_idx, word in enumerate(words_list):
        if word == text_span[curr_idx]:
            indices.append(word_idx)
            if curr_idx+1 == span_len:
                return indices
            curr_idx += 1

    return None


def g(x, y_hat):
    """g for srl as described in paper"""
    spans_bio, spans_parse_tree = collect_spans(x, y_hat)
    words_list = x['metadata'][0]['words']
    num_tokens = len(words_list)
    return_tensor = torch.zeros(1, num_tokens)
    if len(spans_bio) == 0 or len(spans_parse_tree) == 0:
        # exit function if no spans
        return return_tensor

    for span in spans_bio:
        if span not in spans_parse_tree:
            span_indices = check_consecutive(span, words_list)
            if span_indices is not None:
                return_tensor[0][span_indices] = 1.0/num_tokens

    return return_tensor


class GradientBasedInference:
    def __init__(self,
                 model: Model,
                 learning_rate: float = 1e-3,
                 # regularization parameter
                 alpha: float = 0,
                 # to store instances groups (through analyze)
                 store: bool = True
                 ) -> None:

        self.alpha = alpha
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

    def gradient_inference(self, x: Instance, iterations: int, num_samples: int):
        self.stats['total'] += 1
        i = 0
        # revert to original model and init optimizer
        self.model = deepcopy(self.model_copy)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        y_hat = None
        while i < iterations:
            # infer
            y_hat, likelihood = self._infer(x)

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
            g_result = g(x, y_hat)
            if torch.all(g_result == 0):
                if i == 0:
                    self.stats['gzero_start'] += 1
                    self.instances_store.append('gzero', x)
                else:
                    self.stats['gzero_next'] += 1
                    self.instances_store.append('failed', x)
                print("Exit loop due to zero g function")
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
