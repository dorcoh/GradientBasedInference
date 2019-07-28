from typing import Iterable, Callable

import torch
from allennlp.data import DataIterator, Instance
from allennlp.models import Model


class GradientBasedInference:
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator
                 ) -> None:

        self.model = model
        self.iterator = iterator
        self.optimizer = optimizer

    def _compute_loss(self, output_dict) -> torch.Tensor:
        try:
            loss = output_dict["loss"]
        except KeyError:
            raise RuntimeError("The model you are trying to optimize does not contain a"
                               " 'loss' key in the output of model.forward(inputs).")

        # TODO: add GPU support

        return loss

    def _gradient_inference(self, input):
        """
        gradient based inference for one iteration
        :param input: the input to optimize
        """

        # Set the model to "train" mode.
        self.model.train()

        self.optimizer.zero_grad()

        y_hat = self.model(**input)

        loss = self._compute_loss(y_hat)

        if torch.isnan(loss):
            raise ValueError("nan loss encountered")

        loss.backward()

        # TODO: add scaling of gradients with G(y,L)
        #  and update of gradient terms
        # TODO: add regularization term (W_orig-W_curr)/L2(W_orig-W_curr)

        self.inference_loss += loss.item()

        self.optimizer.step()

        return y_hat

    def predict(self, input: Iterable[Instance], g: Callable, iterations: int):
        i = 0
        self.inference_loss = 0.0
        # TODO: store initial weights
        while g() > 0 and i < iterations:
            # redundant loop for generating one instance
            for _input in self.iterator(input):
                y_hat = self._gradient_inference(_input)
                print(self.inference_loss)
                # TODO: modify _input to have y_hat labels then feed it back into the iterator and then gradient_predict

        return y_hat