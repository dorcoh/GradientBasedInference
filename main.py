from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.service.predictors.predictor import Predictor
from allennlp.models.archival import load_archive

from gbi.gradient_inference import GradientBasedInference
from gbi.custom_semantic_role_labeler import CustomSemanticRoleLabeler
from gbi.custom_srl_reader import CustomSrlReader
from gbi.instances_store import load_and_deserialize

import os

# program arguments
load = 'test'
store = False
# hyper-params for gradient inference
regularization = 0
learning_rate = 1
inference_iterations = 15
# TODO: add support for program arguments

# load data, init vocabulary and iterator
test_instances = []
if load == 'test':
    test_datapath = os.getcwd() + '/data/' + load
    srl_reader = CustomSrlReader(token_indexers={"elmo": ELMoTokenCharactersIndexer()})
    test_dataset = srl_reader.subsampled_read(test_datapath)
    test_instances = [i for i in test_dataset]
elif load in ['failed', 'fixed', 'gzero']:
    test_instances = load_and_deserialize(load)

vocab = Vocabulary.from_instances(test_instances)
iterator = BasicIterator(batch_size=1)
iterator.index_with(vocab)

# load pre-trained model
archive = load_archive("srl-model-2018.05.25.tar.gz")
original_predictor = Predictor.from_archive(archive)
model = CustomSemanticRoleLabeler.from_srl(original_predictor._model)


def gbi():
    gbi = GradientBasedInference(model=model,
                                 learning_rate=learning_rate,
                                 alpha=regularization,
                                 store=store)
    for _input in iterator(test_instances, num_epochs=1):
        y_hat = gbi.gradient_inference(_input, iterations=inference_iterations,
                                       num_samples=len(test_instances))
        gbi.print_stats()


def main():
    gbi()


if __name__ == '__main__':
    main()
