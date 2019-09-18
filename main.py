from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive

from gbi.gradient_inference import GradientBasedInference
from gbi.custom_semantic_role_labeler import CustomSemanticRoleLabeler
from gbi.custom_srl_reader import CustomSrlReader
from gbi.instances_store import load_and_deserialize
from utils.args_handler import get_args

import os


def main():
    args = get_args()
    print(args)
    # program arguments
    load = args.load
    store = args.store
    verbose = args.v
    # hyper-params for gradient inference
    regularization = args.a
    learning_rate = args.l
    inference_iterations = args.i
    enable_cuda = args.c

    # load data samples
    instances = []
    # load from data files
    if load in ['test', 'development', 'train', 'selected']:
        datapath = os.getcwd() + '/data/' + load
        srl_reader = CustomSrlReader(token_indexers={"elmo": ELMoTokenCharactersIndexer()})
        test_dataset = srl_reader.subsampled_read(datapath)
        instances = [i for i in test_dataset]
    # load from pickle files
    elif load in ['failed', 'fixed', 'gzero', 'fixed+failed']:
        if load == 'fixed+failed':
            instances = load_and_deserialize('fixed')
            instances += load_and_deserialize('failed')
        else:
            instances = load_and_deserialize(load)
    # init vocabulary and iterator
    vocab = Vocabulary.from_instances(instances)
    iterator = BasicIterator(batch_size=1)
    iterator.index_with(vocab)

    # load pre-trained model
    archive = load_archive("srl-model-2018.05.25.tar.gz")
    original_predictor = Predictor.from_archive(archive)
    model = CustomSemanticRoleLabeler.from_srl(original_predictor._model)

    # init and invoke inference method
    gbi = GradientBasedInference(model=model,
                                 learning_rate=learning_rate,
                                 alpha=regularization,
                                 store=store,
                                 enable_cuda=enable_cuda)
    for instance in iterator(instances, num_epochs=1):
        y_hat = gbi.gradient_inference(instance, iterations=inference_iterations,
                                       num_samples=len(instances), verbose=verbose)
        gbi.print_stats()

    gbi.append_stats(args)


if __name__ == '__main__':
    main()
