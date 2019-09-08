from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.service.predictors.predictor import Predictor
from allennlp.models.archival import load_archive

from gbi.gradient_inference import GradientBasedInference
from gbi.custom_semantic_role_labeler import CustomSemanticRoleLabeler
from gbi.custom_srl_reader import CustomSrlReader
import torch.optim as optim
import os


# load data and init vocabulary
test_datapath = os.getcwd() + '/data/' + 'failed'
srlReader = CustomSrlReader(token_indexers={"elmo": ELMoTokenCharactersIndexer()},
                            num_samples=None)
test_dataset = srlReader.subsampled_read(test_datapath)
test_instances = [i for i in test_dataset]
vocab = Vocabulary.from_instances(test_instances)
iterator = BasicIterator(batch_size=1)
iterator.index_with(vocab)

# load pre-trained model
archive = load_archive("srl-model-2018.05.25.tar.gz")
original_predictor = Predictor.from_archive(archive)
model = CustomSemanticRoleLabeler.from_srl(original_predictor._model)


def gbi():
    # TODO: fix all extreme cases of loader (tree height etc.)
    # TODO: load somehow only 'failed' instances
    # TODO: add metrics calculator
    # TODO: prepare for and run on server
    # TODO: add another experiment, options:
    #  (1) out-of-domain data (need to train special models for it)
    #  (2) different g function with current task
    #  (3) different g function with other task
    inst = test_instances[15]
    # init optimizer
    optimizer = optim.SGD(model.parameters(), lr=1)
    gbi = GradientBasedInference(model=model,
                                 optimizer=optimizer,
                                 alpha=0)
    for _input in iterator([inst], num_epochs=1):
        y_hat = gbi.gradient_inference(_input, iterations=10)
        truth = y_hat['tags'][0] == _input['metadata'][0]['gold_tags']
        print(truth)


def main():
    gbi()
    #iterate_instances()


if __name__ == '__main__':
    main()


# utils

def iterate_instances():
    """
    helper for finding specific instance by conditioning on its words
    """
    for i, inst in enumerate(test_instances):
        predicted = original_predictor.predict_instance(inst)
        predicted_tags = predicted['tags']
        truth = inst.fields['tags'].labels
        if truth != predicted_tags:
            print(inst.fields['tags'].labels)
            print(original_predictor.predict_instance(inst)['tags'])
            pass
        print(i)
        # words = inst['metadata'].metadata['words']

        # truth = 'bus' in words and 'signs' in words
        # if truth:
        #     print(words)
        # else:
        #     continue