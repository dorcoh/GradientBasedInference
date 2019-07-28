from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.service.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.training.trainer import Trainer

from gbi.gradient_inference import GradientBasedInference
from gbi.subsampled_srl_reader import SubsampledSrlReader
import torch.optim as optim
import os


datapath = os.getcwd() + '/data/' + 'test'
res = os.path.exists(datapath)
srlReader = SubsampledSrlReader(token_indexers={"elmo": ELMoTokenCharactersIndexer()},
                                num_samples=1)
test_dataset = srlReader.read(datapath)
instances = [i for i in test_dataset]
vocab = Vocabulary.from_instances(test_dataset)

archive = load_archive("srl-model-2018.05.25.tar.gz")
original_predictor = Predictor.from_archive(archive)
model = original_predictor._model

optimizer = optim.SGD(model.parameters(), lr=0.01)
iterator = BasicIterator(batch_size=1)
iterator.index_with(vocab)


def test():
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=test_dataset,
                      num_epochs=1)
    trainer.train()
    trained_predictor = Predictor(trainer.model, srlReader)

    new = trained_predictor.predict_instance(instances[0])
    orig = original_predictor.predict_instance(instances[0])


def g():
    # TODO: add constraints
    return 1


def gbi():
    gbi = GradientBasedInference(model=model,
                                 optimizer=optimizer,
                                 iterator=iterator)

    gbi.predict(test_dataset, g, iterations=10)


def main():
    # test()
    gbi()


if __name__ == '__main__':
    main()
