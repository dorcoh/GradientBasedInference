from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.service.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.training.trainer import Trainer
from allennlp.data.dataset_readers.semantic_role_labeling import SrlReader
import torch.optim as optim
import os

datapath = os.getcwd() + '/data/' + 'test'
res = os.path.exists(datapath)
srlReader = SrlReader(token_indexers={"elmo": ELMoTokenCharactersIndexer()})
test_dataset = srlReader.read(datapath)

vocab = Vocabulary.from_instances(test_dataset)

archive = load_archive("srl-model-2018.05.25.tar.gz")
predictor = Predictor.from_archive(archive)
# result = predictor.predict_json(
#   {"sentence": "Did Uriah honestly think he could beat the game in under three hours?"}
# )
#print(result)
model = predictor._model

optimizer = optim.SGD(model.parameters(), lr=0.1)
iterator = BasicIterator(batch_size=1)
iterator.index_with(vocab)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=test_dataset,
                  patience=10,
                  num_epochs=1)
trainer.train()
# TODO: (1) check how to init pretrained weights (or find out if already applied)
#  (2) restrict the training to one sample each time
#  (3) amplify the gradient with function g
#  (4) define function g through reading the paper appendix and consulting with Roi