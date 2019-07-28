from allennlp.data.dataset_readers.semantic_role_labeling import SrlReader
from itertools import islice
from overrides import overrides


class SubsampledSrlReader(SrlReader):
    def __init__(self, num_samples, **kwargs):
        super().__init__(**kwargs)
        self.k = num_samples

    @overrides
    def _read(self, datapath):
        yield from islice(super()._read(datapath), self.k)
