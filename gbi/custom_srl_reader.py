import logging

from allennlp.common.file_utils import cached_path
from allennlp.data import Token, Instance
from allennlp.data.dataset_readers.dataset_utils import Ontonotes
from allennlp.data.dataset_readers.semantic_role_labeling import SrlReader
from itertools import islice

from allennlp.data.fields import MetadataField
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class CustomSrlReader(SrlReader):
    """
    custom srl reader features:
    (1) dependency parse tree nodes text spans stored in instance metadata
    (2) limit number of samples
    """
    def subsampled_read(self, datapath, num_samples=None):
        if num_samples is not None:
            # limit data set
            yield from islice(self._read(datapath), num_samples)
        else:
            # no limit
            yield from self._read(datapath)

    @overrides
    def _read(self, file_path: str):
        """OntoNotes custom reader to load spans from dependency pares tree as well"""
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        ontonotes_reader = Ontonotes()
        logger.info("Reading SRL instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info("Filtering to only include file paths containing the %s domain", self._domain_identifier)

        for sentence in self._ontonotes_subset(ontonotes_reader, file_path, self._domain_identifier):

            # skip samples without dep' parse tree
            if not sentence.parse_tree:
                continue

            # extract dep' parse tree spans
            spans = set()
            for subtree in sentence.parse_tree.subtrees():
                if subtree.height() > 0:
                    # TODO: check how to output indices instead of words
                    #  (for extreme cases where different tuples could match)
                    spans.add(tuple(subtree.leaves()))

            tokens = [Token(t) for t in sentence.words]
            if sentence.srl_frames:
                for (_, tags) in sentence.srl_frames:
                    verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                    yield self.text_to_instance_with_spans(tokens, verb_indicator, tags, spans)

    def text_to_instance_with_spans(self, tokens, verb_indicator, tags, spans):
        instance = super().text_to_instance(tokens, verb_indicator, tags)
        metadata_dict = instance.fields['metadata'].metadata
        metadata_dict['spans'] = spans
        instance.fields['metadata'] = MetadataField(metadata_dict)
        return Instance(instance.fields)
