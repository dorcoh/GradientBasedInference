import jsonpickle

from allennlp.data import Token
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer

from gbi.custom_srl_reader import CustomSrlReader


def deserialize(line):
    """deserialize a stringified SRL Instance"""
    srlReader = CustomSrlReader(token_indexers={"elmo": ELMoTokenCharactersIndexer()})
    l_d = jsonpickle.loads(line.rstrip('\n'))
    tokens = [Token(t) for t in l_d['metadata'][0]['words']]
    verb_indicator = l_d['verb_indicator'].tolist()[0]
    tags = l_d['metadata'][0]['gold_tags']
    spans = l_d['metadata'][0]['spans']
    inst = srlReader.text_to_instance_with_spans(tokens=tokens, verb_indicator=verb_indicator,
                                                 tags=tags, spans=spans)
    return inst


def load_and_deserialize(filename):
    """load serialized instances from file"""
    with open(filename, 'r') as handle:
        lines = handle.readlines()

    instances = []
    for line in lines:
        instances.append(deserialize(line))

    return instances


class InstanceStore:
    def __init__(self, disabled=False):
        # g was zero in first iteration
        self.gzero = []
        # g being zero in 2+ iteration or reached end of loop
        self.failed = []
        # sample became true positive
        self.fixed = []

        self.group_names = {
            'gzero': self.gzero,
            'failed': self.failed,
            'fixed': self.fixed
        }

        self.disabled = disabled

    def update_instances_store(self):
        """update store: *append* to file"""
        if not self.disabled:
            for filename, instances_list in self.group_names.items():
                with open(filename, 'a') as handle:
                    while len(instances_list) > 0:
                        handle.write(instances_list.pop() + '\n')

    def append(self, group_name, instance):
        if not self.disabled:
            self.group_names[group_name].append(jsonpickle.dumps(instance))