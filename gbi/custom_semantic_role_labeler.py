from overrides import overrides
from typing import Dict, List, Any

import torch
from allennlp.models import SemanticRoleLabeler


class CustomSemanticRoleLabeler(SemanticRoleLabeler):
    def __init__(self, *args):
        if type(args[0]) is SemanticRoleLabeler:
            self.__dict__ = args[0].__dict__.copy()

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """returns the default SRL forward without loss computation (no tags)"""
        output_dict = super().forward(tokens, verb_indicator, None, metadata)
        return output_dict

    @classmethod
    def from_srl(cls, class_instance):
        return cls(class_instance)

