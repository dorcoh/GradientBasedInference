import torch
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans


def extract_bio_span_indices(predicted_tags):
    if predicted_tags[0]:
        # workaround for batch with size 1
        predicted_tags = predicted_tags[0]
    spans_indices = list()
    current_span_indices = []
    length = len(predicted_tags) - 1
    for i, tag in enumerate(predicted_tags):
        if i != length:
            next_tag = predicted_tags[i + 1]
            # first
            if tag[0] == 'B' and next_tag[0] == 'I':
                current_span_indices.append(i)
                continue
            # others
            elif tag[0] == 'I' and next_tag[0] == 'I':
                current_span_indices.append(i)
                continue
            # collect
            elif tag[0] == 'I' and next_tag[0] != 'I':
                current_span_indices.append(i)
                spans_indices.append(tuple(current_span_indices))
                current_span_indices = []

    return spans_indices


def extract_spans_bio(instance, predicted_tags):
    """extract and return bio spans as tuples in a set, e.g.,
    ('the', 'dog') if relative predicted tags were B-***, I-***"""
    spans_indices = extract_bio_span_indices(predicted_tags)
    sentence = instance['metadata'][0]['words']
    spans = set()
    for indices in spans_indices:
        current_span = [sentence[i] for i in indices]
        spans.add(tuple(current_span))
    return spans


def collect_spans(x, y_hat):
    """get both kind of spans: from bio tag and parse tree"""
    spans_bio = set()
    spans_parse_tree = set()
    if 'spans' in x['metadata'][0]:
        spans_parse_tree = x['metadata'][0]['spans']
        spans_bio = extract_spans_bio(x, y_hat['tags'])
    return spans_bio, spans_parse_tree


def check_consecutive(span, words_list):
    """checks whether text span exists in a sentence, e.g.,
    words_list: ['Blue', 'sky' , 'and' , 'yellow', 'sun'],
     span: ('and', 'yellow'), returns: [2,3] """
    text_span = list(span)
    span_len = len(text_span)
    if span_len == 0:
        return None
    indices = []
    curr_idx = 0
    for word_idx, word in enumerate(words_list):
        if word == text_span[curr_idx]:
            indices.append(word_idx)
            if curr_idx+1 == span_len:
                return indices
            curr_idx += 1

    return None


def g(x, y_hat, verbose=False):
    """g for srl as described in paper"""
    spans_bio, spans_parse_tree = collect_spans(x, y_hat)
    if verbose:
        print("Spans bio: ", spans_bio)
        print("Spans parse tree: ", spans_parse_tree)
    words_list = x['metadata'][0]['words']
    num_tokens = len(words_list)
    return_tensor = torch.zeros(1, num_tokens)
    if len(spans_bio) == 0 or len(spans_parse_tree) == 0:
        # exit function if no spans
        return return_tensor

    for span in spans_bio:
        if span not in spans_parse_tree:
            span_indices = check_consecutive(span, words_list)
            if span_indices is not None:
                return_tensor[0][span_indices] = 1.0/num_tokens

    return return_tensor
