# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 

from __future__ import absolute_import, division, print_function

import csv, json
import logging
import os
import sys
from io import open
import _pickle as cPickle
import torch

logger = logging.getLogger(__name__)


class InputInstance(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, score=None, img_key=None, q_id=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.score = score
        self.img_key = img_key
        self.q_id = q_id


class InputFeat(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, score, img_feat):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.score = score
        self.img_feat = img_feat


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class VQATextProcessor(DataProcessor):
    """ Processor for the VQA Text data set. """

    def get_train_examples(self, data_dir, file_name='train2014_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "train")

        #return self._create_examples(self._read_tsv(os.path.join(data_dir, "train2014_qla.tsv")), "train")

    def get_dev_examples(self, data_dir, file_name='val2014_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "dev")

        #return self._create_examples(self._read_tsv(os.path.join(data_dir, "val2014_qla.tsv")), "dev")

    def get_test_examples(self, data_dir, file_name='test2015_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "test")

    def get_labels(self, label_file):
        """ See base class."""

        ans2label = cPickle.load(open(label_file, 'rb'))
        return list(ans2label.values())
        #return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        examples = []
        for (i, line) in enumerate(lines):
            if set_type!='test' and len(line['an']) == 0: continue

            guid = "%s-%s" % (set_type, str(i))
            text_a = line['q']
            text_b = line['o'].replace(';', ' ').strip() #line['o']
            label = None if set_type.startswith('test') else line['an']
            score = None if set_type.startswith('test') else line['s']
            img_key = line['img_id']
            q_id = int(line['q_id']) if set_type.startswith('test') else 0
            examples.append(InputInstance(guid=guid, text_a=text_a, text_b=text_b, label=label, score=score, img_key=img_key, q_id=q_id))
        return examples

class VQATextAProcessor(DataProcessor):
    """ Processor for the VQA Text data set. """

    def get_train_examples(self, data_dir, file_name='train2014_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "train")

        #return self._create_examples(self._read_tsv(os.path.join(data_dir, "train2014_qla.tsv")), "train")

    def get_dev_examples(self, data_dir, file_name='val2014_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "dev")

        #return self._create_examples(self._read_tsv(os.path.join(data_dir, "val2014_qla.tsv")), "dev")

    def get_test_examples(self, data_dir, file_name='test2015_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "test")

    def get_labels(self, label_file):
        """ See base class."""

        ans2label = cPickle.load(open(label_file, 'rb'))
        return list(ans2label.values())

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        examples = []
        for (i, line) in enumerate(lines):
            if set_type!='test' and len(line['an']) == 0: continue

            guid = "%s-%s" % (set_type, str(i))
            text_a = line['q']
            text_b = None # line['o'] # or None
            label = None if set_type.startswith('test') else line['an']
            score = None if set_type.startswith('test') else line['s']
            img_key = line['img_id']
            q_id = int(line['q_id']) if set_type.startswith('test') else 0
            examples.append(InputInstance(guid=guid, text_a=text_a, text_b=text_b, label=label, score=score, img_key=img_key, q_id=q_id))
        return examples

class GQAProcessor(DataProcessor):
    """ Processor for the GQA data set. """

    def get_train_examples(self, data_dir, file_name='train2014_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "train")

        #return self._create_examples(self._read_tsv(os.path.join(data_dir, "train2014_qla.tsv")), "train")

    def get_dev_examples(self, data_dir, file_name='val2014_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "dev")

        #return self._create_examples(self._read_tsv(os.path.join(data_dir, "val2014_qla.tsv")), "dev")

    def get_test_examples(self, data_dir, file_name='test2015_qla.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "test")

    def get_labels(self, label_file='trainval_testdev_all_ans2label.pkl'):
        """ See base class."""

        ans2label = cPickle.load(open(label_file, 'rb'))
        return list(ans2label.values())

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        examples = []
        for (i, line) in enumerate(lines):
            if set_type!='test' and len(line['an']) == 0: continue

            guid = "%s-%s" % (set_type, str(i))
            text_a = line['q']
            text_b = line['o'] # or None
            label = None if set_type.startswith('test') else line['an']
            score = 0
            img_key = line['img_id']
            q_id = int(line['q_id']) if set_type.startswith('test') else 0
            examples.append(InputInstance(guid=guid, text_a=text_a, text_b=text_b, label=label, score=score, img_key=img_key, q_id=q_id))
        return examples

class NLVRProcessor(DataProcessor):
    """ Processor for the NLVR data set. """

    def get_train_examples(self, data_dir, use_label_seq=True, file_name='nlvr2_train.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "train", use_label_seq)

        #return self._create_examples(self._read_tsv(os.path.join(data_dir, "train2014_qla.tsv")), "train")

    def get_dev_examples(self, data_dir, use_label_seq=True, file_name='nlvr2_dev.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "dev", use_label_seq)

        #return self._create_examples(self._read_tsv(os.path.join(data_dir, "val2014_qla.tsv")), "dev")

    def get_test_examples(self, data_dir, use_label_seq=True, file_name='nlvr2_test1.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "test", use_label_seq)

    def get_labels(self, label_file=None):
        """ See base class."""

        #ans2label = cPickle.load(open(label_file, 'rb'))
        #return list(ans2label.values())
        return [0, 1]

    def _create_examples(self, lines, set_type, use_label_seq=True):
        """ Creates examples for the training and dev sets. """

        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, str(i))
            text_a = line['q']
            text_b = line['o'] if use_label_seq else None
            label = line['label'] #None if set_type.startswith('test') else line['label']
            score = 0
            img_key = line['img_id'] #[line['img_left'], line['img_left']]
            q_id = 0 #int(line['q_id']) if set_type.startswith('test') else 0
            examples.append(InputInstance(guid=guid, text_a=text_a, text_b=text_b, label=label, score=score, img_key=img_key, q_id=q_id))
        return examples

class VCR_Q_A_Processor(DataProcessor):
    """ Processor for the VCR (q -> a) (Det) data set. """

    def get_train_examples(self, data_dir, file_name='vcr_train.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir, file_name='vcr_val.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "dev")

    def get_test_examples(self, data_dir, file_name='vcr_test.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "test")

    def get_labels(self, label_file=None):
        """ See base class."""

        #ans2label = cPickle.load(open(label_file, 'rb'))
        #return list(ans2label.values())
        return [0, 1]

    def _create_examples(self, lines, set_type):
        """ Creates examples for the training and dev sets. """

        examples = []
        for (i, line) in enumerate(lines):
            #if set_type!='test': continue

            guid = "%s-%s" % (set_type, str(i))
            text_a = line['q'] # question
            choices = line['choices']
            label = None if set_type.startswith('test') else line['label']
            img_key = line['img_id']
            q_id = int(line['annot_id'].split('-')[-1]) #int(line['q_id']) if set_type.startswith('test') else 0
            score = line['objects'] if 'objects' in line else None
            examples.append(InputInstance(guid=guid, text_a=text_a, text_b=choices, label=label, score=score, img_key=img_key, q_id=q_id))
        return examples

class VCR_QA_R_Processor(DataProcessor):
    """ Processor for the VCR (qa -> r) QA_R data set. """

    def get_train_examples(self, data_dir, file_name='vcr_train.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir, file_name='vcr_val.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "dev")

    def get_test_examples(self, data_dir, file_name='vcr_test.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "test")

    def get_labels(self, label_file=None):
        """ See base class."""

        #ans2label = cPickle.load(open(label_file, 'rb'))
        #return list(ans2label.values())
        return [0, 1]

    def _create_examples(self, lines, set_type):
        """ Creates examples for the training and dev sets. """

        examples = []
        for (i, line) in enumerate(lines):
            #if set_type!='test': continue

            guid = "%s-%s" % (set_type, str(i))
            text_a = line['q'] + ' ' + line['choices'][line['label']] # question_choice
            choices = line['rational_choices'] # rational_choice
            label = None if set_type.startswith('test') else line['rational_label'] # rational_label
            img_key = line['img_id']
            q_id = int(line['annot_id'].split('-')[-1]) #int(line['q_id']) if set_type.startswith('test') else 0
            examples.append(InputInstance(guid=guid, text_a=text_a, text_b=choices, label=label, score=None, img_key=img_key, q_id=q_id))
        return examples

class VCR_QAR_Processor(DataProcessor):
    """ Processor for the VCR (q->a, qa->r) data set. """

    def get_train_examples(self, data_dir, file_name='vcr_train.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir, file_name='vcr_val.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "dev")

    def get_test_examples(self, data_dir, file_name='vcr_test.json'):
        """ See base class."""

        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, "test")

    def get_labels(self, label_file=None):
        """ See base class."""

        #ans2label = cPickle.load(open(label_file, 'rb'))
        #return list(ans2label.values())
        return [0, 1]

    def _create_examples(self, lines, set_type):
        """ Creates examples for the training and dev sets. """

        examples = []
        for (i, line) in enumerate(lines):
            #if set_type!='test': continue

            guid = "%s-%s-q-a" % (set_type, str(i))
            text_a = line['q'] # question
            choices = line['choices']
            label = None if set_type.startswith('test') else line['label']
            img_key = line['img_id']
            q_id = int(line['annot_id'].split('-')[-1]) #int(line['q_id']) if set_type.startswith('test') else 0
            score = line['objects'] if 'objects' in line else None
            examples.append(InputInstance(guid=guid, text_a=text_a, text_b=choices, label=label, score=score, img_key=img_key, q_id=q_id))

            if set_type == 'train': # qa -> r
                guid = "%s-%s-qa-r" % (set_type, str(i))
                text_a = line['q'] + ' ' + line['choices'][line['label']] # question_choice
                choices = line['rational_choices'] # rational_choice
                label = None if set_type.startswith('test') else line['rational_label'] # rational_label
                img_key = line['img_id']
                q_id = int(line['annot_id'].split('-')[-1]) # int(line['q_id']) if set_type.startswith('test') else 0
                score = line['objects'] if 'objects' in line else None
                examples.append(InputInstance(guid=guid, text_a=text_a, text_b=choices, label=label, score=score, img_key=img_key, q_id=q_id))
        return examples


def convert_examples_to_features_vqa(examples, img_feats, label_list, max_img_seq_length, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label:i for i, label in enumerate(label_list)}

    features = []
    #debug:
    debug_size = 500

    for (ex_index, example) in enumerate(examples[0: ]):
        if len(example.label) == 0: continue
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # image features
        #img_feat = img_feats[example.img_key] # torch
        img_feat = img_feats.item().get(example.img_key) # numpy
        if img_feat.shape[0] > max_img_seq_length:
            img_feat = img_feat[0:max_img_seq_length, ]
            if max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                #segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
        else:
            if max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                #segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
            padding_matrix = torch.zeros((max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            if max_img_seq_length > 0:
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])
                #segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]

        if output_mode == "classification":
            label_id = [label_map[l] for l in example.label]
            score = example.score
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.label, label_id))
            logger.info("score: %s (score = %s)" % (example.score, score))

        features.append(InputFeat(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id, score=score, img_feat=img_feat))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


processors = {
    "vqa_text": VQATextProcessor,
    "vqa_text_a": VQATextAProcessor,
    "gqa": GQAProcessor,
    "nlvr": NLVRProcessor,
    "vcr_q_a": VCR_Q_A_Processor,
    "vcr_qa_r": VCR_QA_R_Processor,
    "vcr_qar": VCR_QAR_Processor,
}

output_modes = {
    "vqa_text": "classification",
    "vqa_text_a": "classification",
    "gqa": "classification",
    "nlvr": "classification",
    "vcr_q_a": "classification",
    "vcr_qa_r": "classification",
    "vcr_qar": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "vqa_text": 3129,
    "vqa_text_a": 3129,
    "gqa": 1853,
    "nlvr": 2,
    "vcr_q_a": 2,
    "vcr_qa_r": 2,
    "vcr_qar": 2,
}