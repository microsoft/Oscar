# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 

from collections import OrderedDict, defaultdict
import json
import numpy as np
import os.path as op
from pprint import pprint
import torch
import re
import subprocess
import tempfile
import time
from typing import Dict, Optional

from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap
from .cider.pyciderevalcap.ciderD.ciderD import CiderD


def evaluate_on_nocaps(split, predict_file, data_dir='data/nocaps/', evaluate_file=None):
    '''
    NOTE: Put the auth file in folder ~/.evalai/
    '''
    if not evaluate_file:
        evaluate_file = op.splitext(predict_file)[0] + '.eval.json'
    if op.isfile(evaluate_file):
        print('{} already exists'.format(evaluate_file))
        with open(evaluate_file, 'r') as fp:
            metrics = json.load(fp)
        return metrics

    image_info_file = op.join(data_dir,
            'nocaps_{}_image_info.json'.format(split))
    image_info = json.load(open(image_info_file))
    open_image_id2id = {}
    for it in image_info['images']:
        open_image_id2id[it['open_images_id']] = it['id']
    predictions = []
    cap_id = 0
    with open(predict_file, 'r') as fp:
        for line in fp:
            p = line.strip().split('\t')
            predictions.append(
                    {'image_id': open_image_id2id[p[0]],
                    'caption': json.loads(p[1])[0]['caption'],
                    'id': cap_id})
            cap_id += 1
    if split == 'test':
        print('Are you sure to submit test split result at: {}'.format(predict_file))
        import ipdb;ipdb.set_trace()
    nocapseval = NocapsEvaluator(phase=split)
    metrics = nocapseval.evaluate(predictions)
    pprint(metrics)
    with open(evaluate_file, 'w') as fp:
        json.dump(metrics, fp)
    return metrics


def evaluate_on_coco_caption(res_file, label_file, outfile=None):
    """
    res_tsv: TSV file, each row is [image_key, json format list of captions].
             Each caption is a dict, with fields "caption", "conf".
    label_file: JSON file of ground truth captions in COCO format.
    """
    assert label_file.endswith('.json')
    if res_file.endswith('.tsv'):
        res_file_coco = op.splitext(res_file)[0] + '_coco_format.json'
        convert_tsv_to_coco_format(res_file, res_file_coco)
    else:
        raise ValueError('unknown prediction result file format: {}'.format(res_file))

    coco = COCO(label_file)
    cocoRes = coco.loadRes(res_file_coco)
    cocoEval = COCOEvalCap(coco, cocoRes, 'corpus')

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    result = cocoEval.eval
    if not outfile:
        print(result)
    else:
        with open(outfile, 'w') as fp:
            json.dump(result, fp, indent=4)
    return result


def convert_tsv_to_coco_format(res_tsv, outfile,
        sep='\t', key_col=0, cap_col=1):
    results = []
    with open(res_tsv) as fp:
        for line in fp:
            parts = line.strip().split(sep)
            key = parts[key_col]
            if cap_col < len(parts):
                caps = json.loads(parts[cap_col])
                assert len(caps) == 1, 'cannot evaluate multiple captions per image'
                cap = caps[0].get('caption', '')
            else:
                # empty caption generated
                cap = ""
            results.append(
                    {'image_id': key,
                    'caption': cap}
                    )
    with open(outfile, 'w') as fp:
        json.dump(results, fp)


class ScstRewardCriterion(torch.nn.Module):
    CIDER_REWARD_WEIGHT = 1

    def __init__(self, cider_cached_tokens='corpus', baseline_type='greedy'):
        self.CiderD_scorer = CiderD(df=cider_cached_tokens)
        assert baseline_type in ['greedy', 'sample']
        self.baseline_type = baseline_type
        self._cur_score = None
        super().__init__()

    def forward(self, gt_res, greedy_res, sample_res, sample_logprobs):
        batch_size = len(gt_res)
        sample_res_size = len(sample_res)
        seq_per_img = sample_res_size // batch_size

        gen_res = []
        gen_res.extend(sample_res)
        gt_idx = [i // seq_per_img for i in range(sample_res_size)]
        if self.baseline_type == 'greedy':
            assert len(greedy_res) == batch_size
            gen_res.extend(greedy_res)
            gt_idx.extend([i for i in range(batch_size)])

        scores = self._calculate_eval_scores(gen_res, gt_idx, gt_res)

        if self.baseline_type == 'greedy':
            baseline = scores[-batch_size:][:, np.newaxis]
        else:
            sc_ = scores.reshape(batch_size, seq_per_img)
            baseline = (sc_.sum(1, keepdims=True) - sc_) / (sc_.shape[1] - 1)

        # sample - baseline
        reward = scores[:sample_res_size].reshape(batch_size, seq_per_img)
        self._cur_score = reward.mean()
        reward = reward - baseline
        reward = reward.reshape(sample_res_size)

        reward = torch.as_tensor(reward, device=sample_logprobs.device, dtype=torch.float)
        loss = - sample_logprobs * reward
        loss = loss.mean()
        return loss

    def get_score(self):
        return self._cur_score

    def _calculate_eval_scores(self, gen_res, gt_idx, gt_res):
        '''
        gen_res: generated captions, list of str
        gt_idx: list of int, of the same length as gen_res
        gt_res: ground truth captions, list of list of str.
            gen_res[i] corresponds to gt_res[gt_idx[i]]
            Each image can have multiple ground truth captions
        '''
        gen_res_size = len(gen_res)

        res = OrderedDict()
        for i in range(gen_res_size):
            res[i] = [self._wrap_sentence(gen_res[i])]

        gts = OrderedDict()
        gt_res_ = [
            [self._wrap_sentence(gt_res[i][j]) for j in range(len(gt_res[i]))]
                for i in range(len(gt_res))
        ]
        for i in range(gen_res_size):
            gts[i] = gt_res_[gt_idx[i]]

        res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
        _, batch_cider_scores = self.CiderD_scorer.compute_score(gts, res_)
        scores = self.CIDER_REWARD_WEIGHT * batch_cider_scores
        return scores

    @classmethod
    def _wrap_sentence(self, s):
        # ensure the sentence ends with <eos> token
        # in order to keep consisitent with cider_cached_tokens
        r = s.strip()
        if r.endswith('.'):
            r = r[:-1]
        r += ' <eos>'
        return r


class NocapsEvaluator(object):
    r"""
    Code from https://github.com/nocaps-org/updown-baseline/blob/master/updown/utils/evalai.py

    A utility class to submit model predictions on nocaps splits to EvalAI, and retrieve model
    performance based on captioning metrics (such as CIDEr, SPICE).

    Extended Summary
    ----------------
    This class and the training script together serve as a working example for "EvalAI in the
    loop", showing how evaluation can be done remotely on privately held splits. Annotations
    (captions) and evaluation-specific tools (e.g. `coco-caption <https://www.github.com/tylin/coco-caption>`_)
    are not required locally. This enables users to select best checkpoint, perform early
    stopping, learning rate scheduling based on a metric, etc. without actually doing evaluation.

    Parameters
    ----------
    phase: str, optional (default = "val")
        Which phase to evaluate on. One of "val" or "test".

    Notes
    -----
    This class can be used for retrieving metrics on both, val and test splits. However, we
    recommend to avoid using it for test split (at least during training). Number of allowed
    submissions to test split on EvalAI are very less, and can exhaust in a few iterations! However,
    the number of submissions to val split are practically infinite.
    """

    def __init__(self, phase: str = "val"):

        # Constants specific to EvalAI.
        self._challenge_id = 355
        self._phase_id = 742 if phase == "val" else 743

    def evaluate(
        self, predictions, iteration: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        r"""
        Take the model predictions (in COCO format), submit them to EvalAI, and retrieve model
        performance based on captioning metrics.

        Parameters
        ----------
        predictions: List[Prediction]
            Model predictions in COCO format. They are a list of dicts with keys
            ``{"image_id": int, "caption": str}``.
        iteration: int, optional (default = None)
            Training iteration where the checkpoint was evaluated.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Model performance based on all captioning metrics. Nested dict structure::

                {
                    "B1": {"in-domain", "near-domain", "out-domain", "entire"},  # BLEU-1
                    "B2": {"in-domain", "near-domain", "out-domain", "entire"},  # BLEU-2
                    "B3": {"in-domain", "near-domain", "out-domain", "entire"},  # BLEU-3
                    "B4": {"in-domain", "near-domain", "out-domain", "entire"},  # BLEU-4
                    "METEOR": {"in-domain", "near-domain", "out-domain", "entire"},
                    "ROUGE-L": {"in-domain", "near-domain", "out-domain", "entire"},
                    "CIDEr": {"in-domain", "near-domain", "out-domain", "entire"},
                    "SPICE": {"in-domain", "near-domain", "out-domain", "entire"},
                }

        """
        # Save predictions as a json file first.
        _, predictions_filename = tempfile.mkstemp(suffix=".json", text=True)
        with open(predictions_filename, "w") as f:
            json.dump(predictions, f)

        submission_command = (
            f"evalai challenge {self._challenge_id} phase {self._phase_id} "
            f"submit --file {predictions_filename}"
        )

        submission_command_subprocess = subprocess.Popen(
            submission_command.split(),
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # This terminal output will have submission ID we need to check.
        submission_command_stdout = submission_command_subprocess.communicate(input=b"N\n")[
            0
        ].decode("utf-8")

        submission_id_regex = re.search("evalai submission ([0-9]+)", submission_command_stdout)
        try:
            # Get an integer submission ID (as a string).
            submission_id = submission_id_regex.group(0).split()[-1]  # type: ignore
        except:
            # Very unlikely, but submission may fail because of some glitch. Retry for that.
            return self.evaluate(predictions)

        if iteration is not None:
            print(f"Submitted predictions for iteration {iteration}, submission id: {submission_id}.")
        else:
            print(f"Submitted predictions, submission_id: {submission_id}")

        # Placeholder stdout for a pending submission.
        result_stdout: str = "The Submission is yet to be evaluated."
        num_tries: int = 0

        # Query every 10 seconds for result until it appears.
        while "CIDEr" not in result_stdout:

            time.sleep(10)
            result_stdout = subprocess.check_output(
                ["evalai", "submission", submission_id, "result"]
            ).decode("utf-8")
            num_tries += 1

            # Raise error if it takes more than 5 minutes.
            if num_tries == 30:
                raise ConnectionError("Unable to get results from EvalAI within 5 minutes!")

        # Convert result to json.
        metrics = json.loads(result_stdout, encoding="utf-8")

        # keys: {"in-domain", "near-domain", "out-domain", "entire"}
        # In each of these, keys: {"B1", "B2", "B3", "B4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"}
        metrics = {
            "in-domain": metrics[0]["in-domain"],
            "near-domain": metrics[1]["near-domain"],
            "out-domain": metrics[2]["out-domain"],
            "entire": metrics[3]["entire"],
        }

        # Restructure the metrics dict for better tensorboard logging.
        # keys: {"B1", "B2", "B3", "B4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"}
        # In each of these, keys: keys: {"in-domain", "near-domain", "out-domain", "entire"}
        flipped_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        for key, val in metrics.items():
            for subkey, subval in val.items():
                flipped_metrics[subkey][key] = subval

        return flipped_metrics

