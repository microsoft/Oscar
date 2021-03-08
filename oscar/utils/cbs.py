# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
# Copyright (c) 2019, Yufei Wang, Karan Desai. Licensed under the MIT license.
# Code is modified from https://github.com/nocaps-org/updown-baseline

import anytree
import base64
import json
import numpy as np
import os.path as op
import torch
from typing import Callable, Dict, List, Optional, Tuple

from oscar.modeling.modeling_utils import BeamHypotheses

StepFunctionType = Callable[
    [torch.Tensor, List[torch.Tensor]], Tuple[torch.Tensor, List[torch.Tensor]]
]


def _enlarge_single_tensor(t, batch_size, num_fsm_states, beam_size):
    # shape: (batch_size * beam_size, *)
    _, *last_dims = t.size()
    return (
        t.view(batch_size, 1, 1, *last_dims)
        .expand(batch_size, num_fsm_states, beam_size, *last_dims)
        .reshape(-1, *last_dims)
    )


class ConstrainedBeamSearch(object):
    r"""
    Implements Constrained Beam Search for decoding the most likely sequences conditioned on a
    Finite State Machine with specified state transitions.
    """

    def __init__(
        self,
        eos_token_ids: List[int],
        max_steps: int = 20,
        beam_size: int = 5,
        per_node_beam_size: Optional[int] = None,
        use_hypo: bool = False,
        tokenizer=None,
    ):
        self._eos_token_ids = eos_token_ids
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or self.beam_size
        self.num_keep_best = 1
        self.length_penalty = 1
        self.use_hypo = use_hypo
        self.tokenizer = tokenizer

    def search(
        self,
        start_predictions: torch.Tensor,
        start_state: List[torch.Tensor],
        step: StepFunctionType,
        fsm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Given a starting state, a step function, and an FSM adjacency matrix, apply Constrained
        Beam Search to find most likely target sequences satisfying specified constraints in FSM.

        .. note::

            If your step function returns ``-inf`` for some log probabilities
            (like if you're using a masked log-softmax) then some of the "best"
            sequences returned may also have ``-inf`` log probability. Specifically
            this happens when the beam size is smaller than the number of actions
            with finite log probability (non-zero probability) returned by the step function.
            Therefore if you're using a mask you may want to check the results from ``search``
            and potentially discard sequences with non-finite log probability.

        Parameters
        ----------
        start_predictions : torch.Tensor
            A tensor containing the initial predictions with shape ``(batch_size, )``. These are
            usually just ``@@BOUNDARY@@`` token indices.
        start_state : ``Dict[str, torch.Tensor]``
            The initial state passed to the ``step`` function. Each value of the state dict
            should be a tensor of shape ``(batch_size, *)``, where ``*`` means any other
            number of dimensions.
        step : ``StepFunctionType``
            A function that is responsible for computing the next most likely tokens, given the
            current state and the predictions from the last time step. The function should accept
            two arguments. The first being a tensor of shape ``(group_size,)``, representing the
            index of the predicted tokens from the last time step, and the second being the
            current state. The ``group_size`` will be ``batch_size * beam_size * num_fsm_states``
            except in the initial step, for which it will just be ``batch_size``. The function is
            expected to return a tuple, where the first element is a tensor of shape
            ``(group_size, vocab_size)`` containing the log probabilities of the tokens for the
            next step, and the second element is the updated state. The tensor in the state should
            have shape ``(group_size, *)``, where ``*`` means any other number of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(predictions, log_probabilities)``, where ``predictions``
            has shape ``(batch_size, num_fsm_states, beam_size, max_steps)``
            and ``log_probabilities`` has shape ``(batch_size, num_fsm_states, beam_size)``.
        """
        # shape: (batch_size, num_fsm_states, num_fsm_states, vocab_size)
        batch_size, num_fsm_states, _, vocab_size = fsm.size()

        # generated hypotheses
        generated_hyps = [
            [BeamHypotheses(self.num_keep_best, self.max_steps, self.length_penalty, early_stopping=False)
            for _ in range(num_fsm_states)]
            for bb in range(batch_size)
        ]

        # List of (batch_size, num_fsm_states, beam_size) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions: List[torch.Tensor] = []

        # List of (batch_size, num_fsm_states, beam_size) tensors. One for each time step. None for
        # the first. Stores the index n for the parent prediction.
        backpointers: List[torch.Tensor] = []

        # Calculate the first timestep. This is done outside the main loop because we are going
        # from a single decoder input (the output from the encoder) to the top `beam_size`
        # decoder outputs per FSM state. On the other hand, within the main loop we are going
        # from the `beam_size` elements of the beam (per FSM state) to `beam_size`^2 candidates
        # from which we will select the top `beam_size` elements for the next iteration.

        curr_ids = (
            start_predictions.expand(batch_size, self.beam_size*num_fsm_states)
            .reshape(batch_size*self.beam_size*num_fsm_states, 1)
        )
        # shape: start_class_log_probabilities (batch_size, vocab_size)
        start_class_logits, state = step(curr_ids, start_state)
        start_class_log_probabilities = torch.nn.functional.log_softmax(start_class_logits, dim=-1)
        start_class_log_probabilities = start_class_log_probabilities[:batch_size, :]
        vocab_size = start_class_log_probabilities.size(-1)

        start_state_predictions = start_class_log_probabilities.view(
            batch_size, 1, vocab_size
        ).expand(batch_size, num_fsm_states, vocab_size)

        start_state_predictions = start_state_predictions.masked_fill(
            (1 - fsm[:, 0, :, :]).to(dtype=torch.bool), float("-inf")
        )

        # (batch_size, num_fsm_states, beam_size)
        start_top_log_probabilities, start_predicted_classes = start_state_predictions.topk(
            self.beam_size
        )
        # shape: (batch_size, num_fsm_states, beam_size)
        last_log_probabilities = start_top_log_probabilities

        predictions.append(start_predicted_classes.view(batch_size, -1))

        log_probs_after_end = torch.full((1, vocab_size), float("-inf")).to(
            start_predictions.device
        )
        log_probs_after_end[:, self._eos_token_ids] = 0.0

        #state = {
            #key: _enlarge_single_tensor(value, batch_size, num_fsm_states, self.beam_size)
            #for (key, value) in state.items()
        #}

        step_state_mask = fsm.view(
            batch_size, num_fsm_states, num_fsm_states, 1, vocab_size
        ).expand(batch_size, num_fsm_states, num_fsm_states, self.beam_size, vocab_size)

        curr_len = curr_ids.shape[1]
        for timestep in range(self.max_steps - curr_len - 1):
            # shape: (batch_size * beam_size * num_fsm_states, )
            last_predictions = predictions[-1].reshape(
                batch_size * self.beam_size * num_fsm_states
            )
            cur_finished = (last_predictions==self._eos_token_ids[0])
            for eos_token in self._eos_token_ids[1:]:
                cur_finished = (cur_finished | (last_predictions==eos_token))
            if cur_finished.all():
                break

            curr_ids = torch.cat([curr_ids, last_predictions.unsqueeze(-1)], dim=1)

            class_logits, state = step(curr_ids, state)
            class_log_probabilities = torch.nn.functional.log_softmax(class_logits, dim=-1)
            #last_predictions_expanded = (
                #last_predictions.view(-1)
                #.unsqueeze(-1)
                #.expand(batch_size * num_fsm_states * self.beam_size, vocab_size)
            #)
            cur_finished_expanded = (
                cur_finished.unsqueeze(-1)
                .expand(batch_size * num_fsm_states * self.beam_size, vocab_size)
            )

            cleaned_log_probabilities = torch.where(
                #last_predictions_expanded == self._eos_token_ids,
                cur_finished_expanded,
                log_probs_after_end,
                class_log_probabilities,
            )
            cleaned_log_probabilities = cleaned_log_probabilities.view(
                batch_size, num_fsm_states, self.beam_size, vocab_size
            )

            device = start_predictions.device
            restricted_predicted_classes = torch.LongTensor(
                batch_size, num_fsm_states, self.beam_size
            ).to(start_predictions.device)
            restricted_beam_log_probs = torch.FloatTensor(
                batch_size, num_fsm_states, self.beam_size
            ).to(start_predictions.device)
            restricted_beam_indices = torch.LongTensor(
                batch_size, num_fsm_states, self.beam_size
            ).to(start_predictions.device)

            expanded_last_log_probabilities = last_log_probabilities.view(
                batch_size, num_fsm_states, self.beam_size, 1
            ).expand(batch_size, num_fsm_states, self.beam_size, self.per_node_beam_size)

            for i in range(num_fsm_states):
                # shape (batch_size, num_fsm_states, self.beam_size, vocab_size)
                state_log_probabilities = cleaned_log_probabilities

                state_log_probabilities = state_log_probabilities.masked_fill(
                    (1 - step_state_mask[:, :, i, :, :]).to(dtype=torch.bool), -1e20
                )
                top_log_probabilities, predicted_classes = state_log_probabilities.topk(
                    self.per_node_beam_size
                )
                summed_top_log_probabilities = (
                    top_log_probabilities + expanded_last_log_probabilities
                )
                # shape: (batch_size, old_num_fsm_states * beam_size * per_node_beam_size)
                reshaped_summed = summed_top_log_probabilities.reshape(batch_size, -1)

                # shape: (batch_size, old_num_fsm_states * beam_size * per_node_beam_size)
                reshaped_predicted_classes = predicted_classes.reshape(batch_size, -1)

                if not self.use_hypo:
                    # shape (batch_size, beam_size)
                    state_beam_log_probs, state_beam_indices = reshaped_summed.topk(self.beam_size)
                    # shape (batch_size, beam_size)
                    state_predicted_classes = reshaped_predicted_classes.gather(1, state_beam_indices)
                else:
                    # shape (batch_size, beam_size*per_node_beam_size)
                    candidate_beam_log_probs, candidate_beam_indices = reshaped_summed.topk(
                            self.beam_size*self.per_node_beam_size, sorted=True, largest=True)
                    # shape (batch_size, beam_size*per_node_beam_size)
                    candidate_predicted_classes = reshaped_predicted_classes.gather(1, candidate_beam_indices)
                    next_batch_beam = []
                    for batch_ex in range(batch_size):
                        next_sent_beam = []
                        for word_id, beam_id, log_prob in zip(candidate_predicted_classes[batch_ex],
                                    candidate_beam_indices[batch_ex],
                                    candidate_beam_log_probs[batch_ex]):
                            if word_id.item() in self._eos_token_ids:
                                generated_hyps[batch_ex][i].add(
                                    curr_ids[batch_ex * self.beam_size*num_fsm_states + beam_id/self.per_node_beam_size, :].clone(),
                                    log_prob.item()
                                )
                            else:
                                next_sent_beam.append((word_id, beam_id, log_prob))
                            if len(next_sent_beam) == self.beam_size:
                                break
                        assert len(next_sent_beam) == self.beam_size
                        next_batch_beam.extend(next_sent_beam)
                    state_predicted_classes = torch.tensor([x[0] for x in next_batch_beam],
                            device=device).reshape(batch_size, self.beam_size)
                    state_beam_indices = torch.tensor([x[1] for x in next_batch_beam],
                            device=device).reshape(batch_size, self.beam_size)
                    state_beam_log_probs = torch.tensor([x[2] for x in next_batch_beam],
                            device=device).reshape(batch_size, self.beam_size)

                restricted_predicted_classes[:, i, :] = state_predicted_classes
                restricted_beam_indices[:, i, :] = state_beam_indices
                restricted_beam_log_probs[:, i, :] = state_beam_log_probs

            restricted_predicted_classes = restricted_predicted_classes.view(batch_size, -1)
            predictions.append(restricted_predicted_classes)

            backpointer = restricted_beam_indices // self.per_node_beam_size
            backpointers.append(backpointer.view(batch_size, -1))

            last_log_probabilities = restricted_beam_log_probs.view(batch_size, num_fsm_states, -1)

            def track_back_state(state_tensor):
                _, *last_dims = state_tensor.size()
                # shape: (batch_size, beam_size, *)
                expanded_backpointer = backpointer.view(
                    batch_size, num_fsm_states * self.beam_size, *([1] * len(last_dims))
                ).expand(batch_size, num_fsm_states * self.beam_size, *last_dims)

                # shape: (batch_size * beam_size, *)
                return (
                    state_tensor.reshape(batch_size, num_fsm_states * self.beam_size, *last_dims)
                    .gather(1, expanded_backpointer)
                    .reshape(batch_size * num_fsm_states * self.beam_size, *last_dims)
                )
            # reorder states
            if state is not None:
                state = tuple(track_back_state(value) for value in state)
            curr_ids = track_back_state(curr_ids)

        last_predictions = predictions[-1].reshape(
            batch_size * self.beam_size * num_fsm_states
        )
        curr_ids = torch.cat([curr_ids, last_predictions.unsqueeze(-1)], dim=1)
        # Reconstruct the sequences.
        # shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # shape: (batch_size, beam_size, 1)
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)

            reconstructed_predictions.append(cur_preds)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)
        all_predictions = all_predictions.view(batch_size, num_fsm_states, self.beam_size, -1)
        assert (all_predictions == curr_ids.reshape(batch_size, num_fsm_states,
                self.beam_size, -1)[:,:,:,1:]).all()

        if self.use_hypo:
            decoded = all_predictions.new(batch_size, num_fsm_states, 1,
                    self.max_steps).fill_(self._eos_token_ids[0])
            scores = last_log_probabilities.new(batch_size, num_fsm_states,
                    1).fill_(-1e5)
            for batch_ex in range(batch_size):
                for i in range(num_fsm_states):
                    beam = all_predictions[batch_ex, i, 0, :]
                    log_prob = last_log_probabilities[batch_ex, i, 0]
                    generated_hyps[batch_ex][i].add(
                        beam.clone(),
                        log_prob.item()
                    )
                    hyps = generated_hyps[batch_ex][i].hyp
                    assert len(hyps) == 1
                    score, sent = hyps[0]
                    decoded[batch_ex, i, 0, :len(sent)] = sent
                    scores[batch_ex, i, 0] = score
            all_predictions = decoded
            last_log_probabilities = scores

        # pad to the same length, otherwise DataParallel will give error
        pad_len = self.max_steps - all_predictions.shape[-1]
        if pad_len > 0:
            padding_ids = all_predictions.new(
                    batch_size, num_fsm_states, self.beam_size,
                    pad_len).fill_(self._eos_token_ids[0])
            all_predictions = torch.cat([all_predictions, padding_ids], dim=-1)

        return all_predictions, last_log_probabilities


def select_best_beam_with_constraints(
    beams: torch.Tensor,
    beam_log_probabilities: torch.Tensor,
    given_constraints: torch.Tensor,
    min_constraints_to_satisfy: int,
    eos_token_ids: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Select the best beam which satisfies specified minimum constraints out of a total number of
    given constraints.

    .. note::

        The implementation of this function goes hand-in-hand with the FSM building implementation
        in :meth:`~updown.utils.constraints.FiniteStateMachineBuilder.build` - it defines which
        state satisfies which (basically, how many) constraints. If the "definition" of states
        change, then selection of beams also changes accordingly.

    Parameters
    ----------
    beams: torch.Tensor
        A tensor of shape ``(batch_size, num_states, beam_size, max_decoding_steps)`` containing
        decoded beams by :class:`~updown.modules.cbs.ConstrainedBeamSearch`. These beams are
        sorted according to their likelihood (descending) in ``beam_size`` dimension.
    beam_log_probabilities: torch.Tensor
        A tensor of shape ``(batch_size, num_states, beam_size)`` containing likelihood of decoded
        beams.
    given_constraints: torch.Tensor
        A tensor of shape ``(batch_size, )`` containing number of constraints given at the start
        of decoding.
    min_constraints_to_satisfy: int
        Minimum number of constraints to satisfy. This is either 2, or ``given_constraints`` if
        they are less than 2. Beams corresponding to states not satisfying at least these number
        of constraints will be dropped. Only up to 3 supported.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Decoded sequence (beam) which has highest likelihood among beams satisfying constraints.
    """
    batch_size, num_states, beam_size, max_decoding_steps = beams.size()

    best_beams: List[torch.Tensor] = []
    best_beam_log_probabilities: List[torch.Tensor] = []

    for i in range(batch_size):
        # fmt: off
        valid_states = [
            s for s in range(2 ** given_constraints[i].item())
            if bin(s).count("1") >= min(given_constraints[i], min_constraints_to_satisfy)
        ]
        # fmt: on

        valid_beams = beams[i, valid_states, 0, :]
        valid_length = torch.ones_like(valid_beams)
        for eos_token_id in eos_token_ids:
            valid_length = valid_length.mul(valid_beams.ne(eos_token_id).long())
        valid_length = valid_length.sum(1) + 1
        valid_beam_log_probabilities = beam_log_probabilities[i, valid_states, 0] / valid_length

        selected_index = torch.argmax(valid_beam_log_probabilities)
        best_beams.append(valid_beams[selected_index, :] )
        best_beam_log_probabilities.append(valid_beam_log_probabilities[selected_index])

    # shape: (batch_size, max_decoding_steps)
    return (torch.stack(best_beams).long().to(beams.device),
            torch.stack(best_beam_log_probabilities).to(beams.device))


def load_wordforms(wordforms_tsvpath):
    wordforms = {}
    with open(wordforms_tsvpath, "r") as fp:
        for line in fp:
            parts = line.strip().split('\t')
            wordforms[parts[0]] = parts[1].split(',')
    return wordforms


class ConstraintBoxesReader(object):
    r"""
    A reader for annotation files containing detected bounding boxes.
    For our use cases, the detections are from an object detector trained using Open Images.
    """
    def __init__(self, boxes_tsvpath):
        self._image_key_to_boxes = {}
        with open(boxes_tsvpath, 'r') as fp:
            for line in fp:
                parts = line.strip().split('\t')
                img_key = parts[0]
                labels = json.loads(parts[1])
                boxes, class_names, scores = [], [], []
                for box in labels:
                    boxes.append(box['rect'])
                    class_names.append(box['class'].lower())
                    scores.append(box['conf'])
                boxes = np.array(boxes)
                scores = np.array(scores)
                self._image_key_to_boxes[img_key] = {"boxes": boxes, "class_names": class_names, "scores": scores}

    def __len__(self):
        return len(self._image_key_to_boxes)

    def __getitem__(self, image_key):
        # Some images may not have any boxes, handle that case too.
        if image_key not in self._image_key_to_boxes:
            return {"boxes": np.array([]), "class_names": [], "scores":
                    np.array([])}
        else:
            return self._image_key_to_boxes[image_key]


class ConstraintFilter(object):
    r"""
    A helper class to perform constraint filtering for providing sensible set of constraint words
    while decoding.

    Extended Summary
    ----------------
    The original work proposing `Constrained Beam Search <https://arxiv.org/abs/1612.00576>`_
    selects constraints randomly.

    We remove certain categories from a fixed set of "blacklisted" categories, which are either
    too rare, not commonly uttered by humans, or well covered in COCO. We resolve overlapping
    detections (IoU >= 0.85) by removing the higher-order of the two objects (e.g. , a "dog" would
    suppress a ‘mammal’) based on the Open Images class hierarchy (keeping both if equal).
    Finally, we take the top-k objects based on detection confidence as constraints.

    Parameters
    ----------
    hierarchy_jsonpath: str
        Path to a JSON file containing a hierarchy of Open Images object classes.
    nms_threshold: float, optional (default = 0.85)
        NMS threshold for suppressing generic object class names during constraint filtering,
        for two boxes with IoU higher than this threshold, "dog" suppresses "animal".
    max_given_constraints: int, optional (default = 3)
        Maximum number of constraints which can be specified for CBS decoding. Constraints are
        selected based on the prediction confidence score of their corresponding bounding boxes.
    """

    # fmt: off
    BLACKLIST: List[str] = [
        "auto part", "bathroom accessory", "bicycle wheel", "boy", "building", "clothing",
        "door handle", "fashion accessory", "footwear", "girl", "hiking equipment", "human arm",
        "human beard", "human body", "human ear", "human eye", "human face", "human foot",
        "human hair", "human hand", "human head", "human leg", "human mouth", "human nose",
        "land vehicle", "mammal", "man", "person", "personal care", "plant", "plumbing fixture",
        "seat belt", "skull", "sports equipment", "tire", "tree", "vehicle registration plate",
        "wheel", "woman", "__background__",
    ]
    # fmt: on

    REPLACEMENTS: Dict[str, str] = {
        "band-aid": "bandaid",
        "wood-burning stove": "wood burning stove",
        "kitchen & dining room table": "table",
        "salt and pepper shakers": "salt and pepper",
        "power plugs and sockets": "power plugs",
        "luggage and bags": "luggage",
    }

    def __init__(
        self, hierarchy_jsonpath, nms_threshold, max_given_constraints
    ):
        def __read_hierarchy(node, parent=None):
            # Cast an ``anytree.AnyNode`` (after first level of recursion) to dict.
            attributes = dict(node)
            children = attributes.pop("Subcategory", [])

            node = anytree.AnyNode(parent=parent, **attributes)
            for child in children:
                __read_hierarchy(child, parent=node)
            return node

        # Read the object class hierarchy as a tree, to make searching easier.
        self._hierarchy = __read_hierarchy(json.load(open(hierarchy_jsonpath)))

        self._nms_threshold = nms_threshold
        self._max_given_constraints = max_given_constraints

    def __call__(self, boxes: np.ndarray, class_names: List[str], scores: np.ndarray) -> List[str]:

        # Remove padding boxes (which have prediction confidence score = 0), and remove boxes
        # corresponding to all blacklisted classes. These will never become CBS constraints.
        keep_indices = []
        for i in range(len(class_names)):
            if scores[i] > 0 and class_names[i] not in self.BLACKLIST:
                keep_indices.append(i)

        boxes = boxes[keep_indices]
        class_names = [class_names[i] for i in keep_indices]
        scores = scores[keep_indices]

        # Perform non-maximum suppression according to category hierarchy. For example, for highly
        # overlapping boxes on a dog, "dog" suppresses "animal".
        keep_indices = self._nms(boxes, class_names)
        boxes = boxes[keep_indices]
        class_names = [class_names[i] for i in keep_indices]
        scores = scores[keep_indices]

        # Retain top-k constraints based on prediction confidence score.
        class_names_and_scores = sorted(list(zip(class_names, scores)), key=lambda t: -t[1])
        class_names_and_scores = class_names_and_scores[: self._max_given_constraints]

        # Replace class name according to ``self.REPLACEMENTS``.
        class_names = [self.REPLACEMENTS.get(t[0], t[0]) for t in class_names_and_scores]

        # Drop duplicates.
        class_names = list(set(class_names))
        return class_names

    def _nms(self, boxes: np.ndarray, class_names: List[str]):
        if len(class_names) == 0:
            return []

        # For object class, get the height of its corresponding node in the hierarchy tree.
        # Less height => finer-grained class name => higher score.
        heights = np.array(
            [
                anytree.search.findall(self._hierarchy, lambda node: node.LabelName.lower() in c)[0].height
                for c in class_names
            ]
        )
        # Get a sorting of the heights in ascending order, i.e. higher scores first.
        score_order = heights.argsort()

        # Compute areas for calculating intersection over union. Add 1 to avoid division by zero
        # for zero area (padding/dummy) boxes.
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Fill "keep_boxes" with indices of boxes to keep, move from left to right in
        # ``score_order``, keep current box index (score_order[0]) and suppress (discard) other
        # indices of boxes having lower IoU threshold with current box from ``score_order``.
        # list. Note the order is a sorting of indices according to scores.
        keep_box_indices = []

        while score_order.size > 0:
            # Keep the index of box under consideration.
            current_index = score_order[0]
            keep_box_indices.append(current_index)

            # For the box we just decided to keep (score_order[0]), compute its IoU with other
            # boxes (score_order[1:]).
            xx1 = np.maximum(x1[score_order[0]], x1[score_order[1:]])
            yy1 = np.maximum(y1[score_order[0]], y1[score_order[1:]])
            xx2 = np.minimum(x2[score_order[0]], x2[score_order[1:]])
            yy2 = np.minimum(y2[score_order[0]], y2[score_order[1:]])

            intersection = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
            union = areas[score_order[0]] + areas[score_order[1:]] - intersection

            # Perform NMS for IoU >= 0.85. Check score, boxes corresponding to object
            # classes with smaller/equal height in hierarchy cannot be suppressed.
            keep_condition = np.logical_or(
                heights[score_order[1:]] >= heights[score_order[0]],
                intersection / union <= self._nms_threshold,
            )

            # Only keep the boxes under consideration for next iteration.
            score_order = score_order[1:]
            score_order = score_order[np.where(keep_condition)[0]]

        return keep_box_indices


class FiniteStateMachineBuilder(object):
    r"""
    A helper class to build a Finite State Machine for Constrained Beam Search, as per the
    state transitions shown in Figures 7 through 9 from our
    `paper appendix <https://arxiv.org/abs/1812.08658>`_.

    The FSM is constructed on a per-example basis, and supports up to three constraints,
    with each constraint being an Open Image class having up to three words (for example
    ``salt and pepper``). Each word in the constraint may have several word-forms (for
    example ``dog``, ``dogs``).

    .. note:: Providing more than three constraints may work but it is not tested.

    **Details on Finite State Machine Representation**

    .. image:: ../_static/fsm.jpg

    The FSM is representated as an adjacency matrix. Specifically, it is a tensor of shape
    ``(num_total_states, num_total_states, vocab_size)``. In this, ``fsm[S1, S2, W] = 1`` indicates
    a transition from "S1" to "S2" if word "W" is decoded. For example, consider **Figure 9**.
    The decoding is at initial state (``q0``), constraint word is ``D1``, while any other word
    in the vocabulary is ``Dx``. Then we have::

        fsm[0, 0, D1] = 0 and fsm[0, 1, D1] = 1    # arrow from q0 to q1
        fsm[0, 0, Dx] = 1 and fsm[0, 1, Dx] = 0    # self-loop on q0

    Consider up to "k" (3) constraints and up to "w" (3) words per constraint. We define these
    terms (as members in the class).

    .. code-block::

        _num_main_states = 2 ** k (8)
        _total_states = num_main_states * w (24)

    First eight states are considered as "main states", and will always be a part of the FSM. For
    less than "k" constraints, some states will be unreachable, hence "useless". These will be
    ignored automatically.

    For any multi-word constraint, we use extra "sub-states" after first ``2 ** k`` states. We
    make connections according to **Figure 7-8** for such constraints. We dynamically trim unused
    sub-states to save computation during decoding. That said, ``num_total_states`` dimension is
    at least 8.

    A state "q" satisfies number of constraints equal to the number of "1"s in the binary
    representation of that state. For example:

      - state "q0" (000) satisfies 0 constraints.
      - state "q1" (001) satisfies 1 constraint.
      - state "q2" (010) satisfies 1 constraint.
      - state "q3" (011) satisfies 2 constraints.

    and so on. Only main states fully satisfy constraints.

    Parameters
    ----------
    tokenizer: BertTokenizer
    wordforms_tsvpath: str
        Path to a TSV file containing two fields: first is the name of Open Images object class
        and second field is a comma separated list of words (possibly singular and plural forms
        of the word etc.) which could be CBS constraints.
    max_given_constraints: int, optional (default = 3)
        Maximum number of constraints which could be given while cbs decoding. Up to three
        supported.
    max_words_per_constraint: int, optional (default = 3)
        Maximum number of words per constraint for multi-word constraints. Note that these are
        for multi-word object classes (for example: ``fire hydrant``) and not for multiple
        "word-forms" of a word, like singular-plurals. Up to three supported.
    """

    def __init__(
        self,
        tokenizer,
        constraint2tokens_tsvpath,
        tokenforms_tsvpath,
        max_given_constraints,
        max_words_per_constraint = 4,
    ):
        self._tokenizer = tokenizer
        self._max_given_constraints = max_given_constraints
        self._max_words_per_constraint = max_words_per_constraint

        self._num_main_states = 2 ** max_given_constraints
        self._num_total_states = self._num_main_states * max_words_per_constraint

        self._wordforms: Dict[str, List[str]] = load_wordforms(tokenforms_tsvpath)
        self._constraint2tokens = load_wordforms(constraint2tokens_tsvpath)

    def build(self, constraints: List[str]):
        r"""
        Build a finite state machine given a list of constraints.

        Parameters
        ----------
        constraints: List[str]
            A list of up to three (possibly) multi-word constraints, in our use-case these are
            Open Images object class names.

        Returns
        -------
        Tuple[torch.Tensor, int]
            A finite state machine as an adjacency matrix, index of the next available unused
            sub-state. This is later used to trim the unused sub-states from FSM.
        """
        assert len(constraints) <= self._max_given_constraints
        fsm = torch.zeros(self._num_total_states, self._num_total_states, dtype=torch.uint8)

        # Self loops for all words on main states.
        fsm[range(self._num_main_states), range(self._num_main_states)] = 1

        fsm = fsm.unsqueeze(-1).repeat(1, 1, self._tokenizer.vocab_size)

        substate_idx = self._num_main_states
        for i, constraint in enumerate(constraints):
            fsm, substate_idx = self._add_nth_constraint(fsm, i + 1, substate_idx, constraint)

        return fsm, substate_idx

    def _add_nth_constraint(self, fsm: torch.Tensor, n: int, substate_idx: int, constraint: str):
        r"""
        Given an (incomplete) FSM matrix with transitions for "(n - 1)" constraints added, add
        all transitions for the "n-th" constraint.

        Parameters
        ----------
        fsm: torch.Tensor
            A tensor of shape ``(num_total_states, num_total_states, vocab_size)`` representing an
            FSM under construction.
        n: int
            The cardinality of constraint to be added. Goes as 1, 2, 3... (not zero-indexed).
        substate_idx: int
            An index which points to the next unused position for a sub-state. It starts with
            ``(2 ** num_main_states)`` and increases according to the number of multi-word
            constraints added so far. The calling method, :meth:`build` keeps track of this.
        constraint: str
            A (possibly) multi-word constraint, in our use-case it is an Open Images object class
            name.

        Returns
        -------
        Tuple[torch.Tensor, int]
            FSM with added connections for the constraint and updated ``substate_idx`` pointing to
            the next unused sub-state.
        """
        #words = constraint.split()
        words = []
        for w in constraint.split():
            words.extend(self._constraint2tokens[w])
        #TODO: set max_words_per_constraint
        #assert len(words) <= self._max_words_per_constraint
        if len(words) > self._max_words_per_constraint:
            words = words[:self._max_words_per_constraint]
        connection_stride = 2 ** (n - 1)

        from_state = 0
        while from_state < self._num_main_states:
            for _ in range(connection_stride):
                word_from_state = from_state
                for i, word in enumerate(words):
                    # fmt: off
                    # Connect to a sub-state for all tokens in multi-word constraint except last.
                    if i != len(words) - 1:
                        fsm = self._connect(
                            fsm, word_from_state, substate_idx, word, reset_state=from_state
                        )
                        word_from_state = substate_idx
                        substate_idx += 1
                    else:
                        fsm = self._connect(
                            fsm, word_from_state, from_state + connection_stride, word,
                            reset_state=from_state,
                        )
                    # fmt: on
                from_state += 1
            from_state += connection_stride
        return fsm, substate_idx

    def _connect(
        self, fsm: torch.Tensor, from_state: int, to_state: int, word: str, reset_state: int = None
    ):
        r"""
        Add a connection between two states for a particular word (and all its word-forms). This
        means removing self-loop from ``from_state`` for all word-forms of ``word`` and connecting
        them to ``to_state``.
        
        Extended Summary
        ----------------
        In case of multi-word constraints, we return back to the ``reset_state`` for any utterance
        other than ``word``, to satisfy a multi-word constraint if all words are decoded
        consecutively. For example: for "fire hydrant" as a constraint between Q0 and Q1, we reach
        a sub-state "Q8" on decoding "fire". Go back to main state "Q1" on decoding "hydrant"
        immediately after, else we reset back to main state "Q0".

        Parameters
        ----------
        fsm: torch.Tensor
            A tensor of shape ``(num_total_states, num_total_states, vocab_size)`` representing an
            FSM under construction.
        from_state: int
            Origin state to make a state transition.
        to_state: int
            Destination state to make a state transition.
        word: str
            The word which serves as a constraint for transition between given two states.
        reset_state: int, optional (default = None)
           State to reset otherwise. This is only valid if ``from_state`` is a sub-state.

        Returns
        -------
        torch.Tensor
            FSM with the added connection.
        """
        wordforms = self._wordforms.get(word, [word])
        #wordform_indices = [self._vocabulary.get_token_index(w) for w in wordforms]
        wordform_indices = self._tokenizer.convert_tokens_to_ids(wordforms)

        for wordform_index in wordform_indices:
            fsm[from_state, to_state, wordform_index] = 1
            fsm[from_state, from_state, wordform_index] = 0

        if reset_state is not None:
            fsm[from_state, from_state, :] = 0
            fsm[from_state, reset_state, :] = 1
            for wordform_index in wordform_indices:
                fsm[from_state, reset_state, wordform_index] = 0

        return fsm

