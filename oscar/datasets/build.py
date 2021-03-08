import os
import logging
import torch
from oscar.utils.misc import get_world_size
from .oscar_tsv import OscarTSVDataset
from transformers.pytorch_transformers import BertTokenizer


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    """
    def __call__(self, batch):
        return list(zip(*batch))


def build_dataset(args):
    """
    Arguments:
        args: configuration.
    """
    full_yaml_file = os.path.join(args.data_dir, args.dataset_file)
    assert os.path.isfile(full_yaml_file)

    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)

    cfg = dict(
        yaml_file=full_yaml_file,
        args=args,
        seq_len=args.max_seq_length,
        on_memory=args.on_memory,
        tokenizer=tokenizer,
    )
    # make dataset from factory
    datasets = [OscarTSVDataset(**cfg)]
    if args.extra_dataset_file:
        full_yaml_file = os.path.join(args.data_dir, args.extra_dataset_file)
        assert os.path.isfile(full_yaml_file)
        cfg['yaml_file'] = full_yaml_file
        cfg['textb_sample_mode'] = args.extra_textb_sample_mode
        datasets.append(OscarTSVDataset(**cfg))

    return datasets


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=shuffle
        )
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


class IterationBasedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def make_batch_data_sampler(
        sampler, images_per_batch, num_iters=None,
        start_iter=0
):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_batch, drop_last=False
    )
    if num_iters is not None and num_iters >= 0:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(args, is_distributed=False, arguments=None):
    num_gpus = get_world_size()
    # figure out start iteration
    if arguments is None:
        start_iter = 0
    else:
        start_iter = arguments['iteration']
    # figure out the batchsize
    grad_accumulate_steps = 1
    if hasattr(args, 'gradient_accumulation_steps'):
        grad_accumulate_steps = args.gradient_accumulation_steps
    assert (
            args.train_batch_size % grad_accumulate_steps == 0
    ), "train_batch_size ({}) must be divisible by the number "
    "of Gradient accumulation ({}) used."\
        .format(args.train_batch_size, grad_accumulate_steps)
    images_per_batch = args.train_batch_size//grad_accumulate_steps
    assert (
        images_per_batch % num_gpus == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
    "of GPUs ({}) used.".format(images_per_batch, num_gpus)
    images_per_gpu = images_per_batch // num_gpus
    logger = logging.getLogger(__name__)
    logger.info("Train with {} images per GPU".format(images_per_gpu))
    shuffle = True
    num_iters = args.max_iters * grad_accumulate_steps

    # build dataset
    datasets = build_dataset(args)

    data_loaders = []
    for i, dataset in enumerate(datasets):
        sampler = make_data_sampler(dataset, shuffle, is_distributed)

        batch_sampler = make_batch_data_sampler(
           sampler, images_per_gpu, num_iters, start_iter
        )
        num_workers = args.num_workers
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=BatchCollator(),
            pin_memory=True,
        )
        data_loaders.append(data_loader)
    return data_loaders
