# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

import logging
import numpy as np
import os
import os.path as op
import shutil
from .misc import mkdir
from .tsv_file import TSVFile


def tsv_writer(values, tsv_file_name, sep='\t'):
    mkdir(os.path.dirname(tsv_file_name))
    tsv_file_name_tmp = tsv_file_name + '.tmp'
    with open(tsv_file_name_tmp, 'wb') as fp:
        assert values is not None
        for value in values:
            assert value is not None
            v = sep.join(map(lambda v: v.decode() if type(v) == bytes else str(v), value)) + '\n'
            v = v.encode()
            fp.write(v)
    os.rename(tsv_file_name_tmp, tsv_file_name)


def concat_files(ins, out):
    out_tmp = out + '.tmp'
    with open(out_tmp, 'wb') as fp_out:
        for i, f in enumerate(ins):
            with open(f, 'rb') as fp_in:
                shutil.copyfileobj(fp_in, fp_out, 1024*1024*10)
    os.rename(out_tmp, out)


def concat_tsv_files(tsvs, out_tsv, generate_lineidx=False):
    concat_files(tsvs, out_tsv)
    if generate_lineidx:
        sizes = [os.stat(t).st_size for t in tsvs]
        sizes = np.cumsum(sizes)
        all_idx = []
        for i, t in enumerate(tsvs):
            for idx in load_list_file(op.splitext(t)[0] + '.lineidx'):
                if i == 0:
                    all_idx.append(idx)
                else:
                    all_idx.append(str(int(idx) + sizes[i - 1]))
        with open(op.splitext(out_tsv)[0] + '.lineidx', 'w') as f:
            f.write('\n'.join(all_idx))


def load_list_file(fname):
    with open(fname, 'r') as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result


def reorder_tsv_keys(in_tsv_file, ordered_keys, out_tsv_file):
    tsv = TSVFile(in_tsv_file, generate_lineidx=True)
    keys = [tsv.seek(i)[0] for i in range(len(tsv))]
    key_to_idx = {key: i for i, key in enumerate(keys)}
    def gen_rows():
        for key in ordered_keys:
            idx = key_to_idx[key]
            yield tsv.seek(idx)
    tsv_writer(gen_rows(), out_tsv_file)


def delete_tsv_files(tsvs):
    for t in tsvs:
        if op.isfile(t):
            try_delete(t)
        line = op.splitext(t)[0] + '.lineidx'
        if op.isfile(line):
            try_delete(line)


def try_once(func):
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.info('ignore error \n{}'.format(str(e)))
    return func_wrapper


@try_once
def try_delete(f):
    os.remove(f)


