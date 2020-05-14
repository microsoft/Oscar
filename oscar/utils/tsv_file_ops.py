# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

import os
from .misc import mkdir


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

