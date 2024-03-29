import sys
from typing import Dict, Tuple

from helmo.util.path_help import add_cephfs_to_path
from helmo.util.interpreter import prepend_repo_2_sys_path

sys.path = prepend_repo_2_sys_path('DeepPavlov')


from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.common.registry import register


@register('one_str_lm_reader')
class OneStrLMReader(DatasetReader):

    def read(self, data_path: str, select: Dict[str, Tuple[int, int]]):
        data_path = add_cephfs_to_path(data_path)
        dataset = {}
        with open(data_path, 'r') as f:
            text = f.read()
        for dt, start_and_size in select.items():
            dataset[dt] = text[start_and_size[0]:start_and_size[0] + start_and_size[1]]
        return dataset
