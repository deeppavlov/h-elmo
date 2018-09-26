from typing import Dict, Tuple
# from deeppavlov.core.data.dataset_reader import DatasetReader
# from deeppavlov.core.common.registry import register

import sys
from util.deal_with_cephfs import add_repo_2_sys_path, add_cephfs_to_path
sys.path = add_repo_2_sys_path('DeepPavlov')
# sys.path.append('/home/anton/DeepPavlov')
# if '/home/anton/dpenv/src/deeppavlov' in sys.path:
#     sys.path.remove('/home/anton/dpenv/src/deeppavlov')


from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.common.registry import register


@register('one_str_lm_reader')
class OneStrLMReader(DatasetReader):

    def read(self, data_path: str, select: Dict[str, Tuple[int, int]]):
        add_cephfs_to_path(data_path)
        dataset = {}
        with open(data_path, 'r') as f:
            text = f.read()
        for dt, start_and_size in select.items():
            dataset[dt] = text[start_and_size[0]:start_and_size[0] + start_and_size[1]]
        return dataset
