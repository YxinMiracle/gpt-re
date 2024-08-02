from typing import List
import torch.utils.data as data
import torch.nn as nn
from model_cofig.config import get_params
from model_pre_data_process.re_model_data_process_utils import build_fine_tuned_re_model_input_data
from process_model.ReBaseData import ReSentBaseData
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer

params = get_params()

auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name, local_files_only=True)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index


# 定义了一个抽象类模板方法，执行论文架构图中左边的部分，存放了有关关系抽取任务的所有数据，有想要的都可以从这个类中获取出来
class KnnRetrievalTemplate:
    def __init__(self, source_data_list: List[ReSentBaseData]):
        self.source_data_list = source_data_list  # type:List[ReSentBaseData]
        # self.relation_name_list = self.get_relation_name_list() # type:List[str]
        # 1. 数据预处理，将所有单词和标签转为对应的id形式，方便模型进行输入
        self.do_pre_data_for_re_model()

    def get_relation_name_list(self) -> List[str]:
        """
        获取整个数据集中的关系列表
        :return:
        """
        relation_name_set = set()  # type:set
        for sent in self.source_data_list:
            relation_name_set.add(sent.relation_type)
        return list(relation_name_set)
