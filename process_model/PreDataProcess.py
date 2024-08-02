# 数据预处理模板类，用户适配PFN Model
from typing import List

from process_model.ReBaseData import ReSentBaseData


class ProcessTrainDataTemplate:
    def __init__(self, data: List[ReSentBaseData]):
        self.sent_data_list = data # type: List[ReSentBaseData]

    # 构建实体2id的数据集 ner2idx.json
    def save_ner2idx_json_data(self):
        for sent_data in self.sent_data_list:
            print(sent_data)

    def do_process(self):

        # 第一步，构建ner的json
        self.save_ner2idx_json_data()
        pass