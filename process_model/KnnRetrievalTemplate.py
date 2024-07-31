from typing import List

from huawei_data_process.reconstructed_context_utils import reconstructed_sent
from process_model.Re_Base_Data import ReBaseData


# 定义了一个抽象类模板方法，执行论文架构图中左边的部分
class KnnRetrievalTemplate:
    def __init__(self, source_data_list: List[ReBaseData]):
        self.source_data_list = source_data_list  # type:List[ReBaseData]
        # 第一步，重构句子
        self.reconstruct_sent()
        self.relation_name_list = self.get_relation_name_list() # type:List[str]
        # self.test()

    def reconstruct_sent(self):
        for article_re_base_data in self.source_data_list:
            article_re_base_data.reconstructed_sent = reconstructed_sent(article_re_base_data)

    def test(self):
        for article_re_base_data in self.source_data_list:
            print(article_re_base_data.reconstructed_sent)

    def get_relation_name_list(self) -> List[str]:
        relation_name_set = set()  # type:set
        for sent in self.source_data_list:
            relation_name_set.add(sent.relation_type)
        return list(relation_name_set)
