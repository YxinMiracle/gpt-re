import pathlib
from typing import List
import os
import pickle
from process_model.ReBaseData import ReSentBaseData

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class LLM_RE:
    K_SHOT_TEMPLATE = "{sent} \n Given the context, the relation between <{head_entity_name}> and <{tail_entity_name}> is <{relation_name}>."

    def __init__(self, origin_sent: ReSentBaseData, similar_sent_list: List[ReSentBaseData], params):
        self.origin_sent = origin_sent
        self.similar_sent_list = similar_sent_list
        self.params = params
        self.root_dir = str(pathlib.Path(__file__).resolve().parent.parent.parent)
        self.goal_label_cache = self._init_goal_label_cache()


    def _init_goal_label_cache(self):
        cache_file_name = self.root_dir + os.path.sep + self.params.data_directory_name + os.path.sep + self.params.gold_label_cache_file_name # type: str
        if os.path.exists(cache_file_name):
            with open(cache_file_name, "rb") as fp:
                goal_label_cache = pickle.load(fp) # type: dict
            return goal_label_cache
        logger.info("没有保存缓存文件，需要进行初始化")
        init_cache_dict = {} # type: dict
        with open(cache_file_name, "wb") as fp:
            pickle.dump(init_cache_dict, fp)



    def gold_label_induced(self):
        """ 金标签提示 """
        demonstration_list = [] # type: list
        for similar_sent in self.similar_sent_list:
            sent = similar_sent.sent # type: str
            head_entity_name = similar_sent.head_entity.entity_name # type: str
            tail_entity_name = similar_sent.tail_entity.entity_name # type: str
            relation_name = similar_sent.relation_type # type: str

            demonstration = self.K_SHOT_TEMPLATE.format(sent=sent,
                                                        head_entity_name=head_entity_name,
                                                        tail_entity_name=tail_entity_name,
                                                        relation_name=relation_name)

            print(demonstration)
