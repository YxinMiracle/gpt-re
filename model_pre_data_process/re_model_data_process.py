import torch
import torch.nn as nn
import torch.utils.data as data
import logging

from sympy.physics.units import force
from transformers import AutoTokenizer

from model_cofig.config import get_params
from process_model.Entity import Entity
from process_model.KnnRetrievalTemplate import KnnRetrievalTemplate

params = get_params()

auto_tokenizer = AutoTokenizer.from_pretrained(params.bert_model_name, local_files_only=True)

logger = logging.getLogger()


def build_find_tuned_re_model_input_data(knn_obj: KnnRetrievalTemplate):
    """
    用来处理re特务模型的输入数据
    :param knn_obj: 这里面有数据处理好的情报数据
    :return:
    """
    result_word_id_list, result_label_id_list = [], []
    for single_sent in knn_obj.source_data_list:
        """
        single_sent为一句话，我需要将他变成
        """
        #
        head_entity = single_sent.head_entity  # type: Entity
        tail_entity = single_sent.tail_entity  # type: Entity
        print(1123123)

        print(head_entity.entity_type, head_entity.entity_name)
        print(tail_entity.entity_type, tail_entity.entity_name)
