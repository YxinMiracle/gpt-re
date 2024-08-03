### 该文件用来处理输入进入关系抽取模型(re_model)的句子
import logging
from typing import List, Tuple

from transformers import AutoTokenizer

from model_cofig.config import get_params
from process_model.Entity import Entity
from process_model.EntityTypeEnum import EntityType
from process_model.ReBaseData import ReSentBaseData

params = get_params()

auto_tokenizer = AutoTokenizer.from_pretrained(params.bert_model_name, local_files_only=True)

logger = logging.getLogger()


def get_head_tail_entities_index_in_sent(sent_token_list: List[str],
                                         head_entity_list: List[str],
                                         tail_entity_list: List[str]) -> Tuple[List[int], List[int]]:
    """
    获取头实体和尾实体在token_list中的连续位置，用于最后取出相应的嵌入进行分类任务。确保连续x
    :param sent_token_list: 句子分词列表
    :param head_entity_list: 头实体的连续词列表
    :param tail_entity_list: 尾实体的连续词列表
    :return: 头实体和尾实体的起始索引组成的元组
    """

    def find_sublist_indices(main_list: List[str], sublist: List[str]) -> List[List[int]]:
        """查找子列表在主列表中的所有连续出现位置的起始索引"""
        sublist_len = len(sublist)
        res_list = []  # type: List[List[int]]
        for i in range(len(main_list) - sublist_len + 1):
            if main_list[i:i + sublist_len] == sublist:
                res_list.append([index for index in range(i, i + sublist_len + 1)])
        return res_list

    head_entity_index_in_sent = find_sublist_indices(sent_token_list, head_entity_list)  # type: List[List[int]]
    tail_entity_index_in_sent = find_sublist_indices(sent_token_list, tail_entity_list)  # type: List[List[int]]

    return head_entity_index_in_sent, tail_entity_index_in_sent


def get_temp_train_data(single_sent: ReSentBaseData) -> Tuple[List[int], List[int], List[int]]:
    """
    处理每个单独的句子，将每个元素转为对应id的形式
    :param single_sent:
    :return:
    """
    head_entity_obj = single_sent.head_entity  # type: Entity
    tail_entity_obj = single_sent.tail_entity  # type: Entity
    fine_tuned_re_model_tokens = single_sent.fine_tuned_re_model_tokens  # type: List[str]

    head_entity_index_in_sent, tail_entity_index_in_sent = get_head_tail_entities_index_in_sent(
        fine_tuned_re_model_tokens,
        head_entity_obj.entity_name_list,
        tail_entity_obj.entity_name_list)

    if len(head_entity_index_in_sent) == 0 or len(tail_entity_index_in_sent) == 0:
        # 如果不符合条件，那么就抛出对应的异常
        raise ValueError("数据处理中遇到条件不符合的数据，进行过滤")

    single_sent_word_id_list = []  # type: List[int]
    single_sent_entity_flag_list = []  # type: List[int]
    single_sent_label_id_list = [RELATION_NAME_LIST.index(single_sent.relation_type)]  # type: List[int]

    token_index = 0  # type: int
    while token_index < len(fine_tuned_re_model_tokens):
        if token_index == head_entity_index_in_sent[0]:
            # 如果说现在处理的词语为头实体
            for head_entity_index in head_entity_index_in_sent:
                token = fine_tuned_re_model_tokens[head_entity_index]  # type: str
                word_list = auto_tokenizer.tokenize(token)  # type: List[str] # 使用bert将单词进行分分词
                word_id_list = auto_tokenizer.convert_tokens_to_ids(word_list)  # type: List[int] # 将单词转为对应的Id
                single_sent_word_id_list.extend(word_id_list)
                single_sent_entity_flag_list.extend(
                    [EntityType.HEAD_ENTITY_ID.get_enum_value()] * len(word_id_list))
            token_index += len(head_entity_index_in_sent)
        elif token_index == tail_entity_index_in_sent[0]:
            # 如果说现在处理的实体为尾实体
            for tail_entity_index in tail_entity_index_in_sent:
                token = fine_tuned_re_model_tokens[tail_entity_index]  # type: str
                word_list = auto_tokenizer.tokenize(token)  # type: List[str] # 使用bert将单词进行分分词
                word_id_list = auto_tokenizer.convert_tokens_to_ids(word_list)  # type: List[int] # 将单词转为对应的Id
                single_sent_word_id_list.extend(word_id_list)
                single_sent_entity_flag_list.extend(
                    [EntityType.TAIL_ENTITY_ID.get_enum_value()] * len(word_id_list))
            token_index += len(tail_entity_index_in_sent)
        else:
            # 如果说现在处理的实体尾其他
            token = fine_tuned_re_model_tokens[token_index]
            word_list = auto_tokenizer.tokenize(token)  # type: List[str] # 使用bert将单词进行分分词
            word_id_list = auto_tokenizer.convert_tokens_to_ids(word_list)  # type: List[int] # 将单词转为对应的Id
            single_sent_word_id_list.extend(word_id_list)
            single_sent_entity_flag_list.extend([EntityType.OTHER_ENTITY_ID.get_enum_value()] * len(word_id_list))
            token_index += 1

    assert len(single_sent_word_id_list) == len(single_sent_entity_flag_list)
    return single_sent_word_id_list, single_sent_entity_flag_list, single_sent_label_id_list


def build_fine_tuned_re_model_input_data(knn_obj) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """
    用来处理re任务模型的整体输入数据
    result_word_id_list 构建句子id列表，便于放入模型中进行学习
    result_entity_flag_list 记录句子中的实体信息，要是头实体就为1，尾实体就为2，其他实体就是0，为此定义了枚举类
    result_label_id_list 记录每个关系对应的类型在RELATION_NAME_LIST列表中的下标位置
    :param knn_obj: 这里面有数据处理好的情报数据
    :return:
    """
    result_word_id_list, result_entity_flag_list, result_label_id_list = [], [], []
    for single_sent in knn_obj.source_data_list:  # 循环每一个句子
        try:
            single_sent_word_id_list, single_sent_entity_flag_list, single_sent_label_id_list = get_temp_train_data(
                single_sent)
            result_word_id_list.append(single_sent_word_id_list)
            result_entity_flag_list.append(single_sent_entity_flag_list)
            result_label_id_list.append(single_sent_label_id_list)
        except ValueError as e:
            logging.error(e)

    return result_word_id_list, result_entity_flag_list, result_label_id_list
