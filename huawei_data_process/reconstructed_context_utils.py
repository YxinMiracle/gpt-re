### 该文件用来处理输入进入SinCSE的句子
from process_model.Re_Base_Data import ReBaseData

# 重构句子的模板
RECONSTRUCTED_BASE_SENT = "The relation between \"{head_entity}\" and \"{tail_entity}\" in the context: \"{input_sent}\""


def reconstructed_sent(article_re_base_data: ReBaseData) -> str:
    """
    重构句子，输入的为原始句子，输出为重构之后的句子
    :param article_re_base_data: ReBaseData
    :return:
    """
    head_entity = article_re_base_data.head_entity.entity_name
    tail_entity = article_re_base_data.tail_entity.entity_name
    sent = article_re_base_data.sent
    return RECONSTRUCTED_BASE_SENT.format(head_entity=head_entity, tail_entity=tail_entity, input_sent=sent)
