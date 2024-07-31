from process_model.Entity import Entity


# 用作存储re任务的从文章中抽出出来的基本格式，单位为一个文章一个model
class ReBaseData:
    def __init__(self, sent: str, head_entity: Entity, tail_entity: Entity, relation_type: str):
        self.sent = sent
        self.head_entity = head_entity
        self.tail_entity = tail_entity
        self.relation_type = relation_type
        self.reconstructed_sent = ""

