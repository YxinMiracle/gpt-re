
# 用做存储对象基本信息
class Entity:
    def __init__(self, entity_name: str, entity_type: str):
        self.entity_name = entity_name
        self.entity_type = entity_type

    def __str__(self):
        print(f"name={self.entity_name},type={self.entity_type}")