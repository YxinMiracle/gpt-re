from typing import List
import os

from huawei_data_process.process_json_data import process_file
from process_model.KnnRetrievalTask import KnnRetrievalTemplate
from process_model.ReBaseData import ReSentBaseData
import json


def get_file_full_path_list_in_directory(directory_name: str) -> List[str]:
    """
    获取某个目录下的所有文件的绝对路径
    :param directory_name:
    :return:
    """
    files = []
    for entry in os.listdir(directory_name):
        full_path = os.path.join(directory_name, entry)
        if os.path.isfile(full_path):
            files.append(os.path.abspath(full_path))
    return files


def get_article_re_task_data(file_path: str, file_id: int) -> List[ReSentBaseData]:
    """
    返回一个文章中所有的能进行关系抽取的句子，会出现有跨句的情况
    :param file_path:
    :param file_id:
    :return:
    """
    file_data = {}  # type: dict
    with open(file_path, "r", encoding="UTF-8") as fp:
        file_data = json.load(fp)
    article_re_task_res = process_file(file_data, file_id)  # type:List[ReSentBaseData]
    return article_re_task_res


if __name__ == '__main__':
    input_dir_name = "./huawei_data_process/huawei_json"
    file_list = get_file_full_path_list_in_directory(input_dir_name)
    total_re_task_data_list = []  # type:List[ReSentBaseData]
    for file_id, file_name in enumerate(file_list):
        article_re_task_res = get_article_re_task_data(file_name, file_id)  # type:List[ReSentBaseData]
        if len(article_re_task_res) == 0: continue
        total_re_task_data_list.extend(article_re_task_res)
    knn_task_obj = KnnRetrievalTemplate(total_re_task_data_list)  # 这里存放的是所有数据集中需要进行关系抽取的句子
