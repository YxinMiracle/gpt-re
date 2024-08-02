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


class Dataset(data.Dataset):
    def __init__(self, inputs, flags, labels):
        self.X = inputs
        self.flags = flags
        self.y = labels

    def __getitem__(self, index):
        return self.X[index], self.flags[index], self.y[index]

    def __len__(self):
        return len(self.X)


def collate_fn(data):
    # 控制bs中每一个句子的长度都是这个bs中最长句子的那个长度
    X, y = zip(*data)
    # X 表示每一个句子
    # y 表示这个句子中对应的label
    lengths = [len(bs_x) for bs_x in X]
    # lengths 表示的是每一个bs中的句子的长度
    max_lengths = max(lengths)
    # 找到这个bs中句子长度的最大值
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(auto_tokenizer.pad_token_id)  # [bs* max_lengths]
    # auto_tokenizer.pad_token_id=0
    padded_y = torch.LongTensor(len(X), max_lengths).fill_(pad_token_label_id)  # [bs* ,max_lengths] 个 pad_token_label_id=-100

    for i, (seq, y_) in enumerate(zip(X, y)):
        length = lengths[i]  # 获取第一个句子
        padded_seqs[i, :length] = torch.LongTensor(seq)
        padded_y[i, :length] = torch.LongTensor(y_)

    return padded_seqs, padded_y


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

    def do_pre_data_for_re_model(self):
        result_word_id_list, result_entity_flag_list, result_label_id_list = build_fine_tuned_re_model_input_data(
            self)  # 这里将这些句子去做关系抽取任务之前的数据处理
        train_dataset = Dataset(result_word_id_list, result_entity_flag_list, result_label_id_list)
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=params.batch_size,
                                      shuffle=params.shuffle,
                                      collate_fn=collate_fn)
        print(train_dataset[0])
