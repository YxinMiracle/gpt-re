import logging
import random
from argparse import ArgumentParser
from datetime import datetime
from typing import List

import numpy as np
import torch.optim as optim
from tqdm import tqdm

from huawei_data_process.process_json_data import process_file
from model.dataloader.dataloader import dataloader
from model.pfn import PFN
from model_cofig.config import get_params
from process_model.PreDataProcess import ProcessTrainDataTemplate
from process_model.ReBaseData import ReSentBaseData
from utils.helper import *
from utils.metrics import micro, macro, f1, LossClass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    设置随机数种子
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    with open(file_path, "r", encoding="UTF-8") as fp:
        file_data = json.load(fp)
    article_re_task_res = process_file(file_data, file_id)  # type:List[ReSentBaseData]
    return article_re_task_res


def evaluate(test_batch, rel2idx, ner2idx, args, test_or_dev, model, BCEloss):
    steps, test_loss = 0, 0
    total_triple_num = [0, 0, 0]
    total_entity_num = [0, 0, 0]
    if args.eval_metric == "macro":
        total_triple_num *= len(rel2idx)
        total_entity_num *= len(ner2idx)

    if args.eval_metric == "micro":
        metric = micro(rel2idx, ner2idx)
    else:
        metric = macro(rel2idx, ner2idx)

    with torch.no_grad():
        for data in test_batch:
            steps += 1
            text = data[0]
            ner_label = data[1].to(device)
            re_label = data[2].to(device)
            mask = data[-1].to(device)

            ner_pred, re_pred = model(text, mask)
            loss = BCEloss(ner_pred, ner_label, re_pred, re_label)
            test_loss += loss

            entity_num = metric.count_ner_num(ner_pred, ner_label)
            triple_num = metric.count_num(ner_pred, ner_label, re_pred, re_label)

            for i in range(len(entity_num)):
                total_entity_num[i] += entity_num[i]
            for i in range(len(triple_num)):
                total_triple_num[i] += triple_num[i]

        triple_result = f1(total_triple_num)
        entity_result = f1(total_entity_num)

        logger.info("------ {} Results ------".format(test_or_dev))
        logger.info("loss : {:.4f}".format(test_loss / steps))
        logger.info(
            "entity: p={:.4f}, r={:.4f}, f={:.4f}".format(entity_result["p"], entity_result["r"], entity_result["f"]))
        logger.info(
            "triple: p={:.4f}, r={:.4f}, f={:.4f}".format(triple_result["p"], triple_result["r"], triple_result["f"]))

    return triple_result, entity_result, test_loss / steps


def train_re_model(params: ArgumentParser, project_root_path: str):
    """
    改函数用来训练关系抽取模型
    :param params: 参数
    :param project_root_path:
    :return:
    """
    output_dir = project_root_path + os.path.sep + params.save_directory_name  # type: str
    if not os.path.exists(output_dir):
        """ 如果路径不存在那么就创建这个文件夹 """
        os.makedirs(output_dir)
    current_datetime = datetime.now()
    output_path_name = current_datetime.strftime('%Y-%m-%d-%H-%M-%S')  # type: str
    logger.addHandler(logging.FileHandler(output_dir + os.path.sep + output_path_name + ".log", 'w'))
    logger.info(params)

    saved_file = save_results(output_dir + os.path.sep + output_path_name + ".txt",
                              header="# epoch \t train_loss \t  dev_loss \t test_loss \t dev_ner \t dev_rel \t test_ner \t test_rel")
    model_file = output_path_name + ".pt"  # type: str

    with open(params.data_directory_name + os.path.sep + params.ner2idx_file_name, "r") as f:
        ner2idx = json.load(f)  # type: dict
    with open(params.data_directory_name + os.path.sep + params.re2idx_file_name, "r") as f:
        rel2idx = json.load(f)  # type: dict

    train_batch, test_batch, dev_batch = dataloader(project_root_path, params, ner2idx, rel2idx)

    if params.do_train:
        logger.info("------Training------")
        input_size = 768  # type: int
        model = PFN(params, input_size, ner2idx, rel2idx)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        if params.eval_metric == "micro":
            metric = micro(rel2idx, ner2idx)
        else:
            metric = macro(rel2idx, ner2idx)

        BCEloss = LossClass()
        best_result = 0
        triple_best = None
        entity_best = None

        for epoch in range(params.epoch):
            steps, train_loss = 0, 0

            model.train()
            for data in tqdm(train_batch):

                steps += 1
                optimizer.zero_grad()

                text = data[0]
                ner_label = data[1].to(device)
                re_label = data[2].to(device)
                mask = data[-1].to(device)

                ner_pred, re_pred = model(text, mask)
                loss = BCEloss(ner_pred, ner_label, re_pred, re_label)

                loss.backward()

                train_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params.clip)
                optimizer.step()

                if steps % params.steps == 0:
                    logger.info("Epoch: {}, step: {} / {}, loss = {:.4f}".format
                                (epoch, steps, len(train_batch), train_loss / steps))

            logger.info("------ Training Set Results ------")
            logger.info("loss : {:.4f}".format(train_loss / steps))

            if params.do_eval:
                model.eval()
                logger.info("------ Testing ------")
                dev_triple, dev_entity, dev_loss = evaluate(dev_batch, rel2idx, ner2idx, params, "dev", model, BCEloss)
                test_triple, test_entity, test_loss = evaluate(test_batch, rel2idx, ner2idx, params, "test", model,
                                                               BCEloss)
                average_f1 = dev_triple["f"] + dev_entity["f"]

                if epoch == 0 or average_f1 > best_result:
                    best_result = average_f1
                    triple_best = test_triple
                    entity_best = test_entity
                    torch.save(model.state_dict(), output_dir + "/" + model_file)
                    logger.info("Best result on dev saved!!!")

                saved_file.save("{} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}".format(epoch,
                                                                                                                  train_loss / steps,
                                                                                                                  dev_loss,
                                                                                                                  test_loss,
                                                                                                                  dev_entity[
                                                                                                                      "f"],
                                                                                                                  dev_triple[
                                                                                                                      "f"],
                                                                                                                  test_entity[
                                                                                                                      "f"],
                                                                                                                  test_triple[
                                                                                                                      "f"]))

        saved_file.save(
            "best test result ner-p: {:.4f} \t ner-r: {:.4f} \t ner-f: {:.4f} \t re-p: {:.4f} \t re-r: {:.4f} \t re-f: {:.4f} ".format(
                entity_best["p"],
                entity_best["r"], entity_best["f"], triple_best["p"], triple_best["r"], triple_best["f"]))


if __name__ == '__main__':
    input_dir_name = "./huawei_data_process/huawei_json"
    file_list = get_file_full_path_list_in_directory(input_dir_name)
    total_re_task_data_list = []  # type:List[ReSentBaseData]
    for file_id, file_name in enumerate(file_list):
        article_re_task_res = get_article_re_task_data(file_name, file_id)  # type:List[ReSentBaseData]
        if len(article_re_task_res) == 0: continue
        total_re_task_data_list.extend(article_re_task_res)
    # ==== 这里已经获得到了所有的句子以及对应的句子里面存在的实体和关系  ======
    params = get_params()
    set_seed(params.seed)
    """ 处理成模型所需形式 """
    p_obj = ProcessTrainDataTemplate(total_re_task_data_list, params)
    p_obj.do_process()
    """ 训练模型 """
    train_re_model(params, p_obj.root_dir)
