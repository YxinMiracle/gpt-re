import argparse

RECONSTRUCTED_BASE_SENT = "The relation between \"{head_entity}\" and \"{tail_entity}\" in the context: \"{input_sent}\""


def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="YxinMiracle RE")
    parser.add_argument("--bert_model_name", type=str, default="bert-base-cased",
                        help="model name (e.g., bert-base-cased, roberta-base)")
    parser.add_argument("--data_directory_name", type=str, default="data", help="data directory name")
    parser.add_argument("--ner2idx_file_name", type=str, default="ner2idx.json", help="ner2idx file name")
    parser.add_argument("--re2idx_file_name", type=str, default="rel2idx.json", help="re2idx file name")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epoch", type=int, default=60, help="Number of epoch")
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--seed", default=7777, type=int)
    params = parser.parse_args()
    return params


if __name__ == '__main__':
    params = get_params()
    print(params.bert_model_name)
