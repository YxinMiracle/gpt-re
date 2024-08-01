import argparse

# 关系抽取任务的所有关系枚举
RELATION_NAME_LIST = ['resolves-to', 'beacons-to', 'exploits', 'targets', 'uses', 'delivers', 'originates-from',
                      'consists-of', 'hashes-to', 'drops', 'alias-of', 'communicates-with', 'controls', 'has',
                      'downloads', 'hosts', 'authored-by', 'compromises', 'variant-of', 'attributed-to', 'located-at',
                      'owns']

RECONSTRUCTED_BASE_SENT = "The relation between \"{head_entity}\" and \"{tail_entity}\" in the context: \"{input_sent}\""


def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="YxinMiracle RE")
    parser.add_argument("--bert_model_name", type=str, default="bert-base-cased",
                        help="model name (e.g., bert-base-cased, roberta-base)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epoch", type=int, default=60, help="Number of epoch")
    parser.add_argument("--shuffle", type=bool, default=True)
    params = parser.parse_args()
    return params


if __name__ == '__main__':
    params = get_params()
    print(params.bert_model_name)
