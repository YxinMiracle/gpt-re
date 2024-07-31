import argparse

# 关系抽取任务的所有关系枚举
RELATION_NAME_LIST = ['resolves-to', 'beacons-to', 'exploits', 'targets', 'uses', 'delivers', 'originates-from',
                      'consists-of', 'hashes-to', 'drops', 'alias-of', 'communicates-with', 'controls', 'has',
                      'downloads', 'hosts', 'authored-by', 'compromises', 'variant-of', 'attributed-to', 'located-at',
                      'owns']


def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="YxinMiracle RE")
    parser.add_argument("--bert_model_name", type=str, default="bert-base-cased",
                        help="model name (e.g., bert-base-cased, roberta-base)")
    params = parser.parse_args()
    return params


if __name__ == '__main__':
    params = get_params()
    print(params.bert_model_name)
