
# this highlights three major columns to check: input, label, and id
Column_to_check = {
    'winogrande': {'input': 'sentence', 'label': lambda x: x[f'option{x["answer"]}'], 'id': 'id'},
    'ceval': {'input': 'question', 'label': lambda x: x[x['answer']], 'id': 'id'},
    'mmlu': {'input': 'question', 'label': lambda x: x[x['answer']], 'id': 'id'},
    'hellaswag': {'input': 'ctx', 'label': lambda x: x['endings'][int(x['label'])], 'id': 'ind'},
    'ARC': {'input': 'question', 'label': lambda x: x['choices']['text'][x['choices']['label'].index(x['answerKey'])], 'id': 'id'},
    'commonsense_qa': {'input': 'question', 'label': lambda x: x['choices']['text'][x['choices']['label'].index(x['answerKey'])], 'id': 'id'}
}

Hf_Name_and_Split = {
    'winogrande': {'hf_name': 'liyucheng/winogrande_val', 'split': 'validation'},
    'ceval': {'hf_name': 'liyucheng/ceval_all', 'split': 'val'},
    'mmlu': {'hf_name': 'liyucheng/mmlu_test', 'split': 'train'},
    'hellaswag': {'hf_name': 'Rowan/hellaswag', 'split': 'validation'},
    'ARC': {'hf_name': 'liyucheng/arc_test', 'split': 'test'},
    'commonsense_qa': {'hf_name': 'commonsense_qa', 'split': 'validation'},
}

# This is used to choose the right Bing market
Dataset_lang = {
    'winogrande': 'en-US',
    'ceval': 'zh-CN',
    'mmlu': 'en-US',
    'hellaswag': 'en-US',
    'ARC': 'en-US',
    'commonsense_qa': 'en-US',
}

Recall_threshold_for_dataset = {
    'winogrande': 0.7,
    'ceval': 0.7,
    'mmlu': 0.7,
    'hellaswag': 0.7,
    'ARC': 0.7,
    'commonsense_qa': 0.7,
}

import numpy as np
from datasets import load_dataset
import re

np.random.seed(42)
from nltk.tokenize import word_tokenize

en_processor = lambda x: [token.lower() for token in word_tokenize(x) if token not in ['.', ',', '?', '!', ';', ':', '"', "'", '']]
zh_processor = lambda x: [token for token in ' '.join(x).split(' ') if token not in [ '，', '。', '？', '！', '；', '：', '“', '”', '‘', '’', '（', '）', '《', '》', '、',]]

def random_sample_ds(ds, n = 100):
    # ds is a dataset object
    # n is the number of samples to be randomly selected
    if len(ds) <= n:
        return ds
    return ds.select(np.random.choice(len(ds), n))

def prepare_query(dataset_name, row):
    """We verbalize the input and label to form a query
    For some benchmarks which have blanks in the input, we replace the blank with the label
    Otherwise, we append the label to the input
    """
    assert dataset_name in Column_to_check.keys(), \
        f'Column_to_check for {dataset_name} is not configed in utils.py'
    id_ = row[Column_to_check[dataset_name]['id']]
    input_ = row[Column_to_check[dataset_name]['input']]
    label = Column_to_check[dataset_name]['label'](row)

    def fill_blanks(question, answers, placeholder = '____'):
        global label

        if placeholder not in question:
            return f'{question} {answers}'
        
        if placeholder == '_':
            return re.sub(r'_', answers, question)
        
        num_blanks = re.findall(r'_{2,}', question)
        if len(num_blanks) > 1:
            if '，' in answers:
                answers = [ans.strip() for ans in answers.split('，')]
            elif ',' in answers:
                answers =[ans.strip() for ans in answers.split(',')]
            elif '；' in answers:
                answers = [ans.strip() for ans in answers.split('；')]
            else:
                return None
                raise ValueError(f'Cannot split {answers} into multiple answers. The question is {question}.')
            if not len(num_blanks) == len(answers):
                return None
                raise ValueError(f'Number of blanks in {question} does not match the number of answers {answers}.')
        else:
            answers = [answers]
        
        label = ' '.join(answers)
        return re.sub(r'_{2,}', lambda match: answers.pop(0), question)

    verbalize = {
        'winogrande': lambda input_, label: fill_blanks(input_, label, '_'), 
        'ceval': lambda input_, label: fill_blanks(input_, label, '____'),
        'mmlu': lambda input_, label: fill_blanks(input_, label),

        'hellaswag': lambda input_, label: f'{input_} {label}',
        'ARC': lambda input_, label: f'{input_} {label}',
        'commonsense_qa': lambda input_, label: f'{input_} {label}',
    }
    return {
        'id': id_,
        'input': input_,
        'label': label,
        'query': verbalize[dataset_name](input_, label),
    }

def prepare_dataset(dataset_names, n = 500):
    dses = {}
    for dataset_name in dataset_names:
        assert dataset_name in Hf_Name_and_Split.keys(), \
            f'Hf_Name_and_Split for {dataset_name} is not defined in utils.py'
        if Hf_Name_and_Split[dataset_name].get('config', None) is not None:
            ds = load_dataset(Hf_Name_and_Split[dataset_name]['hf_name'], Hf_Name_and_Split[dataset_name]['config'], split = Hf_Name_and_Split[dataset_name]['split'])
        else:
            ds = load_dataset(Hf_Name_and_Split[dataset_name]['hf_name'], split = Hf_Name_and_Split[dataset_name]['split'])
        print(dataset_name)
        print(ds)
        print('=====================')
        if n != 'all':
            ds = random_sample_ds(ds, n = n)
        dses[dataset_name] = ds
    return dses