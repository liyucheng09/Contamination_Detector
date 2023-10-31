import json
import os
from nltk.tokenize import word_tokenize
from nltk.translate import meteor_score

import re
import numpy as np

from glob import glob

en_processor = lambda x: [token.lower() for token in word_tokenize(x) if token not in ['.', ',', '?', '!', ';', ':', '"', "'", '']]
zh_processor = lambda x: [token for token in ' '.join(x).split(' ') if token not in [ '，', '。', '？', '！', '；', '：', '“', '”', '‘', '’', '（', '）', '《', '》', '、',]]

def process_search_results(search_results, query, threshold = 0.7, lang = 'en-US'):
    # search_results is a dict
    # threshold is the threshold for meteor score (we focus on recall only here)

    def process_snippet(snippet, query, is_label = False):
        # consider only snippets
        
        fragments = snippet.split('...')
        fragments = [fragment.strip() for fragment in fragments if fragment.strip()]

        match_string = []        
        meteors = []
        for fragment in fragments:
            matches = list(re.finditer(r'<b>(.*?)</b>', fragment))
            if not matches: continue
            match_s = ' '.join([match.group(1) for match in matches])
            match_string.append(match_s)

            processor = en_processor if lang.startswith('en') else zh_processor

            tokenized_query = processor(query)
            tokenized_match = processor(match_s)

            meteor = meteor_score.single_meteor_score(tokenized_query, tokenized_match, alpha=1, gamma=0 if is_label else 0.5)
            meteors.append(meteor)
        
        if not meteors: return 0, ''
        return max(meteors), match_string[np.argmax(meteors)]
    
    def case_type(all_results):
        if all([result['score'] < threshold for result in all_results]):
            return 'clean'
        else:
            matches = [result for result in all_results if result['score'] >= threshold]
            if all([result['score_label'] < threshold for result in matches]):
                return 'input contamination'
            else:
                return 'input-and-label contamination'

    all_results = []
    label = query['label']
    query = search_results['queryContext']['originalQuery']
    if 'webPages' not in search_results:
        return [], all_results, 'clean', 0
    for page in search_results['webPages']['value']:
        name = page['name']
        url = page['url']
        snippet = page['snippet']
        score, match_string = process_snippet(snippet, query)
        label_score, _ = process_snippet(snippet, label, is_label=True)
        all_results.append({
            'query': query,
            'match_string': match_string,
            'score': score,
            'score_label': label_score,
            'name': name,
            'url': url,
            'snippet': snippet,
        })
    
    max_score = max([result['score'] for result in all_results])
    matches = [result for result in all_results if result['score'] >= threshold]
    return matches, all_results, case_type(all_results), max_score

if __name__ == '__main__':

    benchmarks = ['winogrande', 'ceval', 'mmlu', 'hellaswag', 'ARC', 'commonsense_qa']
    

    for benchmark in benchmarks:
        search_results = glob(f'bing_search/{benchmark}/*.json')
        for search_result in search_results:
            with open(search_result, 'r') as f:
                search_result = json.load(f)
            query = search_result['queryContext']['originalQuery']
            matches, all_results, case_type, max_score = process_search_results(search_result, query)
            print(f'{search_result["queryContext"]["originalQuery"]}: {case_type}')
            print(f'max score: {max_score}')
            print('=====================')