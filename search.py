import os
import json
import requests
import datasets
from utils import (Column_to_check, 
                   Dataset_lang, 
                   Recall_threshold_for_dataset, 
                   prepare_query,
                   prepare_dataset,
                   en_processor,
                   zh_processor,)

from nltk.tokenize import word_tokenize
from nltk.translate import meteor_score

import re
import numpy as np

import time
from tqdm import tqdm

from collections import Counter

np.random.seed(42)

def bing_search(query, cache_path = None, mkt = 'en-US'):
    id_ = query['id']
    if cache_path is not None:
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        if os.path.exists(f'{cache_path}/{id_}.json'):
            with open(f'{cache_path}/{id_}.json', 'r') as f:
                return json.load(f)
    Bing_API_Key = os.environ.get('Bing_Key')
    search_url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": Bing_API_Key}

    # adjust the freshness to suit your target models
    params = {"q": query['query'], "textDecorations": True, "textFormat": "HTML", 'responseFilter': 'Webpages', 'mkt': mkt,} # set 'freshness': '2017-01-01..2020-12-30' if you need a specific time range
    
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()

    if cache_path is not None:
        with open(f'{cache_path}/{id_}.json', 'w') as f:
            json.dump(search_results, f, ensure_ascii=False, indent=2)
    
    return search_results

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

    report_path = 'reports/'
    if not os.path.exists(report_path):
        os.makedirs(report_path)

    datasets_to_check = prepare_dataset(['winogrande', 'ceval', 'mmlu', 'hellaswag', 'ARC', 'commonsense_qa'], n = 'all')

    for dataset_name, ds in datasets_to_check.items():

        all_matches = []
        all_results = {}
        count = 0
        for i, row in tqdm(enumerate(ds), desc=f'Processing {dataset_name}'):
            # 3/s if you're using free plan
            # time.sleep(0.3)
            query = prepare_query(dataset_name, row)

            # skip too long queries
            if query['query'] is None or len(query['query']) > 1000: continue

            search_result = bing_search(query, cache_path = f'bing_search/{dataset_name}', mkt = Dataset_lang[dataset_name])
            if search_result is None: continue
            count += 1
            matches, _, case_type, recall_score = process_search_results(search_result, query, threshold=Recall_threshold_for_dataset[dataset_name], lang = Dataset_lang[dataset_name])
            all_results[query['id']] = (case_type, recall_score)
            if matches: all_matches.append(matches)
        
        # output the contamination report
        print(f'Dataset: {dataset_name} - Num: {count}')
        print(dict(Counter([case[0] for case in all_results.values()])))

        # save the report
        with open(f'{report_path}/{dataset_name}_report.json', 'w') as f:
            json.dump({
                'dataset_name': dataset_name,
                'matches': all_matches,
            }, f, ensure_ascii=False, indent=2)
        with open(f'{report_path}/{dataset_name}_annotations.json', 'w') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
