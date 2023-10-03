import os
import json
import requests
import datasets
from utils import Column_to_check, Dataset_lang, Recall_threshold_for_dataset

from nltk.tokenize import word_tokenize
from nltk.translate import meteor_score

import re
import numpy as np

import time
from tqdm import tqdm

np.random.seed(42)

def bing_search(query, id_ = None, cache_path = None, mkt = 'en-US'):
    if cache_path is not None:
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        if os.path.exists(f'{cache_path}/{id_}.json'):
            with open(f'{cache_path}/{id_}.json', 'r') as f:
                return json.load(f)
    Bing_API_Key = os.environ.get('Bing_Key')

    search_url = "https://api.bing.microsoft.com/v7.0/search"

    headers = {"Ocp-Apim-Subscription-Key": Bing_API_Key}
    params = {"q": query, "textDecorations": True, "textFormat": "HTML", 'responseFilter': 'Webpages', 'mkt': mkt}

    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()

    if cache_path is not None:
        with open(f'{cache_path}/{id_}.json', 'w') as f:
            json.dump(search_results, f, ensure_ascii=False, indent=2)
    
    return search_results

def process_search_results(search_results, threshold = 0.7, lang = 'en-US'):
    # search_results is a dict
    # threshold is the threshold for meteor score (we focus on recall only here)

    def process_snippet(snippet, query):
        # consider only snippets

        fragments = snippet.split('...')
        fragments = [fragment.strip() for fragment in fragments]

        match_string = []        
        meteors = []
        for fragment in fragments:
            matchs = list(re.finditer(r'<b>(.*?)</b>', fragment))
            match_s = ' '.join([match.group(1) for match in matchs])
            match_string.append(match_s)

            processor = str.lower if lang.startswith('en') else lambda x: ' '.join(x)
            meteor = meteor_score.single_meteor_score(query, match_s, alpha=1, preprocess=processor)
            meteors.append(meteor)
        
        return max(meteors), match_string[np.argmax(meteors)]

    all_results = []    
    query = search_results['queryContext']['originalQuery']
    if 'webPages' not in search_results:
        return [], all_results
    for page in search_results['webPages']['value']:
        name = page['name']
        url = page['url']
        snippet = page['snippet']
        score, match_string = process_snippet(snippet, query)
        all_results.append({
            'query': query,
            'match_string': match_string,
            'score': score,
            'name': name,
            'url': url,
            'snippet': snippet,
        })
    
    matches = [result for result in all_results if result['score'] >= threshold]
    
    return matches, all_results

def random_sample_ds(ds, n = 100):
    # ds is a dataset object
    # n is the number of samples to be randomly selected
    if len(ds) <= n:
        return ds
    return ds.select(np.random.choice(len(ds), n))

if __name__ == '__main__':

    report_path = 'reports/'
    if not os.path.exists(report_path):
        os.makedirs(report_path)

    datasets_to_check = {
        'winogrande': random_sample_ds(datasets.load_dataset('winogrande', 'winogrande_xs', split = 'test')),
        'ceval': random_sample_ds(datasets.load_dataset('liyucheng/ceval_all', split = 'test')),
        'mmlu': random_sample_ds(datasets.load_dataset('liyucheng/mmlu_mini', split = 'test')),
    }

    for dataset_name, ds in datasets_to_check.items():
        assert dataset_name in Column_to_check.keys() and Column_to_check[dataset_name] in ds.column_names, \
            f'Column_to_check for {dataset_name} is not defined in utils.py or the column does not exist in the dataset'

        all_matches = []
        for i, row in tqdm(enumerate(ds), desc=f'Processing {dataset_name}'):
            # limit per second for free plan is 3/s
            time.sleep(0.34)

            query = row[Column_to_check[dataset_name]]
            # skip too long queries
            if len(query) > 1000: continue

            search_result = bing_search(query, id_ = i, cache_path = f'bing_search/{dataset_name}', mkt = Dataset_lang[dataset_name])
            matches, _ = process_search_results(search_result, threshold=Recall_threshold_for_dataset[dataset_name],lang = Dataset_lang[dataset_name])
            if matches: all_matches.append(matches)
        
        # output the contamination report based on all_matches
        print(f'Total number of samples in {dataset_name}: {len(ds)}')
        print(f'Total number of samples with matches: {len(all_matches)}')
        with open(f'{report_path}/{dataset_name}.json', 'w') as f:
            json.dump({
                'dataset_name': dataset_name,
                'matches': all_matches,
            }, f, ensure_ascii=False, indent=2)
        
