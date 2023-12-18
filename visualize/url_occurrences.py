import matplotlib.pyplot as plt
from urllib.parse import urlparse
from collections import defaultdict
import json

def extract_domain(link):
    """
    Extracts the main domain from a given URL
    """
    parsed_uri = urlparse(link)
    domain = '{uri.netloc}'.format(uri=parsed_uri)
    # if "www." prefix exists, remove it to get the main domain
    return domain.replace('www.', '')

def plot_domains(links, benchmark):
    """
    Plots the main domains from the list of links
    """
    # Extract and count domains
    domain_counts = defaultdict(int)
    for link in links:
        domain = extract_domain(link)
        # if domain == 'huggingface.co': continue
        domain_counts[domain] += 1

    # Sort domains by their frequencies
    sorted_domains = sorted(domain_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    sorted_domains = list(reversed(sorted_domains))

    domains = [item[0] for item in sorted_domains]
    counts = [item[1] for item in sorted_domains]

    # Plot
    plt.figure(figsize=(4, 6), dpi=150)
    plt.barh(domains, counts, color='skyblue')
    plt.xlabel('Count')
    # plt.ylabel('Domain')
    plt.title('Comtamination Occurrences')
    plt.tight_layout()
    plt.savefig(f'{benchmark}_urls.png')

if __name__ == '__main__':
    benchmarks = [ 'hellaswag']
    # benchmarks = ['winogrande', 'ceval', 'mmlu', 'hellaswag', 'ARC', 'commonsense_qa']

    for benchmark in benchmarks:
        report = f'reports/{benchmark}_report.json'
        with open(report, 'r') as f:
            data = json.load(f)
        matches = data['matches']
        links = []
        for match in matches:
            for link in match:
                links.append(link['url'])
        plot_domains(links, benchmark)
