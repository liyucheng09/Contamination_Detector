import json
from sklearn.metrics import accuracy_score
from glob import glob
import matplotlib.pyplot as plt

def contaminated_examples():
    pass

def load_model_predictions():
    models = glob('model_predictions/*.json')

    model_predictions = {}
    for model in models:
        with open(model, 'r') as f:
            data = json.load(f)
        model_name = model.split('/')[-1].split('.')[0][:-len('_autogptq')]
        model_predictions[model_name] = data
    return model_predictions

if __name__ == '__main__':

    # benchmarks = ['winogrande', 'ceval', 'mmlu', 'hellaswag', 'ARC', 'commonsense_qa']
    benchmarks = ['mmlu', 'hellaswag', 'ARC', 'commonsense_qa']
    model_predictions = load_model_predictions()
    recall_gold_by_benchmark = {(0.2*i, benchmark):[] for i in range(1, 6) for benchmark in benchmarks}
    recall_pred_by_benchmark = {(0.2*i, benchmark):[] for i in range(1, 6) for benchmark in benchmarks}
    recall_gold_by_model = {(0.2*i, model): [] for i in range(1, 6) for model in model_predictions}
    recall_pred_by_model = {(0.2*i, model): [] for i in range(1, 6) for model in model_predictions}
    for benchmark in benchmarks:
        print(benchmark)
        annotation = f'reports/{benchmark}_annotations.json'
        with open(annotation, 'r') as f:
           annotation = json.load(f)
        goldens = {}
        preds = {}
        types = {}
            
        for id, case in annotation.items():
            case, recall = case
            for model in model_predictions:
                if model not in goldens:
                    goldens[model] = []
                    preds[model] = []
                    types[model] = []
                try:
                    gold = model_predictions[model][benchmark][id]['gold']
                except:
                    continue
                pred = model_predictions[model][benchmark][id]['pred']
                goldens[model].append(gold)
                preds[model].append(pred)
                types[model].append(case)
                # gold and pred to different recall buckets
                if model != 'llama_2_70b':
                    continue   
                for i in range(1, 6):
                    if recall <= 0.2*i:
                        recall_gold_by_benchmark[(0.2*i, benchmark)].append(str(gold))
                        recall_pred_by_benchmark[(0.2*i, benchmark)].append(str(pred))
                        recall_gold_by_model[(0.2*i, model)].append(str(gold))
                        recall_pred_by_model[(0.2*i, model)].append(str(pred))
                        break
        print('=====================')
        for model_name in goldens.keys():
            # clean results
            print(f'acc for {model_name} on clean set of {benchmark}:')
            refs = [str(gold) for gold, case in zip(goldens[model_name], types[model_name]) if case == 'clean']
            preds_ = [str(pred) for pred, case in zip(preds[model_name], types[model_name]) if case == 'clean']
            print(accuracy_score(refs, preds_))

            # contaminated results
            print(f'acc for {model_name} on contaminated set of {benchmark}:')
            refs = [str(gold) for gold, case in zip(goldens[model_name], types[model_name]) if case != 'clean']
            preds_ = [str(pred) for pred, case in zip(preds[model_name], types[model_name]) if case != 'clean']
            print(accuracy_score(refs, preds_))

            # input-and-label contaminated results
            print(f'acc for {model_name} on input-and-label contaminated set of {benchmark}:')
            refs = [str(gold) for gold, case in zip(goldens[model_name], types[model_name]) if case == 'input-and-label contamination']
            preds_ = [str(pred) for pred, case in zip(preds[model_name], types[model_name]) if case == 'input-and-label contamination']
            print(accuracy_score(refs, preds_))
    
    name_map = {
        'ceval': 'C-Eval',
        'mmlu': 'MMLU',
        'hellaswag': 'HellaSwag',
        'commonsense_qa': 'CommonsenseQA',
        'ARC': 'ARC',
    }
    # recall
    # for i in range(1, 6):
    #     print(f'recall for {i*0.2}:')
    #     print('=====================')
    #     for benchmark in benchmarks:
    #         print(f'{benchmark}: {len(recall_gold_by_benchmark[(0.2*i, benchmark)])}')
    #         acc = accuracy_score(recall_gold_by_benchmark[(0.2*i, benchmark)], recall_pred_by_benchmark[(0.2*i, benchmark)])
    #         print(acc)
    #     print('=====================')
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    import seaborn as sns
    colors = sns.color_palette("colorblind", len(benchmarks))
    markers = ['o', 's', '^', 'v', '+',]
    line_styles = ['-',]
    # line_styles = ['-', '--', '-.', ':']
    for index, benchmark in enumerate(benchmarks):
        print(f'{benchmark}:')
        print('=====================')
        accs = []
        for i in range(1, 6):
            num_cases = len(recall_gold_by_benchmark[(0.2*i, benchmark)])
            print(f'{i*0.2}: ---- {num_cases}')
            acc = accuracy_score(recall_gold_by_benchmark[(0.2*i, benchmark)], recall_pred_by_benchmark[(0.2*i, benchmark)])
            print('Acc: ', acc)
            if num_cases > 50:
                accs.append((i*0.2, acc))
        ax.plot([acc[0] for acc in accs], [acc[1] for acc in accs], label=f'{name_map[benchmark]}', color=colors[index % len(colors)], marker=markers[index % len(markers)], linestyle=line_styles[index % len(line_styles)], markersize=5)

    ax.set_xticks([0.2*i for i in range(1, 6)], fontsize=10)
    ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=4, shadow=False)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    # ax.axvspan(0.4, 0.6, color='gray', alpha=0.2)  
    # plt.grid(True, linestyle='--', axis='y', which='major')
    plt.tight_layout()
    plt.show()
        # for model in model_predictions:
        #     print(f'{model}:')
        #     print(accuracy_score(recall_gold_by_model[(0.2*i, model)], recall_pred_by_model[(0.2*i, model)]))