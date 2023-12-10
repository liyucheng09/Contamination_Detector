import json
from sklearn.metrics import accuracy_score
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd

def load_model_predictions():
    """Load model predictions from JSON files."""
    models = glob('model_predictions/*.json')
    model_predictions = {}
    for model in models:
        with open(model, 'r') as f:
            data = json.load(f)
        model_name = model.split('/')[-1].split('.')[0].replace('_autogptq', '')
        model_predictions[model_name] = data
    return model_predictions

def calculate_accuracy(goldens, preds, types, contamination_type):
    """Calculate accuracy for a specific contamination type."""
    refs = [str(gold) for gold, case in zip(goldens, types) if case == contamination_type]
    preds_ = [str(pred) for pred, case in zip(preds, types) if case == contamination_type]
    return accuracy_score(refs, preds_)

def create_accuracy_table(model_predictions, benchmarks):
    """Create a human-friendly table of model accuracies."""
    results = []
    for benchmark in benchmarks:
        annotation_path = f'reports/{benchmark}_annotations.json'
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        for model_name, model_data in model_predictions.items():
            goldens, preds, types = [], [], []
            for id, (case, _) in annotation.items():
                try:
                    gold = model_data[benchmark][id]['gold']
                    pred = model_data[benchmark][id]['pred']
                except KeyError:
                    continue
                goldens.append(gold)
                preds.append(pred)
                types.append(case)

            # Add accuracies to results
            results.append({
                'Model': model_name,
                'Benchmark': benchmark,
                'Clean Accuracy': calculate_accuracy(goldens, preds, types, 'clean'),
                'Input-only Contaminated Accuracy': calculate_accuracy(goldens, preds, types, 'input contamination'),
                'Input-Label Contaminated Accuracy': calculate_accuracy(goldens, preds, types, 'input-and-label contamination')
            })
    return pd.DataFrame(results)

def save_results_to_json(df, filename):
    """Save the dataframe results to a JSON file."""
    df.to_json(filename, orient='records', lines=True)

def save_recall_accuracy_figure(model_predictions, benchmarks, filename):
    """Plot and save the recall vs accuracy figure."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)

    # Define a colorblind-friendly color palette
    colors = [
        "#0072B2",  # Blue
        "#D55E00",  # Orange
        "#009E73",  # Green
        "#CC79A7",  # Pink
        "#F0E442",  # Yellow
        "#56B4E9"   # Sky Blue
    ]
    markers = ['o', 's', '^', 'v', '+']
    line_styles = ['-']

    for index, benchmark in enumerate(benchmarks):
        accs = []
        annotation_path = f'reports/{benchmark}_annotations.json'
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        for i in range(1, 6):
            recall_level = 0.2 * i
            recall_gold = []
            recall_pred = []
            for id, (case, recall) in annotation.items():
                if recall <= recall_level:
                    for model_name in model_predictions:
                        try:
                            gold = model_predictions[model_name][benchmark][id]['gold']
                            pred = model_predictions[model_name][benchmark][id]['pred']
                            recall_gold.append(str(gold))
                            recall_pred.append(str(pred))
                        except KeyError:
                            continue

            if len(recall_gold) > 50:
                acc = accuracy_score(recall_gold, recall_pred)
                accs.append((recall_level, acc))

        if accs:
            ax.plot([acc[0] for acc in accs], [acc[1] for acc in accs], label=benchmark,
                    color=colors[index % len(colors)], marker=markers[index % len(markers)],
                    linestyle=line_styles[index % len(line_styles)], markersize=5)

    ax.set_xticks([0.2 * i for i in range(1, 6)])
    ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=4, shadow=False)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)

if __name__ == '__main__':
    benchmarks = ['mmlu', 'hellaswag', 'ARC', 'commonsense_qa']
    model_predictions = load_model_predictions()

    # Create and save accuracy table
    accuracy_table = create_accuracy_table(model_predictions, benchmarks)
    print(accuracy_table)
    save_results_to_json(accuracy_table, 'model_accuracy_results.json')

    # Save the recall vs accuracy figure
    save_recall_accuracy_figure(model_predictions, benchmarks, 'recall_accuracy_plot.png')
