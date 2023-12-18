<p align="center">
    <img src="https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/logo.png" alt="Logo of Contamination Detector" width="auto" height="160" />
</p>

# Contamination Detector for LLMs Evaluation

Data Contamination is a pervasive and critical issue in the evaluation of Large Language Models (LLMs). Our **Contamination Detector** is designed to identify and analyze potential contamination issues without needing access to the LLMs' training data, enabling the community to audit LLMs evaluation results and conduct robust evaluation.

**News!!**

- Our new preprint: [An open source data contamination report for large language models](https://arxiv.org/abs/2310.17589)!

# Our Methods: check potential contamination via search engine

Contamination Detector checks whether test examples appear on the internet via **Bing search** and **Common Crawl index**. We categorize test samples into three subsets:
1. **Clean** set: the question and reference answer do not appear online.
2. **Input-only contaminated** set: the question appears online, but not its answer.
3. **Input-and-label contaminated** set: both question and answer appear online.


If either the "question" or "answer" of a test example is found online, this sample may have been included in the LLM's training data. As a result, LLMs might gain an **unfair advantage by 'remembering' these samples**, rather than genuinely **understanding or solving them**.

We now support the following popular LLMs benchmarks:

- MMLU
- CEval
- Winogrande
- ARC
- Hellaswag
- CommonsenseQA

# Get start: Test LLMs' degree of contamination

1. Clone the repository and install the required packages:

```
git clone https://github.com/liyucheng09/Contamination_Detector.git
cd Contamination_Detector/
pip install -r requirements.txt
```

2. We need model predictions to further analyze their data contamination issue. We have prepared model predictions for the following LLMs:

- LLaMA 7,13,30,65B
- Llama-2 7,13,70B
- Qwen-7b
- Baichuan2-7B
- Mistral-7B
- Mistral Instruct 7B
- Yi 6B

That you can download directly without going through the inference:

```
wget https://github.com/liyucheng09/Contamination_Detector/releases/download/v0.1.1rc2/model_predictions.zip
unzip model_predictions.zip
```

If you hope to conduct the analysis on your own prediction data, format your model prediction as following and put under `model_predictions/`:

```
{
  "mmlu": {
    "business_ethics 0": {
      "gold": "C",
      "pred": "A"
    },
    "business_ethics 1": {
      "gold": "B",
      "pred": "A"
    },
    "business_ethics 2": {
      "gold": "D",
      "pred": "A"
    },
    "business_ethics 3": {
      "gold": "D",
      "pred": "D"
    },
    "business_ethics 4": {
      "gold": "B",
      "pred": "B"
    },
    .....
```

3. Generate contamination analysis table:

```
python clean_dirty_comparison.py
```

This will use the contamination annotation under `reports/` to generate models' performance on the clean, input-only contaminated, and input-and-label contaminated subsets.

See how the performance of Llama-2 70B differs on the three subsets.

| Dataset   | Condition          | Llama-2 70B |
|-----------|--------------------|--------------|
| MMLU      | Clean              | .6763       |
| MMLU      | All Dirty          | .6667 ↓      |
| MMLU      | Input-label Dirty  | .7093 ↑      |
| Hellaswag | Clean              | .7726       |
| Hellaswag | All Dirty          | .8348 ↑      |
| Hellaswag | Input-label Dirty  | .8455 ↑      |
| ARC       | Clean              | .4555       |
| ARC       | All Dirty          | .5632 ↑      |
| ARC       | Input-label Dirty  | .5667 ↑      |
| Average   | Clean              | .6348       |
| Average   | All Dirty          | .6882 ↑      |
| Average   | Input-label Dirty  | .7072 ↑      |

Other than this table, `clean_dirty_comparison.py` also produces a figure illustrating how the performance change with the recall score (the extent of contamination for a sample).


# Audit your own evaluation data

To check potential contamination in your benchmark, we have a script to identify potential contaminated test samples in your data:

Set up your benchmark in `utils.py`, this requires you to specify how to load your benchmark and verbalization methods, etc.

Then run the following to produce contamination reports for your benchmark:

```
python search.py
```

To run this script, you will need a free access token for Bing search API. You could obtain one via [this](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api). A free access key allow 1000 calls monthly. Student will receive $100 funding if you're creating a new account.

Set the key via `export Bing_Key = [YOUR API KEY]` in terminal.


`search.py` will generate a report under `reports/` such as `reports/mmlu_report.json` that highlight all matches online, for example:
```
[
  {
    "input": "The economy is in a deep recession. Given this economic situation which of the following statements about monetary policy is accurate?",
    "match_string": "The economy is in a deep recession. Given this economic situation, which of the following statements about monetary policy is accurate policy recession policy",
    "score": 0.900540825748582,
    "name": "<b>AP Macroeconomics Question 445: Answer and Explanation</b> - CrackAP.com",
    "contaminated_url": "https://www.crackap.com/ap/macroeconomics/question-445-answer-and-explanation.html",
  },
...
```

Reports for six popular multi-choice QA benchmarks are ready to access under `/reports`.

To visualize the results, please move to [visualize](https://github.com/liyucheng09/Contamination_Detector/tree/master/visualize).

**Check contamination examples: MMLU at [here](https://htmlpreview.github.io/?https://github.com/liyucheng09/Contamination_Detector/blob/master/reports/mmlu.html), and C-Eval at [here](https://htmlpreview.github.io/?https://github.com/liyucheng09/Contamination_Detector/blob/master/reports/ceval.html)**

If you cannot accessing Huggingface Hub for the benchmark datasets, download them as json files [here](https://github.com/liyucheng09/Contamination_Detector/releases/tag/v0.1.1).

## Citation:

Consider cite our project if you find it helpful:
```
@article{Li2023AnOS,
  title={An Open Source Data Contamination Report for Large Language Models},
  author={Yucheng Li},
  journal={ArXiv},
  year={2023},
  volume={abs/2310.17589},
}
```

## Issues

Open an issue or contact me via email if you encounter any problems in your use.
