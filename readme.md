<p align="center">
    <img src="https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/logo.png" alt="Logo of Contamination Detector" width="auto" height="160" />
</p>

# Contamination Detector for LLMs Evaluation

Data Contamination is a real crisis for Large Language Models (LLMs) evaluation. **Contamination Detector** aids in identifying and analyzing such potential contamination without requiring access to the LLMs' training data, enabling even small teams and individuals to conduct robust evaluation.

**News!!**

- Our paper: [open sourced data contamination report for llama series models](https://arxiv.org/abs/2310.17589) shows the overall steps to conduct data contamination analysis with this tool .
- Paper: [Quantifying Memorisation via Perplexity](https://arxiv.org/abs/2309.10677) showing the basic idea of perplexity verificaiton.

# Methods

- **Internet Presence Verification**:

Contamination Detector checks whether test examples are present on the internet via **Bing search** and **Common Crawl index**. If both question and answer appear online, there is a big chance that this test sample was included in Common Crawl, and thus was memorized by LLMs.

<figure>
  <img src="https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/case.png" width="700"/>
</figure>

*>> An example of contaminated sample from MMLU test set (in Llama training data).↑*

Check out the extensive analysis on Llama series models at [here](https://arxiv.org/abs/2310.17589).

<!-- | Dataset       | #Total | #Online | #CommonCrawl (All Dirty) | #Input-only Contamination | #Input-and-label Contamination |
|---------------|-------:|--------:|-------------------------:|--------------------------:|-------------------------------:|
| Hellaswag     |   9315 |     805 | 805 (8.6%)               | 30(0.3%)                  | 775 (8.3%)                    |
| ARC           |   1172 |     102 | 90 (7.7%)                | 28 (2.4%)                 | 62 (5.3%)                     |
| CommonsenseQA |   1221 |      19 | 16 (1.3%)                | 0                         | 16 (1.3%)                     |
| MMLU          |  11322 |    1307 | 1213 (8.7%)              | 355 (2.5%)                | 858 (6.1%)                    |
| Winogrande    |   1267 |      13 | 13 (1.0%)                | 0                         | 13 (1.0%)                     |
| C-Eval        |   1335 |     712 | 3 (0.2%)                 | 0                         | 3 (0.2%)                      |


*>> How many test samples are leaked to Llama's training data? Analysis for popular multi-choice QA benchmarks.↑* -->

- **Test Memorization via Perplexity**:

Contamination Detector also verifies whether the model exhibits **memorization** behaviors on test benchmarks. This could be used to check whether model are **cheating by optimizing on benchmarks**. Just simply comparing the perplexity on training and test split of benchmarks.

You can also use PPL to detect data contamination:

<figure>
<img src="https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/xsum.png" width = 700>
</figure>

*>> The results show LLaMA models have memorized XSum test set partially (ppl is lower than clean data, but higher than fully memorised baseline).↑*


## 0.To start

Clone the repository:

```
git clone https://github.com/liyucheng09/Contamination_Detector.git
```

Install the required packages:

```
pip install -r requirements.txt
```

## 1. Run Internet Presence Verification

To check whether test examples are accessible on the internet:

Simply run this to produce a report for a benchmark:

```
python search.py
```

---

To run this script, you need have the bing search results for the benchmarks you want to analyze.

**you can directly download my search results** [here](https://github.com/liyucheng09/Contamination_Detector/releases/tag/v0.1.0), so you don't have to pay for accessing Bing.

There are six popular multi-choice QA benchmarks you could directly download: `['winogrande', 'ceval', 'mmlu', 'hellaswag', 'ARC', 'commonsense_qa']`. After this, unzip them under `bing_search/`.

Otherwise, you will need a free access token for Bing search API. You could obtain one via [this](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api). A free access key allow 1000 calls monthly. Student will receive $100 funding if you're creating a new account.

Set the key via `export Bing_Key = [YOUR API KEY]` in terminal.

---
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

If you cannot accessing Huggingface Hub for the benchmark datasets, download them as json files [here](https://github.com/liyucheng09/Contamination_Detector/releases/tag/v0.1.0).

## 2. Compare Model Performance on Clean and Dirty test sets

This will generate the comparison of accuracy between the *clean and dirty benchmark subsets* for various models.

But first you need prepare:
1. the model predictions on these benchmarks.

Download predictions of all Llama series models on these benchmark at [here](https://github.com/liyucheng09/Contamination_Detector/releases/tag/v0.1.0). Unzip and put them under `model_predictions/`.

2. get the benchmark reports (generated in step 1) ready.

Then run:
```
python clean_dirty_comparison.py
```

See how contamination affect the evaluation of Llama-2 70B, more results in the report.

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


## 3. Run Perplexity Verification

Read this paper first to get the basic idea of perplexity test: [Quantifying Memorisation via Perplexity](https://arxiv.org/abs/2309.10677).

Run this to conduct the perplexity comparison:

```
python perplexity.py
```

Customize `perplexity.py` to specify models and benchmarks to test. It will write the results under `reports/`.

Then you could generate the figure to visualize the results via:

```
python visualize/visualize_perplexity.py
```

**Check perplexity analysis of QA (BoolQ, SQuAD, QuAD) benchmarks [here](https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/qa.png)**

**Important:** you should choose apporiate data source for the memorised and clean baselinse if you are doing data contamination analysis. For example, qa benchmarks often use wikipedia passages, so you should use wikipedia as the base for the two baselines. 

Again, read [this paper](https://arxiv.org/abs/2309.10677) before you start.

## Issues

Open an issue or contact me via email if you encounter any problems in your use.