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

*The extent of contamination of XSum test set.*

We found the perplexity on XSum is between the memorized and clean baseline, which indicate XSum is partially contaminated (memorized).


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

To run this script, you will need a free access token for Bing search API. You could obtain one via [this](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api). A free access key allow 1000 calls monthly. Student will receive $100 funding if you're creating a new account.

Set the key via `export Bing_Key = [YOUR API KEY]` in terminal.

Or, **you can directly download my search results** [here](https://github.com/liyucheng09/Contamination_Detector/releases/tag/v0.1.0), so you don't have to pay for accessing Bing.

It will generate a report under `reports/` that highlight all matches online, for example:
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

You could visualize contamination examples after generating the reports via:

```
python visualize/visualize_search.py
```

This will highlight the matched part of benchmark samples.

**`MMLU` - An example of contamination visualization**

**Question:** The economy is in a deep recession. Given this economic situation which of the following statements about monetary policy is accurate?

**Matches:**
| Page Name | Overlapping | Match Ratio | URL |
|-|-|-|-|
| AP Macroeconomics Question 445: Answer and Explanation - CrackAP.com | **The economy is in a deep recession. Given this economic situation**, **which of the following statements about monetary policy is accurate?** A. Expansionary policy would only worsen the recession. B. Expansionary policy greatly increases aggregate demand if investment is sensitive to changes in the interest rate. | 0.901 | [Link](https://www.crackap.com/ap/macroeconomics/question-445-answer-and-explanation.html) |  
| AP Macroeconomics Practice Test 21 - CrackAP.com | **Given this economic situation**, **which of the following statements about monetary policy is accurate?** A. Expansionary policy would only worsen the recession. B. Expansionary policy greatly increases aggregate demand if investment is sensitive to changes in the interest rate. | 0.615 | [Link](https://www.crackap.com/ap/macroeconomics/test41.html) |

It will hightlight the overlapping part of the benchmark and internet pages.

**Check more contamination examples: MMLU at [here](https://htmlpreview.github.io/?https://github.com/liyucheng09/Contamination_Detector/blob/master/reports/mmlu.html), and C-Eval at [here](https://htmlpreview.github.io/?https://github.com/liyucheng09/Contamination_Detector/blob/master/reports/ceval.html)**

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

**Important:** you should choose apporiate data source for the memorised and clean baselinse. For example, qa benchmarks often use wikipedia passages, so you should use wikipedia as the base for the two baselines. 

Again, read [this paper](https://arxiv.org/abs/2309.10677) before you start.

## Issues

Open an issue or contact me via email if you encounter any problems in your use.