<p align="center">
    <img src="https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/logo.png" alt="Logo of Contamination Detector" width="auto" height="160" />
</p>

# Contamination Detector for LLMs Evaluation

Data Contamination is a real crisis for Large Language Models (LLMs) evaluation. **Contamination Detector** aids in identifying and analyzing such potential contamination without requiring access to the LLMs' training data, enabling even small teams and individuals to conduct robust evaluation.

**Updates!**

- Check out our latest [open source data contamination report for llama series models](https://arxiv.org/abs/2310.17589)!

# Methods

- **Internet Presence Verification**:

Contamination Detector goes through the benchmark and verify whether test examples (both inputs and labels) are present on the internet via **Bing search** and **Common Crawl index**.

<figure>
  <img src="https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/links.png" width="400"/>
  <figcaption style="text-align: center; font-style: italic; padding-top: 8px;">The source of contamination for the MMLU benchmark.</figcaption>
</figure>


<figure>
<img src="https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/benchmarks.png" width=600>
<figcaption style="text-align: center; font-style: italic; padding-top: 8px;">Contamination in popular multi-choice QA benchmarks tested on Llama models</figcaption>
</figure>

Check the [contamination report of llama](https://arxiv.org/abs/2310.17589) for more details.

- **Test Memorization via Perplexity**:

Contamination Detector also audit the entire benchmark on a specific LLM, to verify whether the model exhibits **memorization** behaviors on test benchmarks. This is done by comparing the perplexity of the benchmark against memorised and fresh data.

<figure>
<img src="https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/xsum.png" width = 500>
<figcaption>The extent of contamination of XSum test set.</figcaption>
</figure>

We found the perplexity on XSum is between the memorized and clean baseline, which indicate XSum is partially contaminated.

## Why Choose Contamination Detector

Traditional methods for analyzing contamination often depend on identifying overlaps between training and test data. However, many modern LLMs utilize closed-source training data, restricting analysis largely to internal reports from big tech companies or research groups.

Contamination Detector uses a novel approach by directly auditing models and benchmarks, thereby **avoiding the need for access to the training data.** And there is no need for the massive storage demands for hosting the entire training data as well. This provides opportunities for small teams and individuals to conduct their own contamination analyses.

## 0.To start

Clone the repository:

```
git clone https://github.com/liyucheng09/Contamination_Detector.git
```

Install the required packages:

```
pip install -r requirements.txt
```

## Run Internet Presence Verification

To check whether test examples are accessible on the internet:

Simply run this to produce a report for a benchmark:

```
python search.py
```

To run this script, you will need a free access token for Bing search API. You could obtain one via [this](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api). A free access key allow 1000 calls monthly.

Set the key via `export Bing_Key = [YOUR API KEY]` in terminal.

Or, you can directly download my search results [here](https://github.com/liyucheng09/Contamination_Detector/releases/tag/v0.1.0), so you don't have to pay for accessing Bing.

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

You could visualize contamination examples via:

```
python visualize_search.py
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

**Check more comtamination examples: MMLU at [here](https://htmlpreview.github.io/?https://github.com/liyucheng09/Contamination_Detector/blob/master/reports/mmlu.html), and C-Eval at [here](https://htmlpreview.github.io/?https://github.com/liyucheng09/Contamination_Detector/blob/master/reports/ceval.html)**

If you cannot accessing Huggingface Hub for the benchmarks, download as json file [here](https://github.com/liyucheng09/Contamination_Detector/releases/tag/v0.1.0).

## Compare Model Performance on Clean and Dirty test set

This will generate the comparison of accuracy between the clean and dirty benchmark subsets for various models.

But first you need prepare:
1. the model predictions on these benchmarks.
2. get the benchmark reports ready (the internet presence reports).

Download predictions of all Llama series models on these benchmark at [here](https://github.com/liyucheng09/Contamination_Detector/releases/tag/v0.1.0). Unzip and put them under `model_predictions/`.

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


## Test Memorization via Perplexity

Read this paper first to get the basic idea of perplexity test: [Estimating Contamination via Perplexity](https://arxiv.org/abs/2309.10677)!

Run this to conduct the perplexity comparison:

```
python perplexity.py
```

You could specify models and benchmarks to test. It will write the results under `reports/`.

Then you could generate the figure to visualize the results via:

```
python visualize_search.py
```

**Check perplexity analysis of QA (BoolQ, SQuAD, QuAD) benchmarks [here](https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/qa.png)**

**Important:** you should choose apporiate data source for the memorised and clean baselinse. For example, qa benchmarks often use wikipedia passages, so you should use wikipedia as the base for the two baselines. 

Again, read [this paper](https://arxiv.org/abs/2309.10677) before you start.

## Issues

Open an issue or contact me via email if you encounter any problems in your use.