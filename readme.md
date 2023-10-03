<p align="center">
    <img src="https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/logo.png" alt="Logo of Contamination Detector" width="auto" height="150" />
</p>

# Contamination Detector for LLMs Evaluation

In the realm of Large Language Models (LLMs), Data Contamination is a ubiquitous issue. **Contamination Detector** aids in identifying and analyzing such potential contamination without requiring access to the LLMs' training data, enabling even small teams and individuals to conduct robust analyses.

## Features

Traditional methods for analyzing contamination often depend on identifying overlaps between training and test data. However, many modern LLMs utilize closed-source training data, restricting analysis largely to internal reports from big tech companies or research groups.

Contamination Detector uses a novel approach by directly auditing models and benchmarks, thereby **avoiding the need for access to the training data.** This provides opportunities for small teams and individuals to conduct their own contamination analyses.

We implement two principal approaches:
- **Internet Presence Verification**: Using `search.py`, verify whether test examples (both inputs and labels) are present on the internet, signifying a potential inclusion in web-scraped training data like Common Crawl.
- **Memorization Behavior Examination**: Employing `perplexity.py`, ascertain whether a model displays memorization behaviors on test benchmarks by comparing the perplexity of benchmarks against two specific baselines: memorized and clean.

## 0.To start

Clone the repository:

```
git clone https://github.com/liyucheng09/Contamination_Detector.git
```

Install the required packages:

```
pip install -r requirements.txt
```

## `search.py`

To check whether test examples are accessible on the internet:

Simply run this to produce a report for a benchmark:

```
python search.py
```

To run this script, you will need a free access token for Bing search API. You could obtain one via [this](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api).

**`MMLU` - An example of contamination**

**Question:** The economy is in a deep recession. Given this economic situation which of the following statements about monetary policy is accurate?

**Matches:**
| Page Name | Overlapping | Match Ratio | URL |
|-|-|-|-|
| **AP Macroeconomics Question 445: Answer and Explanation** - CrackAP.com | **The economy is in a deep recession. Given this economic situation**, **which of the following statements about monetary policy is accurate?** A. Expansionary policy would only worsen the recession. B. Expansionary policy greatly increases aggregate demand if investment is sensitive to changes in the interest rate. | 0.901 | [Link](https://www.crackap.com/ap/macroeconomics/question-445-answer-and-explanation.html) |  
| **AP Macroeconomics Practice Test 21** - CrackAP.com | **Given this economic situation**, **which of the following statements about monetary policy is accurate?** A. Expansionary policy would only worsen the recession. B. Expansionary policy greatly increases aggregate demand if investment is sensitive to changes in the interest rate. | 0.615 | [Link](https://www.crackap.com/ap/macroeconomics/test41.html) |

We found the question, choices, and answers are all presented in the domain `www.crackap.com/`, which indicates a high risk that this example is contaminated.

**Check more comtamination examples of MMLU [here](https://github.com/liyucheng09/Contamination_Detector/blob/master/reports/mmlu.html)**

**Check comtamination examples of C-Eval [here](https://github.com/liyucheng09/Contamination_Detector/blob/master/reports/ceval.html)**

Results for some benchmarks:
- `MMLU`: 29 out of 100 have high risk of contamination.
- `C-Eval`: 35 out of 100 have high risk of contamination.
- `Winograde`: only 1 out of 100 have high risk of contamination.

## `perplexity.py`

We also propose to analyze contamination by measure how much a model **exhibits memorizations** on a certain benchmark - we compare the perplexity of a benchmark against two tailored baselines: the memorized and clean baseline.

- The memorized baseline contains samples model already learned during training. 
- The clean baseline consists of materials the model has never seen before. 

Comparing the perplexity of benchmark against the two baselines, we could quantify too what extent the model memorized the benchmark.

Let's take XSum as an example:

![](https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/xsum.png)

XSum is comprised from BBC News articles. So the memorized baseline here contains BBC News published during the period of training data construction (Aug 2022 for LLaMA, Sep 2019 for GPT-3). The clean baseline is collected from latest BBC News published after June 2023.

We found the perplexity on XSum is between the memorized and clean baseline, which indicate XSum is partially contaminated.

simply run:

```
python perplexity.py
```

to view the perplexity comparison.

Check this paper for more details about this method: [Estimating Contamination via Perplexity: Quantifying Memorisation in Language Model Evaluation](https://arxiv.org/abs/2309.10677).

## Visualization tools

You could generate visualization report to better understand data contaminatioin.

try:

```
python visualize_search.py
```

This onnly works after you run `python search.py`.

Will add visualization for `perplexity.py` shortly.

## Issues

Open an issue or contact me via email if you encounter any problems in your use.