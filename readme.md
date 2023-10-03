<p align="center">
    <img src="https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/logo.png" alt="Logo of Contamination Detector" width="auto" height="150" />
</p>

# Contamination Detector for LLMs Evaluation

Due to the massive training corpus used in modern LLMs, it's getting more common for training data to unintentionally include parts of benchmark tests. The community calls this phenomenon **Data Contamination**.

Comtamination Detector is a tool to identify potential data contamination issues in Large Language Model (LLM) evaluations.

## Features

Traditional ways for comtamination analysis is extremely limited - these methods rely on finding the overlappings between the training data and the test set. Unfortunately, training data of modern LLMs are often closed sourced. That's why existing contamination analysises are often internal reports alongside the release of LLMs from big tech companies or research teams.

The main feature of our method is we **do not need the access of the training data**, instead, our method audits models and benchmarks directly. This provides opportunities for the small teams or individuals to conduct contamination analysis.

Two main approaches:
- Check whether test examples can be found on the internet -- with `search.py`.
- Check whether models exhibit memorisation behaviour on test benchmarks -- with `perplexity.py`.

## 0.To start

Clone the repository:

`git clone https://github.com/liyucheng09/Contamination_Detector.git`

Install the required packages:

`pip install -r requirements.txt`

## `search.py`

We verify whether test examples **(inputs and labels)** are accessible on the internet. If a example is crawlable on the internet, it is very likely included in Common Crawl and involved in model training.

An example from MMLU:

**MMLU - Test sample**

Question: The economy is in a deep recession. Given this economic situation which of the following statements about monetary policy is accurate?

| Page Name | Overlapping | Match Ratio | URL |
|-|-|-|-|
| **AP Macroeconomics Question 445: Answer and Explanation** - CrackAP.com | **The economy is in a deep recession. Given this economic situation**, **which of the following statements about monetary policy is accurate?** A. Expansionary policy would only worsen the recession. B. Expansionary policy greatly increases aggregate demand if investment is sensitive to changes in the interest rate. | 0.901 | [Link](https://www.crackap.com/ap/macroeconomics/question-445-answer-and-explanation.html) |  
| **AP Macroeconomics Practice Test 21** - CrackAP.com | **Given this economic situation**, **which of the following statements about monetary policy is accurate?** A. Expansionary policy would only worsen the recession. B. Expansionary policy greatly increases aggregate demand if investment is sensitive to changes in the interest rate. | 0.615 | [Link](https://www.crackap.com/ap/macroeconomics/test41.html) |

We found the question, choices, and answers are presented in the domain `www.crackap.com/` at the same time, which indicates a high risk that this example is contaminated.

Simply run:

`python search.py`

to produce a report for a benchmark.

To run this script, you will need a free access token for Bing search API. You could obtain one via [this](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api).

Results for some benchmarks:
- `MMLU`: 29 out of 100 have high risk of contamination.
- `C-Eval`: 35 out of 100 have high risk of contamination.
- `Winograde`: only 1 out of 100 have high risk of contamination.

## `perplexity.py`

To verify how much a model **exhibits memorizations** on a certain test set, we compare the perplexity of the benchmark against two baselines: the memorized and clean baseline.

The memorized baseline consists of materials presented during the model's training stage. The clean baseline consists of materials the model has never seen before. Comparing the perplexity of benchmark against the two baselines, we could quantify too what extent the model memorized the benchmark.

Let's take XSum as an example:

![](https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/xsum.png)

XSum is comprised from BBC News articles. So the memorized baseline here contains BBC News published during the period of training data construction (Aug 2022 for LLaMA, Sep 2019 for GPT-3). The clean baseline is collected from latest BBC News published after June 2023.

We found the perplexity on XSum is between the memorized and clean baseline, which indicate XSum is partially contaminated.

simply run:

`python perplexity.py`

to view the perplexity comparison.

## Visualization tools

You could generate visualization report to better understand data contaminatioin.

try:

`python visualize_search.py`

This onnly works after you run `python search.py`.

Will add visualization for `perplexity.py` shortly.