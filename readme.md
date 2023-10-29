<p align="center">
    <img src="https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/logo.png" alt="Logo of Contamination Detector" width="auto" height="160" />
</p>

# Contamination Detector for LLMs Evaluation

In the realm of Large Language Models (LLMs), Data Contamination is a ubiquitous issue. **Contamination Detector** aids in identifying and analyzing such potential contamination without requiring access to the LLMs' training data, enabling even small teams and individuals to conduct robust evaluation.

- **Internet Presence Verification**:

Contamination Detector goes through the benchmark and verify whether test examples (both inputs and labels) are present on the internet.

An example of contamination from **MMLU**:

```
[
  {
    "input": "The economy is in a deep recession. Given this economic situation which of the following statements about monetary policy is accurate?",
    "match_string": "The economy is in a deep recession. Given this economic situation, which of the following statements about monetary policy is accurate policy recession policy",
    "score": 0.900540825748582,
    "name": "<b>AP Macroeconomics Question 445: Answer and Explanation</b> - CrackAP.com",
    "contaminated_url": "https://www.crackap.com/ap/macroeconomics/question-445-answer-and-explanation.html",
  },
  {
    "input": "The economy is in a deep recession. Given this economic situation which of the following statements about monetary policy is accurate?",
    "match_string": "Given this economic situation which of the following statements about monetary policy is accurate policy recession policy",
    "score": 0.615243730628346,
    "name": "<b>AP Macroeconomics Practice Test 21</b> - CrackAP.com",
    "contaminated_url": "https://www.crackap.com/ap/macroeconomics/test41.html",
  }
]
```

The results show this example including the question, choices, and answer is accessible on the internet and thus has high risk of containation.

- **Memorization Behavior Examination**:

Contamination Detector also audit the entire benchmark on a specific LLM, to verify whether the model exhibits memorization behaviors on test benchmarks. This is done by comparing the perplexity of the benchmark against memorised and fresh data.

![](https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/xsum.png)

This is an contamination illustration from **XSum**. We found the perplexity on XSum is between the memorized and clean baseline, which indicate XSum is partially contaminated.

## Why Choose Contamination Detector

Traditional methods for analyzing contamination often depend on identifying overlaps between training and test data. However, many modern LLMs utilize closed-source training data, restricting analysis largely to internal reports from big tech companies or research groups.

Contamination Detector uses a novel approach by directly auditing models and benchmarks, thereby **avoiding the need for access to the training data.** This provides opportunities for small teams and individuals to conduct their own contamination analyses.

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

To run this script, you will need a free access token for Bing search API. You could obtain one via [this](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api).

It will generate a report under `reports/`.

After you have the report for example `mmlu.json`, you could try this to visualize several samples:

```
python visualize_search.py
```

**`MMLU` - An example of contamination visualization**

**Question:** The economy is in a deep recession. Given this economic situation which of the following statements about monetary policy is accurate?

**Matches:**
| Page Name | Overlapping | Match Ratio | URL |
|-|-|-|-|
| AP Macroeconomics Question 445: Answer and Explanation - CrackAP.com | **The economy is in a deep recession. Given this economic situation**, **which of the following statements about monetary policy is accurate?** A. Expansionary policy would only worsen the recession. B. Expansionary policy greatly increases aggregate demand if investment is sensitive to changes in the interest rate. | 0.901 | [Link](https://www.crackap.com/ap/macroeconomics/question-445-answer-and-explanation.html) |  
| AP Macroeconomics Practice Test 21 - CrackAP.com | **Given this economic situation**, **which of the following statements about monetary policy is accurate?** A. Expansionary policy would only worsen the recession. B. Expansionary policy greatly increases aggregate demand if investment is sensitive to changes in the interest rate. | 0.615 | [Link](https://www.crackap.com/ap/macroeconomics/test41.html) |

It will hightlight the overlapping part of the benchmark and internet pages.

**Check more contamination examples of MMLU [here](https://htmlpreview.github.io/?https://github.com/liyucheng09/Contamination_Detector/blob/master/reports/mmlu.html)**

**Check contamination examples of C-Eval [here](https://htmlpreview.github.io/?https://github.com/liyucheng09/Contamination_Detector/blob/master/reports/ceval.html)**

Results for some benchmarks:
- `MMLU`: 29 out of 100 have high risk of contamination.
- `C-Eval`: 35 out of 100 have high risk of contamination.
- `Winograde`: only 1 out of 100 have high risk of contamination.

## Run Memorization Behavior Examination

simply run to view the perplexity comparison:

```
python perplexity.py
```

You could specify models and benchmarks to test. It will write the results under `/reports/`.

Then you could generate the figure with:

```
python visualize_search.py
```

**Check perplexity analysis of QA (BoolQ, SQuAD, QuAD) benchmarks [here](https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/qa.png)**

**Important:** you should choose apporiate data source for the memorised and clean baselinse. For example, qa benchmarks often use wikipedia passages, so you should use wikipedia as the base for the two baselines.
Check more details about this method in this paper: [Estimating Contamination via Perplexity: Quantifying Memorisation in Language Model Evaluation](https://arxiv.org/abs/2309.10677).

## Issues

Open an issue or contact me via email if you encounter any problems in your use.
