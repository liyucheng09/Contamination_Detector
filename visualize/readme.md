# Visualizing

## 1. Visualize Search Results

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

## 2. Perplexity

You could generate the figure to visualize the perplexity results via:

```
python visualize/visualize_perplexity.py
```

**Check perplexity analysis example of QA (BoolQ, SQuAD, QuAD) benchmarks [here](https://github.com/liyucheng09/Contamination_Detector/blob/master/pics/qa.png)**