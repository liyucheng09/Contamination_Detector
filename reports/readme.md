# Data Contamination Reports for Popular Multi-choice QA Benchmark

 - `mmlu_report.json`
 - `ceval_report.json`
 - `hellaswag_report.json`
 - `commonsense_qa_report.json`
 - `ARC_report.json`
 - `winogrande_report.json`


## file structure

```
{
  "dataset_name": "mmlu",
  "matches": [
    [
      {
        "query": "Power imbalance, where for example one party possess more resources, unfair distribution of Resources, where one party gains more from the relationship, and CSOs being Co-opted, are all limitations and risks of business-CSO collaborations.",
        "match_string": "where for example one party possess more resources unfair distribution where one party gains more from the relationship and CSOs being are all limitations and risks of business-CSO collaborations Power imbalance",
        "score": 0.9084593254575504,
        "score_label": 0.7499999999999999,
        "name": "<b>Chapter 10 Multiple Choice Questions</b> - <b>Business</b> Ethics 5e Student ...",
        "url": "https://learninglink.oup.com/access/content/cranebe5e-student-resources/cranebe5e-chapter-10-multiple-choice-questions",
        "snippet": "_____, <b>where for example</b> <b>one</b> <b>party</b> <b>possess</b> <b>more</b> <b>resources</b>, <b>unfair</b> <b>distribution</b> of _____, <b>where one</b> <b>party</b> <b>gains</b> <b>more</b> <b>from the relationship</b>, <b>and CSOs</b> <b>being</b> _____, <b>are all</b> <b>limitations</b> <b>and risks</b> <b>of business-CSO</b> <b>collaborations</b>. <b>Power</b> <b>imbalance</b>, Benefits, Hoodwinked ..."
      }
    ],

```

Each list in `matches` represents a test instance in corresponding benchmark.
And each dict represents a match from Bing Search. 

The `score` indicate the ratio of overlapping between the page and the test instance.

The `score_label` indicate the overlapping of answer with the given page.