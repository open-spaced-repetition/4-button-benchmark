# 4 Button Benchmark

## Introduction

This benchmark reuses some of the code from the [SRS benchmark](https://github.com/open-spaced-repetition/srs-benchmark). However, it has a different goal: 
instead of measuring how good spaced repetition algorithms are at predicting binary pass/fail probability of recall, it measures how well they can predict the distribution of all 4 grades that are used in Anki.

## Dataset

The dataset for the this benchmark comes from 10 thousand Anki users. In total, this dataset contains information about ~727 million reviews of flashcards. The full dataset is hosted on Hugging Face Datasets: [open-spaced-repetition/anki-revlogs-10k](https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k).

## Evaluation

### Data Split

In the SRS benchmark, we use a tool called `TimeSeriesSplit`. This is part of the [sklearn](https://scikit-learn.org/) library used for machine learning. The tool helps us split the data by time: older reviews are used for training and newer reviews for testing. That way, we don't accidentally cheat by peeking into the future. In practice, we use past study sessions to predict future ones. This makes `TimeSeriesSplit` a good fit for our benchmark.

### Metrics

We use [cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy) with 4 classes. Lower values are better.

### Methods

1) method1: FSRS-7 is optimized as usual, to predict binary pass/fail probability of recall (`R` for short). Then historical usage rates of Hard/Good/Easy are estimated. Then `p(Again)=1 - R`, `p(Hard)=R*p(Hard|pass)`, 
`p(Good)=R*p(Good|pass)`, `p(Easy)=R*p(Easy|pass)`. Pass means "Hard, Good or Easy", `p(Hard|pass) + p(Good|pass) + p(Easy|pass) = 1`. This is a simple method that assumes that lower/higher probability of recall doesn't affect 
how likely the user is to press Hard or Easy.
2) method2: 8 parameters are estimated from user's data. They define how p(Hard|pass) and p(Easy|pass) depend on R. As R increases, p(Hard|pass) decreases and p(Easy|pass) increases. This is a more sophisticated method that
takes into account that how likely the user is to press Hard/Good/Easy depends on the probability of recall. At lower R recall requires more effort, so Hard becomes more likely. At higher R recall requires less effort, 
so Easy becomes more likely.


## Result

Total number of collections (each from one Anki user): _________.

Total number of reviews for evaluation: _________.

The best result is highlighted in **bold**.

| Method | CrossEntropy4 |
| --- | --- |
