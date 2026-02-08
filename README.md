# Multilevel Modeling for Hospital Length of Stay Prediction

This repo contains the R code from my research on predicting hospital length of stay (LOS). The main idea: most prediction models only look at patient characteristics, but hospitals and regions matter too.

## What's this about?

We wanted to see if adding hospital-level info (bed size, teaching status, etc.) and regional factors could improve LOS predictions. Turns out, it does. Hierarchical models that account for these factors outperform simpler approaches.

The data comes from the 2019 National Inpatient Sample (NIS), which has about 7 million hospital stays. We sampled 1 million records for this analysis.

## Models

We compared three approaches:

1. **Model 1** — Just patient stuff (age, diagnoses, severity, etc.)
2. **Model 2** — Patient + hospital characteristics (but treating hospital as a regular variable)
3. **Model 3** — Hierarchical model with hospital as a random effect

Model 3 captures the fact that patients within the same hospital tend to have correlated outcomes. This is the key insight.

## Getting started

You'll need R (4.0+) and these packages:

```r
source("install_packages.R")
```

Then open `src/main_analysis.R` and update the data path at the top:

```r
DATA_PATH <- "your/path/to/NIS_data.csv"
```

Note: I can't share the raw NIS data here due to the data use agreement, but you can get it from [HCUP](https://www.hcup-us.ahrq.gov/nisoverview.jsp).

## Quick results

Adding hospital and regional factors improved R² compared to patient-only models. The hierarchical model (Model 3) had the best fit, especially for hospitals with unusual case mixes.

Full results are in the paper (see citation below).

## Citation

If you use this code:

```

```

## Questions?

Feel free to open an issue or email me at ma7684@g.rit.edu
