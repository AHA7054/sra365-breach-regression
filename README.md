# What Drives the Cost of a Data Breach?

A regression analysis of 500 simulated data breach incidents, built from my coursework in SRA 365 (Statistics for Security and Risk Analysis) at Penn State.

**[Read the full analysis →](analysis.md)**

## Key findings

- **The scale of a breach drives its cost.** The number of records affected explains about 20% of the variation in financial loss on its own. Detection time adds a small but significant effect, and the other factors add little once records are known.
- **Sensitive data share does not predict detection speed.** The logistic model performs no better than always guessing "slow," a good example of why accuracy is misleading with imbalanced classes.
- **Data quality matters.** About 16% of incidents look like placeholder rows. The regression results hold up without them, but most of the categorical associations do not.

## Methods

| Section | Method |
|---|---|
| Distributions and data quality | Descriptive statistics, z-score outlier screening, placeholder detection |
| Simple regression | OLS, 85% confidence and prediction intervals |
| Assumptions | Durbin-Watson, Breusch-Pagan, Shapiro-Wilk, residual and Q-Q plots, standardized residuals |
| Multiple regression | Best-subsets selection, hierarchical models, nested F-tests, AIC, VIF |
| Logistic regression | Binomial GLM, odds ratios, predicted probabilities, classification table |
| Categorical | Chi-square and Fisher's exact tests with standardized residuals |
| Sensitivity check | Refitting models without placeholder rows |

## Repository structure

```
├── analysis.qmd      # source: code and narrative
├── analysis.md       # rendered report (GitHub displays this)
├── analysis_files/   # figures for the rendered report
└── data/
    ├── breaches.dat              # continuous variables, used for regression
    └── breaches_categorical.csv  # adds categorical versions, used for logistic and chi-square
```

## Data

The datasets were provided for SRA 365 and describe simulated (not real) breach incidents.

| Variable | Description |
|---|---|
| `fin_loss` | Financial loss (scaled) |
| `num_records` | Number of records affected (scaled) |
| `num_people` | Number of people affected (scaled) |
| `per_sensitive` | Percent of records that were sensitive |
| `dys_detect` | Days until the breach was detected |
| `dys_impact` | Days the breach had an impact |
| `cost_controls` | Cost of security controls (scaled) |
| `*_v2` | Categorical versions of the variables above |

## Reproducing

Requires R and [Quarto](https://quarto.org) (included with RStudio).

```r
install.packages(c("ggplot2", "car", "lmtest", "leaps", "gmodels", "knitr", "rmarkdown"))
```

Then open `analysis.qmd` in RStudio and click **Render**.
