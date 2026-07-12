# MSPTE Framework: Fuzzy Cognitive Modeling of Factors Affecting Educational Robot Effectiveness in STEM Learning

This repository contains the technical/analytical implementation accompanying the manuscript:

> **A Meta-Synthesis and Fuzzy Cognitive Modeling of Factors Influencing the Effectiveness of Educational Robots in STEM Learning: Developing and Validating the MSPTE Framework**

It covers the quantitative/computational half of the study — the Fuzzy Cognitive Map (FCM) construction, simulation, and validation that builds on top of the qualitative meta-synthesis (51 empirical studies) to produce and validate the **MSPTE framework**.

## Overview

The study proceeds in two phases:

1. **Qualitative meta-synthesis** (Sandelowski & Barroso framework) — interpretive integration of 51 studies into 11 sub-themes nested under 4 main themes, each coded for facilitator/barrier polarity and frequency.
2. **Fuzzy Cognitive Mapping** — the qualitative model is quantified into a weighted directed graph (16 nodes: 4 main themes + 11 sub-themes + 1 outcome node), combining:
   - **Statistical weighting** derived directly from the meta-synthesis frequency and polarity data
   - **TF-IDF + cosine similarity** (machine learning / NLP) to discover latent cross-theme edges
   - **Kosko's fuzzy inference rule** (with decay) to simulate steady-state activation
   - **Scenario simulation** (best case, worst case, targeted interventions) and **sensitivity analysis** over the inference parameters (λ, γ)
   - **Network centrality analysis** and statistical validation (Pearson/Spearman correlation between node centrality and raw thematic frequency)

## Repository Structure

```
.
├── fcm_analysis_english.py     # Main analysis script (single-file pipeline)
├── مضامین-t__1_.docx           # Input: comprehensive meta-synthesis table (required)
├── مفاهیم.docx                 # Input: full description of the 85 concepts (optional, documentation only)
└── outputs/                    # Generated on run (see below)
```

## Requirements

- Python 3.8+
- Dependencies (installed automatically by the first cell of the script):
  ```
  pip install python-docx scikit-learn networkx scipy pandas matplotlib
  ```

## Usage

Designed to run as a notebook (e.g., Google Colab) split into `# %%` cells, or as a standard script.

1. Place `مضامین-t__1_.docx` (and optionally `مفاهیم.docx`) in the working directory.
   - In Colab, upload via:
     ```python
     from google.colab import files
     uploaded = files.upload()
     ```
2. Run the cells in order:
   ```bash
   python fcm_analysis_english.py
   ```
3. All tables (CSV) and figures (PNG) are written to the current directory.

### Input file format

`مضامین-t__1_.docx` must contain the meta-synthesis table as its **second table** (`d.tables[1]`), with columns:

| Main theme | Sub-theme | Concepts (name (freq): description / ...) | Sources | Total frequency |
|---|---|---|---|---|

Facilitator/barrier polarity is parsed directly from the Persian markers **"تسهیل‌گر"** (facilitator) and **"مانع"** (barrier) inside each concept's description — these markers must be preserved verbatim in the source document for the parser to work.

## Outputs

| File | Description |
|---|---|
| `table1_subtheme_stats.csv` | Facilitator/barrier/net-polarity statistics per sub-theme |
| `table2_weight_matrix.csv` | Full 16×16 FCM adjacency (weight) matrix |
| `table3_scenarios.csv` | Steady-state effectiveness under each simulated scenario |
| `table4_sensitivity.csv` | Sensitivity of results to λ (sigmoid steepness) and γ (memory/decay) |
| `table5_centrality.csv` | Node centrality, role (transmitter/receiver/ordinary) |
| `table6_validation.txt` | Pearson/Spearman correlation between centrality and raw frequency |
| `fig1_fcm_network.png` | Full FCM network graph |
| `fig2_weight_heatmap.png` | Adjacency matrix heatmap |
| `fig3_scenario_comparison.png` | Bar chart comparing scenario outcomes |
| `fig4_convergence.png` | Convergence trajectories across scenarios |
| `fig5_centrality.png` | Node centrality decomposition (in/out-degree) |
| `fig6_sensitivity.png` | Sensitivity surface across λ and γ |

## Optional: LLM-assisted semantic validation

Section `[12]` contains an **optional, disabled-by-default** module that uses the Claude API as a secondary semantic check on edge relationships. It is not used in, and does not alter, the study's primary results, which rest solely on the statistical/TF-IDF pipeline above. It only activates if an `ANTHROPIC_API_KEY` environment variable is set.

## Reproducibility notes

- All edge weights are deterministically derived from the input document; the only stochastic-adjacent step is TF-IDF vectorization, which is itself deterministic given fixed input text.
- λ = 0.7, γ = 0.1 were selected via the sensitivity analysis in section `[8]` as the unsaturated stable operating point; the full grid (λ ∈ {0.3–2.0}, γ ∈ {0.05–0.3}) is reported in `table4_sensitivity.csv` for transparency.

## Citation

If you use this code, please cite the associated manuscript:

> Pedrami, M., Saffari, Y. & (forthcoming). *A Meta-Synthesis and Fuzzy Cognitive Modeling of Factors Influencing the Effectiveness of Educational Robots in STEM Learning: Developing and Validating the MSPTE Framework.* [Journal, volume/issue/DOI to be added upon publication].


## Authors
- **Mohammad Pedrami** — First-author — Meta-synthesis analysis & data preparing
  ORCID: https://orcid.org/0009-0002-5611-8238
  
- **Dr. Yasaman Saffari** — Corresponding-author — Artificial Intelligence Researcher — FCM analysis & code implementation
  Personal website: https://yasamansaffarii.github.io

- Iran
