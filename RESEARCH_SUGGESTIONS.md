# Research & Academic Suggestions

**Repo**: Hemant — Network Intrusion Detection (UNSW-NB15 + Synthetic IPv6)
**Date**: 2026-04-15

---

## Table of Contents

1. [Understanding of the Research Direction](#understanding-of-the-research-direction)
2. [Strengths of the Current Work](#strengths-of-the-current-work)
3. [Research Gaps & Weaknesses](#research-gaps--weaknesses)
4. [Recommended Next Moves](#recommended-next-moves)
5. [Stretch Goals & Publication Angle](#stretch-goals--publication-angle)

---

## Understanding of the Research Direction

Based on the repo contents, the thesis appears to follow this arc:

1. **Classical baselines** (LR multiclass + binary) on a synthetic IPv6 dataset
2. **Literature CNN benchmark reproduction** on UNSW-NB15 (3 architectures from 2 papers)
3. **Transfer evaluation** — test UNSW-trained CNNs on the synthetic IPv6 dataset to measure domain shift
4. **(Upcoming)** Design a custom CNN and compare it against all baselines and benchmarks

The core research question seems to be: **Can CNN-based intrusion detection models trained on standard benchmarks (UNSW-NB15) generalize to synthetic IPv6 network traffic, and can a custom architecture close the transfer gap?**

---

## Strengths of the Current Work

- **The two-stage evaluation design is sound**: training on UNSW-NB15, evaluating in-domain, then measuring transfer drop to synthetic data is a clean and defensible experimental setup.
- **Reproducing multiple literature methods before proposing your own** is methodologically correct and demonstrates awareness of the field.
- **Paper-vs-reproduced comparison tables** add transparency and show scientific maturity — explicitly acknowledging that reproduced numbers may differ from reported numbers.
- **Macro-F1 as the primary metric** (not accuracy) is the right choice for imbalanced intrusion detection data. The thesis-level awareness of this is clear throughout.
- **Threshold selection on validation for binary** avoids data leakage and shows proper experimental discipline.
- **The drop analysis framework** (UNSW accuracy vs synthetic accuracy, per-model) is a useful contribution structure.

---

## Research Gaps & Weaknesses

### 1. The research question and contribution are not explicitly stated

The CNN notebook says "build benchmark CNNs before moving to our own CNN design," but neither the README nor any notebook states:
- What is the **specific hypothesis** being tested?
- What is the **novel contribution** of this work?
- What **gap in the literature** does this address?

Without this, the work reads as "we tried some methods on some data" rather than "we tested hypothesis X and found Y."

**Suggestion**: Write a 1-page research statement (even as a markdown file) that explicitly states:
- The problem (IPv6 intrusion detection is understudied)
- The gap (existing models are trained on IPv4 benchmarks like UNSW-NB15 — how well do they transfer?)
- The hypothesis (e.g., "domain-adapted CNNs can close the transfer gap between IPv4-trained models and IPv6 traffic")
- The contribution (the synthetic dataset itself? the transfer analysis? the custom CNN?)

### 2. The synthetic IPv6 dataset lacks documentation of its generation process

The synthetic dataset (`synthetic_ipv6_grounded_v3_32x32`) is central to the thesis but there is no documentation of:
- How was it generated? (simulation? transformation from UNSW-NB15? GAN? rule-based?)
- What does "grounded" mean in the dataset name?
- What does "v3" imply — were there prior versions? What changed?
- What does "32x32" refer to — image resolution? feature count?
- What are the key differences from UNSW-NB15 in terms of feature schema?
- Why is this dataset a valid proxy for real IPv6 intrusion traffic?

If the synthetic dataset is your original contribution, it needs a **datacard** or **dataset paper** section explaining:
- Generation methodology
- Feature alignment with UNSW-NB15
- Known limitations and biases
- Class distribution rationale (why is Benign so dominant?)

### 3. The class imbalance invalidates several conclusions

The synthetic dataset has:

| Class | Total | Test |
|-------|-------|------|
| Benign | 2829 | 425 |
| Exploits | 50 | 7 |
| Worms | 2 | 1 |
| Analysis | 3 | 1 |
| Backdoor | 5 | 0 |

This is not just "imbalanced" — several classes are **statistically unobservable**. With 0-1 test samples, any metric for those classes is meaningless. The current work flags these as "unstable" but continues to include them in macro-F1 calculations, which drags the macro-F1 down due to zero-division artifacts rather than real model failures.

**Suggestion**:
- Report metrics **with and without** unstable classes (e.g., "macro-F1 over stable classes only: X")
- Consider whether the extremely rare classes (Worms, Analysis, Backdoor) should be merged into an "Other" category for evaluation purposes
- If the synthetic dataset generation is under your control, generate more minority samples — a 1400:1 ratio makes multiclass evaluation scientifically unsound

### 4. Transfer evaluation needs deeper analysis

The drop analysis (UNSW accuracy minus synthetic accuracy) is a good start, but it only tells you *that* performance drops, not *why*. To make the transfer analysis publishable:

- **Feature-level analysis**: Which features exist in both datasets? Which are missing or renamed? The `align_to_reference_columns()` function fills missing columns with NaN — how many columns are typically missing? This is a key driver of transfer drop.
- **Per-class transfer analysis**: Does the model fail uniformly across classes, or do some classes transfer well and others don't? A per-class F1 comparison table (UNSW vs synthetic) would be very informative.
- **Distribution shift visualization**: Use t-SNE, UMAP, or PCA to visualize the feature distributions of UNSW vs synthetic test sets in the same embedding space. This shows whether the domains overlap or are clearly separated.
- **Confidence calibration**: Are the model's predicted probabilities on synthetic data well-calibrated? Low confidence on synthetic data (even for correct predictions) would indicate the model "knows" it's out of distribution.

### 5. The LR baselines and CNN benchmarks are disconnected

Notebooks 01 and 02 evaluate LR on the synthetic dataset only. The CNN notebook evaluates on UNSW-NB15 then transfers to synthetic. But there is no **unified comparison table** across all methods:

| Model | UNSW Macro-F1 | Synthetic Macro-F1 | Drop |
|-------|---------------|---------------------|------|
| LR (multiclass) | — | 0.398 | — |
| Systems2024 Arch1 | ? | ? | ? |
| Systems2024 Arch2 | ? | ? | ? |
| MobileNetV2 | ? | ? | ? |
| *Custom CNN* | ? | ? | ? |

Without this table, a reviewer cannot compare the methods. The LR baselines should also be run on UNSW-NB15 for a fair comparison.

### 6. No statistical significance testing

Five seeds provide a mean and std, but:
- The LR experiments have std = 0 (as discussed)
- The CNN experiments use 5 seeds, but there's no paired test (e.g., paired t-test or Wilcoxon signed-rank) to determine whether differences between models are statistically significant

**Suggestion**: For every pairwise model comparison you plan to claim ("model A outperforms model B"), run a statistical test. With 5 seeds, power is limited — consider 10+ seeds if feasible.

### 7. Image-based CNN (MobileNetV2) approach lacks justification

The Noever & Noever approach converts tabular features into 16x16 grayscale images and feeds them to MobileNetV2. This is an interesting idea from the literature, but:
- The conversion is lossy (padding to 256 features, reshaping into a grid)
- The spatial structure of a 16x16 image has no inherent meaning for network flow features (pixel adjacency is arbitrary)
- MobileNetV2 was designed for natural images — its inductive biases (translation invariance, hierarchical spatial features) don't align with tabular data

This should be **discussed as a limitation** of the benchmark, not just implemented silently. It also creates a natural motivation for your custom CNN: "image-based approaches impose arbitrary spatial structure; we propose [alternative approach]."

### 8. No ablation or feature importance analysis

None of the experiments analyze which features drive the predictions:
- Which features matter most for the LR models? (`coef_` is available)
- Which features, when removed, cause the largest performance drop?
- Are the features that matter on UNSW-NB15 the same ones that matter on synthetic data?

Feature importance analysis would strengthen the transfer story significantly: "the model relies on feature X, which has different distributions across domains, explaining the transfer drop."

---

## Recommended Next Moves

### Move 1: Write the unified comparison framework (before building the custom CNN)

Before adding complexity, create a single notebook or script that:
1. Runs LR (binary + multiclass) on **both** UNSW-NB15 and synthetic
2. Loads CNN benchmark results from artifacts
3. Produces a single comparison table with all models, both datasets, both tasks
4. Computes per-model transfer drop

This becomes Table 1 of the thesis. Everything after this is about beating the numbers in this table.

### Move 2: Deepen the transfer analysis

Add to the CNN benchmark notebook:
- A **feature overlap audit**: how many UNSW features survive in the synthetic dataset?
- A **per-class transfer F1 table**: which attack types transfer well and which don't?
- A **distribution shift visualization** (UMAP/t-SNE on the vectorized features, colored by dataset)
- Count how many synthetic samples get classified as "Unknown" or with <50% confidence

This analysis alone could be a standalone section of the thesis.

### Move 3: Design the custom CNN with a clear hypothesis

Don't just "make a CNN." Choose a design motivated by the transfer analysis:

- **If the problem is feature mismatch**: Design a feature-agnostic architecture (e.g., attention over features rather than convolution, so the model doesn't rely on positional feature ordering)
- **If the problem is class imbalance**: Incorporate focal loss, SMOTE-based augmentation, or prototypical networks for few-shot attack classes
- **If the problem is domain shift**: Use domain adaptation techniques (e.g., DANN — Domain-Adversarial Neural Network, or MMD loss) to align UNSW and synthetic feature distributions
- **If the image representation is the bottleneck**: Propose a better tabular-to-image mapping (e.g., feature correlation-based pixel arrangement instead of arbitrary ordering)

Each of these is a testable hypothesis and a potential contribution.

### Move 4: Address the class imbalance systematically

This is both a practical and academic issue. Consider:

- **Data-level**: SMOTE, ADASYN, or conditional generation (CTGAN) for minority classes
- **Loss-level**: Focal loss, class-balanced loss (Cui et al., 2019)
- **Evaluation-level**: Report per-class F1 for all classes with sufficient support; use AUPRC instead of F1 for binary; use Cohen's kappa alongside accuracy
- **Architectural**: Prototypical networks or metric learning for few-shot attack classes

An ablation study comparing 2-3 of these approaches would be a strong experimental section.

### Move 5: Run LR baselines on UNSW-NB15

This is a small effort with high payoff. Currently LR is only evaluated on synthetic data. Running the same LR pipeline on UNSW-NB15 lets you:
- Complete the unified comparison table
- Determine whether LR is competitive with CNNs on tabular data (it often is — this is a known finding in the tabular ML literature)
- If LR matches or beats CNNs on UNSW-NB15, that's an important finding worth discussing

### Move 6: Add a non-deep-learning tree-based baseline

XGBoost or LightGBM on tabular data is the strongest known baseline for tabular classification (see Grinsztajn et al., "Why do tree-based models still outperform deep learning on tabular data?", NeurIPS 2022). Omitting this comparison is a vulnerability that a reviewer will flag.

Adding XGBoost takes minimal code (same preprocessing pipeline, swap the classifier) and provides:
- A strong tabular baseline to justify the CNN approach
- Built-in feature importance (SHAP values)
- A calibration reference for the CNN models

---

## Stretch Goals & Publication Angle

### If the synthetic dataset is novel

If this synthetic IPv6 dataset is your original creation, the strongest publication angle is a **benchmark paper**:
- "We present SynIPv6-NB15, a synthetic IPv6 intrusion detection dataset grounded in UNSW-NB15. We evaluate transfer performance of N existing methods and show that [finding]."
- Include a datacard, generation methodology, and baseline results
- Target: a workshop paper at a security conference (e.g., ACSAC, RAID workshop tracks) or a dataset paper at NeurIPS Datasets & Benchmarks

### If the custom CNN is the contribution

The publication angle shifts to:
- "We propose [architecture name], a CNN for intrusion detection that [key innovation]. We show it outperforms literature benchmarks on UNSW-NB15 and transfers better to IPv6 traffic."
- The key innovation must be clearly stated: domain adaptation? attention-based feature weighting? better tabular-to-image conversion?
- Target: IEEE/ACM conference on network security or a journal like IEEE TIFS, Computers & Security

### If the transfer analysis is the contribution

This is viable even without a novel architecture:
- "We systematically evaluate the transferability of CNN-based intrusion detectors from IPv4 (UNSW-NB15) to IPv6 network traffic and identify [specific failure modes]."
- Requires the deeper analysis from Move 2 above
- Target: empirical study venue (e.g., ACM IMC, PAM, or a security measurements workshop)

### Literature to engage with

Beyond the two papers already cited, consider:

1. **Tabular vs deep learning**: Grinsztajn et al., "Why do tree-based models still outperform deep learning on tabular data?", NeurIPS 2022 — directly relevant to justifying CNN over XGBoost
2. **Domain adaptation for security**: Singla et al., "Preparing Network Intrusion Detection Deep Learning Models with Minimal Data Using Adversarial Domain Adaptation", AsiaCCS 2020
3. **UNSW-NB15 analysis**: Moustafa & Slay, "UNSW-NB15: A comprehensive data set for network intrusion detection systems", MilCIS 2015 — the original dataset paper
4. **Class imbalance in IDS**: Douzas & Bacao, "Effective data generation for imbalanced learning using conditional generative adversarial networks", Expert Systems with Applications, 2018
5. **Feature importance in IDS**: Kasongo & Sun, "A Deep Learning Method with Wrapper Based Feature Extraction for Wireless Intrusion Detection System", Computers & Security, 2020
6. **IPv6 security**: Zulkiflee et al., "A Taxonomy of IPv6 Transition-Based Threats: Analysis and Implications for Security", IEEE Access, 2023 — helps position the IPv6 angle

---

## Summary: Priority-Ordered Next Moves

| Order | Action | Effort | Research Value |
|-------|--------|--------|----------------|
| 1 | Write a 1-page research statement (hypothesis, gap, contribution) | Low | Critical — frames everything |
| 2 | Document the synthetic dataset generation process | Low | Critical — reviewers will ask |
| 3 | Build the unified comparison table (all models, both datasets) | Medium | High — becomes the main results table |
| 4 | Run LR + XGBoost on UNSW-NB15 | Low | High — completes the baseline picture |
| 5 | Deepen transfer analysis (feature overlap, per-class, UMAP) | Medium | High — potential standalone contribution |
| 6 | Design custom CNN with a motivated hypothesis | High | Core contribution |
| 7 | Address class imbalance with at least one data/loss-level technique | Medium | Strengthens evaluation credibility |
| 8 | Add statistical significance tests (10+ seeds, paired tests) | Low | Required for publication |
| 9 | Write limitation discussion (image inductive bias, rare classes, synthetic validity) | Low | Shows academic maturity |
