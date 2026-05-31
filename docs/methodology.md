# Methodology

## Research Question

*Do managers who substitute earlier in a match produce better post-substitution outcomes than managers who substitute later, after controlling for the game state at the moment of the decision?*

This document describes every statistical and machine learning method used to answer that question. It covers what each method is, why it was chosen, how it was applied, and what assumptions it requires. Numerical findings and results are reported separately.

The central identification challenge runs through every section below: managers do not substitute randomly. Teams that are losing substitute earlier; teams that are winning substitute later. A naive comparison of early and late substitutions would not measure the effect of timing — it would measure the effect of being behind on the scoreboard. Every method in this pipeline exists, in some form, to break that conflation.

---

## 1. Directed Acyclic Graph (DAG)

### What it is

A Directed Acyclic Graph is a formal tool for mapping assumed causal relationships between variables before any data is analyzed. Nodes represent variables; directed edges represent assumed causal pathways. Three node types matter for causal inference:

- **Confounders** influence both the treatment (when a manager substitutes) and the outcome (what happens after). They must be controlled.
- **Mediators** sit on the causal pathway between treatment and outcome — they are caused by the substitution itself. Controlling for them blocks the very effect being estimated and introduces bias.
- **Colliders** are caused by both treatment and outcome. Conditioning on them opens spurious associations between variables that are otherwise unrelated.

### Why I used it

Without an explicit causal graph, analysts routinely over-control — including mediators that sit on the causal pathway — or under-control, omitting genuine confounders. Both mistakes bias estimates in different directions, and neither mistake is detectable from the data alone. The DAG was constructed before any modeling began, using domain knowledge about football tactics to commit to a causal structure.

### How it was applied

Seven confounders were identified — variables that plausibly affect both *when* a manager decides to substitute and *how the team performs* afterward:

- **Scoreline at substitution minute** — teams losing substitute earlier and face a harder recovery task
- **Goal differential (xG proxy)** — captures match tempo beyond raw score
- **Home / away status** — home teams substitute differently and have different outcome baselines
- **Opponent quality (rolling form)** — managers respond to opponent strength; outcomes depend on it
- **Substitute player quality** — stronger bench allows earlier substitution; better players improve outcomes
- **Player fatigue (rolling workload)** — fatigue drives substitution timing and affects the outgoing player's impact
- **Substitutions already used** — constrains the available timing window

Three categories of variables were explicitly excluded from adjustment:

- **Mediators** — tactical shape change, pressing intensity, and momentum shift. These occur *because* a substitution was made. Controlling for them would absorb the effect being studied.
- **Colliders** — final match outcome and whether the team ultimately won. Conditioning on these variables opens non-causal associations between substitution timing and game-state variables.

### Assumptions and limitations

DAGs are constructed from domain knowledge, not data. If a variable is misclassified — a mediator treated as a confounder, or a confounder omitted — no subsequent statistical method can correct the resulting bias. The DAG makes these assumptions explicit and therefore auditable, which is an improvement over implicit modeling choices, but it does not eliminate the need for careful domain reasoning.

---

## 2. Rule-Based Substitution Classification

### What it is

Substitution reasons — injury, tactical decision, disciplinary response — are not labeled in the raw data. **Proxy classification** using observable signals assigns a likely reason to each substitution based on its timing and context. This approach is standard in observational sports research when ground-truth labels are unavailable.

### Why I used it

A manager substituting because a player has collapsed with a hamstring injury is not making a timing optimization decision — the substitution is forced. Including injury and disciplinary substitutions in a study of tactical timing would contaminate the treatment variable, mixing voluntary decisions with involuntary ones. Filtering to tactical substitutions only sharpens the research question to what the manager actually controlled.

### How it was applied

A four-class hierarchical classifier was applied, where the first matching rule wins:

- **Rule 1 — `injury_proxy`**: first substitution in the match AND made before minute 60
- **Rule 2 — `disciplinary_proxy`**: player substituted out received a yellow card in the same match before the substitution minute
- **Rule 3 — `red_card_match`**: a red card occurred in the match before the substitution minute
- **Rule 4 — `tactical`**: all remaining substitutions

Only observations classified as `tactical` were retained for analysis. The first-half exclusion in Rule 1 follows Del Corral et al. (2008), who used the same logic on the grounds that tactical substitutions before the hour mark are rare enough to treat early firsts as injury signals.

### Assumptions and limitations

Rule-based classification introduces misclassification in both directions. Some substitutions before minute 60 are genuine tactical decisions — for example, a poor individual performance or a red card in the opposing team forcing a formation shift. Some post-60 substitutions are injury-driven but do not trigger Rule 1 because they are not the first substitution. Sensitivity analyses that vary the minute threshold in Rule 1 are advisable to test whether classification choices materially alter the results.

---

## 3. Propensity Score Estimation

### What it is

A **propensity score** is the probability that a unit received treatment given its observed covariates: P(W = 1 | X). Rosenbaum and Rubin (1983) showed that conditioning on the propensity score is sufficient to control for all measured confounders simultaneously — it collapses a multi-dimensional covariate space into a single scalar, making matching tractable.

### Why I used it

With seven confounders, direct exact matching across all variables simultaneously fails in practice — the probability of finding an exact or near-exact match on all dimensions simultaneously falls rapidly as dimensionality grows. The propensity score reduces the matching problem from seven dimensions to one while preserving the key balancing property that Rosenbaum and Rubin proved: units with the same propensity score have the same distribution of covariates, in expectation.

### How it was applied

A logistic regression model was fitted to predict P(substitution at this minute = 1) for every observation in the dataset (both substitution moments and no-substitution moments):

- **Model**: `sklearn LogisticRegression`, L2 regularization, C = 1.0, lbfgs solver
- **Target**: binary indicator — substitution made at this game-minute or not
- **Covariates**: `score_diff_at_sub`, `opponent_points_l4`, `log1p(player_quality)`, `player_out_minutes_l30`, `subs_used_before`, `is_home`, `is_winning`, `is_losing`, league fixed effects (one-hot), substitution direction (one-hot), player position (one-hot)

Key design decisions:

- **Substitution slot excluded**: collinearity with match minute (r = 0.55) made it redundant and would have created near-multicollinearity in the model.
- **Player quality and fatigue excluded from the propensity model**: control observations — game moments where no substitution occurred — have no substitution player by definition. Including these variables would make the propensity model predict treatment with near-perfect accuracy (AUC → 1.0), destroying the overlap required for matching. They enter the outcome model instead.
- **Log transformation of player quality**: the raw distribution of goals-plus-assists per 90 minutes showed skewness of +7.6 in exploratory analysis. The log1p transformation brings this distribution closer to symmetry.
- **League fixed effects**: exploratory analysis confirmed a systematic gap in mean substitution timing across the five leagues. League is both a confounder (affects timing) and an outcome modifier (affects goal-scoring rates), satisfying the criteria for inclusion.

### Assumptions and limitations

Logistic regression assumes log-odds of treatment are linear in the covariates. Nonlinear relationships between game state and substitution probability — for example, sharp changes at particular scorelines — may not be captured. An AUC around 0.54 indicates genuine distributional overlap between substitution moments and no-substitution moments, which is a desirable property for matching rather than a sign of poor model fit.

---

## 4. Nearest Neighbor Propensity Score Matching

### What it is

**Propensity score matching** constructs a quasi-experimental comparison group. For each treated observation (an actual substitution), the algorithm finds a control observation (a game moment where no substitution occurred) that has a nearly identical propensity score — and therefore, by Rosenbaum and Rubin's theorem, a nearly identical expected distribution of covariates. The result approximates what a randomized experiment would produce.

### Why I used it

Matching addresses confounding directly: it constructs treated and control groups that are comparable on all measured confounders, rather than adjusting for them statistically after the fact. Unlike regression adjustment alone, matching does not extrapolate beyond the region where treated and control observations overlap — it restricts inference to comparable situations.

### How it was applied

- **Algorithm**: nearest neighbor without replacement, using a KDTree on logit-transformed propensity scores
- **Caliper**: 0.026 — computed as 0.2 times the standard deviation of the logit propensity score, following Austin (2011)
- **Within-league constraint**: treated observations were only matched to control observations from the same league. Exploratory analysis confirmed that substitution behavior and outcome baselines differ meaningfully across leagues; cross-league matching would compare tactically incomparable situations
- **Logit transformation for caliper**: the propensity score is bounded between 0 and 1 and is compressed near the extremes. Austin (2011) recommends applying the caliper on the logit scale, where distances are more uniform across the full distribution

The logit-scale caliper was:

```
caliper = 0.2 × std(logit(propensity_score))
```

**Common support** was assessed by plotting propensity score distributions for treated and control groups in each league before matching. Observations outside the region of overlap were trimmed before matching proceeded.

**Balance assessment** used standardized mean differences (described separately in Section 11) computed before and after matching for all covariates, with a threshold of SMD < 0.10 to declare balance.

### Assumptions and limitations

Matching controls only for measured confounders. Private manager information — a player's undisclosed fitness status, scouting intelligence about the opposition, tactical information not captured in public statistics — cannot be matched on. This residual threat is quantified in the Rosenbaum sensitivity analysis. No-replacement matching means each control observation is used at most once; if the control pool is thin in certain regions of propensity score space, some treated observations may receive poor-quality matches despite passing the caliper.

---

## 5. Average Treatment Effect on the Treated (ATT)

### What it is

The **Average Treatment Effect on the Treated (ATT)** answers: for the units that actually received treatment, what was the causal effect of that treatment? This differs from the Average Treatment Effect (ATE), which asks what the effect would be for a randomly selected unit from the entire population.

### Why I used it

The ATT is the correct estimand for this research question. We want to know what happened to teams that actually made tactical substitutions — not what would happen to all matches, including those where substitutions were not made and may not have been appropriate. Propensity score matching is designed to estimate the ATT; the matched control observations form the counterfactual for the treated group specifically.

### How it was applied

The **matched pair difference estimator** was applied directly to the matched pairs:

```
d_i = Y_treated_i − Y_control_i
ATT = mean(d_i)
SE  = std(d_i) / sqrt(n)
```

The standard error uses the paired-difference form, which accounts for within-pair correlation and is the appropriate choice for matched designs. A two-sided paired t-test assessed the null hypothesis that ATT = 0.

**Outcome windows**:

- **Primary**: `goal_diff_next15` — goal differential in the 15 minutes following the matched moment
- **Secondary**: `goal_diff_next30` — restricted to the non-truncated subset where at least 30 minutes remained after the substitution minute

**Truncation handling**: the tactical substitution window (minutes 55–85) and a 30-minute outcome window are mathematically incompatible — a substitution at minute 70 has only 20 minutes remaining. Pairs where either observation's outcome window extended past the match end were excluded from the primary analysis. The 15-minute outcome window was chosen to recover the maximum number of usable pairs while keeping the outcome window within match time for most of the sample.

### Assumptions and limitations

The paired t-test assumes that pair differences are approximately normally distributed. With a sample of over nine thousand pairs, the central limit theorem makes this assumption mild regardless of the underlying distribution. The 15-minute outcome window may not capture delayed effects of substitutions — tactical changes sometimes take multiple minutes to alter team shape and pressing patterns before influencing goal-scoring probability.

---

## 6. Doubly Robust Estimation

### What it is

**Doubly robust estimation** combines propensity score adjustment with outcome regression. The key property, established by Robins and Rotnitzky (1995), is that the estimate is consistent if *either* the propensity model *or* the outcome model is correctly specified — not necessarily both. A single misspecification does not automatically invalidate the estimate.

### Why I used it

The matched pair ATT relies entirely on the propensity model for confounding control. Doubly robust estimation adds a second layer of protection against propensity model misspecification. It also allows player quality and fatigue to enter the outcome model — covariates that could not be included in the propensity model without destroying common support.

**Why player quality and fatigue enter here, not in the propensity model**: control observations — game moments where no substitution occurred — have no substitution player. These variables are structurally null for all control rows, not missing by happenstance. Including them in the propensity model would allow the model to perfectly separate treated from control observations, eliminating overlap and making matching impossible. In the outcome model, they are valid predictors for treated observations and can be set to zero for control observations without distorting propensity estimation.

### How it was applied

OLS regression was fitted on the stacked long-format dataset (treated rows and matched control rows combined):

```
goal_diff_next15 ~ treated
  + score_diff_at_sub + is_home + opponent_points_l4
  + subs_used_before + is_winning + is_losing
  + log1p_player_quality + player_out_minutes_l30
  + C(league) + C(sub_direction)
```

**Standard errors**: HC3 heteroskedasticity-robust standard errors, appropriate given that the outcome variable (goal differential in 15 minutes) is a low-variance discrete measure with a large proportion of zero values.

The **doubly robust ATT** is the coefficient on the `treated` indicator after conditioning on all covariates. Comparison of this estimate to the matched pair ATT from Step 5 tests whether the two modeling approaches converge on the same conclusion.

### Assumptions and limitations

OLS assumes the outcome model is correctly specified up to a linear approximation. Nonlinear relationships between covariates and goal outcomes — for example, threshold effects at specific scorelines — may not be fully captured. The doubly robust property is asymptotic; in finite samples both the propensity and outcome models contribute to the estimate's behavior.

---

## 7. Placebo Test

### What it is

A **placebo test** (also called a falsification test) validates a causal pipeline by checking whether it produces spurious effects where none should exist. It does not test the magnitude of the finding — it tests whether the research design is functioning correctly.

### Why I used it

Propensity score matching pipelines can generate artifacts if the control pool is systematically different from the treated pool in ways the propensity model does not capture — even after matched balance is verified on measured covariates. A clean placebo result, where the pipeline returns an ATT near zero when applied to a fake treatment assignment, provides evidence that the design is not generating spurious signals from structural features of the data.

### How it was applied

- **Placebo treatment**: randomly assigned at minute 35, well inside the first half where tactical substitutions essentially never occur in professional football
- **Assignment**: observations at the placebo minute were randomly split 50/50 into fake treated and fake control groups (seed 42)
- **Pipeline**: identical propensity estimation, matching, and ATT estimation steps applied to the fake treatment labels
- **Expected result**: ATT ≈ 0, p > 0.05 — if no real substitution effect exists at minute 35, the pipeline should detect none

If the placebo ATT is large and significant, the pipeline is producing artifacts rather than measuring real treatment effects.

### Assumptions and limitations

The placebo test validates the absence of gross spurious correlations driven by design structure. It does not rule out all sources of bias. A passing placebo confirms the pipeline is not manufacturing effects from noise, but does not confirm the absence of subtle unmeasured confounding in the main analysis.

---

## 8. Rosenbaum Sensitivity Analysis

### What it is

**Rosenbaum sensitivity analysis** quantifies how robust a causal finding is to unmeasured confounding. It asks: how strong would an unmeasured confounder need to be to explain away the observed result? The output is **Γ (Gamma)** — the factor by which an unmeasured variable would need to increase the odds of receiving treatment to render the finding non-significant, under worst-case assumptions. The methodology is developed in Rosenbaum (2002), *Observational Studies*.

### Why I used it

Propensity score matching controls only for observed covariates. Private manager knowledge — player fitness not captured in public tracking data, opposition scouting, injury information not yet publicly disclosed — cannot be matched on. Rosenbaum bounds provide a formal, citable statement about how sensitive the conclusion is to this irreducible threat. Without this step, any causal claim from observational data lacks quantification of its vulnerability to hidden bias.

### How it was applied

- **Base test**: Wilcoxon signed-rank test on matched pair differences — nonparametric and therefore appropriate for the discrete, low-variance outcome distribution
- **Gamma range**: 1.0 to 3.0, tested at standard increments
- **Computation**: for each Γ, an upper-bound p-value is computed using a normal approximation to the Wilcoxon statistic under worst-case hidden confounding. The critical Γ is the smallest value at which the upper-bound p-value exceeds 0.05

Robustness thresholds:

- **Γ < 1.3** — fragile
- **Γ 1.3–1.5** — moderate robustness
- **Γ 1.5–2.0** — robust
- **Γ > 2.0** — very robust

### Assumptions and limitations

The bound is conservative — it assumes worst-case hidden confounding at every Γ level. Actual sensitivity is often less severe. When the primary ATT is not statistically significant, Γ = 1.00 is the correct and expected output: there is no significant finding for the sensitivity analysis to bound. This is an honest result, not a methodological failure.

---

## 9. Causal Forest and CATE Estimation

### What it is

A **causal forest** extends the random forest to heterogeneous treatment effect estimation. Where propensity score matching produces one average effect, a causal forest produces a separate treatment effect estimate for every observation — the **Conditional Average Treatment Effect (CATE_i = E[Y(1) − Y(0) | X = x_i])**. The method was introduced by Wager and Athey (2018) in the *Journal of the American Statistical Association*.

The distinction from a standard random forest is direct:

- **Random forest** — estimates E[Y | X = x]: the predicted outcome given covariates
- **Causal forest** — estimates E[Y(1) − Y(0) | X = x]: the predicted treatment effect given covariates

### Why I used it

A null average ATT raises a specific question that a single matched estimator cannot answer: is the null uniform across all subgroups, or does it mask large positive and negative effects that cancel to zero? A causal forest produces a distribution of individual treatment effect estimates whose spread and structure reveal whether meaningful heterogeneity exists beneath the average.

### How it was applied

- **Implementation**: `econml.grf.RegressionForest` fitted on matched pair differences: Y = `t_goal_diff_next15 − c_goal_diff_next15`. Using pair differences as the outcome is the correct approach for matched pair data where confounding has already been addressed by matching
- **Hyperparameters**: n_estimators = 2,000; min_samples_leaf = 10; max_features = sqrt; random_state = 42; n_jobs = −1
- **Feature set**: same covariates as the propensity model, plus `t_minute`. Match minute is included here as an **effect modifier** — the causal forest tests whether the size of the treatment effect varies with timing within the substitution window. This is a legitimate use distinct from including minute as a confounder in the propensity model

**Three heterogeneity tests** were applied to assess whether CATE variation reflects real signal or fitted noise:

- **Test A — Quintile calibration**: fit on 70% train, predict on 30% test; group test observations by quintile of predicted CATE; check whether actual outcomes increase monotonically with predicted CATE. A spread of actual means across quintiles greater than 0.10 goal differential indicates real heterogeneity; less than 0.05 indicates noise
- **Test B — Best linear predictor (BLP)**: regress actual pair differences on predicted CATEs on the test set; a significant slope (p < 0.05) means predicted CATE variation tracks actual outcomes
- **Test C — Variance decomposition**: `Var(CATE) / Var(Y)` — quantifies what share of total outcome variance is attributable to treatment effect heterogeneity, assessing practical alongside statistical significance

### Assumptions and limitations

The regression forest on pair differences assumes matching adequately addressed confounding. Residual confounding would bias CATE estimates without detection. Goal differential in 15 minutes is a discrete low-variance measure — the outcome distribution limits how much signal any estimator can extract. The three heterogeneity tests exist specifically to distinguish real treatment effect variation from overfitting.

---

## 10. SHAP Value Analysis

### What it is

**SHAP (SHapley Additive exPlanations)** values decompose a model's prediction for each individual observation into additive contributions from each feature, grounded in Shapley values from cooperative game theory. Lundberg and Lee (2017, NeurIPS) introduced the unified framework connecting Shapley values to modern machine learning models.

For a prediction ŷ_i with global mean ŷ:

```
ŷ_i = ŷ + φ_i1 + φ_i2 + ... + φ_ip
```

where φ_ij is the SHAP value for feature j for observation i.

### Why I used it

The causal forest produces thousands of individual CATE estimates. SHAP identifies which covariates most drive variation in those estimates — answering the question: conditional on the null average effect, what observable factors push individual treatment effect estimates higher or lower? This provides interpretable structure to what would otherwise be an opaque collection of numbers.

**A critical interpretive constraint**: SHAP values in this context explain variation in *CATE estimates*, not causal effects of the features themselves. A high SHAP contribution from player quality means that feature pushes the forest's treatment effect estimate up or down. It does not mean player quality causally affects match outcomes — that would require a separate study with player quality as the treatment variable. This distinction must be maintained in all reporting.

### How it was applied

- **Implementation**: `shap.TreeExplainer` applied to the fitted `RegressionForest` — exact SHAP values for tree-based models, no approximation required
- **Output**: SHAP values matrix (n × p) — one value per observation per feature
- **Visualizations**:
  - *Beeswarm summary plot*: features ranked by mean |SHAP value|, each point one observation, colored by feature value direction
  - *Dependence plots*: top 4 features by mean |SHAP|, showing how SHAP values shift as the feature value changes, with automatic interaction highlighting

### Assumptions and limitations

TreeExplainer produces exact SHAP values for tree ensembles. The interpretive limitation is not computational but conceptual: SHAP values from a model fitted to treatment effect estimates inherit the limitations of those estimates. If the causal forest CATE estimates are noisy (as expected when heterogeneity is modest), SHAP values will explain noise as well as signal. Features with high mean |SHAP| values should be interpreted as drivers of CATE *variation*, not as causal moderators of the treatment effect without further analysis.

---

## 11. Covariate Balance Assessment

### What it is

**Standardized Mean Difference (SMD)** measures the normalized gap between treated and control group means on each covariate. It is the standard diagnostic for propensity score matching quality (Austin 2009):

```
SMD = (mean_T − mean_C) / sqrt((var_T + var_C) / 2)
```

SMD is expressed in standard deviation units, making it comparable across covariates with different scales.

### Why I used it

A propensity score match is only as valid as the balance it achieves. Matching on the scalar propensity score does not guarantee balance on individual covariates — particularly if the propensity model is misspecified or if within-league control pools are thin. SMD before and after matching makes balance empirically verifiable rather than assumed.

### How it was applied

SMD was computed for all covariates before and after matching. The accepted threshold for declaring balance is **SMD < 0.10** (Austin 2009).

Results were visualized using a **love plot**:

- Y-axis: covariates sorted by pre-matching SMD
- X-axis: SMD value
- Pre-matching: open circles
- Post-matching: filled circles
- Reference lines at SMD = 0.10 and SMD = 0.20
- Background shading: green zone (balanced), yellow zone (marginal), red zone (imbalanced)

Two variables — player quality and player fatigue — were excluded from the balance verdict. Their SMDs are structurally large because control rows have no substitution player by design; their values are always zero for control observations. This is not a matching failure; it is a consequence of what the control group represents.

### Assumptions and limitations

SMD < 0.10 is a widely used rule of thumb, not a formal statistical threshold. Small residual imbalances on measured covariates can persist even when all SMDs clear the threshold. The doubly robust estimation step provides an additional layer of adjustment against residual imbalance that matching alone does not fully eliminate.

---

*All analyses were conducted in Python. Core libraries: pandas, numpy, scikit-learn (propensity model, KDTree matching), statsmodels (doubly robust OLS), scipy (Wilcoxon test, Rosenbaum bounds), econml (RegressionForest), shap (SHAP values), matplotlib (all visualizations).*
