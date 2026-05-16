# Phase 3 Results: Propensity Score Matching

**Research question:** Do managers who substitute earlier (minutes 55–70) produce better post-substitution outcomes than managers who substitute later (minutes 71–85), after controlling for the game state at the moment of the decision?

**Data:** Big 5 European leagues — Premier League, La Liga, Bundesliga, Serie A, Ligue 1 — across the 2022–23 and 2023–24 seasons.

---

## What Phase 3 Does

Substitution timing is not random. Managers substitute earlier when losing and later when winning, which means a straight comparison between early and late subs would just be measuring the effect of being behind on the scoreline — not the manager's decision. Propensity score matching fixes this by finding, for every substitution that actually happened, a game moment from the same league where no substitution occurred but the situation on the pitch was nearly identical: same scoreline direction, similar recent opponent quality, same number of substitutions already used, and same home/away status. Once those pairs are formed, any difference in what happens next is attributable to the substitution itself rather than the circumstances around it.

---

## Sample Flow

We started with 31,545 treated rows (actual substitutions) and 205,822 control rows (game moments where no sub occurred). From there, 481 matches involving red cards were removed entirely — a red card forces an early substitution and would contaminate the timing comparison. That cut took the treated count to around 27,000. Then injury and disciplinary substitutions were filtered out because those are not voluntary tactical choices. The final analysis sample is **20,768 tactical substitutions** paired against a control pool of **178,161 no-sub moments** — roughly 8.6 control rows available per treated observation.

---

## Propensity Model

A logistic regression was fitted to predict: *given the current game state, how likely is a substitution to happen at this moment?* Each observation gets a score between 0 and 1 — its propensity score.

The strongest predictor by a clear margin is whether the team is losing. Losing teams substitute more aggressively at any given minute, which confirms the core confounding concern. Winning margin (score difference) is the second most important — bigger lead, later substitution. League effects show that Serie A and Bundesliga managers are slightly more likely to substitute at any given moment than Premier League managers. Home/away status and substitution slot context contribute smaller signals.

The model's AUC came in at 0.54, which sounds low but is actually desirable here. Game state alone should not perfectly predict who substitutes — if it did, there would be no overlap between the groups and matching would be impossible. A modest AUC means the two groups are similar enough that we can find real comparisons.

One thing worth noting: player-specific features like who came on, their quality rating, and fatigue load were intentionally left out of the propensity model. Those variables only exist for actual substitutions — there is no "player coming on" for a control game moment, because no sub happened. Including them would make the model predict treatment perfectly (AUC = 1.0), which destroys the overlap the matching depends on.

---

## Common Support

Before matching, we check that the treated and control propensity score distributions actually overlap — if they live in completely different ranges, there are no valid matches.

![PS Distribution](ps_distribution.png)

The plot above shows both groups in every league. Purple is substitutions, green is no-sub moments. In all five leagues, the two distributions sit on top of each other across the same narrow range (roughly 0.06 to 0.16). This is the overlap we need. The spiky shape comes from the binary variables — when a team goes from level to losing, the propensity score jumps by a fixed amount, creating clusters at specific values rather than a smooth curve.

Across all five leagues, 99.9–100% of treated observations fall inside the overlap region, meaning almost every substitution has at least one valid control match available.

---

## Matching Results

Each substitution was matched to its nearest no-sub game moment within the same league, using a caliper of 0.026 (following Austin 2011 — 0.2 times the standard deviation of the logit-transformed propensity score). Each control observation can only be used once.

The matching rate was essentially perfect. Premier League, La Liga, and Ligue 1 matched 100% of treated observations. Bundesliga matched 99.9% (2 unmatched) and Serie A matched 99.9% (4 unmatched). Across all leagues the result is **20,762 matched pairs** from 20,768 treated observations. The mean propensity score distance within pairs is 0.0000 — the matched control moments had virtually the same probability of a substitution as the treated moments that actually received one.

---

## Balance Assessment

After matching, the check is whether the two groups are genuinely similar on each covariate. The standard metric is Standardized Mean Difference (SMD) — the gap between group means expressed in standard deviation units. An SMD below 0.10 is the accepted threshold for balance.

![Love Plot](love_plot.png)

The love plot above shows each variable as a row. The hollow red circle is the gap before matching; the filled green dot is the gap after. Every green dot sits well inside the green zone, nowhere near the 0.10 threshold.

The variable that needed matching the most was whether the team was losing — before matching, losing-game moments were overrepresented among substitutions (SMD of 0.066). After matching that dropped to 0.010. Number of substitutions already used went from 0.031 to 0.005. Winning status, home advantage, scoreline difference, and opponent quality were already fairly balanced before matching and became even closer after. All six game-state covariates pass. The overall balance verdict is **PASS**.

Two variables — player quality and player fatigue — are reported separately in the raw CSV but deliberately excluded from this verdict. Their gaps are structurally large because control rows have no substitution player by design, so these values are always zero for the control group. That is not a matching failure; it is a property of what the control group represents.

---

## Early Outcome Signal

Looking at the 20,762 matched pairs and filtering to pairs where the 30-minute outcome window was not truncated by the end of the match, the raw numbers are:

- Substitution moments: average goal difference over the next 30 minutes of **+0.033**
- Matched no-sub moments: average goal difference of **−0.036**
- Raw difference: **+0.069 goals**

This is a preliminary signal only, not a formal estimate. It suggests that the 30-minute stretch following a substitution tended to go slightly better for the team than a comparable stretch where no substitution was made. Phase 4 puts proper standard errors on this and runs the formal causal tests.

---

