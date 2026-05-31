# Project Walkthrough

## The Project in One Paragraph

This project asked whether football managers who substitute earlier in a match produce better outcomes than managers who substitute later — after accounting for the situation they were already in. The research question sounds simple, but answering it honestly is not. Teams that are losing substitute earlier; teams that are winning substitute later. A raw comparison would just be measuring the effect of falling behind, not the effect of timing. Across five major European leagues and two seasons, the project found no average causal effect of substitution timing on goal differential in the 15 minutes following a substitution. The strongest predictor of post-substitution outcomes was the quality of the player coming on — not when they came on.

---

## The Core Challenge — Why This Is Hard

Imagine two managers. One substitutes at minute 62 because his team is losing 1-0 and he needs a goal. The other substitutes at minute 78 because his team is winning and wants fresh legs to protect the lead. If you just compare what happened after each substitution, the losing manager looks worse on average. But that is because of the situation, not the decision.

This is the problem every phase of the project was built around. The technical term for this is confounding — when the thing that drives the decision (game state) is also the thing that affects the outcome. Ignoring it produces a biased answer.

To get an unbiased answer, you need to compare substitutions to comparable non-substitution moments — situations with the same score, same opponent quality, same home/away status. Only then can any difference be attributed to the substitution itself.

Every design choice in this project — what data to collect, how to classify substitutions, how to build the control group, which variables to match on — traces back to this one challenge.

---

## Phase by Phase

### Phase 1 — Collecting the Data

The goal of data collection was not just to gather match events. It was to gather enough context around every substitution to reconstruct what a manager was facing when they made the decision.

The original plan included shot-level expected goals (xG) — a measure of how many goals a team should have scored based on the quality of their chances. xG would have been a better outcome variable than raw goals, and a better measure of game state than the scoreline alone. The problem was practical: the Python library used to scrape FBref (soccerdata) did not expose the xG columns. The StatsBomb dataset had xG but only for 66 matches — far too few for a study of this scale. After investigating both options, the decision was to use goal differential as the outcome instead. It is noisier than xG but unbiased, and it is consistent with how the prior literature approached the same measurement problem.

Beyond substitution events, three other data sources were needed. Match events — goals and yellow cards by minute — were required to reconstruct the exact scoreline at any moment and to identify which players had been booked. Player appearance records across prior matches were needed to build a fatigue proxy for the player being removed. Season statistics for each substitute were needed to measure the quality of the player coming on.

Joining these sources together surfaced a team name mismatch problem. FBref uses "Manchester Utd" where Understat uses "Manchester United." Fuzzy string matching resolved most cases, but every automated match was spot-checked to avoid silently linking the wrong teams.

The final collection covered 3,585 matches, 31,545 substitution events, player minutes across the season, match-level goal and card events, and season statistics for every player who featured as a substitute.

---

### Phase 2 — Building the Analytical Dataset

The raw data was match events. The analysis needed one row per game-moment observation, with game state, player context, and outcome variables all attached. That transformation took the most engineering work of any phase.

The first problem was substitution classification. Not every substitution in the data was a tactical timing decision. A manager who substitutes at minute 28 because a player pulled a hamstring is not choosing when to act — he has no choice. Including forced substitutions in a study of timing decisions would contaminate the treatment.

FBref does not label substitutions by reason. A set of proxy rules was built: first substitutions before minute 60 were flagged as likely injury-driven; any substitution where the outgoing player held a yellow card was flagged as disciplinary; matches involving a red card before the substitution were removed entirely. These are proxies, not ground truth. Some genuine tactical decisions get removed. Some injury subs slip through. That imperfection was documented and accepted as the best approach available without manual labelling.

The second problem was constructing the control group. For every real substitution, the analysis needed comparison points — game moments where no substitution occurred but the situation looked similar. These do not exist ready-made in any dataset. For each match, every minute between 50 and 90 where no substitution happened was converted into a row with all the same game state information as a real substitution row. This produced roughly 200,000 control observations to compare against 31,000 treated ones.

Building the outcome variable required scanning forward from each observation minute and counting goals for and against in the next 15 minutes. Own goals had to be attributed to the conceding team rather than the scoring team — a small fix that would have introduced systematic error if missed.

The result was a single dataset of 237,000 rows and 46 columns, ready for the matching phase.

---

### EDA — Looking Before Modeling

Before any model was built, the data was examined carefully. This was not optional. Several decisions made during EDA changed the design of every subsequent phase.

The substitution timing distribution showed that 19% of tactical subs fell outside the originally planned 50-90 minute control window. The window was expanded to capture them.

The correlation matrix flagged that substitution slot number and the actual minute of substitution moved together strongly. Including both in the matching model would cause instability. Slot number was dropped; the count of substitutions already made was kept instead.

The distribution of substitute player quality was heavily right-skewed — a small number of elite attackers extended the tail far to the right. A log transformation was applied to compress that tail before modeling.

Mean substitution timing varied by about 2.5 minutes across the five leagues. That confirmed an intuition: Serie A and Bundesliga managers tend to substitute at systematically different moments than Premier League managers. This meant matching should happen within leagues only, never across them.

---

### Phase 3 — Finding Comparable Situations

Matching was the process of finding, for every real substitution, a game moment from another match where no substitution occurred but the situation looked nearly identical. The goal was to isolate the timing decision from everything else that was happening.

The first design decision was to match only within leagues. The EDA had confirmed that substitution behavior differs across leagues. A Premier League moment and a Serie A moment are not truly comparable situations, even if the scoreline is the same.

The second decision was to set a maximum match distance — a caliper. If the closest available control observation was still too different from a treated observation, the treated observation was left unmatched rather than paired with a poor comparison. A poor match would introduce exactly the kind of bias the whole process was trying to remove.

Player quality and fatigue were excluded from the matching model entirely. This was a forced choice. Control observations — minutes where no substitution occurred — have no substitution player. Those variables are structurally undefined for control rows. Including them would have made the two groups perfectly distinguishable, collapsing the overlap needed for matching to work. The decision was to exclude them from matching and bring them into the outcome model in Phase 4.

After matching was complete, every covariate was checked to confirm the two groups were genuinely similar. The love plot — a chart showing the gap between groups on each variable before and after matching — confirmed balance on all six measured covariates. The result was 20,762 matched pairs, with essentially a 100% match rate across all five leagues.

---

### Phase 4 — Measuring the Effect

The plan was to measure what happened in the 30 minutes after each substitution. The problem revealed itself when the filter was applied.

Most substitutions happen between minutes 55 and 85. A substitution at minute 70 only has 20 minutes of match remaining. At minute 78, only 12 minutes remain. When pairs were filtered to those where both observations had a full 30-minute window ahead, 96% of the matched pairs were excluded. What remained was 755 pairs from 20,762 — a tiny, unrepresentative sliver of the data that contained only early substitutions.

The fix was to switch the primary outcome to 15 minutes. A substitution at minute 75 still has a full 15-minute window ahead. That change recovered 43.7% of the matched pairs — around 9,000 pairs — which was enough for a reliable analysis. The 30-minute analysis was preserved as a secondary check on the 755 surviving pairs.

Four validation checks were run before reporting any result. The first ran the entire pipeline on fake treatment assignments in the first half — where tactical subs essentially never occur — to confirm the pipeline would not produce spurious effects. It found nothing. The second used a second estimation method that incorporated player quality and fatigue into the outcome model directly, providing protection against any misspecification in the matching model. The third ran a formal sensitivity analysis to bound how large an unmeasured factor would need to be to overturn the finding. The fourth confirmed that the balance achieved in Phase 3 held in the 9,000-pair subsample used for the primary analysis.

---

### Phase 5 — Asking Whether the Average Hides Something

A null average effect leaves one important question open. What if substituting early is genuinely helpful in some situations and harmful in others — and those effects happen to cancel out across the full sample? A single average number would never reveal that.

Phase 5 used a causal forest to estimate a separate treatment effect for every individual matched pair, rather than one average across all of them. Three tests then assessed whether the variation in those individual estimates reflected real signal or just noise in a discrete, low-variance outcome. For the specific methods and why they were chosen, see [docs/methodology.md](methodology.md).

---

## What the Project Taught

The biggest practical lesson was about the gap between the data you want and the data you can get. The xG problem arrived in Phase 1 and forced a design change before any modeling began. The truncation wall arrived in Phase 4 and invalidated the original outcome window. Both were solvable — but solving them required understanding why they were problems, not just working around them.

A null result is not a failed project. It is an answer. Prior research found the same thing using a different method. This project confirmed that result using a more rigorous causal design, across five leagues and two seasons. Replicating a null with better methods is a contribution — it narrows the space of plausible conclusions.

The most valuable single step in the entire project was drawing the causal graph before touching the data. Deciding which variables to control for, which to exclude because they sit on the causal pathway, and which would create new problems if conditioned on — all of that came from the graph, not from the data. Without it, the analysis could have produced a result that looked right but was biased in ways that would not be visible in the output.

With shot-level xG for all 3,585 matches, the outcome variable would be substantially cleaner. The truncation problem would still exist — it is geometric, not a data quality issue — but the surviving pairs would measure something more precise than raw goals in a 15-minute window.
