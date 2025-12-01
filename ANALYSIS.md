# Empirical Analysis of CFR Variants in Multiplayer Games

**Course**: 6.S890 - Algorithmic Game Theory
**Project**: Multi-Agent CFR Analysis
**Research Question**: What can be said about the average strategies of no-regret players in multiplayer games? Do they empirically perform well across a wide range of domains?

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Background and Motivation](#background-and-motivation)
3. [Algorithm Implementations](#algorithm-implementations)
4. [Experimental Setup](#experimental-setup)
5. [Results by Game](#results-by-game)
6. [Cross-Game Analysis](#cross-game-analysis)
7. [Key Finding: The CFR+ Multiplayer Poker Anomaly](#key-finding-the-cfr-multiplayer-poker-anomaly)
8. [Discussion](#discussion)
9. [Conclusions and Future Work](#conclusions-and-future-work)

---

## Executive Summary

This project systematically compares four variants of Counterfactual Regret Minimization (CFR) across five different game environments. Our key finding is an **anomalous behavior of CFR+ in multiplayer poker games**: while CFR+ consistently outperforms vanilla CFR in 2-player games and non-poker domains, it **underperforms** in 3-player poker games (both Kuhn and Leduc).

### Main Results Table

| Game | Players | Winner | CFR+ NashConv | Vanilla NashConv |
|------|---------|--------|---------------|------------------|
| Kuhn Poker | 2 | CFR+ | 0.004 | 0.016 |
| Kuhn Poker | 3 | **Vanilla** | 0.007 | **0.004** |
| Leduc Poker | 2 | CFR+ | 0.12 | 0.38 |
| Leduc Poker | 3 | **Vanilla** | 0.34 | **0.26** |
| Goofspiel | 2 | CFR+ | 0.000000 | 0.000439 |
| Prisoner's Dilemma | 2 | CFR+ | ~0 | ~0 |
| Coordination | 3 | CFR+ | 0.000034 | 0.000463 |

**Key Insight**: CFR+'s regret flooring mechanism, designed to accelerate convergence, appears to interfere with learning in multiplayer poker environments specifically.

---

## Background and Motivation

### The Problem of No-Regret Learning in Multiplayer Games

In two-player zero-sum games, CFR has strong theoretical guarantees:
- The average strategy converges to a Nash Equilibrium
- Convergence rate is O(1/√T) for vanilla CFR, O(1/T) for CFR+

However, in **multiplayer (n > 2) general-sum games**:
- Nash Equilibria may not exist or may not be unique
- CFR guarantees only convergence to a **Coarse Correlated Equilibrium (CCE)**
- The "equilibrium selection problem" arises when multiple equilibria exist

### Research Questions

1. Does CFR+ maintain its convergence advantage in multiplayer settings?
2. Does full-width (deterministic) CFR behave differently from Monte Carlo CFR?
3. How do different game structures affect CFR performance?

---

## Algorithm Implementations

We implemented four CFR variants in `solvers.py`:

### 1. MCCFR (Monte Carlo CFR) - Vanilla

**Chance Node Handling**: Samples ONE outcome per iteration
```python
if state.is_chance_node():
    outcomes = state.chance_outcomes()
    action = np.random.choice([o[0] for o in outcomes], p=[o[1] for o in outcomes])
    state.apply_action(action)
    return self.cfr(state, player, iteration, reach_probs)
```

**Regret Update**: Standard cumulative regrets
```python
self.regret_sum[info_state][action] += regrets[i] * p_opp
```

**Strategy Averaging**: Uniform weights
```python
self.strategy_sum[info_state][action] += strategy[i] * p_self
```

### 2. MCCFR+ (Monte Carlo CFR Plus)

Same chance node handling as MCCFR, but with two key modifications:

**Regret Flooring**: Clamp cumulative regrets to 0
```python
self.regret_sum[info_state][action] = max(0, self.regret_sum[info_state][action] + regrets[i] * p_opp)
```

**Linear Averaging**: Weight strategies by iteration number
```python
weight = iteration + 1
self.linear_strategy_sum[info_state][action] += weight * strategy[i] * p_self
```

### 3. Full-Width CFR (Textbook) - Vanilla

**Chance Node Handling**: Computes FULL EXPECTATION over all outcomes
```python
if state.is_chance_node():
    expected_value = 0.0
    for action, prob in state.chance_outcomes():
        child_state = state.child(action)
        child_value = self.cfr_full_traversal(child_state, player, iteration, reach_probs, chance_reach * prob)
        expected_value += prob * child_value
    return expected_value
```

This eliminates sampling variance entirely, giving cleaner convergence curves.

### 4. Full-Width CFR+ (Textbook Plus)

Combines full-width traversal with CFR+ improvements (regret flooring + linear averaging).

### Theoretical Convergence Rates

| Variant | Convergence Rate | Notes |
|---------|------------------|-------|
| Vanilla CFR | O(1/√T) | Slower but more stable |
| CFR+ | O(1/T) | Faster due to regret flooring |

The O(1/T) rate means CFR+ should reach a given NashConv in approximately √T fewer iterations than vanilla CFR.

---

## Experimental Setup

### Games Tested

| Game | Players | Actions | Chance Nodes | Type |
|------|---------|---------|--------------|------|
| Kuhn Poker (2P) | 2 | Pass/Bet | Card deal | Zero-sum |
| Kuhn Poker (3P) | 3 | Pass/Bet | Card deal | General-sum |
| Leduc Poker (2P) | 2 | Fold/Call/Raise | Card deal + community | Zero-sum |
| Leduc Poker (3P) | 3 | Fold/Call/Raise | Card deal + community | General-sum |
| Goofspiel (2P) | 2 | Card selection | Prize card | Zero-sum |
| Prisoner's Dilemma | 2 | Cooperate/Defect | None | General-sum |
| Coordination | 3 | A/B/C | None | General-sum |

### Metrics

1. **NashConv**: Sum of each player's incentive to deviate from the current strategy
   - NashConv = 0 implies Nash Equilibrium
   - Lower is better

2. **Coordination Rate** (for coordination game): Percentage of simulated games where all players choose the same action

3. **Training Time**: Wall-clock time for comparison

### Implementation Details

- Framework: OpenSpiel (Google DeepMind)
- Iterations: 1000-5000 depending on game size
- Logging: Every 100-500 iterations
- Policy: TabularPolicy for exact representation

---

## Results by Game

### Experiment 1: 2-Player Kuhn Poker (Baseline)

**File**: `01_kuhn_poker_2p.ipynb`

**Results after 5000 iterations**:

| Algorithm | Final NashConv |
|-----------|----------------|
| MCCFR (Vanilla) | 0.016 |
| MCCFR+ | 0.006 |
| Full-Width CFR | 0.016 |
| Full-Width CFR+ | **0.004** |

**Observations**:
- CFR+ variants outperform vanilla variants (as expected)
- Full-Width versions show smoother convergence curves (no sampling noise)
- MCCFR ≈ Full-Width CFR in final NashConv (Monte Carlo variance averages out)
- The O(1/T) vs O(1/√T) difference is clearly visible

**Conclusion**: Results match theoretical expectations. CFR+ is strictly better in 2-player zero-sum games.

---

### Experiment 2: 3-Player Kuhn Poker

**File**: `02_kuhn_poker_3p.ipynb`

**Results after 5000 iterations**:

| Algorithm | Final NashConv |
|-----------|----------------|
| MCCFR (Vanilla) | 0.007 |
| MCCFR+ | 0.014 |
| Full-Width CFR | **0.004** |
| Full-Width CFR+ | 0.007 |

**ANOMALY DETECTED**:
- Full-Width CFR Vanilla (0.004) **beats** Full-Width CFR+ (0.007)
- This is the opposite of theoretical predictions!

**Observations**:
- The pattern flip occurs in both MC and Full-Width versions
- Full-Width Vanilla achieves the best overall performance
- CFR+'s regret flooring appears to hurt rather than help

---

### Experiment 3: Leduc Poker (2P and 3P)

**File**: `03_leduc_poker_scalability.ipynb`

#### 2-Player Leduc (1000 iterations)

| Algorithm | Final NashConv | Time (s) |
|-----------|----------------|----------|
| MCCFR (Vanilla) | 0.58 | 1.83 |
| MCCFR+ | 0.30 | 1.83 |
| Full-Width CFR | 0.38 | 52.68 |
| Full-Width CFR+ | **0.12** | 51.80 |

**Observations**:
- CFR+ wins in 2-player (as expected)
- Full-Width is ~30x slower but achieves ~5x better NashConv per iteration
- Trade-off: For fixed time budget, MCCFR may be competitive

#### 3-Player Leduc (500 iterations)

| Algorithm | Final NashConv | Time (s) |
|-----------|----------------|----------|
| MCCFR (Vanilla) | 0.45 | 3.37 |
| MCCFR+ | 0.51 | 3.27 |
| Full-Width CFR | **0.26** | 298.23 |
| Full-Width CFR+ | 0.34 | 298.36 |

**ANOMALY CONFIRMED**:
- Full-Width Vanilla (0.26) beats Full-Width CFR+ (0.34)
- Pattern matches 3-player Kuhn Poker exactly
- The anomaly is consistent across different poker variants

---

### Experiment 4: General-Sum Games (Goofspiel, Prisoner's Dilemma)

**File**: `04_general_sum_games.ipynb`

#### Goofspiel (1000 iterations)

| Algorithm | Final NashConv |
|-----------|----------------|
| MCCFR (Vanilla) | 0.000669 |
| MCCFR+ | 0.000141 |
| Full-Width CFR | 0.000439 |
| Full-Width CFR+ | **0.000000** |

**Observations**:
- CFR+ wins decisively
- Full-Width CFR+ achieves perfect equilibrium (NashConv = 0)
- **No anomaly** - CFR+ works as expected

#### Prisoner's Dilemma

| Algorithm | Final NashConv |
|-----------|----------------|
| All variants | ~0 |

**Observations**:
- All variants converge to (Defect, Defect) Nash Equilibrium
- No chance nodes, so MC = Full-Width
- CFR+ slightly faster convergence

---

### Experiment 5: Coordination Game

**File**: `05_equilibrium_selection.ipynb`

**Game Structure**:
- 3 players, 3 actions (A, B, C)
- Payoff = 1 if all match, 0 otherwise
- Multiple equilibria: (A,A,A), (B,B,B), (C,C,C)

**Results after 5000 iterations**:

| Algorithm | Final NashConv | Coordination Rate |
|-----------|----------------|-------------------|
| MCCFR (Vanilla) | 0.000463 | 99.90% |
| MCCFR+ | **0.000034** | 100.00% |
| Full-Width CFR | 0.000463 | 100.00% |
| Full-Width CFR+ | **0.000034** | 100.00% |

**Key Observations**:
1. **MC = Full-Width**: Identical results because coordination game has NO chance nodes
2. **CFR+ wins**: ~14x better NashConv than vanilla
3. **Perfect coordination achieved**: All algorithms successfully coordinate (>99.9%)
4. **No anomaly**: CFR+ works as expected in this 3-player game

---

## Cross-Game Analysis

### Pattern Summary

| Game Type | Players | Chance Nodes | Private Info | CFR+ vs Vanilla |
|-----------|---------|--------------|--------------|-----------------|
| Poker | 2 | Yes | Yes | **CFR+ wins** |
| Poker | 3 | Yes | Yes | **Vanilla wins** |
| Goofspiel | 2 | Yes | No | CFR+ wins |
| PD | 2 | No | No | CFR+ wins |
| Coordination | 3 | No | Yes* | CFR+ wins |

*Players don't see others' actions before choosing, but no random chance.

### Identifying the Pattern

The anomaly appears **specifically** in:
- 3+ player games
- WITH chance nodes (card deals)
- WITH hidden private information

It does NOT appear in:
- 2-player games (any type)
- Games without chance nodes (Coordination, PD)
- Games with chance but no hidden info (Goofspiel - prize cards revealed)

---

## Key Finding: The CFR+ Multiplayer Poker Anomaly

### Hypothesis: Regret Flooring Interference

CFR+'s regret flooring (`max(0, regret)`) was designed to prevent "unlearning" good actions. In 2-player zero-sum games, this is always beneficial because:
- If an action was bad, its negative regret should be forgotten
- The opponent's optimal response is fixed (minimax)

However, in **multiplayer poker**:
- Each player's optimal response depends on BOTH other players
- The strategy space is more complex (not a simple saddle point)
- Negative regrets may carry **useful information** about multi-way interactions

### Evidence Supporting This Hypothesis

1. **Pattern specificity**: The anomaly only appears in 3-player poker, not in other 3-player games (Coordination)

2. **Private information matters**: In poker, players have hidden cards. The interaction between 3+ players with private information creates complex strategic dependencies that vanilla CFR handles better.

3. **Chance nodes matter**: The anomaly doesn't appear in games without chance nodes, suggesting the interaction between regret flooring and stochastic information revelation is key.

### Alternative Hypotheses

1. **Linear averaging interference**: The linear weighting might over-weight later strategies before they've stabilized in multiplayer settings

2. **Correlated equilibrium dynamics**: In >2 player games, the equilibrium concept changes. CFR+'s aggressive regret minimization might overshoot the CCE

3. **Implementation artifact**: While unlikely (both MC and Full-Width show the same pattern), we cannot fully rule out subtle bugs

---

## Discussion

### Implications for Practice

1. **Game-dependent algorithm selection**: Practitioners should not blindly apply CFR+ to all games. For multiplayer poker specifically, vanilla CFR may be preferable.

2. **Pluribus context**: The Pluribus poker AI used MCCFR (not CFR+) with additional mechanisms. Our findings suggest this choice may have been empirically motivated.

3. **Scalability trade-offs**:
   - Full-Width CFR is impractical for large games (30x slower in Leduc)
   - But it provides cleaner theoretical insights
   - MCCFR is necessary for real poker but introduces noise

### Limitations

1. **Game coverage**: We tested only 5 game types. More games (especially >3 players) would strengthen conclusions.

2. **Iteration counts**: Limited by computational resources. Longer runs might reveal different asymptotic behavior.

3. **No theoretical proof**: Our finding is empirical. A rigorous theoretical explanation remains future work.

4. **Single random seed**: Results may vary with different initializations (especially for MCCFR).

### Connection to Course Material

This project directly addresses the course theme:
> "What can be said about the average strategies of no-regret players in multiplayer games?"

Our answer: **It depends on the game structure.** The theoretical guarantees of CFR+ do not uniformly transfer to multiplayer settings. The relationship between regret minimization and equilibrium convergence is more nuanced than the 2-player theory suggests.

---

## Conclusions and Future Work

### Main Conclusions

1. **CFR+ is not universally superior**: In 3-player poker games, vanilla CFR outperforms CFR+ by a significant margin.

2. **Game structure matters**: The combination of (a) >2 players, (b) chance nodes, and (c) private information appears to be the trigger for the anomaly.

3. **Full-Width vs Monte Carlo**: Full-Width CFR provides cleaner convergence and better per-iteration performance, but at 30x computational cost. The choice depends on the time budget.

4. **Coordination is achievable**: Despite the "Tower of Babel" problem, CFR successfully coordinates in pure coordination games (but this is a simplified setting).

### Future Work

1. **Characterize the boundary**: Test on 4-player, 5-player games. At what point does the anomaly emerge? Is it gradual or sharp?

2. **Ablation study**: Test regret flooring and linear averaging separately to identify which component causes the anomaly.

3. **Theoretical investigation**: Develop a formal explanation for why negative regrets carry useful information in multiplayer poker.

4. **Modified CFR+**: Design a "soft" regret flooring mechanism that might preserve the benefits of CFR+ while avoiding the multiplayer poker anomaly.

5. **Other poker variants**: Test on Texas Hold'em abstractions to verify the finding scales to realistic poker.

---

## Appendix A: Code Structure

```
Multiagent_Final/
├── solvers.py                    # CFR implementations
│   ├── CFRSolver                 # MCCFR Vanilla
│   ├── CFRPlusSolver             # MCCFR+
│   ├── FullWidthCFRSolver        # Textbook CFR
│   └── FullWidthCFRPlusSolver    # Textbook CFR+
├── 01_kuhn_poker_2p.ipynb        # Experiment 1
├── 02_kuhn_poker_3p.ipynb        # Experiment 2
├── 03_leduc_poker_scalability.ipynb  # Experiment 3
├── 04_general_sum_games.ipynb    # Experiment 4
├── 05_equilibrium_selection.ipynb    # Experiment 5
└── ANALYSIS.md                   # This document
```

## Appendix B: Key Algorithm Differences

| Feature | Vanilla CFR | CFR+ |
|---------|-------------|------|
| Regret storage | Cumulative (can be negative) | Floored at 0 |
| Strategy averaging | Uniform weights | Linear weights (t) |
| Convergence rate | O(1/√T) | O(1/T) |
| Memory of bad actions | Preserved | Forgotten |

## Appendix C: NashConv Definition

NashConv (Nash Convergence) measures distance from Nash Equilibrium:

```
NashConv(σ) = Σᵢ max_{σ'ᵢ} [uᵢ(σ'ᵢ, σ₋ᵢ) - uᵢ(σ)]
```

Where:
- σ is the current strategy profile
- σ'ᵢ is player i's best response to opponents' strategies
- uᵢ is player i's expected utility

NashConv = 0 implies no player can improve by unilateral deviation (Nash Equilibrium).

---

*Document generated as part of 6.S890 Final Project*
