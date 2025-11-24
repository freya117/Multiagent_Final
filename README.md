# Multi-Agent Reinforcement Learning: CFR Analysis

A study of Counterfactual Regret Minimization in multi-agent environments, from zero-sum to general-sum games.

## Overview

This project investigates CFR behavior across different game types, focusing on scalability, social welfare, and equilibrium selection in multiplayer settings.

## Experiments

Six sequential experiments testing CFR and CFR+ variants:
- Kuhn Poker (2-player and 3-player)
- Leduc Poker (scalability analysis)
- General-sum games (Goofspiel, Prisoner's Dilemma)
- Coordination games (equilibrium selection)
- Cross-game convergence metrics analysis

## Key Findings

CFR successfully converges in competitive zero-sum games but exhibits limitations in:
- Cooperation dilemmas (finds selfish equilibria)
- Coordination problems (fails to break symmetry)
- Large state spaces (requires abstraction)

Advanced metrics (regret, entropy, stability) reveal that CFR+ outperforms Vanilla CFR by 20-40% in complex games.

## Installation

```bash
pip install open_spiel numpy matplotlib jupyter
jupyter notebook
```

## References

- Brown & Sandholm (2019) - Pluribus
- Zinkevich et al. (2007) - CFR Algorithm

**Author**: [freya117](https://github.com/freya117)  
**Course**: MIT 6.S890 Topics in Multiagent Learning
