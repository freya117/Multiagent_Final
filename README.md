# Multi-Agent Reinforcement Learning: CFR Analysis

A comprehensive study of **Counterfactual Regret Minimization (CFR)** in multi-agent environments, from zero-sum games to general-sum settings.

## 📊 Project Overview

This project systematically investigates CFR's behavior across increasingly challenging game-theoretic environments to answer:
1. **Scalability**: How does CFR handle multiplayer complexity?
2. **Social Welfare**: Does CFR find cooperative or selfish equilibria in general-sum games?
3. **Equilibrium Selection**: Can independent learners coordinate when multiple equilibria exist?

## 🎮 Experiments

### Experiment Series (6 Notebooks)

| # | Game | Players | Focus |
|---|------|---------|-------|
| 01 | Kuhn Poker | 2 | Baseline validation |
| 02 | Kuhn Poker | 3 | Multiplayer scaling |
| 03 | Leduc Poker | 2-3 | State space explosion |
| 04 | Goofspiel & Prisoner's Dilemma | 3 | General-sum dynamics |
| 05 | Coordination Game | 3 | Equilibrium selection |
| 06 | **Cross-Game Analysis** | - | **Advanced metrics** |

## 🔬 Key Findings

### ✅ Where CFR Excels
- **Competitive games** (zero-sum, constant-sum)
- **Sequential decision-making** with imperfect information
- CFR+ consistently outperforms Vanilla CFR (20-40% improvement in complex games)

### ❌ Where CFR Fails
- **Cooperation dilemmas**: Prisoner's Dilemma → 67% social welfare loss
- **Coordination problems**: 84.6% miscoordination rate in symmetric equilibria
- **Massive state spaces**: Requires abstraction for practical scalability

## 📈 Advanced Metrics

We introduced three new convergence metrics beyond standard NashConv:

1. **Average Regret**: Measures learning efficiency ($O(1/\sqrt{T})$ decay)
2. **Strategy Entropy**: Quantifies randomness in policies
3. **Strategy Stability**: Tracks policy oscillation ($||\pi_t - \pi_{t-1}||$)

**Result**: CFR+ shows 30-40% faster regret minimization and 20% smoother convergence compared to Vanilla CFR.

## 🛠️ Technical Stack

- **Framework**: OpenSpiel (DeepMind)
- **Algorithms**: Vanilla CFR, CFR+ (Regret Matching+ with Linear Averaging)
- **Languages**: Python 3.8+
- **Dependencies**: `numpy`, `matplotlib`, `pyspiel`

## 📚 Project Structure

```
.
├── 01_kuhn_poker_2p.ipynb          # Baseline experiments
├── 02_kuhn_poker_3p.ipynb          # Multiplayer analysis
├── 03_leduc_poker_scalability.ipynb # Complexity scaling
├── 04_general_sum_games.ipynb      # Social welfare study
├── 05_equilibrium_selection.ipynb  # Coordination failure
├── 06_advanced_metrics.ipynb       # Cross-game convergence analysis
├── solvers.py                      # CFR/CFR+ implementations
└── scripts/
    ├── run_experiments.py          # Batch experiment runner
    └── verify_notebooks.py         # Automated testing
```

## 🚀 Getting Started

### Installation
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install open_spiel numpy matplotlib jupyter
```

### Run Experiments
```bash
jupyter notebook  # Open any of the 01-06 notebooks
# Or run all experiments:
python scripts/run_experiments.py
```

## 📖 Key Insights

### The "Tragedy of the Commons"
In 3-Player Prisoner's Dilemma:
- ✅ Converged to perfect Nash Equilibrium (NashConv = 0.0000)
- ❌ Social welfare = 3.0 (all defect) vs optimal 9.0 (all cooperate)
- **No-regret learning optimizes individual rationality, not group welfare**

### The "Tower of Babel" Problem
In 3-Player Coordination Game:
- 3 equivalent Nash equilibria: (A,A,A), (B,B,B), (C,C,C)
- Independent CFR agents only coordinated **15.4% of the time**
- **Symmetry-breaking requires explicit coordination mechanisms**

## 🔮 Future Work

1. **Team-CFR**: Joint optimization for coordinated agents
2. **Opponent Modeling**: Exploit non-equilibrium play
3. **Deep CFR**: Neural network generalization for massive games

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

Built using **OpenSpiel** (DeepMind) and inspired by research from:
- Brown & Sandholm (2019) - Pluribus: Superhuman AI for Multiplayer Poker
- Zinkevich et al. (2007) - Regret Minimization in Games with Incomplete Information

---

**Author**: [freya117](https://github.com/freya117)  
**Course**: Multi-Agent Systems Final Project
