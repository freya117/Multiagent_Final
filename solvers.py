import numpy as np
import pyspiel
from collections import defaultdict
from open_spiel.python.algorithms import exploitability


# =============================================================================
# MONTE CARLO CFR (MCCFR) - Chance Sampling Variant
# =============================================================================

class CFRSolver:
    def __init__(self, game):
        self.game = game
        self.regret_sum = defaultdict(lambda: np.zeros(self.game.num_distinct_actions()))
        self.strategy_sum = defaultdict(lambda: np.zeros(self.game.num_distinct_actions()))
        self.info_set_map = {}  # Maps info state string to action list
        
        # Pre-populate info set map (optional, but good for debugging)
        # In a real large game, we'd do this dynamically
        
    def get_strategy(self, info_state, legal_actions):
        regrets = self.regret_sum[info_state]
        # Filter regrets for legal actions only
        # (Assuming regrets are stored by action index)
        
        # Normalize positive regrets
        positive_regrets = np.maximum(regrets, 0)
        sum_positive_regrets = np.sum(positive_regrets)
        
        num_actions = len(legal_actions)
        strategy = np.zeros(num_actions)
        
        if sum_positive_regrets > 0:
            for i, action in enumerate(legal_actions):
                strategy[i] = positive_regrets[action] / sum_positive_regrets
        else:
            strategy = np.ones(num_actions) / num_actions
            
        return strategy

    def get_average_strategy(self, info_state):
        strategy_sum = self.strategy_sum[info_state]
        sum_strategy = np.sum(strategy_sum)
        
        if sum_strategy > 0:
            return strategy_sum / sum_strategy
        else:
            # Default to uniform if no strategy accumulated
            # We need to know legal actions here, which is tricky if we just have the state string
            # For simplicity, we'll return normalized strategy sum or uniform
            return np.ones_like(strategy_sum) / len(strategy_sum)

    def train(self, num_iterations, log_every=1000):
        history = {
            "nash_conv": [],
            "avg_regret": [],
            "strategy_entropy": [],
            "strategy_stability": []
        }
        
        previous_policy_vec = None

        for i in range(num_iterations):
            for player in range(self.game.num_players()):
                # Initial reach probabilities: 1.0 for everyone
                reach_probs = np.ones(self.game.num_players())
                self.cfr(self.game.new_initial_state(), player, i, reach_probs)
            
            if (i + 1) % log_every == 0:
                current_policy = self.get_policy()
                
                # 1. NashConv
                nash_conv = exploitability.nash_conv(self.game, current_policy)
                history["nash_conv"].append(nash_conv)
                
                # 2. Average Regret (Mean positive regret per info state, normalized by iterations)
                total_pos_regret = 0
                count = 0
                for info_state, regrets in self.regret_sum.items():
                     total_pos_regret += np.sum(np.maximum(regrets, 0))
                     count += 1
                # Normalize by iteration count to see convergence
                avg_regret = (total_pos_regret / max(1, count)) / (i + 1)
                history["avg_regret"].append(avg_regret)

                # 3. Strategy Entropy (Measure of randomness)
                total_entropy = 0
                count = 0
                # current_policy.policy_table() returns {state_str: [(action, prob), ...]}
                policy_table = current_policy.policy_table() if callable(current_policy.policy_table) else current_policy.policy_table
                
                for info_state, probs in policy_table.items():
                    p_vec = np.array([p[1] for p in probs])
                    p_vec = p_vec[p_vec > 0] # Avoid log(0)
                    ent = -np.sum(p_vec * np.log(p_vec))
                    total_entropy += ent
                    count += 1
                avg_entropy = total_entropy / max(1, count)
                history["strategy_entropy"].append(avg_entropy)

                # 4. Strategy Stability (L2 distance from previous policy)
                # Flatten policy to vector for comparison
                current_policy_vec = []
                sorted_keys = sorted(policy_table.keys())
                for k in sorted_keys:
                    probs = sorted(policy_table[k], key=lambda x: x[0])
                    current_policy_vec.extend([p[1] for p in probs])
                current_policy_vec = np.array(current_policy_vec)

                if previous_policy_vec is not None:
                    # Ensure vectors are same length (info sets might grow)
                    if len(current_policy_vec) == len(previous_policy_vec):
                        stability = np.linalg.norm(current_policy_vec - previous_policy_vec)
                    else:
                        stability = 0.0 # Info sets changed, skip this step
                    history["strategy_stability"].append(stability)
                else:
                    history["strategy_stability"].append(0.0)
                
                previous_policy_vec = current_policy_vec

                if (i + 1) % (log_every * 10) == 0:
                    print(f"Iteration {i + 1}/{num_iterations} - NashConv: {nash_conv:.6f}")
                    
        return history

    def cfr(self, state, player, iteration, reach_probs):
        if state.is_terminal():
            return state.returns()[player]
        
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = np.random.choice([o[0] for o in outcomes], 
                                    p=[o[1] for o in outcomes])
            state.apply_action(action)
            return self.cfr(state, player, iteration, reach_probs)
        
        current_player = state.current_player()
        info_state = state.information_state_string(current_player)
        legal_actions = state.legal_actions()
        
        # Store legal actions for this info state
        if info_state not in self.info_set_map:
            self.info_set_map[info_state] = legal_actions
            
        num_actions = len(legal_actions)
        
        strategy = self.get_strategy(info_state, legal_actions)
        
        action_values = np.zeros(num_actions)
        
        for i, action in enumerate(legal_actions):
            new_state = state.child(action)
            
            new_reach_probs = reach_probs.copy()
            new_reach_probs[current_player] *= strategy[i]
                
            action_values[i] = self.cfr(new_state, player, iteration, new_reach_probs)
        
        expected_value = np.dot(strategy, action_values)
        
        if current_player == player:
            regrets = action_values - expected_value
            
            # Counterfactual reach probability: product of all OTHER players' reach probs
            # P_{-i} = Prod_{j != i} P_j
            # We can calculate this by dividing total product by P_i (if P_i > 0)
            # Or just loop and multiply. Loop is safer for zeros.
            p_opp = 1.0
            for p in range(self.game.num_players()):
                if p != player:
                    p_opp *= reach_probs[p]
                
            for i, action in enumerate(legal_actions):
                self.regret_sum[info_state][action] += regrets[i] * p_opp
        
        # Update strategy sum for ALL players (standard CFR)
        # Note: In some variants, we only update for the current player
        if current_player == player:
             p_self = reach_probs[player]
                 
             for i, action in enumerate(legal_actions):
                self.strategy_sum[info_state][action] += strategy[i] * p_self
        
        return expected_value

    def get_policy(self):
        policy_dict = {}
        for info_state, legal_actions in self.info_set_map.items():
            avg_strat = self.get_average_strategy(info_state)
            # Map back to action indices
            # avg_strat is indexed by action ID if we initialized it that way, 
            # OR it's indexed by legal_action index.
            # In __init__, we used num_distinct_actions, so it's indexed by action ID.
            
            action_probs = []
            for i, action in enumerate(legal_actions):
                # We need to be careful about indexing. 
                # self.strategy_sum is size [num_distinct_actions]
                # So we just access [action]
                prob = self.strategy_sum[info_state][action]
                action_probs.append((int(action), float(prob)))
            
            # Normalize
            total_prob = sum(p[1] for p in action_probs)
            if total_prob > 0:
                action_probs = [(a, p / total_prob) for a, p in action_probs]
            else:
                action_probs = [(a, 1.0 / len(legal_actions)) for a, p in action_probs]
                
            policy_dict[info_state] = action_probs
            
        return pyspiel.TabularPolicy(policy_dict)


class FullWidthCFRSolver:
    """
    Textbook-accurate Full-Width CFR (no Monte Carlo sampling).

    This implements the exact CFR algorithm from the lecture notes:
    1. NextStrategy(): Ask each decision point's regret minimizer for its strategy
    2. ObserveUtility(): Compute counterfactual utilities and feed back to regret minimizers

    Key difference from MCCFR: At chance nodes, we compute the FULL EXPECTATION
    over all outcomes, not a single sample.
    """

    def __init__(self, game):
        self.game = game
        self.regret_sum = defaultdict(lambda: np.zeros(self.game.num_distinct_actions()))
        self.strategy_sum = defaultdict(lambda: np.zeros(self.game.num_distinct_actions()))
        self.info_set_map = {}

    def get_strategy(self, info_state, legal_actions):
        """
        NextStrategy() for a single decision point.
        Uses regret matching: normalize positive regrets to get probability distribution.
        """
        regrets = self.regret_sum[info_state]
        positive_regrets = np.maximum(regrets, 0)
        sum_positive_regrets = np.sum(positive_regrets)

        num_actions = len(legal_actions)
        strategy = np.zeros(num_actions)

        if sum_positive_regrets > 0:
            for i, action in enumerate(legal_actions):
                strategy[i] = positive_regrets[action] / sum_positive_regrets
        else:
            # Uniform if no positive regrets
            strategy = np.ones(num_actions) / num_actions

        return strategy

    def train(self, num_iterations, log_every=1000):
        """
        Main training loop following textbook CFR:
        For each iteration:
            1. Build current strategy (NextStrategy)
            2. Compute counterfactual values (ObserveUtility)
            3. Update regrets
        """
        history = {
            "nash_conv": [],
            "avg_regret": [],
            "strategy_entropy": [],
            "strategy_stability": []
        }

        previous_policy_vec = None

        for iteration in range(num_iterations):
            # For each player, traverse the full game tree
            for player in range(self.game.num_players()):
                self.cfr_full_traversal(
                    self.game.new_initial_state(),
                    player,
                    iteration,
                    reach_probs=np.ones(self.game.num_players()),
                    chance_reach=1.0
                )

            # Logging
            if (iteration + 1) % log_every == 0:
                current_policy = self.get_policy()

                # NashConv
                nash_conv = exploitability.nash_conv(self.game, current_policy)
                history["nash_conv"].append(nash_conv)

                # Average Regret
                total_pos_regret = 0
                count = 0
                for info_state, regrets in self.regret_sum.items():
                    total_pos_regret += np.sum(np.maximum(regrets, 0))
                    count += 1
                avg_regret = (total_pos_regret / max(1, count)) / (iteration + 1)
                history["avg_regret"].append(avg_regret)

                # Strategy Entropy
                total_entropy = 0
                count = 0
                policy_table = current_policy.policy_table()

                for info_state, probs in policy_table.items():
                    p_vec = np.array([p[1] for p in probs])
                    p_vec = p_vec[p_vec > 0]
                    if len(p_vec) > 0:
                        ent = -np.sum(p_vec * np.log(p_vec))
                        total_entropy += ent
                    count += 1
                avg_entropy = total_entropy / max(1, count)
                history["strategy_entropy"].append(avg_entropy)

                # Strategy Stability
                current_policy_vec = []
                sorted_keys = sorted(policy_table.keys())
                for k in sorted_keys:
                    probs = sorted(policy_table[k], key=lambda x: x[0])
                    current_policy_vec.extend([p[1] for p in probs])
                current_policy_vec = np.array(current_policy_vec)

                if previous_policy_vec is not None and len(current_policy_vec) == len(previous_policy_vec):
                    stability = np.linalg.norm(current_policy_vec - previous_policy_vec)
                else:
                    stability = 0.0
                history["strategy_stability"].append(stability)

                previous_policy_vec = current_policy_vec

                if (iteration + 1) % (log_every * 10) == 0:
                    print(f"Iteration {iteration + 1}/{num_iterations} - NashConv: {nash_conv:.6f}")

        return history

    def cfr_full_traversal(self, state, player, iteration, reach_probs, chance_reach):
        """
        Full-width CFR traversal (textbook accurate).

        Key difference from MCCFR: At chance nodes, compute EXPECTATION over all outcomes
        instead of sampling a single outcome.

        Returns: Expected utility for the given player from this state
        """
        # Terminal node: return utility
        if state.is_terminal():
            return state.returns()[player]

        # Chance node: compute FULL expectation (not sampled!)
        if state.is_chance_node():
            expected_value = 0.0
            for action, prob in state.chance_outcomes():
                child_state = state.child(action)
                # Multiply chance_reach by this outcome's probability
                child_value = self.cfr_full_traversal(
                    child_state, player, iteration, reach_probs, chance_reach * prob
                )
                expected_value += prob * child_value
            return expected_value

        # Decision node
        current_player = state.current_player()
        info_state = state.information_state_string(current_player)
        legal_actions = state.legal_actions()

        # Store legal actions for this info state
        if info_state not in self.info_set_map:
            self.info_set_map[info_state] = legal_actions

        num_actions = len(legal_actions)

        # Step 1: Get current strategy from regret minimizer (NextStrategy)
        strategy = self.get_strategy(info_state, legal_actions)

        # Step 2: Compute counterfactual values for each action
        action_values = np.zeros(num_actions)

        for i, action in enumerate(legal_actions):
            child_state = state.child(action)

            # Update reach probabilities
            new_reach_probs = reach_probs.copy()
            new_reach_probs[current_player] *= strategy[i]

            # Recurse
            action_values[i] = self.cfr_full_traversal(
                child_state, player, iteration, new_reach_probs, chance_reach
            )

        # Expected value under current strategy
        expected_value = np.dot(strategy, action_values)

        # Step 3: Update regrets (ObserveUtility) - only for the traversing player
        if current_player == player:
            # Counterfactual regrets = action_value - expected_value
            # Weighted by opponent reach probability (and chance reach)
            regrets = action_values - expected_value

            # Counterfactual reach: product of all OTHER players' reach * chance reach
            cf_reach = chance_reach
            for p in range(self.game.num_players()):
                if p != player:
                    cf_reach *= reach_probs[p]

            # Update cumulative regrets
            for i, action in enumerate(legal_actions):
                self.regret_sum[info_state][action] += regrets[i] * cf_reach

        # Step 4: Update strategy sum (for computing average strategy)
        # Weight by player's own reach probability
        if current_player == player:
            player_reach = reach_probs[player] * chance_reach
            for i, action in enumerate(legal_actions):
                self.strategy_sum[info_state][action] += strategy[i] * player_reach

        return expected_value

    def get_policy(self):
        """Convert accumulated strategy sums to a TabularPolicy."""
        policy_dict = {}
        for info_state, legal_actions in self.info_set_map.items():
            action_probs = []
            for action in legal_actions:
                prob = self.strategy_sum[info_state][action]
                action_probs.append((int(action), float(prob)))

            # Normalize
            total_prob = sum(p[1] for p in action_probs)
            if total_prob > 0:
                action_probs = [(a, p / total_prob) for a, p in action_probs]
            else:
                action_probs = [(a, 1.0 / len(legal_actions)) for a, p in action_probs]

            policy_dict[info_state] = action_probs

        return pyspiel.TabularPolicy(policy_dict)


class FullWidthCFRPlusSolver(FullWidthCFRSolver):
    """
    Textbook-accurate Full-Width CFR+ (no Monte Carlo sampling).

    CFR+ improvements over vanilla CFR:
    1. Regret flooring: Clamp cumulative regrets to 0 after each update
    2. Linear averaging: Weight strategies by iteration number (later = more weight)
    """

    def __init__(self, game):
        super().__init__(game)
        self.linear_strategy_sum = defaultdict(lambda: np.zeros(self.game.num_distinct_actions()))

    def cfr_full_traversal(self, state, player, iteration, reach_probs, chance_reach):
        """
        Full-width CFR+ traversal with regret flooring and linear averaging.
        """
        # Terminal node
        if state.is_terminal():
            return state.returns()[player]

        # Chance node: full expectation
        if state.is_chance_node():
            expected_value = 0.0
            for action, prob in state.chance_outcomes():
                child_state = state.child(action)
                child_value = self.cfr_full_traversal(
                    child_state, player, iteration, reach_probs, chance_reach * prob
                )
                expected_value += prob * child_value
            return expected_value

        # Decision node
        current_player = state.current_player()
        info_state = state.information_state_string(current_player)
        legal_actions = state.legal_actions()

        if info_state not in self.info_set_map:
            self.info_set_map[info_state] = legal_actions

        num_actions = len(legal_actions)

        # Get strategy (same as vanilla - regret matching with floored regrets)
        strategy = self.get_strategy(info_state, legal_actions)

        # Compute action values
        action_values = np.zeros(num_actions)

        for i, action in enumerate(legal_actions):
            child_state = state.child(action)
            new_reach_probs = reach_probs.copy()
            new_reach_probs[current_player] *= strategy[i]
            action_values[i] = self.cfr_full_traversal(
                child_state, player, iteration, new_reach_probs, chance_reach
            )

        expected_value = np.dot(strategy, action_values)

        # Update regrets with FLOORING (CFR+ key feature #1)
        if current_player == player:
            regrets = action_values - expected_value

            cf_reach = chance_reach
            for p in range(self.game.num_players()):
                if p != player:
                    cf_reach *= reach_probs[p]

            for i, action in enumerate(legal_actions):
                # CFR+: Add regret, then floor at 0
                self.regret_sum[info_state][action] = max(
                    0,
                    self.regret_sum[info_state][action] + regrets[i] * cf_reach
                )

        # Update strategy sum with LINEAR WEIGHTING (CFR+ key feature #2)
        if current_player == player:
            weight = iteration + 1  # Linear weight: iteration 0 -> weight 1, etc.
            player_reach = reach_probs[player] * chance_reach
            for i, action in enumerate(legal_actions):
                self.linear_strategy_sum[info_state][action] += weight * strategy[i] * player_reach

        return expected_value

    def get_policy(self):
        """Use linear_strategy_sum for CFR+."""
        policy_dict = {}
        for info_state, legal_actions in self.info_set_map.items():
            action_probs = []
            for action in legal_actions:
                prob = self.linear_strategy_sum[info_state][action]
                action_probs.append((int(action), float(prob)))

            total_prob = sum(p[1] for p in action_probs)
            if total_prob > 0:
                action_probs = [(a, p / total_prob) for a, p in action_probs]
            else:
                action_probs = [(a, 1.0 / len(legal_actions)) for a, p in action_probs]

            policy_dict[info_state] = action_probs

        return pyspiel.TabularPolicy(policy_dict)


class CFRPlusSolver(CFRSolver):
    def __init__(self, game):
        super().__init__(game)
        self.linear_strategy_sum = defaultdict(lambda: np.zeros(self.game.num_distinct_actions()))

    def cfr(self, state, player, iteration, reach_probs):
        if state.is_terminal():
            return state.returns()[player]
        
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = np.random.choice([o[0] for o in outcomes], 
                                    p=[o[1] for o in outcomes])
            state.apply_action(action)
            return self.cfr(state, player, iteration, reach_probs)
        
        current_player = state.current_player()
        info_state = state.information_state_string(current_player)
        legal_actions = state.legal_actions()
        
        if info_state not in self.info_set_map:
            self.info_set_map[info_state] = legal_actions
        
        # CFR+ uses Regret Matching+ (floor regrets at 0)
        # We can reuse get_strategy because it already floors at 0
        strategy = self.get_strategy(info_state, legal_actions)
        
        action_values = np.zeros(len(legal_actions))
        
        for i, action in enumerate(legal_actions):
            new_state = state.child(action)
            
            new_reach_probs = reach_probs.copy()
            new_reach_probs[current_player] *= strategy[i]

            action_values[i] = self.cfr(new_state, player, iteration, new_reach_probs)
        
        expected_value = np.dot(strategy, action_values)
        
        if current_player == player:
            regrets = action_values - expected_value
            
            p_opp = 1.0
            for p in range(self.game.num_players()):
                if p != player:
                    p_opp *= reach_probs[p]

            for i, action in enumerate(legal_actions):
                # CFR+: Add regret, then floor at 0
                self.regret_sum[info_state][action] = max(0, self.regret_sum[info_state][action] + regrets[i] * p_opp)
        
        # Linear Averaging: Weight = iteration + 1 (since iteration starts at 0)
        # Update strategy sum for the current player
        if current_player == player:
            weight = iteration + 1
            p_self = reach_probs[player]
            
            for i, action in enumerate(legal_actions):
                self.linear_strategy_sum[info_state][action] += weight * strategy[i] * p_self
        
        return expected_value

    def get_average_strategy(self, info_state):
        # Override to use linear_strategy_sum
        # This is just a helper, the real work is in get_policy
        return self.linear_strategy_sum[info_state]

    def get_policy(self):
        policy_dict = {}
        for info_state, legal_actions in self.info_set_map.items():
            # Use linear_strategy_sum
            action_probs = []
            for i, action in enumerate(legal_actions):
                prob = self.linear_strategy_sum[info_state][action]
                action_probs.append((int(action), float(prob)))
            
            # Normalize
            total_prob = sum(p[1] for p in action_probs)
            if total_prob > 0:
                action_probs = [(a, p / total_prob) for a, p in action_probs]
            else:
                action_probs = [(a, 1.0 / len(legal_actions)) for a, p in action_probs]
                
            policy_dict[info_state] = action_probs
            
        return pyspiel.TabularPolicy(policy_dict)
