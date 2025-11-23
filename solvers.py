import numpy as np
import pyspiel
from collections import defaultdict
from open_spiel.python.algorithms import exploitability

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
                self.cfr(self.game.new_initial_state(), player, i)
            
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

    def cfr(self, state, player, iteration):
        if state.is_terminal():
            return state.returns()[player]
        
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = np.random.choice([o[0] for o in outcomes], 
                                    p=[o[1] for o in outcomes])
            state.apply_action(action)
            return self.cfr(state, player, iteration)
        
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
            action_values[i] = self.cfr(new_state, player, iteration)
        
        expected_value = np.dot(strategy, action_values)
        
        if current_player == player:
            regrets = action_values - expected_value
            for i, action in enumerate(legal_actions):
                self.regret_sum[info_state][action] += regrets[i]
        
        # Update strategy sum for ALL players (standard CFR)
        # Note: In some variants, we only update for the current player
        if current_player == player:
             for i, action in enumerate(legal_actions):
                self.strategy_sum[info_state][action] += strategy[i]
        
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

class CFRPlusSolver(CFRSolver):
    def __init__(self, game):
        super().__init__(game)
        self.linear_strategy_sum = defaultdict(lambda: np.zeros(self.game.num_distinct_actions()))

    def cfr(self, state, player, iteration):
        if state.is_terminal():
            return state.returns()[player]
        
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = np.random.choice([o[0] for o in outcomes], 
                                    p=[o[1] for o in outcomes])
            state.apply_action(action)
            return self.cfr(state, player, iteration)
        
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
            action_values[i] = self.cfr(new_state, player, iteration)
        
        expected_value = np.dot(strategy, action_values)
        
        if current_player == player:
            regrets = action_values - expected_value
            for i, action in enumerate(legal_actions):
                # CFR+: Add regret, then floor at 0
                self.regret_sum[info_state][action] = max(0, self.regret_sum[info_state][action] + regrets[i])
        
        # Linear Averaging: Weight = iteration + 1 (since iteration starts at 0)
        # Update strategy sum for the current player
        if current_player == player:
            weight = iteration + 1
            for i, action in enumerate(legal_actions):
                self.linear_strategy_sum[info_state][action] += weight * strategy[i]
        
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
