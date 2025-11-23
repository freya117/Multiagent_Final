import pyspiel
import numpy as np
from collections import defaultdict
from open_spiel.python.algorithms import exploitability

class CFRSolver:
    """
    Vanilla Counterfactual Regret Minimization solver.
    """
    def __init__(self, game):
        self.game = game
        self.num_players = game.num_players()
        self.regret_sum = defaultdict(lambda: np.zeros(self.game.num_distinct_actions()))
        self.strategy_sum = defaultdict(lambda: np.zeros(self.game.num_distinct_actions()))
        self.info_set_map = {}
        
    def train(self, num_iterations, log_every=100):
        nash_conv_history = []
        for iteration in range(num_iterations):
            for player in range(self.num_players):
                state = self.game.new_initial_state()
                self.cfr(state, player, iteration)
            
            if (iteration + 1) % log_every == 0:
                current_policy = self.get_policy()
                nash_conv = exploitability.nash_conv(self.game, current_policy)
                nash_conv_history.append(nash_conv)
        return nash_conv_history
    
    def cfr(self, state, player, iteration):
        if state.is_terminal():
            return state.returns()[player]
        
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = np.random.choice([o[0] for o in outcomes], p=[o[1] for o in outcomes])
            state.apply_action(action)
            return self.cfr(state, player, iteration)
        
        current_player = state.current_player()
        info_state = state.information_state_string(current_player)
        legal_actions = state.legal_actions()
        num_actions = len(legal_actions)
        
        if info_state not in self.info_set_map:
            self.info_set_map[info_state] = legal_actions
        
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
        
        for i, action in enumerate(legal_actions):
            self.strategy_sum[info_state][action] += strategy[i]
        
        return expected_value
    
    def get_strategy(self, info_state, legal_actions):
        regrets = self.regret_sum[info_state]
        positive_regrets = np.maximum(regrets[legal_actions], 0)
        sum_positive_regret = np.sum(positive_regrets)
        if sum_positive_regret > 0:
            return positive_regrets / sum_positive_regret
        else:
            return np.ones(len(legal_actions)) / len(legal_actions)

    def get_average_strategy(self, info_state):
        legal_actions = self.info_set_map[info_state]
        strategy_sum = self.strategy_sum[info_state][legal_actions]
        sum_strategy = np.sum(strategy_sum)
        if sum_strategy > 0:
            return strategy_sum / sum_strategy
        else:
            return np.ones(len(legal_actions)) / len(legal_actions)

    def get_policy(self):
        policy_dict = {}
        for info_state, legal_actions in self.info_set_map.items():
            avg_strat = self.get_average_strategy(info_state)
            action_probs = []
            for i, action in enumerate(legal_actions):
                action_probs.append((int(action), float(avg_strat[i])))
            policy_dict[info_state] = action_probs
        return pyspiel.TabularPolicy(policy_dict)


class CFRPlusSolver(CFRSolver):
    """
    CFR+ Solver:
    1. Regrets are floored at 0 (Regret Matching+).
    2. Average strategy is weighted by iteration number (Linear Averaging).
    """
    def __init__(self, game):
        super().__init__(game)
        # We need to store iteration weight for linear averaging
        self.linear_strategy_sum = defaultdict(lambda: np.zeros(self.game.num_distinct_actions()))

    def cfr(self, state, player, iteration):
        # Override to use linear averaging and CFR+ logic
        if state.is_terminal():
            return state.returns()[player]
        
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = np.random.choice([o[0] for o in outcomes], p=[o[1] for o in outcomes])
            state.apply_action(action)
            return self.cfr(state, player, iteration)
        
        current_player = state.current_player()
        info_state = state.information_state_string(current_player)
        legal_actions = state.legal_actions()
        num_actions = len(legal_actions)
        
        if info_state not in self.info_set_map:
            self.info_set_map[info_state] = legal_actions
        
        # Get strategy using Regret Matching+
        strategy = self.get_strategy(info_state, legal_actions)
        
        action_values = np.zeros(num_actions)
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
        # Note: In standard CFR+, we update the average strategy for BOTH players? 
        # Usually it's updated for the current player, or all players if traversing.
        # Here we traverse for one player, but strategy sum is usually updated for the traverser.
        # Wait, in the vanilla implementation above, I updated strategy_sum for ALL nodes visited.
        # That's technically "External Sampling" style if I'm sampling chance.
        # But here I am sampling chance.
        
        # For CFR+, we typically weight by max(0, iteration - d) where d is a delay.
        # We'll use simple linear weighting: weight = iteration + 1.
        weight = iteration + 1
        for i, action in enumerate(legal_actions):
            self.linear_strategy_sum[info_state][action] += weight * strategy[i]
        
        return expected_value

    def get_average_strategy(self, info_state):
        # Override to use linear_strategy_sum
        legal_actions = self.info_set_map[info_state]
        strategy_sum = self.linear_strategy_sum[info_state][legal_actions]
        sum_strategy = np.sum(strategy_sum)
        if sum_strategy > 0:
            return strategy_sum / sum_strategy
        else:
            return np.ones(len(legal_actions)) / len(legal_actions)
