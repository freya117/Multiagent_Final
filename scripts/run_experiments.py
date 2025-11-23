import pyspiel
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from open_spiel.python.algorithms import exploitability

class CFRSolver:
    """
    Counterfactual Regret Minimization solver for n-player games.
    """
    def __init__(self, game):
        self.game = game
        self.num_players = game.num_players()
        
        # Regret and strategy tables indexed by information set string
        self.regret_sum = defaultdict(lambda: np.zeros(self.game.num_distinct_actions()))
        self.strategy_sum = defaultdict(lambda: np.zeros(self.game.num_distinct_actions()))
        self.info_set_map = {}  # Maps info state string to action list
        
        print(f"=== CFR Solver Initialized ===")
        print(f"Game: {game.get_type().long_name}")
        print(f"Players: {self.num_players}")
        print(f"Max actions: {game.num_distinct_actions()}")
        print()
    
    def train(self, num_iterations, log_every=100):
        """
        Train CFR for a specified number of iterations.
        Returns a list of NashConv values over time.
        """
        print(f"=== Starting CFR Training for {num_iterations} iterations ===\n")
        
        nash_conv_history = []
        
        for iteration in range(num_iterations):
            # Run CFR from the root for each player
            for player in range(self.num_players):
                state = self.game.new_initial_state()
                self.cfr(state, player, iteration)
            
            # Log progress and compute NashConv
            if (iteration + 1) % log_every == 0:
                current_policy = self.get_policy()
                nash_conv = exploitability.nash_conv(self.game, current_policy)
                nash_conv_history.append(nash_conv)
                if (iteration + 1) % (log_every * 10) == 0:
                    print(f"Iteration {iteration + 1}/{num_iterations} - NashConv: {nash_conv:.6f}")
        
        print(f"\n=== Training Complete ===")
        return nash_conv_history
    
    def cfr(self, state, player, iteration):
        """
        Counterfactual Regret Minimization recursive algorithm.
        """
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
            # Create list of (action, prob) tuples
            action_probs = []
            for i, action in enumerate(legal_actions):
                action_probs.append((int(action), float(avg_strat[i])))
            policy_dict[info_state] = action_probs
            
        return pyspiel.TabularPolicy(policy_dict)

if __name__ == "__main__":
    print("-" * 50)
    print("RUNNING 2-PLAYER KUHN POKER EXPERIMENT")
    print("-" * 50)
    game_2p = pyspiel.load_game("kuhn_poker")
    solver_2p = CFRSolver(game_2p)
    nash_conv_2p = solver_2p.train(10000, log_every=1000)
    print(f"Final NashConv (2P): {nash_conv_2p[-1]:.6f}")
    
    print("\n" + "-" * 50)
    print("RUNNING 3-PLAYER KUHN POKER EXPERIMENT")
    print("-" * 50)
    game_3p = pyspiel.load_game("kuhn_poker", {"players": 3})
    solver_3p = CFRSolver(game_3p)
    # 3-player is harder, so maybe more iterations or just see the trend
    nash_conv_3p = solver_3p.train(50000, log_every=5000)
    print(f"Final NashConv (3P): {nash_conv_3p[-1]:.6f}")
