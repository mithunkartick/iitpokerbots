import random
from pkbot.engine import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid

# ------------------------------------------------------------------------
# Global Deck Constants for Monte Carlo Simulation
# ------------------------------------------------------------------------
SUITS = ['s', 'h', 'd', 'c']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
FULL_DECK = [r+s for r in RANKS for s in SUITS]
RANK_VALUES = {r: i for i, r in enumerate(RANKS, 2)}

def fast_heuristic_evaluator(cards):
    """
    A lightweight evaluator for starter bots. 
    Returns a score tuple: (Max Frequency of a Rank, Highest Rank with that Frequency).
    E.g., Three Jacks = (3, 11).
    """
    if not cards:
        return (0, 0)
    ranks = [RANK_VALUES[card[0]] for card in cards]
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    max_count = max(counts.values())
    best_rank = max(r for r, count in counts.items() if count == max_count)
    return (max_count, best_rank)

def monte_carlo_equity(my_cards, board_cards, num_simulations=50):
    """
    Estimates the probability of winning the hand via random rollouts.
    """
    known_cards = set(my_cards + board_cards)
    remaining_deck = [c for c in FULL_DECK if c not in known_cards]
    
    wins = 0
    ties = 0

    for _ in range(num_simulations):
        # Sample remaining cards needed to complete opponent's hand and the board
        drawn_cards = random.sample(remaining_deck, 2 + (5 - len(board_cards)))
        
        opp_cards = drawn_cards[:2]
        simulated_board = board_cards + drawn_cards[2:]
        
        my_score = fast_heuristic_evaluator(my_cards + simulated_board)
        opp_score = fast_heuristic_evaluator(opp_cards + simulated_board)
        
        if my_score > opp_score:
            wins += 1
        elif my_score == opp_score:
            ties += 1

    return (wins + (0.5 * ties)) / num_simulations

# ------------------------------------------------------------------------
# The Main Player Class
# ------------------------------------------------------------------------

class Player:
    def __init__(self):
        self.hands_played = 0

    def on_hand_start(self, game_info, current_state):
        """
        Called at the beginning of every round.
        """
        self.hands_played += 1
        self.time_left = game_info.time_bank

    def get_move(self, game_info, current_state):
        """
        Called whenever the engine needs your action.
        """
        street = current_state.street
        
        # 1. Time Management Safety Net
        if self.time_left < 1.0:
            return self.panic_move(current_state)

        # 2. Handle the Sneak Peek Auction
        if street == 'auction':
            return self.handle_auction(current_state)

        # 3. Handle Betting Streets
        my_cards = current_state.my_hand
        rank1 = RANK_VALUES[my_cards[0][0]]
        rank2 = RANK_VALUES[my_cards[1][0]]
        is_pocket_pair = (rank1 == rank2)
        high_card = max(rank1, rank2)

        if street == 'preflop':
            return self.play_preflop(current_state, is_pocket_pair, high_card)
        else:
            return self.play_postflop(current_state)

    def on_hand_end(self, game_info, current_state):
        """
        Called when the round finishes.
        """
        pass # Add opponent tracking logic here later

    # --- Strategy Methods ---

    def handle_auction(self, current_state):
        pot_size = current_state.pot
        my_chips = current_state.my_chips
        my_cards = current_state.my_hand
        
        rank1 = RANK_VALUES[my_cards[0][0]]
        rank2 = RANK_VALUES[my_cards[1][0]]
        
        # Bid higher (20% of pot) if we need info on marginal hands, lower (5%) if we have strong pairs
        if rank1 == rank2 and rank1 >= 10:
            bid = int(pot_size * 0.05)
        else:
            bid = int(pot_size * 0.20)
            
        safe_bid = min(max(10, bid), my_chips)
        
        if current_state.can_act(ActionBid):
            return ActionBid(safe_bid)
        return self.panic_move(current_state)

    def play_preflop(self, current_state, is_pocket_pair, high_card):
        # Raise with strong hands
        if is_pocket_pair or high_card >= 12:
            if current_state.can_act(ActionRaise):
                min_raise, _ = current_state.raise_bounds
                return ActionRaise(min_raise)
        
        # Call with decent hands
        if high_card >= 10:
            if current_state.can_act(ActionCall):
                return ActionCall()
                
        # Fold or check trash hands
        if current_state.can_act(ActionCheck):
            return ActionCheck()
        return ActionFold()

    def play_postflop(self, current_state):
        my_cards = current_state.my_hand
        board_cards = current_state.board
        cost_to_call = current_state.cost_to_call
        pot_size = current_state.pot
        
        # Scale down simulations if time is running out
        sims = 50 if self.time_left > 5.0 else 15
        win_prob = monte_carlo_equity(my_cards, board_cards, num_simulations=sims)
        
        pot_odds = cost_to_call / (pot_size + cost_to_call) if cost_to_call > 0 else 0.0
            
        # Value bet / Raise if we have a huge advantage
        if win_prob > 0.75:
            if current_state.can_act(ActionRaise):
                min_r, _ = current_state.raise_bounds
                return ActionRaise(min_r)
                
        # Call if it is mathematically profitable
        if win_prob > pot_odds:
            if current_state.can_act(ActionCall):
                return ActionCall()
            if current_state.can_act(ActionCheck):
                return ActionCheck()
                
        # Give up if odds are bad
        if current_state.can_act(ActionCheck):
            return ActionCheck()
        return ActionFold()

    def panic_move(self, current_state):
        if current_state.can_act(ActionCheck):
            return ActionCheck()
        return ActionFold()