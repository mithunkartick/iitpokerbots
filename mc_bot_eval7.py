import random
import eval7
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

# ------------------------------------------------------------------------
# Core Monte Carlo Engine (eval7 Powered)
# ------------------------------------------------------------------------
# Create this globally once so we don't recalculate it
DECK_STRINGS = [r+s for r in '23456789TJQKA' for s in 'shdc']

def monte_carlo_equity(my_cards_str, board_cards_str, opp_revealed_str, num_simulations=100):
    """
    Hyper-optimized MC Equity Calculator.
    Moves all heavy list operations and deck building OUTSIDE the loop.
    """
    my_cards = [eval7.Card(c) for c in my_cards_str]
    board_cards = [eval7.Card(c) for c in board_cards_str]
    opp_revealed = [eval7.Card(c) for c in opp_revealed_str]
    
    known_cards = set(my_cards + board_cards + opp_revealed)
    
    # Build the remaining deck ONCE
    remaining_deck = [eval7.Card(c) for c in DECK_STRINGS if eval7.Card(c) not in known_cards]
    
    wins = 0
    ties = 0

    cards_needed_board = 5 - len(board_cards)
    cards_needed_opp = 2 - len(opp_revealed)
    total_needed = cards_needed_board + cards_needed_opp

    for _ in range(num_simulations):
        # Sample all required cards in one single optimized C-call
        drawn = random.sample(remaining_deck, total_needed)
        
        sim_board = board_cards + drawn[:cards_needed_board]
        sim_opp = opp_revealed + drawn[cards_needed_board:]
        
        my_val = eval7.evaluate(my_cards + sim_board)
        opp_val = eval7.evaluate(sim_opp + sim_board)
        
        if my_val > opp_val:
            wins += 1
        elif my_val == opp_val:
            ties += 1

    return (wins + (0.5 * ties)) / num_simulations

# ------------------------------------------------------------------------
# The Main Player Class
# ------------------------------------------------------------------------
class Player(BaseBot):
    def __init__(self):
        super().__init__()
        self.hands_played = 0

    def on_hand_start(self, game_info, current_state):
        self.hands_played += 1
        self.time_left = getattr(game_info, 'time_bank', 20.0)

    def on_hand_end(self, game_info, current_state):
        pass 

    def get_move(self, game_info, current_state):
        street = current_state.street.lower()
        valid_actions = current_state.legal_actions
        
        # 1. Time Management Safety Net
        if self.time_left < 1.0:
            return ActionCheck() if ActionCheck in valid_actions else ActionFold()

        # 2. Extract State Context
        my_cards = [str(c) for c in current_state.my_hand]
        board_cards = [str(c) for c in current_state.board]
        opp_revealed = [str(c) for c in getattr(current_state, 'opp_revealed_cards', [])]
        
        # 3. Handle the Sneak Peek Auction
        if ActionBid in valid_actions:
            return self.handle_auction(current_state, my_cards, board_cards, opp_revealed)

        # 4. Calculate True Equity (Tournament Speed)
        sims = 100 if street in ['preflop', 'pre-flop'] else 200
        
        if self.time_left < 5.0: 
            sims = 30  # Hard panic mode to ensure we finish the match
            
        equity = monte_carlo_equity(my_cards, board_cards, opp_revealed, num_simulations=sims)

        # 5. Execute Sizing Strategy
        return self.play_street(current_state, equity)

    # --- Strategy Methods ---

    def handle_auction(self, current_state, my_cards, board_cards, opp_revealed):
        pot_size = current_state.pot
        my_chips = current_state.my_chips
        
        # Run a quick 200-sim equity check just on our hole cards
        base_equity = monte_carlo_equity(my_cards, board_cards, opp_revealed, num_simulations=200)
        
        # If we have a monster, bid decently to protect it. If marginal, bid small.
        if base_equity > 0.65:
            bid = int(pot_size * 0.15)
        elif base_equity > 0.50:
            bid = int(pot_size * 0.08)
        else:
            bid = 0 # Starve the pot if we have trash
            
        max_bid = int(my_chips)
        safe_bid = max(0, min(bid, max_bid))
        return ActionBid(safe_bid)

    def play_street(self, current_state, equity):
        valid_actions = current_state.legal_actions
        cost_to_call = current_state.cost_to_call
        pot_size = current_state.pot
        
        pot_odds = cost_to_call / max(1.0, float(pot_size + cost_to_call))
        
        # Extract Raise Bounds Safely
        min_r = getattr(current_state, 'min_raise', 20)
        max_r = getattr(current_state, 'max_raise', int(current_state.my_chips))
        if hasattr(current_state, 'raise_bounds') and current_state.raise_bounds is not None:
            min_r, max_r = current_state.raise_bounds

        # Tier 1: The Shove (> 90% Equity)
        if equity > 0.90 and ActionRaise in valid_actions:
            return ActionRaise(int(max_r))
            
        # Tier 2: The Value Bet (> 75% Equity)
        if equity > 0.75 and ActionRaise in valid_actions:
            target = int(pot_size * 0.75)
            return ActionRaise(int(max(min_r, min(target, max_r))))
            
        # Tier 3: The Probe/Min-Raise (> 60% Equity)
        if equity > 0.60 and ActionRaise in valid_actions:
            return ActionRaise(int(min_r))

        # Tier 4: The Profitable Call
        if equity > pot_odds + 0.05: # Adding a 5% margin of safety
            if ActionCall in valid_actions: return ActionCall()
            if ActionCheck in valid_actions: return ActionCheck()

        # Tier 5: Give Up
        if ActionCheck in valid_actions: return ActionCheck()
        return ActionFold()

# ------------------------------------------------------------------------
# Engine Binding
# ------------------------------------------------------------------------
# Bypassing the engine's internal module aliasing bug
import pkbot.runner
pkbot.runner.BaseBot = BaseBot 

if __name__ == '__main__':
    run_bot(Player(), parse_args())