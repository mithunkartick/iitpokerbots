import random
import eval7
import numpy as np
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

# Globally initialize deck string for speed
DECK_STRINGS = [r+s for r in '23456789TJQKA' for s in 'shdc']
RANK_VALUES = {r: i for i, r in enumerate('23456789TJQKA', 2)}

def monte_carlo_equity(my_cards_str, board_cards_str, opp_revealed_str, num_simulations=100):
    """
    Hyper-optimized C-level MC Simulator. Only runs Post-Flop.
    """
    my_cards = [eval7.Card(c) for c in my_cards_str]
    board_cards = [eval7.Card(c) for c in board_cards_str]
    opp_revealed = [eval7.Card(c) for c in opp_revealed_str] # FIXED TYPO: Card, not Caard
    
    known_cards = set(my_cards + board_cards + opp_revealed)
    remaining_deck = [eval7.Card(c) for c in DECK_STRINGS if eval7.Card(c) not in known_cards]
    
    wins, ties = 0, 0
    cards_needed_board = 5 - len(board_cards)
    cards_needed_opp = 2 - len(opp_revealed)
    total_needed = cards_needed_board + cards_needed_opp

    for _ in range(num_simulations):
        drawn = random.sample(remaining_deck, total_needed)
        sim_board = board_cards + drawn[:cards_needed_board]
        sim_opp = opp_revealed + drawn[cards_needed_board:]
        
        my_val = eval7.evaluate(my_cards + sim_board)
        opp_val = eval7.evaluate(sim_opp + sim_board)
        
        if my_val > opp_val: wins += 1
        elif my_val == opp_val: ties += 1

    return (wins + (0.5 * ties)) / num_simulations

class Player(BaseBot):
    def __init__(self):
        super().__init__()
        self.hands_played = 0
        self.opp_bids = []         # Tracks auction sniping
        self.equity_cache = {}     # ADDED CACHE to prevent time-bank bleed

    def on_hand_start(self, game_info, current_state):
        self.hands_played += 1
        self.time_left = getattr(game_info, 'time_bank', 20.0)
        self.equity_cache.clear()  # Wipe memory for the new hand

    def on_hand_end(self, game_info, current_state):
        if hasattr(current_state, 'auction_bids') and current_state.auction_bids:
            opp_bid = current_state.auction_bids[1] if current_state.dealer == 0 else current_state.auction_bids[0]
            self.opp_bids.append(opp_bid)
            if len(self.opp_bids) > 20: self.opp_bids.pop(0)

    def get_move(self, game_info, current_state):
        street = current_state.street.lower()
        valid_actions = current_state.legal_actions
        
        if self.time_left < 1.0:
            return ActionCheck() if ActionCheck in valid_actions else ActionFold()

        my_cards = [str(c) for c in current_state.my_hand]
        board_cards = [str(c) for c in current_state.board]
        opp_revealed = [str(c) for c in getattr(current_state, 'opp_revealed_cards', [])]
        
        # 1. THE AUCTION SNIPER
        if ActionBid in valid_actions:
            return self.handle_auction(current_state, my_cards)

        # 2. STATE MEMOIZATION (Saves time during raise wars)
        state_hash = (tuple(my_cards), tuple(board_cards), tuple(opp_revealed))
        
        if state_hash in self.equity_cache:
            equity = self.equity_cache[state_hash]
        else:
            if street in ['preflop', 'pre-flop']:
                equity = self.fast_preflop_estimate(my_cards)
            else:
                sims = 300 if self.time_left > 5.0 else 50
                equity = monte_carlo_equity(my_cards, board_cards, opp_revealed, num_simulations=sims)
            self.equity_cache[state_hash] = equity

        return self.play_street(current_state, equity, street)

    # --- STRATEGY UPGRADES ---

    def fast_preflop_estimate(self, my_cards):
        r1, s1 = my_cards[0][0], my_cards[0][1]
        r2, s2 = my_cards[1][0], my_cards[1][1]
        
        val1, val2 = RANK_VALUES[r1], RANK_VALUES[r2]
        is_pair = (val1 == val2)
        is_suited = (s1 == s2)
        high, low = max(val1, val2), min(val1, val2)
        
        # Mathematically tightened Pre-Flop LUT
        if is_pair:
            if val1 >= 11: return 0.82  # AA, KK, QQ, JJ
            if val1 >= 8: return 0.70   # TT, 99, 88
            return 0.55                 # Low pairs
        
        if is_suited:
            if low >= 11: return 0.65   # AKs, AQs, KQs, JQs
            if low >= 9: return 0.55    # Decent Suited Connectors
            return 0.45                 # Trash suited
            
        if low >= 11: return 0.60       # Premium Offsuit (AKo, AQo)
        if high >= 12: return 0.50      # High Card K/A
        
        # THE FIX: Trash is explicitly penalized so it mathematically folds to raises
        return 0.25

    def handle_auction(self, current_state, my_cards):
        pot_size = current_state.pot
        my_chips = current_state.my_chips
        equity = self.fast_preflop_estimate(my_cards)
        
        # Don't pay for information if our hand is absolute garbage
        if equity < 0.40: return ActionBid(0)
        
        if len(self.opp_bids) >= 3:
            median_bid = int(np.median(self.opp_bids))
            # Cap the sniper so we don't overpay if the opponent is manic bidding
            max_rational_bid = int(pot_size * 0.30)
            target_bid = min(median_bid + 1, max_rational_bid)
        else:
            target_bid = int(pot_size * 0.10) if equity > 0.60 else 0
            
        safe_bid = max(0, min(target_bid, int(my_chips)))
        return ActionBid(safe_bid)

    def play_street(self, current_state, equity, street):
        valid_actions = current_state.legal_actions
        pot_odds = current_state.cost_to_call / max(1.0, float(current_state.pot + current_state.cost_to_call))
        
        min_r = getattr(current_state, 'min_raise', 20)
        max_r = getattr(current_state, 'max_raise', int(current_state.my_chips))
        if hasattr(current_state, 'raise_bounds') and current_state.raise_bounds is not None:
            min_r, max_r = current_state.raise_bounds

        # Aggression
        if equity > 0.85 and ActionRaise in valid_actions: return ActionRaise(int(max_r))
        if equity > 0.70 and ActionRaise in valid_actions: return ActionRaise(int(max(min_r, min(current_state.pot * 0.75, max_r))))
        if equity > 0.60 and ActionRaise in valid_actions: return ActionRaise(int(min_r))
        
        # Call Logic with Implied Odds tracking
        margin = 0.05 # Standard 5% safety margin
        if street in ['flop', 'turn'] and 0.30 <= equity <= 0.45:
            margin = -0.05 # Implied odds discount for strong draws

        if equity > pot_odds + margin:
            if ActionCall in valid_actions: return ActionCall()
            if ActionCheck in valid_actions: return ActionCheck()

        if ActionCheck in valid_actions: return ActionCheck()
        return ActionFold()

# --- Engine Binding ---
import pkbot.runner
pkbot.runner.BaseBot = BaseBot 

if __name__ == '__main__':
    run_bot(Player(), parse_args())