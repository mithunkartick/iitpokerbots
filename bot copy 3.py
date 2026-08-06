import random
import numpy as np
import base64
import bz2
import pickle
from treys import Card, Evaluator
from pkbot.engine import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.runner import parse_args, run_bot

# 1. Global Pre-computed Data & Initialization
evaluator = Evaluator()

SUITS = ['s', 'h', 'd', 'c']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
FULL_DECK = [r+s for r in RANKS for s in SUITS]
RANK_VALUES = {r: i for i, r in enumerate(RANKS, 2)}
DECK_MAP = {card: i for i, card in enumerate(FULL_DECK)}
BRAIN_PAYLOAD = '<insert_brain_payload_here>'

class FoldedCFRBrain:
    def __init__(self, payload):
        """Unpacks the base64 string back into NumPy arrays instantly."""
        byte_data = base64.b64decode(payload)
        decompressed_data = bz2.decompress(byte_data)
        self.weights = pickle.loads(decompressed_data)

    def relu(self, x):
        return np.maximum(0, x)

    def predict(self, state_vector):
        """O(1) NumPy Forward Pass. Zero TensorFlow overhead."""
        x = np.array(state_vector, dtype=np.float16)
        
        # Hidden Layer 1 (Weights 0, Biases 1)
        x = np.dot(x, self.weights[0]) + self.weights[1]
        x = self.relu(x)
        
        # Hidden Layer 2 (Weights 2, Biases 3)
        x = np.dot(x, self.weights[2]) + self.weights[3]
        x = self.relu(x)
        
        # Hidden Layer 3 (Weights 4, Biases 5)
        x = np.dot(x, self.weights[4]) + self.weights[5]
        x = self.relu(x)
        
        # Output Layer (Weights 6, Biases 7)
        out = np.dot(x, self.weights[6]) + self.weights[7]
        return out

# O(1) Preflop Lookup Table (Win Probability vs Random Hand)
PREFLOP_EQUITY = {
    "AA": 0.852, "KK": 0.824, "QQ": 0.799, "JJ": 0.775, "TT": 0.751, "99": 0.716, 
    "88": 0.687, "77": 0.662, "66": 0.632, "55": 0.603, "44": 0.570, "33": 0.536, "22": 0.503,
    "AKs": 0.670, "AQs": 0.661, "AJs": 0.654, "ATs": 0.647, "A9s": 0.630, "A8s": 0.621, "A7s": 0.610, 
    "A6s": 0.599, "A5s": 0.599, "A4s": 0.589, "A3s": 0.580, "A2s": 0.570, "KQs": 0.633, "KJs": 0.625,
    "KTs": 0.618, "K9s": 0.600, "K8s": 0.582, "K7s": 0.573, "K6s": 0.563, "K5s": 0.553, "K4s": 0.542,
    "K3s": 0.531, "K2s": 0.520, "QJs": 0.603, "QTs": 0.595, "Q9s": 0.577, "Q8s": 0.558, "Q7s": 0.540,
    "Q6s": 0.530, "Q5s": 0.519, "Q4s": 0.508, "Q3s": 0.497, "Q2s": 0.485, "JTs": 0.575, "J9s": 0.556,
    "J8s": 0.538, "J7s": 0.518, "J6s": 0.500, "J5s": 0.489, "J4s": 0.478, "J3s": 0.467, "J2s": 0.455,
    "T9s": 0.540, "T8s": 0.521, "T7s": 0.502, "T6s": 0.482, "T5s": 0.463, "T4s": 0.452, "T3s": 0.440,
    "T2s": 0.428, "98s": 0.506, "97s": 0.487, "96s": 0.467, "95s": 0.446, "94s": 0.426, "93s": 0.415,
    "92s": 0.403, "87s": 0.475, "86s": 0.454, "85s": 0.434, "84s": 0.413, "83s": 0.392, "82s": 0.380,
    "76s": 0.446, "75s": 0.425, "74s": 0.403, "73s": 0.381, "72s": 0.359, "65s": 0.420, "64s": 0.398,
    "63s": 0.375, "62s": 0.352, "54s": 0.397, "53s": 0.374, "52s": 0.351, "43s": 0.360, "42s": 0.337,
    "32s": 0.323,
    "AKo": 0.653, "AQo": 0.644, "AJo": 0.636, "ATo": 0.628, "A9o": 0.610, "A8o": 0.599, "A7o": 0.587,
    "A6o": 0.575, "A5o": 0.574, "A4o": 0.563, "A3o": 0.553, "A2o": 0.542, "KQo": 0.613, "KJo": 0.604,
    "KTo": 0.596, "K9o": 0.577, "K8o": 0.556, "K7o": 0.546, "K6o": 0.535, "K5o": 0.523, "K4o": 0.511,
    "K3o": 0.499, "K2o": 0.487, "QJo": 0.581, "QTo": 0.572, "Q9o": 0.552, "Q8o": 0.531, "Q7o": 0.511,
    "Q6o": 0.499, "Q5o": 0.487, "Q4o": 0.474, "Q3o": 0.462, "Q2o": 0.449, "JTo": 0.550, "J9o": 0.529,
    "J8o": 0.509, "J7o": 0.487, "J6o": 0.467, "J5o": 0.455, "J4o": 0.443, "J3o": 0.430, "J2o": 0.417,
    "T9o": 0.512, "T8o": 0.491, "T7o": 0.469, "T6o": 0.447, "T5o": 0.426, "T4o": 0.414, "T3o": 0.401,
    "T2o": 0.387, "98o": 0.474, "97o": 0.452, "96o": 0.430, "95o": 0.406, "94o": 0.385, "93o": 0.372,
    "92o": 0.358, "87o": 0.440, "86o": 0.417, "85o": 0.394, "84o": 0.370, "83o": 0.347, "82o": 0.333,
    "76o": 0.409, "75o": 0.385, "74o": 0.360, "73o": 0.336, "72o": 0.312, "65o": 0.380, "64o": 0.355,
    "63o": 0.330, "62o": 0.304, "54o": 0.354, "53o": 0.328, "52o": 0.302, "43o": 0.312, "42o": 0.286,
    "32o": 0.270
}

# ==========================================
# 3. THE PLAYER CLASS
# ==========================================
class Player:
    def __init__(self):
        try:
            self.brain = FoldedCFRBrain(BRAIN_PAYLOAD)
            self.brain_active = True
        except Exception:
            self.brain_active = False 
            
        # --- THE OPPONENT TRACKING SUITE ---
        self.hands_played = 0
        self.opp_vpip_count = 0  # Voluntarily Put In Pot (Are they a Calling Station?)
        self.opp_raise_count = 0 # Aggression frequency (Are they a Maniac or a Nit?)

    def evaluate_hand_strength(self, current_state):
        """Calculates pure mathematical equity to authorize exploitative overrides."""
        if current_state.street == 'preflop':
            key = self.get_hand_key(current_state.my_hand)
            return PREFLOP_EQUITY.get(key, 0.40)
            
        # Postflop: Use Treys to rank the hand against the board
        t_board = [Card.new(c) for c in current_state.board]
        t_hole = [Card.new(c) for c in current_state.my_hand]
        
        # Sneak peek inclusion if we won the auction
        if hasattr(current_state, 'sneak_peek_card') and current_state.sneak_peek_card:
            if getattr(current_state, 'auction_won', False):
                t_hole.append(Card.new(current_state.sneak_peek_card))
                
        # Get raw score (1 is Royal Flush, 7462 is worst) and normalize to 0.0 - 1.0
        try:
            my_rank = evaluator.evaluate(t_hole, t_board)
            return 1.0 - (my_rank / 7462.0)
        except:
            return 0.5 # Fallback if board parsing fails

    def on_hand_start(self, game_info, current_state):
        self.hands_played += 1

    def get_move(self, game_info, current_state):
        self.time_left = game_info.time_bank
        
        if self.time_left < 1.0 or not self.brain_active:
            return self.panic_move(current_state)

        # 1. Ask the Brain for baseline GTO Regrets
        state_vector = self.vectorize_state(current_state)
        regrets = self.brain.predict(state_vector)
        
        # 2. Convert to probabilities
        pos_regrets = np.maximum(regrets, 0)
        sum_pos = np.sum(pos_regrets)
        strategy = pos_regrets / sum_pos if sum_pos > 0 else np.ones(4) / 4.0
            
        # 3. Greedy Argmax - Default to the most profitable GTO action
        chosen_action_idx = np.argmax(strategy)
        
        # ==========================================
        # THE EXPLOITATIVE OVERRIDE ENGINE
        # ==========================================
        # We need at least 10 hands of data before we start profiling
        if self.hands_played > 10 and current_state.street != 'auction':
            vpip = self.opp_vpip_count / self.hands_played
            aggression = self.opp_raise_count / self.hands_played
            hand_strength = self.evaluate_hand_strength(current_state)

            # EXPLOIT PROFILE 1: THE "CALLING STATION" (VPIP > 65%)
            # They play too many hands and refuse to fold.
            if vpip > 0.65:
                # Override 1A: Never run GTO bluffs. If we are weak, just fold/check.
                if hand_strength < 0.45 and chosen_action_idx in [1, 2, 3]:
                    chosen_action_idx = 0 
                # Override 1B: Maximize value. If we have the nuts, jam it. They will call.
                elif hand_strength > 0.85 and chosen_action_idx < 3:
                    chosen_action_idx = 3 # All-In

            # EXPLOIT PROFILE 2: THE "NIT" (Aggression < 10%)
            # They are terrified of risking chips unless they have an unbeatable hand.
            elif aggression < 0.10:
                # Override 2A: Respect their rare aggression. If they bet, fold anything but a monster.
                if current_state.cost_to_call > 0 and hand_strength < 0.85:
                    chosen_action_idx = 0
                # Override 2B: Steal their blinds. If they check to us, aggressively probe.
                elif current_state.cost_to_call == 0 and hand_strength > 0.35 and chosen_action_idx == 0:
                    chosen_action_idx = 1 # 26% Pot bet to steal

        # 4. Execute the final, profile-adjusted move
        return self.map_index_to_action(chosen_action_idx, current_state)

    def on_hand_end(self, game_info, current_state):
        """Silently logs opponent behavior at the conclusion of every hand."""
        my_wager = getattr(current_state, 'my_wager', 0)
        opp_wager = getattr(current_state, 'opp_wager', 0)
        
        # Baseline blind is usually 10 chips. If they put in more, they voluntarily played.
        if opp_wager > 10:
            self.opp_vpip_count += 1
            
        # If their final wager was higher than ours, they took aggressive action.
        if opp_wager > my_wager and opp_wager > 20:
            self.opp_raise_count += 1

    # ==========================================
    # 4. DEEP CFR STATE MAPPING
    # ==========================================
    def vectorize_state(self, current_state):
        """Converts pkbot state to the 110-feature Deep CFR tensor."""
        # 1. Hole Cards (52 bits)
        hand_vec = np.zeros(52, dtype=np.float16)
        for c in current_state.my_hand:
            if c in DECK_MAP: hand_vec[DECK_MAP[c]] = 1.0
            
        # 2. Board Cards + Sneak Peek (52 bits)
        board_vec = np.zeros(52, dtype=np.float16)
        visible_board = list(current_state.board)
        
        # If we won the auction, the sneak peek might be passed as an attribute or a 3rd hole card
        auction_won = getattr(current_state, 'auction_won', False)
        if hasattr(current_state, 'sneak_peek_card') and current_state.sneak_peek_card:
            visible_board.append(current_state.sneak_peek_card)
            auction_won = True
            
        for c in visible_board:
            if c in DECK_MAP: board_vec[DECK_MAP[c]] = 1.0
            
        # 3. Game Features (6 bits)
        street_map = {'preflop':0.0, 'flop':0.25, 'auction':0.5, 'turn':0.75, 'river':1.0}
        my_wager = getattr(current_state, 'my_wager', 0)
        opp_wager = getattr(current_state, 'opp_wager', 0)
        
        game_features = np.array([
            current_state.pot / 2000.0,
            max(my_wager, opp_wager) / 2000.0,
            1.0 if auction_won else 0.0,
            street_map.get(current_state.street, 1.0),
            getattr(current_state, 'my_stack', 10000) / 10000.0,
            getattr(current_state, 'opp_stack', 10000) / 10000.0
        ], dtype=np.float16)
        
        return np.concatenate([hand_vec, board_vec, game_features])

    def decode_chips(self, action_idx, pot, stack):
        """Translates the 4 abstract action indices back into chips."""
        # Index 0: Fold / Bid Zero
        # Index 1: 26% Pot (Slightly overbids the standard 25% bots)
        # Index 2: 105% Pot (Slightly overbids the standard 100% bots)
        # Index 3: All-In
        bids = [0, int(pot * 0.26), int(pot * 1.05), stack]
        return bids[action_idx]

    def map_index_to_action(self, action_idx, current_state):
        """Converts the chosen abstract action into an engine-legal Action."""
        my_stack = getattr(current_state, 'my_stack', 10000)
        pot = current_state.pot
        
        # A. Handle the Auction Street
        if current_state.street == 'auction':
            bid_amount = self.decode_chips(action_idx, pot, my_stack)
            if current_state.can_act(ActionBid):
                return ActionBid(min(bid_amount, my_stack))
            return ActionCheck() # Fallback

        # B. Handle Betting Streets
        if action_idx == 0:
            if current_state.can_act(ActionCheck): 
                return ActionCheck()
            return ActionFold()

        target_investment = self.decode_chips(action_idx, pot, my_stack)
        cost_to_call = current_state.cost_to_call

        if target_investment <= cost_to_call:
            if current_state.can_act(ActionCall): return ActionCall()
            if current_state.can_act(ActionCheck): return ActionCheck()
            return ActionFold()
        else:
            if current_state.can_act(ActionRaise):
                min_r, max_r = current_state.raise_bounds
                safe_raise = max(min_r, min(target_investment, max_r))
                return ActionRaise(safe_raise)
            elif current_state.can_act(ActionCall):
                return ActionCall()
            return ActionFold()

    # ==========================================
    # 5. SAFEGUARDS
    # ==========================================
    def get_hand_key(self, cards):
        if len(cards) < 2: return "72o"
        c1, c2 = cards[0], cards[1]
        r1, s1, r2, s2 = c1[0], c1[1], c2[0], c2[1]
        if RANK_VALUES.get(r1, 0) < RANK_VALUES.get(r2, 0):
            r1, r2 = r2, r1
        if r1 == r2: return f"{r1}{r2}"
        elif s1 == s2: return f"{r1}{r2}s"
        else: return f"{r1}{r2}o"

    def panic_move(self, current_state):
        if current_state.can_act(ActionCheck):
            return ActionCheck()
        if current_state.can_act(ActionCall):
            if current_state.street == 'preflop':
                key = self.get_hand_key(current_state.my_hand)
                if PREFLOP_EQUITY.get(key, 0.40) > 0.55:
                    return ActionCall()
            else:
                pot_odds = current_state.cost_to_call / max(1, current_state.pot + current_state.cost_to_call)
                if pot_odds < 0.15:
                    return ActionCall()
        return ActionFold()
    
if __name__ == '__main__':
    run_bot(Player(), parse_args())