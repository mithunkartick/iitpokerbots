import random
from engine import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid

class Player:
    def __init__(self):
        self.hands_played = 0
        self.epoch_40_pnl = []
        self.epoch_250_start_bankroll = 0
        
        # --- Strategic Memory ---
        self.raised_flop = False    # For Double-Barreling
        self.aof_shove_count = 0    # Tracking All-in or Fold bots
        self.is_aof_bot = False     # Triggered by 40-round loop
        
        # --- Dynamic Knobs ---
        self.value_mult = 1.2
        self.probe_mult = 0.25
        self.call_tolerance = 0.40  # Increased base tolerance to reach showdowns
        self.auction_bid_pct = 0.20

    def _get_rank_value(self, rank_str):
        mapping = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
        return mapping.get(rank_str, 0)

    def evaluate_board(self, board):
        """Analyzes community cards to determine how 'wet' (dangerous/coordinated) it is."""
        if len(board) < 3: return "DRY"
        suits = [c[1] for c in board]
        ranks = sorted([self._get_rank_value(c[0]) for c in board])
        
        # 3 or more of the same suit is highly coordinated
        if max([suits.count(s) for s in set(suits)]) >= 3: return "WET"
        
        # 3 connected cards (e.g., 7-8-9) is coordinated
        consecutive = 1
        for i in range(len(ranks)-1):
            if ranks[i+1] == ranks[i] + 1: consecutive += 1
        if consecutive >= 3: return "WET"
        
        return "DRY"

    def classify_hand(self, my_cards, board):
        cards = my_cards + board
        if not cards: return 5
        ranks = [c[0] for c in cards]
        rank_counts = {r: ranks.count(r) for r in set(ranks)}
        
        # The Nuts / Near-Nuts
        if any(c >= 4 for c in rank_counts.values()) or (3 in rank_counts.values() and 2 in rank_counts.values()):
            return 0 
        # Monster
        if any(c == 3 for c in rank_counts.values()) or max([ [c[1] for c in cards].count(s) for s in set([c[1] for c in cards])]) >= 5: 
            return 1
        # Strong (Two Pair)
        if sum(1 for c in rank_counts.values() if c == 2) >= 2: 
            return 2
        # Pair
        if 2 in rank_counts.values(): 
            return 3
        # High Equity Draw
        suits = [c[1] for c in cards]
        if max([suits.count(s) for s in set(suits)]) == 4:
            return 4
        return 5

    def get_move(self, game_info, current_state):
        street = current_state.street
        board = current_state.board
        pot = current_state.pot
        cost = current_state.cost_to_call
        my_chips = current_state.my_chips

        # Reset memory on new hand
        if street == 'preflop': 
            self.raised_flop = False

        tier = self.classify_hand(current_state.my_hand, board)
        board_texture = self.evaluate_board(board)

        # --- 1. AUCTION VISION ---
        if street == 'auction':
            bid = int(pot * self.auction_bid_pct)
            return ActionBid(min(bid, current_state.opp_chips))

        if current_state.opp_revealed_cards:
            opp_rank = self._get_rank_value(current_state.opp_revealed_cards[0][0])
            if opp_rank < 8 and tier >= 4: tier = 2  # Bully weak revealed cards
            elif opp_rank >= 12 and tier == 3: tier = 4 # Caution against big cards

        # --- 2. THE AOF EXPLOITER LOGIC ---
        # If we know they are an AoF bot, we mathematically expand our calling range.
        if self.is_aof_bot and cost > pot * 2.0:
            # We call their shoves with ANY pair (Tier 3) or better. 
            # We don't wait for Tier 1. This crushes blind shoves.
            if tier <= 3: return ActionCall()
            return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()

        # --- 3. PROACTIVE BETTING & DOUBLE BARRELING ---
        if cost == 0:
            if current_state.can_act(ActionRaise):
                # The Double Barrel: If we raised the flop with trash, and the turn is DRY, fire again!
                if street == 'turn' and self.raised_flop and board_texture == "DRY" and tier >= 4:
                    size = min(int(pot * 0.40), current_state.raise_bounds[1]) # 40% pot bluff
                    return ActionRaise(max(current_state.raise_bounds[0], size))

                # Value & Probes
                if tier <= 2:
                    size = min(int(pot * self.value_mult), current_state.raise_bounds[1])
                    move = ActionRaise(max(current_state.raise_bounds[0], size))
                else:
                    size = min(int(pot * self.probe_mult), current_state.raise_bounds[1])
                    move = ActionRaise(max(current_state.raise_bounds[0], size))
                
                if street == 'flop': self.raised_flop = True
                return move
            return ActionCheck()

        # --- 4. SHOWDOWN ENFORCEMENT (Cost > 0) ---
        # "Take it to showdown unless defeat is certain."
        if cost > 0:
            # Tier 0 and 1: Never fold. Raise if possible.
            if tier <= 1:
                if current_state.can_act(ActionRaise):
                    return ActionRaise(min(int(pot * self.value_mult), current_state.raise_bounds[1]))
                return ActionCall()

            # The Showdown Pull: If it's the River, and we have ANY piece of the board (Tier 3/4), 
            # we are highly likely to call to see their cards, unless they shove a massive amount.
            if street == 'river' and tier <= 4:
                # We will call up to 75% of the pot on the river just to keep them honest
                if cost <= pot * 0.75: return ActionCall()

            # Normal Street Floats: Use dynamic call_tolerance
            if cost <= (pot * self.call_tolerance):
                return ActionCall()
            
            # Absolute defeat certainty (Massive bet + Trash hand)
            return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()

    def on_hand_start(self, game_info, current_state):
        if self.hands_played == 0: self.epoch_250_start_bankroll = game_info.bankroll

    def on_hand_end(self, game_info, current_state):
        self.hands_played += 1
        self.epoch_40_pnl.append(current_state.payoff)
        
        # Track AoF behavior (shoving more than 80% of stack)
        if current_state.opp_wager >= current_state.opp_chips * 0.8:
            self.aof_shove_count += 1
        
        # --- 40-ROUND LOOP ---
        if self.hands_played % 40 == 0:
            net_40 = sum(self.epoch_40_pnl)
            
            # AoF Detection Protocol
            if self.aof_shove_count >= 5: # If they shoved 5+ times in 40 hands, they are an AoF bot
                self.is_aof_bot = True
            else:
                self.is_aof_bot = False
            self.aof_shove_count = 0 # Reset for next 40 rounds
            
            # Adjust Showdown & Float willingness based on profitability
            if net_40 > 0:
                self.call_tolerance = min(0.60, self.call_tolerance + 0.05) # Loosen up, we're winning
                self.value_mult = min(2.0, self.value_mult + 0.1)
            else:
                self.call_tolerance = max(0.20, self.call_tolerance - 0.05) # Tighten slightly, but don't become a rock
                self.probe_mult = max(0.15, self.probe_mult - 0.05)
                
            self.epoch_40_pnl = []

        # --- 250-ROUND LOOP ---
        if self.hands_played % 250 == 0:
            net_250 = game_info.bankroll - self.epoch_250_start_bankroll
            if net_250 < -300:
                # Still taking hits. Reduce probe frequency but keep showdown calling intact for Tier 3+.
                self.probe_mult = 0.10
                self.call_tolerance = 0.25 
            elif net_250 > 500:
                self.probe_mult = 0.35
                self.call_tolerance = 0.50
            self.epoch_250_start_bankroll = game_info.bankroll