from argparse import Action
import random

from engine import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid

class Player:
    def __init__(self):
        self.hands_played = 0
        self.epoch_40_pnl = []
        self.epoch_250_start_bankroll = 0
        self.opp_chips = 5000
        self.start_stack = 5000
        
        # --- Memory & Tracking ---
        self.raised_flop = False
        self.is_aof_bot = False
        self.shove_count = 0
        
        # --- Showdown Knobs ---
        self.call_tolerance = 0.65  
        self.bravery_boost = 1.0    
        self.auction_bid_pct = 0.30 
        
        # --- THE BLUFF ENGINE ---
        self.bluff_prob = 0.30       # Starts at 30% chance to run a massive bluff
        self.bluff_decay = 0.85      # Multiplier to reduce bluffing every 40 rounds
        self.min_bluff_prob = 0.05   # Never goes below 5% to remain unpredictable

    def _get_rank_value(self, rank_str):
        mapping = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
        return mapping.get(rank_str, 0)

    def classify_preflop(self, my_cards):
        """Analyzes hole cards regardless of the board."""
        rank_vals = sorted([self._get_rank_value(c[0]) for c in my_cards], reverse=True)
        is_suited = my_cards[0][1] == my_cards[1][1]
        
        # Premium: Pocket Pairs or High Connectors (K-Q, A-J, etc.)
        if rank_vals[0] == rank_vals[1]: return 1 # Pocket Pair
        if rank_vals[0] >= 12 and rank_vals[1] >= 10: return 2 # High Cards (K-Q, A-Q, etc)
        if rank_vals[0] >= 10 and is_suited: return 3 # Suited Connectors
        return 5 # Actual Trash

    def classify_hand(self, my_cards, board):
        cards = my_cards + board
        ranks = [c[0] for c in cards]
        rank_counts = {r: ranks.count(r) for r in set(ranks)}
        
        if any(c >= 3 for c in rank_counts.values()): return 1 # Monster+
        if sum(1 for c in rank_counts.values() if c == 2) >= 2: return 2 # Two Pair
        if 2 in rank_counts.values(): return 3 # Any Pair
        
        # TIER 6: The "Showdown Specialist" (Ace or King High)
        my_ranks = [self._get_rank_value(c[0]) for c in my_cards]
        if max(my_ranks) >= 13: return 6 
        
        return 5 # Pure Trash

    def get_move(self, game_info, current_state):
        print("udhaaaaaaaaaaaay")        
        street = current_state.street
        pot = current_state.pot
        cost = current_state.cost_to_call
        maxi = current_state.raise_bounds[1]
        mini = current_state.raise_bounds[0]
        absolute_max = min(current_state.my_chips, maxi, current_state.opp_chips)
        
        # 1. EVALUATE STRENGTH FIRST (The "Source of Truth")
        if street == 'preflop':
            tier = self.classify_preflop(current_state.my_hand)
        else:
            tier = self.classify_hand(current_state.my_hand, current_state.board)
        print("tier:", tier)    
        # 2. AUCTION LOGIC (Highest Priority)
        if street == 'auction':
            # Always participate if we have a decent hand
            if random.random() < 0.6:
                return ActionBid(min(int(pot * 0.1), self.opp_chips))
            else:
                return ActionBid(min(10, self.opp_chips))

        # 3. PRE-FLOP DEFENSE (The Fix for your Jacks/Ace-high folds)
        elif street == 'preflop':
            # If there is a raise (cost > 0)
            if cost > 0:
                # If we have a pair (Tier 1) or strong High Cards (Tier 2), NEVER FOLD.
                # We call or raise.
                if tier <= 2:
                    if random.random() < 0.4:
                        return ActionRaise(min(int(pot *2), absolute_max)) if current_state.can_act(ActionRaise) else ActionCall()
                    else:
                        return ActionCall() if current_state.can_act(ActionCall) else ActionCall()
                # If we are the BB (cost is just the difference), call with almost anything
                if current_state.my_chips <= 0 and cost <= 50: # Assume BB defense
                    return ActionCall() if current_state.can_act(ActionCall) else ActionCall()
            
            # If no raise (Check), try to raise with strong hands
            elif cost == 0 and tier <= 2:
                return ActionRaise(min(int(pot *3), absolute_max)) if current_state.can_act(ActionRaise) else ActionCall()
            else:
                if random.random()<0.7:
                    if current_state.can_act(ActionCall):
                        return ActionCall()
                    elif current_state.can_act(ActionRaise):
                        return ActionRaise(min(int(pot * 2), absolute_max)) if current_state.can_act(ActionRaise) else ActionCall()
                    else:
                        return ActionCall()
                else:
                    return ActionRaise(min(int(pot *2), absolute_max)) if current_state.can_act(ActionRaise) else ActionCall()

        # 4. POST-FLOP ACTION
        # Only reach here if street is NOT 'preflop' and NOT 'auction'
        elif cost == 0:
            if current_state.can_act(ActionCheck):
                return ActionCheck()
            else:
                return ActionRaise(min(int(pot * 0.03), absolute_max)) if current_state.can_act(ActionRaise) else ActionCall()
            
            
        # If we have a Pair (3) or better, we do not fold to normal bets.
        elif tier <= 3:
            if random.random()<0.93:
                return ActionRaise(min(int(pot * 0.3), absolute_max)) if current_state.can_act(ActionRaise) else ActionCall()
            elif current_state.can_act(ActionCall):
                return ActionCall()
            else:
                return ActionRaise(min(int(pot * 0.02), absolute_max)) if current_state.can_act(ActionRaise) else ActionCall()
           
            
        # Only fold if we have pure trash (Tier 5) AND it's expensive
        elif tier == 5 and cost > (pot * 0.3):
            if random.random()<0.4:
                return ActionFold() if current_state.can_act(ActionFold) else ActionCheck()
            else:
               if random.random()<0.2:
                   return ActionRaise(min(int(pot * 0.3), absolute_max)) if current_state.can_act(ActionRaise) else ActionCall()
               return ActionRaise(min(int(pot * 0.02), absolute_max)) if current_state.can_act(ActionRaise) else ActionCall()
        
        elif current_state.can_act(ActionCall):
            return ActionCall()
        
        else:
            if random.random()>0.1:
                return ActionRaise(min(max(int(pot * 0.02), mini), maxi)) if current_state.can_act(ActionRaise) else ActionCall()
            else:
                return ActionRaise(min(max(int(pot * 0.15), mini), maxi)) if current_state.can_act(ActionRaise) else ActionCall()

    def on_hand_start(self, game_info, current_state):
        if self.hands_played == 0: self.epoch_250_start_bankroll = game_info.bankroll

    def on_hand_end(self, game_info, current_state):
        self.opp_chips -= current_state.payoff
        self.hands_played += 1
        self.epoch_40_pnl.append(current_state.payoff)
        if current_state.opp_wager > (current_state.opp_chips * 0.7): self.shove_count += 1
        
        # --- 40-ROUND ADAPTATION & DECAY ---
        if self.hands_played % 40 == 0:
            net = sum(self.epoch_40_pnl)
            self.is_aof_bot = self.shove_count >= 4
            
            # DECAY THE BLUFF: Slowly reduce the chaos
            self.bluff_prob = max(self.min_bluff_prob, self.bluff_prob * self.bluff_decay)
            
            if net < 0:
                self.call_tolerance = max(0.40, self.call_tolerance - 0.05)
                self.auction_bid_pct = min(0.50, self.auction_bid_pct + 0.05)
            else:
                self.call_tolerance = min(0.80, self.call_tolerance + 0.02)
            
            self.epoch_40_pnl = []
            self.shove_count = 0

        # --- 250-ROUND COURSE CORRECTION ---
        if self.hands_played % 250 == 0:
            if (game_info.bankroll - self.epoch_250_start_bankroll) < -500:
                self.call_tolerance = 0.30 
                self.bravery_boost = 0.7
            self.epoch_250_start_bankroll = game_info.bankroll