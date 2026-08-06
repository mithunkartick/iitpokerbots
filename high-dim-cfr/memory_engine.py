from treys import Deck, Evaluator, Card
from pkbot.states import GameState, PokerState, STARTING_STACK
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid

class MemoryEngine:
    """
    Decoupled In-Memory Engine. 
    Constructs genuine pkbot states without socket overhead.
    """
    def __init__(self):
        self.evaluator = Evaluator()

    def play_hand(self, p1_bot, p2_bot):
        deck = Deck()
        p1_hand = [Card.int_to_str(c) for c in deck.draw(2)]
        p2_hand = [Card.int_to_str(c) for c in deck.draw(2)]
        board = []
        
        p1_wager = 10 # SB
        p2_wager = 20 # BB
        
        # --- FLOP ---
        board.extend([Card.int_to_str(c) for c in deck.draw(3)])
        
        # --- THE SNEAK PEEK AUCTION ---
        p1_state_auc = self._build_poker_state(3, True, p1_wager, p2_wager, p1_hand, p2_hand, board, [], [], active=0)
        p2_state_auc = self._build_poker_state(3, True, p1_wager, p2_wager, p1_hand, p2_hand, board, [], [], active=1)
        
        p1_bid = p1_bot.get_auction_bid(p1_state_auc)
        p2_bid = p2_bot.get_auction_bid(p2_state_auc)
        
        p1_rev, p2_rev = [], []
        
        # Second-Price Resolution
        if p1_bid > p2_bid:
            p1_wager += p2_bid
            p1_rev = [p2_hand[0]]
        elif p2_bid > p1_bid:
            p2_wager += p1_bid
            p2_rev = [p1_hand[0]]
        else: # Tie
            p1_wager += p1_bid; p2_wager += p2_bid
            p1_rev = [p2_hand[0]]; p2_rev = [p1_hand[0]]
            
        # --- TURN & RIVER ---
        board.extend([Card.int_to_str(c) for c in deck.draw(2)]) 
        
        p1_state_riv = self._build_poker_state(5, False, p1_wager, p2_wager, p1_hand, p2_hand, board, p1_rev, p2_rev, active=0)
        p2_state_riv = self._build_poker_state(5, False, p1_wager, p2_wager, p1_hand, p2_hand, board, p1_rev, p2_rev, active=1)
        
        p1_action_idx, p1_bet_amt = p1_bot.get_action(p1_state_riv)
        p2_action_idx, p2_bet_amt = p2_bot.get_action(p2_state_riv)
        
        # Fold Handling (Index 0)
        if p1_action_idx == 0 and p2_action_idx != 0: return -p1_wager
        if p2_action_idx == 0 and p1_action_idx != 0: return p2_wager
        
        final_invested = min(p1_wager + p1_bet_amt, p2_wager + p2_bet_amt)
        
        # --- SHOWDOWN ---
        p1_eval = [Card.new(c) for c in p1_hand]
        p2_eval = [Card.new(c) for c in p2_hand]
        b_eval = [Card.new(c) for c in board]
        
        p1_score = self.evaluator.evaluate(p1_eval, b_eval)
        p2_score = self.evaluator.evaluate(p2_eval, b_eval)
        
        if p1_score < p2_score: return final_invested
        elif p2_score < p1_score: return -final_invested
        else: return 0

    def _build_poker_state(self, street_val, is_auction, p1_wag, p2_wag, p1_h, p2_h, brd, p1_r, p2_r, active):
        """ Instantiates the official pkbot GameState and PokerState wrappers """
        gs = GameState(
            dealer=1,
            street=street_val,
            auction=is_auction,
            bids=[],
            wagers=[p1_wag, p2_wag],
            chips=[STARTING_STACK - p1_wag, STARTING_STACK - p2_wag],
            hands=[p1_h, p2_h],
            opp_hands=[p1_r, p2_r],
            community_cards=brd,
            parent_state=None
        )
        return PokerState(gs, active)
