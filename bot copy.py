import random
from pkbot.engine import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid

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

def monte_carlo_equity(my_cards, board_cards, opp_revealed=None, num_simulations=50):
    """
    Estimates win probability. Now accepts known opponent cards to restrict the sample space.
    """
    if opp_revealed is None:
        opp_revealed = []
        
    import random
    from pkbot.engine import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid

    SUITS = ['s', 'h', 'd', 'c']
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    FULL_DECK = [r+s for r in RANKS for s in SUITS]
    RANK_VALUES = {r: i for i, r in enumerate(RANKS, 2)}

    def fast_heuristic_evaluator(cards):
        if not cards:
            return (0, 0)
        ranks = [RANK_VALUES[card[0]] for card in cards]
        counts = {}
        for r in ranks:
            counts[r] = counts.get(r, 0) + 1
        max_count = max(counts.values())
        best_rank = max(r for r, count in counts.items() if count == max_count)
        return (max_count, best_rank)

    def monte_carlo_equity(my_cards, board_cards, opp_revealed=None, num_simulations=50):
        if opp_revealed is None:
            opp_revealed = []
        known_cards = set(my_cards + board_cards + opp_revealed)
        remaining_deck = [c for c in FULL_DECK if c not in known_cards]
        my_treys = [Card.new(c) for c in my_cards]
        board_treys = [Card.new(c) for c in board_cards]
        opp_rev_treys = [Card.new(c) for c in opp_revealed]
        wins = 0
        ties = 0
        cards_needed_for_opp = 2 - len(opp_revealed)
        cards_needed_for_board = 5 - len(board_cards)
        total_to_draw = cards_needed_for_opp + cards_needed_for_board

        for _ in range(num_simulations):
            drawn = random.sample(remaining_deck, total_to_draw)
            opp_sim_treys = opp_rev_treys + [Card.new(c) for c in drawn[:cards_needed_for_opp]]
            sim_board = board_treys + [Card.new(c) for c in drawn[cards_needed_for_opp:]]
            my_score = evaluator.evaluate(sim_board, my_treys)
            opp_score = evaluator.evaluate(sim_board, opp_sim_treys)
            if my_score < opp_score:
                wins += 1
            elif my_score == opp_score:
                ties += 1
        return (wins + (0.5 * ties)) / max(1, num_simulations)


    class Player:
        def __init__(self):
            self.hands_played = 0
            self.opp_actions = 0
            self.opp_raises = 0
            self.aggression_factor = 0.0
            self.preflop_ev_table = {
        "22": 0.5097,
        "32s": 0.3538,
        "32o": 0.3299,
        "42s": 0.3773,
        "42o": 0.3334,
        "52s": 0.3706,
        "52o": 0.3367,
        "62s": 0.3735,
        "62o": 0.3469,
        "72s": 0.3802,
        "72o": 0.3471,
        "82s": 0.4128,
        "82o": 0.3652,
        "92s": 0.4331,
        "92o": 0.3822,
        "T2s": 0.4586,
        "T2o": 0.4156,
        "J2s": 0.4692,
        "J2o": 0.4282,
        "Q2s": 0.4986,
        "Q2o": 0.4757,
        "K2s": 0.5378,
        "K2o": 0.5017,
        "A2s": 0.5711,
        "A2o": 0.5549,
        "33": 0.5347,
        "43s": 0.3915,
        "43o": 0.3431,
        "53s": 0.3941,
        "53o": 0.3572,
        "63s": 0.3987,
        "63o": 0.3603,
        "73s": 0.4117,
        "73o": 0.3592,
        "83s": 0.4146,
        "83o": 0.3738,
        "93s": 0.4319,
        "93o": 0.3875,
        "T3s": 0.4528,
        "T3o": 0.4278,
        "J3s": 0.4901,
        "J3o": 0.4666,
        "Q3s": 0.5051,
        "Q3o": 0.4743,
        "K3s": 0.55,
        "K3o": 0.5142,
        "A3s": 0.597,
        "A3o": 0.5617,
        "44": 0.5679,
        "54s": 0.4281,
        "54o": 0.3885,
        "64s": 0.4129,
        "64o": 0.378,
        "74s": 0.4121,
        "74o": 0.3804,
        "84s": 0.4307,
        "84o": 0.3883,
        "94s": 0.4508,
        "94o": 0.395,
        "T4s": 0.4624,
        "T4o": 0.4303,
        "J4s": 0.4879,
        "J4o": 0.4571,
        "Q4s": 0.5255,
        "Q4o": 0.4823,
        "K4s": 0.5511,
        "K4o": 0.5207,
        "A4s": 0.5878,
        "A4o": 0.5725,
        "55": 0.6204,
        "65s": 0.4309,
        "65o": 0.3875,
        "75s": 0.4351,
        "75o": 0.4116,
        "85s": 0.4435,
        "85o": 0.4103,
        "95s": 0.4611,
        "95o": 0.4335,
        "T5s": 0.4721,
        "T5o": 0.4422,
        "J5s": 0.4966,
        "J5o": 0.4653,
        "Q5s": 0.5148,
        "Q5o": 0.5008,
        "K5s": 0.5554,
        "K5o": 0.5325,
        "A5s": 0.6057,
        "A5o": 0.5869,
        "66": 0.6267,
        "76s": 0.4501,
        "76o": 0.4195,
        "86s": 0.4577,
        "86o": 0.4261,
        "96s": 0.4697,
        "96o": 0.4476,
        "T6s": 0.498,
        "T6o": 0.4688,
        "J6s": 0.5219,
        "J6o": 0.4764,
        "Q6s": 0.5271,
        "Q6o": 0.5094,
        "K6s": 0.571,
        "K6o": 0.5595,
        "A6s": 0.5893,
        "A6o": 0.572,
        "77": 0.6678,
        "87s": 0.48,
        "87o": 0.4406,
        "97s": 0.4907,
        "97o": 0.4788,
        "T7s": 0.4993,
        "T7o": 0.4847,
        "J7s": 0.5285,
        "J7o": 0.5014,
        "Q7s": 0.5366,
        "Q7o": 0.5227,
        "K7s": 0.5666,
        "K7o": 0.5442,
        "A7s": 0.6085,
        "A7o": 0.5908,
        "88": 0.6884,
        "98s": 0.5139,
        "98o": 0.4798,
        "T8s": 0.52,
        "T8o": 0.4935,
        "J8s": 0.5439,
        "J8o": 0.5212,
        "Q8s": 0.5565,
        "Q8o": 0.5328,
        "K8s": 0.5806,
        "K8o": 0.5519,
        "A8s": 0.6068,
        "A8o": 0.5938,
        "99": 0.7279,
        "T9s": 0.541,
        "T9o": 0.5122,
        "J9s": 0.5542,
        "J9o": 0.5308,
        "Q9s": 0.5797,
        "Q9o": 0.5595,
        "K9s": 0.5897,
        "K9o": 0.5722,
        "A9s": 0.6354,
        "A9o": 0.6047,
        "TT": 0.7497,
        "JTs": 0.5803,
        "JTo": 0.5693,
        "QTs": 0.5994,
        "QTo": 0.571,
        "KTs": 0.6151,
        "KTo": 0.5994,
        "ATs": 0.6444,
        "ATo": 0.6246,
        "JJ": 0.7838,
        "QJs": 0.5979,
        "QJo": 0.5808,
        "KJs": 0.6176,
        "KJo": 0.6028,
        "AJs": 0.6561,
        "AJo": 0.6302,
        "QQ": 0.7918,
        "KQs": 0.6384,
        "KQo": 0.6106,
        "AQs": 0.6552,
        "AQo": 0.6434,
        "KK": 0.826,
        "AKs": 0.6692,
        "AKo": 0.653,
        "AA": 0.8501
    }

        def on_hand_start(self, game_info, current_state):
            self.hands_played += 1
            self.time_left = game_info.time_bank

        def get_move(self, game_info, current_state):
            street = current_state.street
            if self.time_left < 1.0:
                return self.panic_move(current_state)
            if street == 'auction':
                return self.handle_auction(current_state)
            my_cards = current_state.my_hand
            rank1 = RANK_VALUES[my_cards[0][0]]
            rank2 = RANK_VALUES[my_cards[1][0]]
            is_pocket_pair = (rank1 == rank2)
            high_card = max(rank1, rank2)
            if current_state.opp_wager > current_state.my_wager:
                self.opp_raises += 1
                self.opp_actions += 1
            self.aggression_factor = self.opp_raises / max(1, self.opp_actions)
            if street == 'preflop':
                return self.play_preflop(current_state, is_pocket_pair, high_card)
            else:
                return self.play_postflop(current_state)

        def on_hand_end(self, game_info, current_state):
            pass

        def handle_auction(self, current_state):
            pot_size = current_state.pot
            my_chips = current_state.my_chips
            my_cards = current_state.my_hand
            board_cards = current_state.board
            baseline_equity = monte_carlo_equity(my_cards, board_cards, opp_revealed=[], num_simulations=20)
            uncertainty = 1.0 - (2.0 * abs(baseline_equity - 0.5))
            bid_percentage = 0.05 + (0.25 * uncertainty)
            calculated_bid = int(pot_size * bid_percentage)
            safe_bid = max(0, min(calculated_bid, my_chips))
            if current_state.can_act(ActionBid):
                return ActionBid(safe_bid)
            return self.panic_move(current_state)

        def play_preflop(self, current_state, is_pocket_pair, high_card):
            hand_key = get_hand_string(my_cards)
            expected_value = self.preflop_ev_table.get(hand_key, -1.0)
            if expected_value > 1.0:
                return ActionRaise(min_raise)
            elif expected_value > 0.0:
                return ActionCall()
            return ActionFold()

        def play_postflop(self, current_state):
            my_cards = current_state.my_hand
            board_cards = current_state.board
            cost_to_call = current_state.cost_to_call
            pot_size = current_state.pot
            revealed_cards = current_state.opp_revealed_cards
            sims = 50 if self.time_left > 5.0 else 15
            win_prob = monte_carlo_equity(my_cards, board_cards, opp_revealed=revealed_cards, num_simulations=sims)
            pot_odds = cost_to_call / (pot_size + cost_to_call) if cost_to_call > 0 else 0.0
            if win_prob > 0.75:
                if current_state.can_act(ActionRaise):
                    min_r, max_r = current_state.raise_bounds
                    target_raise = int(pot_size * 0.5) if win_prob < 0.90 else pot_size
                    safe_raise = max(min_r, min(target_raise, max_r))
                    return ActionRaise(safe_raise)
            if win_prob > pot_odds:
                if current_state.can_act(ActionCall):
                    return ActionCall()
                if current_state.can_act(ActionCheck):
                    return ActionCheck()
            if current_state.can_act(ActionCheck):
                return ActionCheck()
            return ActionFold()

        def panic_move(self, current_state):
            if current_state.can_act(ActionCheck):
                return ActionCheck()
            return ActionFold()