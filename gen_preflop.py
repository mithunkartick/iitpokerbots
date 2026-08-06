import json
import random
from treys import Card, Evaluator

evaluator = Evaluator()
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']
FULL_DECK_STRINGS = [r+s for r in RANKS for s in SUITS]

def simulate_hand_equity(hand_strings, num_simulations=5000):
    """
    Runs Monte Carlo rollouts for a specific starting hand against a random opponent.
    """
    hand_treys = [Card.new(c) for c in hand_strings]
    remaining_deck = [c for c in FULL_DECK_STRINGS if c not in hand_strings]
    
    wins = 0
    ties = 0

    for _ in range(num_simulations):
        drawn = random.sample(remaining_deck, 7)
        opp_treys = [Card.new(c) for c in drawn[:2]]
        board_treys = [Card.new(c) for c in drawn[2:]]
        
        my_score = evaluator.evaluate(board_treys, hand_treys)
        opp_score = evaluator.evaluate(board_treys, opp_treys)
        
        if my_score < opp_score:
            wins += 1
        elif my_score == opp_score:
            ties += 1

    return (wins + (0.5 * ties)) / num_simulations

def generate_169_hands():
    """
    Generates the 169 unique Texas Hold'em starting hands.
    - 13 Pocket Pairs (e.g., 'AA', 'KK')
    - 78 Suited Hands (e.g., 'AKs', 'T9s')
    - 78 Offsuit Hands (e.g., 'AKo', 'T9o')
    """
    equity_table = {}
    total_hands = 169
    current = 0
    
    print(f"Starting Monte Carlo simulation for all {total_hands} starting hands...")
    
    for i in range(len(RANKS)):
        for j in range(i, len(RANKS)):
            rank1 = RANKS[j]
            rank2 = RANKS[i]
            
            if rank1 == rank2:
                key = f"{rank1}{rank2}"
                cards = [f"{rank1}s", f"{rank2}h"]
                equity = simulate_hand_equity(cards)
                equity_table[key] = round(equity, 4)
                
            else:
                key_suited = f"{rank1}{rank2}s"
                cards_suited = [f"{rank1}s", f"{rank2}s"]
                equity_suited = simulate_hand_equity(cards_suited)
                equity_table[key_suited] = round(equity_suited, 4)
                
                key_offsuit = f"{rank1}{rank2}o"
                cards_offsuit = [f"{rank1}s", f"{rank2}h"]
                equity_offsuit = simulate_hand_equity(cards_offsuit)
                equity_table[key_offsuit] = round(equity_offsuit, 4)
                
            current += 1 if rank1 == rank2 else 2
            if current % 10 == 0 or current == total_hands:
                print(f"Processed {current}/{total_hands} hands...")

    return equity_table

if __name__ == "__main__":
    final_table = generate_169_hands()
    
    with open("preflop_equity.json", "w") as f:
        json.dump(final_table, f, indent=4)
        
    print("\nSuccess! Saved to preflop_equity.json")
    print("Example Data:", json.dumps({k: final_table[k] for k in list(final_table.keys())[:5]}, indent=4))