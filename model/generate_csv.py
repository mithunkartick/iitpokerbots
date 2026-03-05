import os
import csv
import re

# We use the same preflop equity table we generated earlier to classify the opponent's hand
RANK_VALUES = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'T':10, 'J':11, 'Q':12, 'K':13, 'A':14}
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

import os
import csv
import re

# 1. Pre-computed Data 
RANK_VALUES = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'T':10, 'J':11, 'Q':12, 'K':13, 'A':14}
PREFLOP_EQUITY = {
    "AA": 0.852, "KK": 0.824, "QQ": 0.799, "JJ": 0.775, "TT": 0.751, "99": 0.716, "88": 0.687, "AKs": 0.670, 
    "AQs": 0.661, "AKo": 0.653, "AQo": 0.644, "AJs": 0.654, "AJo": 0.636, "KQs": 0.633, "KQo": 0.613,
    # ... (Paste your full 169-hand dictionary here) ...
}

def get_hand_key(cards_str):
    """Converts '[Qs Ks]' to 'KQs'"""
    cards = cards_str.strip('[]').split()
    r1, s1 = cards[0][0], cards[0][1]
    r2, s2 = cards[1][0], cards[1][1]
    
    if RANK_VALUES[r1] < RANK_VALUES[r2]:
        r1, r2 = r2, r1
        
    if r1 == r2: return f"{r1}{r2}"
    elif s1 == s2: return f"{r1}{r2}s"
    else: return f"{r1}{r2}o"

def calculate_target_class(cards_str):
    """Buckets the opponent's true hand into 0 (Trash), 1 (Marginal), 2 (Monster)"""
    hand_key = get_hand_key(cards_str)
    equity = PREFLOP_EQUITY.get(hand_key, 0.40)
    
    if equity > 0.65: return 2
    elif equity > 0.48: return 1
    return 0

def parse_competition_logs(log_dir, my_bot):
    dataset = []
    
    for filename in os.listdir(log_dir):
        if filename.endswith(".log"):
            filepath = os.path.join(log_dir, filename)
            
            with open(filepath, 'r') as file:
                # Read the very first line of the file
                first_line = file.readline().strip() 
                
                # Regex to find "BotA vs BotB" format. \S+ handles names with special characters.
                match = re.search(r'(\S+)\s+vs\s+(\S+)', first_line)
                if not match:
                    print(f"Skipping {filename}: Could not find match header.")
                    continue
                    
                bot1, bot2 = match.groups()
                
                # Dynamic Opponent Detection:
                # If bot1 is Pokerastinot, the opponent is bot2. Otherwise, the opponent is bot1.
                opp_bot = bot2 if bot1 == my_bot else bot1
                
                print(f"Processing {filename} -> You: {my_bot} | Opponent: {opp_bot}")
                
                current_hand = {}
                street = "preflop"
                
                for line in file:
                    line = line.strip()
                    
                    if line.startswith("Round #"):
                        if 'target_class' in current_hand:
                            dataset.append([
                                current_hand.get('preflop_wager', 0),
                                current_hand.get('flop_wager', 0),
                                current_hand.get('auction_bid', 0),
                                current_hand['target_class']
                            ])
                        
                        current_hand = {'preflop_wager': 0, 'flop_wager': 0, 'auction_bid': 0}
                        street = "preflop"
                        
                    elif f"{opp_bot} received" in line:
                        cards_str = re.search(r'\[(.*?)\]', line).group(0)
                        current_hand['target_class'] = calculate_target_class(cards_str)
                        
                    elif line.startswith("Flop"): street = "flop"
                    elif line.startswith("Turn"): street = "turn"
                    
                    elif line.startswith(f"{opp_bot} raises to"):
                        amount = int(line.split()[-1])
                        if street == "preflop": current_hand['preflop_wager'] = amount
                        elif street == "flop": current_hand['flop_wager'] = amount
                    
                    elif line.startswith(f"{opp_bot} calls"):
                        if street == "preflop" and current_hand['preflop_wager'] == 0:
                            current_hand['preflop_wager'] = 20
                            
                    elif line.startswith(f"{opp_bot} bids"):
                        current_hand['auction_bid'] = int(line.split()[-1])
                        
    return dataset

if __name__ == "__main__":
    log_directory = "bot-engine-2026/logs/iter-1-mar-1-logfiles" 
    
    # Plugged in your exact bot name here!
    my_bot_name = "Pokerastinot"
    
    print("Parsing master logs...")
    data = parse_competition_logs(log_directory, my_bot_name)
    
    with open("training_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Preflop_Wager", "Flop_Wager", "Auction_Bid", "Target_Class"])
        writer.writerows(data)
        
    print(f"Success! Extracted {len(data)} perfectly labeled hands.")