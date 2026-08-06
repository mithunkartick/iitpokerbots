import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import random
import json
import time
import eval7
import sys
from collections import deque

# Import EVERYTHING directly from engine to ensure types match internal GameState checks
from engine import (
    GameState, HandResult, STARTING_STACK, BIG_BLIND, SMALL_BLIND,
    ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
)

# ==========================================
# 1. PURE GAME THEORY ABSTRACTION
# ==========================================
# 8 Actions: 4 for the Sneak Peek Auction, 4 for No-Limit Betting
ACTION_DIM = 8

def map_index_to_action(idx, state, active_player):
    valid_actions = state.get_valid_actions()
    
    # --- SNEAK PEEK AUCTION ---
    if state.auction:
        min_bid, max_bid = state.get_bid_limits()
        pot = (STARTING_STACK - state.chips[0]) + (STARTING_STACK - state.chips[1])
        if idx == 0: return ActionBid(0)
        if idx == 1: return ActionBid(min(max_bid, int(pot * 0.1)))
        if idx == 2: return ActionBid(min(max_bid, int(pot * 0.5)))
        if idx == 3: return ActionBid(max_bid)
        return ActionBid(0) # Safety Fallback

    # --- NO-LIMIT BETTING ---
    if idx == 4: return ActionFold() if ActionFold in valid_actions else ActionCheck()
    if idx == 5: return ActionCall() if ActionCall in valid_actions else ActionCheck()
    
    if ActionRaise in valid_actions:
        min_raise, max_raise = state.get_raise_limits()
        pot = (STARTING_STACK - state.chips[0]) + (STARTING_STACK - state.chips[1])
        
        # Exponential Abstraction: Pot-sized and All-In.
        # This naturally restricts tree depth via stack exhaustion, preserving Nash Equilibrium.
        if idx == 6: return ActionRaise(min(max_raise, max(min_raise, int(pot))))
        if idx == 7: return ActionRaise(max_raise)
        
    return ActionCheck() if ActionCheck in valid_actions else ActionFold()

def get_valid_action_mask(state):
    """Dynamically masks illegal actions strictly according to engine.py."""
    mask = torch.zeros(ACTION_DIM)
    if state.auction:
        mask[0:4] = 1.0 # Only Bids allowed during the auction street
    else:
        valid_actions = state.get_valid_actions()
        mask[4:6] = 1.0 # Fold/Check and Call/Check always allowed if valid
        if ActionRaise in valid_actions:
            mask[6:8] = 1.0 # Only allow Pot-Sized and All-In raises
    return mask

# ==========================================
# 2. STRICT STATE ENCODING
# ==========================================
def encode_cards(cards):
    vec = np.zeros(52, dtype=np.float32)
    for card in cards:
        card_str = str(card) # Eval7 fix: cast object to string
        rank_idx = "23456789TJQKA".index(card_str[0].upper())
        suit_idx = "cdhs".index(card_str[1].lower())
        vec[rank_idx * 4 + suit_idx] = 1.0
    return vec

def encode_state(state, active_player):
    """Translates the Poker state into a 164-dimensional float32 tensor."""
    hole_vec = encode_cards(state.hands[active_player])
    board_vec = encode_cards(state.deck.peek(state.street) if state.street > 0 else [])
    revealed_vec = encode_cards(state.opp_hands[active_player])
    
    my_chips = state.chips[active_player] / STARTING_STACK
    opp_chips = state.chips[1 - active_player] / STARTING_STACK
    my_wager = state.wagers[active_player] / STARTING_STACK
    opp_wager = state.wagers[1 - active_player] / STARTING_STACK
    
    pot = ((STARTING_STACK - state.chips[0]) + (STARTING_STACK - state.chips[1])) / (STARTING_STACK * 2)
    street_norm = state.street / 5.0
    is_bb = 1.0 if (state.dealer != active_player) else 0.0
    is_auction = 1.0 if state.auction else 0.0

    numeric_features = np.array([my_chips, opp_chips, my_wager, opp_wager, pot, street_norm, is_bb, is_auction], dtype=np.float32)
    state_vector = np.concatenate([hole_vec, board_vec, revealed_vec, numeric_features])
    return torch.FloatTensor(state_vector)

# ==========================================
# 3. DEEP NEURAL NETWORK
# ==========================================
class DeepCFRNetwork(nn.Module):
    def __init__(self):
        super(DeepCFRNetwork, self).__init__()
        self.fc1 = nn.Linear(164, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, ACTION_DIM)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.out(x)
    
def safe_apply_action(state, action):
    """
    Wraps the engine's apply_action to prevent in-place mutation bugs.
    Creates fresh copies of the mutable lists so tree branches don't corrupt each other.
    """
    safe_bids = list(state.bids)
    safe_opp_hands = [list(state.opp_hands[0]), list(state.opp_hands[1])]
    
    safe_state = GameState(
        state.dealer, state.street, state.auction, safe_bids, 
        state.wagers, state.chips, state.hands, safe_opp_hands, 
        state.deck, state.parent_state
    )
    return safe_state.apply_action(action)

# ==========================================
# 4. EXTERNAL SAMPLING MCCFR WORKER
# ==========================================
def traverse_mccfr(state, traversing_player, p0, p1, regret_net, strategy_net, adv_queue, strat_queue):
    # Base Case: Terminal Showdown or Fold
    if isinstance(state, HandResult):
        return state.payoffs[traversing_player] / STARTING_STACK
        
    active_player = state.dealer % 2
    
    # --- UNIFIED DECISION NODE ---
    # The auction is now treated natively. State encoding perfectly hides the sealed bid.
    state_tensor = encode_state(state, active_player)
    mask = get_valid_action_mask(state)
    
    with torch.no_grad():
        regrets = regret_net(state_tensor.unsqueeze(0)).squeeze(0)
    
    positive_regrets = torch.clamp(regrets, min=0.0) * mask
    sum_regrets = torch.sum(positive_regrets)
    
    if sum_regrets > 0:
        strategy = positive_regrets / sum_regrets
    else:
        strategy = mask / torch.sum(mask)

    if active_player == traversing_player:
        # EXTERNAL SAMPLING: Explore ALL valid actions (including the 4 bids!)
        action_values = torch.zeros(ACTION_DIM)
        node_value = 0.0
        
        for a in range(ACTION_DIM):
            if mask[a] == 1:
                engine_action = map_index_to_action(a, state, active_player)
                
                # Apply the safe wrapper to prevent engine memory corruption
                next_state = safe_apply_action(state, engine_action)
                
                val = traverse_mccfr(next_state, traversing_player, p0, p1, regret_net, strategy_net, adv_queue, strat_queue)
                action_values[a] = val
                node_value += strategy[a] * val
                
        # Calculate regrets and push to GPU! The network finally learns to bid.
        cf_regrets = (action_values - node_value) * mask
        weight = p1 if traversing_player == 0 else p0
        
        adv_queue.put((state_tensor.numpy(), (cf_regrets * weight).numpy()))
        strat_queue.put((state_tensor.numpy(), (strategy * (p0 if traversing_player == 0 else p1)).numpy()))
        
        return node_value
    else:
        # EXTERNAL SAMPLING: Sample ONE action for the opponent
        action_idx = torch.multinomial(strategy, 1).item()
        engine_action = map_index_to_action(action_idx, state, active_player)
        
        next_state = safe_apply_action(state, engine_action)
        
        if active_player == 0: p0 *= strategy[action_idx].item()
        else: p1 *= strategy[action_idx].item()
            
        return traverse_mccfr(next_state, traversing_player, p0, p1, regret_net, strategy_net, adv_queue, strat_queue)

def data_generation_worker(worker_id, regret_net, strategy_net, adv_queue, strat_queue, games_played_counter):
    # CRITICAL: Prevents 1600 threads from choking your CPU cache
    torch.set_num_threads(1) 
    
    round_num = 1
    
    while True:
        deck = eval7.Deck()
        deck.shuffle()
        hands = [deck.deal(2), deck.deal(2)]
        dealer_idx = (round_num + 1) % 2 
        
        wagers = [0, 0]
        chips = [STARTING_STACK, STARTING_STACK]
        wagers[dealer_idx], chips[dealer_idx] = SMALL_BLIND, STARTING_STACK - SMALL_BLIND
        wagers[1 - dealer_idx], chips[1 - dealer_idx] = BIG_BLIND, STARTING_STACK - BIG_BLIND
        
        state = GameState(dealer_idx, 0, False, [None, None], wagers, chips, hands, [[], []], deck, None)
        traversing_player = round_num % 2
        
        try:
            traverse_mccfr(state, traversing_player, 1.0, 1.0, regret_net, strategy_net, adv_queue, strat_queue)
        except Exception as e:
            # We must print the error now so it doesn't fail silently
            print(f"[WORKER {worker_id}] Error: {str(e)}")
            
        round_num += 1
        
        # Update the UI counter EVERY hand so you can see it moving instantly
        with games_played_counter.get_lock():
            games_played_counter.value += 1

# ==========================================
# 5. DUAL-GPU LEARNER & EXPORTER
# ==========================================
def learner_process(regret_net, strategy_net, adv_queue, strat_queue, dev_regret, dev_strat, games_played_counter):
    regret_opt = optim.Adam(regret_net.parameters(), lr=0.001)
    strat_opt = optim.Adam(strategy_net.parameters(), lr=0.001)
    
    adv_buffer = deque(maxlen=200000)
    strat_buffer = deque(maxlen=200000)
    
    batch_size = 1024
    print(f"\n[MAIN] GPU Learner initialized.")
    print(f"[MAIN] Regret Network assigned to: {torch.cuda.get_device_name(dev_regret)} ({dev_regret})")
    print(f"[MAIN] Strategy Network assigned to: {torch.cuda.get_device_name(dev_strat)} ({dev_strat})")
    print(f"[MAIN] Target duration: 4.75 hours. Waiting for buffers to fill...\n")
    
    start_time = time.time()
    last_print_time = start_time
    MAX_TRAIN_TIME = 2 * 3600 
    
    current_regret_loss, current_strat_loss = 0.0, 0.0
    
    while time.time() - start_time < MAX_TRAIN_TIME:
        # Drain queues into memory buffers
        while not adv_queue.empty(): adv_buffer.append(adv_queue.get())
        while not strat_queue.empty(): strat_buffer.append(strat_queue.get())
            
        # GPU 0: Train Regret Network
        if len(adv_buffer) > batch_size:
            batch = random.sample(adv_buffer, batch_size)
            states, targets = zip(*batch)
            states = torch.FloatTensor(np.array(states)).to(dev_regret)
            targets = torch.FloatTensor(np.array(targets)).to(dev_regret)
            
            regret_opt.zero_grad()
            preds = regret_net(states)
            loss = F.mse_loss(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(regret_net.parameters(), max_norm=1.0)
            regret_opt.step()
            current_regret_loss = loss.item()
            
        # GPU 1: Train Strategy Network
        if len(strat_buffer) > batch_size:
            batch = random.sample(strat_buffer, batch_size)
            states, targets = zip(*batch)
            states = torch.FloatTensor(np.array(states)).to(dev_strat)
            targets = torch.FloatTensor(np.array(targets)).to(dev_strat)
            
            strat_opt.zero_grad()
            preds = strategy_net(states)
            loss = torch.sum(-targets * F.log_softmax(preds, dim=-1), dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(strategy_net.parameters(), max_norm=1.0)
            strat_opt.step()
            current_strat_loss = loss.item()

        # Monitoring Printout
        current_time = time.time()
        if current_time - last_print_time >= 30:
            elapsed = current_time - start_time
            hrs, rem = divmod(elapsed, 3600)
            mins, secs = divmod(rem, 60)
            
            print(f"[LEARNER STATUS | {int(hrs):02d}:{int(mins):02d}:{int(secs):02d}]")
            print(f"  -> Total Hands Traversed : {games_played_counter.value:,}")
            print(f"  -> Buffer Sizes          : Advantage [{len(adv_buffer)}/200K] | Strategy [{len(strat_buffer)}/200K]")
            print(f"  -> Current Loss          : Regret {current_regret_loss:.5f} | Strategy {current_strat_loss:.5f}")
            print("-" * 60)
            last_print_time = current_time

    print("\n[MAIN] Target time reached. Exporting Final Strategy weights...")
    weights_dict = {name: tensor.cpu().numpy().tolist() for name, tensor in strategy_net.state_dict().items()}
    print(weights_dict)
    with open("bot_weights.json", "w") as f:
        json.dump(weights_dict, f)
    print("[MAIN] Export successful to 'bot_weights.json'.")

# ==========================================
# 6. EXECUTION ENTRY POINT
# ==========================================
if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    print("==================================================")
    print("      SNEAK PEEK HOLD'EM - DEEP MCCFR TRAINER     ")
    print("==================================================")
    
    # 1. HARD-FAIL CUDA CHECK
    if not torch.cuda.is_available():
        print("[CRITICAL ERROR] PyTorch cannot detect your GPUs!")
        sys.exit(1)
        
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        print(f"[WARNING] Only {gpu_count} GPU(s) detected. Defaulting both networks to cuda:0.")
        dev_regret = torch.device("cuda:0")
        dev_strat = torch.device("cuda:0")
    else:
        print(f"[MAIN] Verified Hardware: 2x GPUs detected. Engaging Dual-GPU Training.")
        dev_regret = torch.device("cuda:0")
        dev_strat = torch.device("cuda:1")
    
    # Shared CPU networks for the 40 worker cores to read from
    shared_regret_net = DeepCFRNetwork().share_memory()
    shared_strat_net = DeepCFRNetwork().share_memory()
    
    # Specific GPU networks for the central Learner to train
    learner_regret_net = DeepCFRNetwork().to(dev_regret)
    learner_strat_net = DeepCFRNetwork().to(dev_strat)
    learner_regret_net.load_state_dict(shared_regret_net.state_dict())
    learner_strat_net.load_state_dict(shared_strat_net.state_dict())
    
    # Multi-processing Queues and Counters
    adv_queue = mp.Queue(maxsize=200000)
    strat_queue = mp.Queue(maxsize=200000)
    games_played_counter = mp.Value('i', 0)
    
    num_cpus = 40
    print(f"[MAIN] Spawning {num_cpus} Data Generation Workers...")
    workers = []
    for i in range(num_cpus):
        p = mp.Process(target=data_generation_worker, args=(i, shared_regret_net, shared_strat_net, adv_queue, strat_queue, games_played_counter))
        p.start()
        workers.append(p)
        
    try:
        # Launch the asynchronous Learner on the main thread
        learner_process(learner_regret_net, learner_strat_net, adv_queue, strat_queue, dev_regret, dev_strat, games_played_counter)
    except KeyboardInterrupt:
        print("\n[MAIN] Training interrupted manually! Executing emergency export...")
        weights_dict = {name: tensor.cpu().numpy().tolist() for name, tensor in learner_strat_net.state_dict().items()}
        print(weights_dict)
        with open("bot_weights.json", "w") as f:
            json.dump(weights_dict, f)
        print("[MAIN] Emergency export saved to 'bot_weights.json'.")
            
    # Cleanup
    for p in workers:
        p.terminate()