import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import random
import time
import eval7
import traceback
import base64
import zlib
import io
import json

from engine import GameState, HandResult, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from engine import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid

# ==========================================
# HEAVYWEIGHT CONSTANTS
# ==========================================
ACTIONS = 5  
STATE_DIM = 217
BATCH_SIZE = 8192
EPOCHS = 3500  # Scaled for ~4.5 hours on dual V100s

DECK_STRINGS = [r+s for r in '23456789TJQKA' for s in 'shdc']
DECK_MAP = {card: i for i, card in enumerate(DECK_STRINGS)}

def vectorize_state(state: GameState, active_idx: int):
    my_chips = state.chips[active_idx]
    opp_chips = state.chips[1-active_idx]
    pot = (STARTING_STACK - my_chips) + (STARTING_STACK - opp_chips)
    cost_to_call = state.wagers[1-active_idx] - state.wagers[active_idx]
    
    pot_ratio = min(1.0, pot / 20000.0) 
    stack_ratio = min(1.0, my_chips / 10000.0)
    cost_ratio = min(1.0, cost_to_call / max(1, pot))
    
    if getattr(state, 'auction', False):
        street_val = 0.25
    else:
        street_map = {0: 0.0, 3: 0.5, 4: 0.75, 5: 1.0}
        street_val = street_map.get(state.street, 0.0)
        
    button_val = 1.0 if active_idx == 0 else 0.0 

    hole_vec = np.zeros(52)
    board_vec = np.zeros(52)
    opp_revealed_vec = np.zeros(52)
    my_exposed_vec = np.zeros(52)  # THE NEW 52-DIM VECTOR
    
    for c in state.hands[active_idx]: hole_vec[DECK_MAP[str(c)]] = 1.0
        
    board_cards = state.deck.peek(state.street) if state.street > 0 else []
    for c in board_cards: board_vec[DECK_MAP[str(c)]] = 1.0
        
    opp_revealed = state.opp_hands[active_idx]
    my_exposed = state.opp_hands[1-active_idx]
    
    auc_status = np.zeros(4)
    if state.street <= 3 and not opp_revealed and not my_exposed:
        auc_status[0] = 1.0 
    elif len(opp_revealed) > 0:
        auc_status[1] = 1.0 
        for c in opp_revealed: opp_revealed_vec[DECK_MAP[str(c)]] = 1.0
    elif len(my_exposed) > 0:
        auc_status[2] = 1.0
        for c in my_exposed: my_exposed_vec[DECK_MAP[str(c)]] = 1.0
    else:
        auc_status[3] = 1.0 

    vec = np.concatenate([
        [pot_ratio, stack_ratio, cost_ratio, street_val, button_val],
        auc_status, hole_vec, board_vec, opp_revealed_vec, my_exposed_vec
    ])
    return vec

class PokerNet(nn.Module):
    def __init__(self):
        super(PokerNet, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, ACTIONS)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.size(0) == 1: self.eval() 
        else: self.train()
        x = self.relu(self.fc1(x))
        if x.size(0) > 1: x = self.bn1(x)
        x = self.relu(self.fc2(x))
        if x.size(0) > 1: x = self.bn2(x)
        return self.fc4(self.relu(self.fc3(x)))

def get_strategy(regrets):
    pos_regrets = torch.clamp(regrets, min=0.0)
    sum_regrets = torch.sum(pos_regrets)
    return pos_regrets / sum_regrets if sum_regrets > 0 else torch.ones(ACTIONS) / ACTIONS

def map_abstract_action(state: GameState, action_idx):
    valid_actions = state.get_valid_actions() 
    
    if ActionBid in valid_actions:
        min_bid, max_bid = state.get_bid_limits() 
        pot = (STARTING_STACK - state.chips[0]) + (STARTING_STACK - state.chips[1])
        bid_fractions = [0.0, 0.20, 0.40, 0.75, 1.0]
        amt = int(pot * bid_fractions[action_idx])
        return ActionBid(max(min_bid, min(amt, max_bid))) 
        
    if action_idx == 0:
        return ActionCheck() if ActionCheck in valid_actions else ActionFold() 
    if action_idx == 1:
        if ActionCall in valid_actions: return ActionCall() 
        return ActionCheck() if ActionCheck in valid_actions else ActionFold() 
        
    if ActionRaise in valid_actions: 
        min_r, max_r = state.get_raise_limits() 
        pot = (STARTING_STACK - state.chips[0]) + (STARTING_STACK - state.chips[1])
        if action_idx == 2: target = int(pot * 0.5)
        elif action_idx == 3: target = int(pot * 1.0)
        else: target = max_r
        return ActionRaise(max(min_r, min(target, max_r))) 
        
    if ActionCall in valid_actions: return ActionCall() 
    return ActionCheck() if ActionCheck in valid_actions else ActionFold() 

def traverse(state, learning_player, p0_prob, p1_prob, local_model, opp_model, data_queue, depth=0):
    if isinstance(state, HandResult):
        return state.payoffs[learning_player] / float(STARTING_STACK)

    if depth > 15: return 0.0 

    active_idx = state.dealer % 2 
    state_arr = vectorize_state(state, active_idx)
    state_tensor = torch.FloatTensor(state_arr).unsqueeze(0)

    current_model = local_model if active_idx == learning_player else opp_model
    with torch.no_grad(): regrets = current_model(state_tensor).squeeze(0)
    strategy = get_strategy(regrets).numpy()

    if active_idx == learning_player:
        ev_vals = np.zeros(ACTIONS)
        node_val = 0.0
        for a_idx in range(ACTIONS):
            next_state = state.apply_action(map_abstract_action(state, a_idx)) 
            ev = traverse(next_state, learning_player, 
                          p0_prob * strategy[a_idx] if active_idx == 0 else p0_prob,
                          p1_prob * strategy[a_idx] if active_idx == 1 else p1_prob, 
                          local_model, opp_model, data_queue, depth + 1)
            ev_vals[a_idx] = ev
            node_val += strategy[a_idx] * ev
            
        regret_updates = np.maximum(0.0, ev_vals - node_val)
        data_queue.put((state_arr, regret_updates))
        return node_val
    else:
        a_idx = np.random.choice(ACTIONS, p=strategy)
        next_state = state.apply_action(map_abstract_action(state, a_idx)) 
        return traverse(next_state, learning_player, 
                        p0_prob * strategy[a_idx] if active_idx == 0 else p0_prob,
                        p1_prob * strategy[a_idx] if active_idx == 1 else p1_prob, 
                        local_model, opp_model, data_queue, depth + 1)

def cpu_worker(worker_id, data_queue, shared_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    local_model, opp_model = PokerNet(), PokerNet()
    games_played = 0
    
    while True:
        try:
            if games_played % 50 == 0:
                if 'latest_weights' in shared_dict:
                    local_model.load_state_dict({k: torch.tensor(v) for k, v in shared_dict['latest_weights'].items()})
                if 'ghost_weights' in shared_dict and len(shared_dict['ghost_weights']) > 0:
                    opp_model.load_state_dict({k: torch.tensor(v) for k, v in random.choice(shared_dict['ghost_weights']).items()})

            deck = eval7.Deck()
            deck.shuffle()
            hands = [deck.deal(2), deck.deal(2)] 
            wagers = [SMALL_BLIND, BIG_BLIND] 
            chips = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND] 
            
            root_state = GameState(0, 0, False, [None, None], wagers, chips, hands, [[], []], deck, None) 
            traverse(root_state, random.choice([0, 1]), 1.0, 1.0, local_model, opp_model, data_queue, depth=0)
            games_played += 1

        except Exception:
            time.sleep(5)

def run_master():
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    data_queue = manager.Queue(maxsize=100000)
    shared_dict = manager.dict()
    shared_dict['ghost_weights'] = []
    
    NUM_WORKERS = max(1, mp.cpu_count() - 2)
    print(f"🔄 Spawning {NUM_WORKERS} Workers...", flush=True)
    workers = [mp.Process(target=cpu_worker, args=(i, data_queue, shared_dict)) for i in range(NUM_WORKERS)]
    for p in workers: p.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master_model = PokerNet().to(device)
    optimizer = optim.Adam(master_model.parameters(), lr=0.001)
    
    shared_dict['latest_weights'] = {k: v.cpu().numpy() for k, v in master_model.state_dict().items()}

    for epoch in range(EPOCHS):
        batch_states, batch_targets = [], []
        while len(batch_states) < BATCH_SIZE:
            if not data_queue.empty():
                s, r = data_queue.get()
                batch_states.append(s)
                batch_targets.append(r)
        
        states = torch.tensor(np.array(batch_states), dtype=torch.float32, device=device)
        targets = torch.tensor(np.array(batch_targets), dtype=torch.float32, device=device)
        
        optimizer.zero_grad()
        loss = nn.MSELoss()(master_model(states), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(master_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        safe_dict = {k: v.cpu().numpy() for k, v in master_model.state_dict().items()}
        shared_dict['latest_weights'] = safe_dict
        
        if epoch % 50 == 0:
            ghost_pool = list(shared_dict['ghost_weights'])
            ghost_pool.append(safe_dict)
            if len(ghost_pool) > 10: ghost_pool.pop(0)
            shared_dict['ghost_weights'] = ghost_pool
            print(f"🏆 Epoch {epoch}/{EPOCHS} | Loss: {loss.item():.4f}", flush=True)

    for p in workers: p.terminate()
    
    print("\n🗜️ Generating JSON-Safe BRAIN_PAYLOAD...")
    safe_weights = {k: master_model.state_dict()[k].cpu().half().numpy().tolist() 
                    for k in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias', 'fc4.weight', 'fc4.bias']}
    
    json_text = json.dumps(safe_weights).encode('utf-8')
    brain_payload = base64.b64encode(zlib.compress(json_text, level=9)).decode('utf-8')

    with open("/scratch/mithun_kb_ph.iitr/brain_payload_217.txt", "w") as f:
        f.write(f'BRAIN_PAYLOAD = "{brain_payload}"\n')
    torch.save(master_model.state_dict(), "/scratch/mithun_kb_ph.iitr/pokerastinot_217_final.pth")

if __name__ == '__main__':
    run_master()