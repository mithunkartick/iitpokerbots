import os
# ==========================================
# HPC CLUSTER SAFETY LOCKS (PREVENT THREAD EXPLOSION)
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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

# 1. Direct Engine Imports (NO PKBOT WRAPPERS)
from engine import GameState, HandResult, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from engine import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid

# ==========================================
# CONSTANTS & VECTORIZER
# ==========================================
ACTIONS = 4  
STATE_DIM = 165
BATCH_SIZE = 4096
EPOCHS = 1000

DECK_STRINGS = [r+s for r in '23456789TJQKA' for s in 'shdc']
DECK_MAP = {card: i for i, card in enumerate(DECK_STRINGS)}

def vectorize_state(state: GameState, active_idx: int):
    """ Reads directly from the raw engine.GameState physics block """
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

    hole_vec = np.zeros(52); board_vec = np.zeros(52); revealed_vec = np.zeros(52)
    
    for c in state.hands[active_idx]: 
        hole_vec[DECK_MAP[str(c)]] = 1.0
        
    board_cards = state.deck.peek(state.street) if state.street > 0 else []
    for c in board_cards: 
        board_vec[DECK_MAP[str(c)]] = 1.0
        
    opp_revealed = state.opp_hands[active_idx]
    auc_status = np.zeros(4)
    if state.street <= 3 and not opp_revealed:
        auc_status[0] = 1.0 
    elif len(opp_revealed) > 0:
        auc_status[1] = 1.0 
        for c in opp_revealed: 
            revealed_vec[DECK_MAP[str(c)]] = 1.0
    else:
        auc_status[2] = 1.0 

    vec = np.concatenate([
        [pot_ratio, stack_ratio, cost_ratio, street_val, button_val],
        auc_status, hole_vec, board_vec, revealed_vec
    ])
    return vec

# ==========================================
# NEURAL NETWORK ARCHITECTURE
# ==========================================
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
        bid_fractions = [0.0, 0.15, 0.33, 1.0]
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
        target = int(pot * 0.56) if action_idx == 2 else int(pot * 1.03)
        return ActionRaise(max(min_r, min(target, max_r))) 
        
    if ActionCall in valid_actions: return ActionCall() 
    return ActionCheck() if ActionCheck in valid_actions else ActionFold() 

# ==========================================
# GAME TREE TRAVERSAL
# ==========================================
def traverse(state, learning_player, p0_prob, p1_prob, local_model, opp_model, data_queue, depth=0):
    if isinstance(state, HandResult):
        # THE FIX: Normalize raw chip payoffs to a [-1.0, 1.0] scale
        return state.payoffs[learning_player] / float(STARTING_STACK)

    if depth > 15:
        return 0.0 

    active_idx = state.dealer % 2 
    state_arr = vectorize_state(state, active_idx)
    state_tensor = torch.FloatTensor(state_arr).unsqueeze(0)

    current_model = local_model if active_idx == learning_player else opp_model
    
    with torch.no_grad():
        regrets = current_model(state_tensor).squeeze(0)
    strategy = get_strategy(regrets).numpy()

    if active_idx == learning_player:
        ev_vals = np.zeros(ACTIONS)
        node_val = 0.0
        
        for a_idx in range(ACTIONS):
            engine_action = map_abstract_action(state, a_idx)
            next_state = state.apply_action(engine_action) 
            
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
        engine_action = map_abstract_action(state, a_idx)
        next_state = state.apply_action(engine_action) 
        return traverse(next_state, learning_player, 
                        p0_prob * strategy[a_idx] if active_idx == 0 else p0_prob,
                        p1_prob * strategy[a_idx] if active_idx == 1 else p1_prob, 
                        local_model, opp_model, data_queue, depth + 1)

# ==========================================
# ASYNCHRONOUS CPU WORKER
# ==========================================
def cpu_worker(worker_id, data_queue, shared_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    local_model = PokerNet()
    opp_model = PokerNet()
    
    print(f"✅ [Worker {worker_id}] Successfully booted and standing by.", flush=True)
    games_played = 0
    worker_start = time.time()

    while True:
        try:
            if games_played % 50 == 0:
                if games_played > 0:
                    elapsed = time.time() - worker_start
                    print(f"⚡ [Worker {worker_id}] Processed 50 hands in {elapsed:.2f}s", flush=True)
                    worker_start = time.time()

                if 'latest_weights' in shared_dict:
                    np_dict = shared_dict['latest_weights']
                    tensor_dict = {k: torch.tensor(v) for k, v in np_dict.items()}
                    local_model.load_state_dict(tensor_dict)
                    
                if 'ghost_weights' in shared_dict and len(shared_dict['ghost_weights']) > 0:
                    np_ghost = random.choice(shared_dict['ghost_weights'])
                    tensor_ghost = {k: torch.tensor(v) for k, v in np_ghost.items()}
                    opp_model.load_state_dict(tensor_ghost)

            deck = eval7.Deck()
            deck.shuffle()
            hands = [deck.deal(2), deck.deal(2)] 
            wagers = [SMALL_BLIND, BIG_BLIND] 
            chips = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND] 
            
            root_state = GameState(0, 0, False, [None, None], wagers, chips, hands, [[], []], deck, None) 
            learning_player = random.choice([0, 1])
            
            traverse(root_state, learning_player, 1.0, 1.0, local_model, opp_model, data_queue, depth=0)
            games_played += 1

        except Exception as e:
            print(f"\n❌ [Worker {worker_id}] FATAL CRASH: {str(e)}", flush=True)
            traceback.print_exc()
            time.sleep(10)

# ==========================================
# MULTI-GPU MASTER LOOP
# ==========================================
def run_master():
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    data_queue = manager.Queue(maxsize=50000)
    shared_dict = manager.dict()
    shared_dict['ghost_weights'] = []
    
    NUM_WORKERS = max(1, mp.cpu_count() - 2)
    print(f"🔄 [Master] Hardware detected. Spawning {NUM_WORKERS} Asynchronous CPU Workers...", flush=True)
    
    workers = [mp.Process(target=cpu_worker, args=(i, data_queue, shared_dict)) for i in range(NUM_WORKERS)]
    for p in workers: p.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 [Master] Booting. Hardware Accelerated on: {device.type.upper()}", flush=True)
    
    master_model = PokerNet().to(device)
    optimizer = optim.Adam(master_model.parameters(), lr=0.001)
    
    safe_dict = {k: v.cpu().numpy() for k, v in master_model.state_dict().items()}
    shared_dict['latest_weights'] = safe_dict

    start_time = time.time()
    for epoch in range(EPOCHS):
        batch_states, batch_targets = [], []
        last_print = 0
        
        while len(batch_states) < BATCH_SIZE:
            if not data_queue.empty():
                s, r = data_queue.get()
                batch_states.append(s)
                batch_targets.append(r)
                
                if len(batch_states) % 500 == 0 and len(batch_states) != last_print:
                    print(f"⏳ [Master] Gathering Batch Queue: {len(batch_states)} / {BATCH_SIZE}...", flush=True)
                    last_print = len(batch_states)
        
        states = torch.tensor(np.array(batch_states), dtype=torch.float32, device=device)
        targets = torch.tensor(np.array(batch_targets), dtype=torch.float32, device=device)
        
        optimizer.zero_grad()
        predictions = master_model(states)
        loss = nn.MSELoss()(predictions, targets)
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        loss.backward()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(master_model.parameters(), max_norm=1.0)
        optimizer.step()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        with torch.no_grad():
            weight_norm = sum(p.norm().item() for p in master_model.parameters() if p.requires_grad)
            
        print(f"⚙️ [Optimizer] Loss: {loss.item():.5f} | Raw Grad Norm: {grad_norm:.4f} | Total Weight Norm: {weight_norm:.2f}")
        
        safe_dict = {k: v.cpu().numpy() for k, v in master_model.state_dict().items()}
        shared_dict['latest_weights'] = safe_dict
        
        if epoch % 50 == 0:
            ghost_pool = list(shared_dict['ghost_weights'])
            ghost_pool.append(safe_dict)
            if len(ghost_pool) > 10: ghost_pool.pop(0)
            shared_dict['ghost_weights'] = ghost_pool
            
            elapsed = (time.time() - start_time) / 60.0
            print(f"\n=================================================")
            print(f"🏆 Epoch {epoch}/{EPOCHS} Complete | Loss: {loss.item():.4f} | Time: {elapsed:.1f}m")
            print(f"=================================================\n", flush=True)

    print("✅ [Master] Training Complete. Terminating workers...")
    for p in workers: p.terminate()
    

    # ==========================================
    # BRAIN PAYLOAD EXPORTER
    # ==========================================
    print("\n🗜️ Generating BRAIN_PAYLOAD...")
    # Downcast to FP16 to halve the file size
    half_dict = {k: v.cpu().half() for k, v in master_model.state_dict().items()}
    buffer = io.BytesIO()
    torch.save(half_dict, buffer)
    raw_bytes = buffer.getvalue()
    
    # Compress and Encode
    compressed_bytes = zlib.compress(raw_bytes, level=9)
    brain_payload = base64.b64encode(compressed_bytes).decode('utf-8')

    print(f'BRAIN_PAYLOAD = "{brain_payload}"\n')
    
    with open("brain_payload.txt", "w") as f:
        f.write(f'BRAIN_PAYLOAD = "{brain_payload}"\n')
        
    print(f"✅ Payload successfully written to 'brain_payload.txt'")
    print(f"📉 Raw Size: {len(raw_bytes)/1024:.2f} KB | Compressed: {len(compressed_bytes)/1024:.2f} KB")
    
    torch.save(master_model.state_dict(), "/scratch/mithun_kb_ph.iitr/pokerastinot_v100_final.pth")
    print("💾 Model saved to '/scratch/mithun_kb_ph.iitr/pokerastinot_v100_final.pth' successfully.")

if __name__ == '__main__':
    run_master()