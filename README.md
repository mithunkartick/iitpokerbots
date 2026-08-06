# iitpokerbots
Building a Pokerbot using RL and Basic Game Theory to tackle a specific version of poker.

# IIT Pokerbots: The Journey of building a Deep CFR Pokerbot

Welcome to the development documentation for our competition bot built for the **IIT Pokerbots: Sneak Peek No-Limit Texas Hold'em Tournament**. 

This repository logs our progression from simple statistical models to an advanced **Deep Counterfactual Regret Minimization (Deep CFR)** strategy. Although our final submission took a different path due to training constraints, this document details the full scope of our engineering efforts.

---

## Project Overview & Ruleset

The tournament features a unique variant of **Heads-Up No-Limit Texas Hold'em** with a custom **Auction Phase** ("Sneak Peek"):
* **Starting Stacks:** 400 chips (200 BB)
* **Blinds:** 1 SB / 2 BB
* **The Auction (Sneak Peek):** Prior to the Turn, players submit secret bids. The highest bidder pays their bid to the pot and inspects one of the opponent’s hole cards.
* **Timing Constraints:** 2.0 seconds max per query, with a total 20.0-second time bank per hand/match.

---

## The Development Progression

### Phase 1: Pure Monte Carlo Engine (O(N) Evaluation)
* **Strategy:** Built an `eval7`-backed Monte Carlo simulator to evaluate showdown equity post-flop.
* **Improvements Made:** 
  * Implemented an O(1) pre-flop look-up table (LUT) to eliminate pre-flop simulation overhead.
  * Optimized deck-sampling using low-level C calls within `eval7` to hoard processing power for later streets.
* **The Ceiling:** While fast, pure Monte Carlo only calculates **Showdown Equity**—it has zero concept of **Fold Equity**. It played predictably ("Nit style"), making it vulnerable to aggressive stealing and bluff trapping.

### Phase 2: Heuristic "Aggro-MC" Iteration
* **Strategy:** Embedded rule-based bluffs (Continuation Bets, Semi-Bluffs on flush/straight draws, and "Sneak Peek" info bluffs).
* **The Pitfall:** Adding hardcoded bluffs against deterministic bots created a classic Game Theory leak. When opponent bots called our bluffs, they held mathematically strong hands, causing our bot to bleed chips into traps despite winning more individual hands.

### Phase 3: Transition to Deep CFR
* **Strategy:** Switched from Monte Carlo heuristics to a Deep Counterfactual Regret Minimization model to naturally calculate **Nash Equilibrium**.
* **Architecture Highlights:**
  * **217-Dimensional State Vector:** Custom-engineered vectorizer encoding pot size, relative stack sizes, legal bounds, 52-bit card vectors, street indicators, and Sneak Peek auction statuses.
  * **Dual-Head Neural Network:** Predicts both categorical action regrets (Fold, Check/Call, Raise, Bid) and continuous action sizing (Bet/Bid multipliers).
  * **Server/Client Adaptation:** Built an `EngineStateAdapter` in `bot.py` to seamlessly map runtime `PokerState` objects into the internal `GameState` tensor expected by the neural network without performance drops.

---

## System Architecture

```text
├── bot.py                        # Tournament runtime entry point (BaseBot subclass)
├── train_deep_cfr_iit.py         # Deep CFR training pipeline and evaluation loops
├── src/
│   ├── core/
│   │   ├── deep_cfr.py           # MCCFR tree traversal and prioritized memory
│   │   └── model.py              # PyTorch PokerNetwork (217-D input -> 4 actions + sizing)
│   └── utils/
│       ├── logging.py            # Error tracking and game log parsers
│       └── settings.py           # Global strict checking flags
└── models/                       # Checkpoints (.pt weights)
```
---

## Final Results & Reflections

Ultimately, we were unable to complete the training for the Deep CFR agent in time for the tournament deadline. As a result, our final submission relied on our Phase 1 `eval7` Monte-Carlo strategy, which secured us a rank within the top 250. 

While we strongly believe the Deep CFR approach would have performed tremendously better had the training completed, the engineering journey itself was incredibly rewarding. Building out the state adapters, understanding the limits of heuristic poker, and designing a game-theory optimal architecture was a massive learning experience.

---

## Acknowledgments & Methodology

We extend our sincere gratitude to the **PARAMGANGA facility provided by the Institute Computer Centre, IITR**, for generously providing us with the computational resources required to attempt training the Deep CFR model. 

This project was built using an iterative engineering process. Minimal to moderate assistance from Large Language Models (LLMs) was utilized throughout development to:
* Accelerate boilerplate integration between external libraries (`eval7`, `pkbot`, PyTorch).
* Debug edge-case tensor shape mismatches and state serialization bugs.
* Benchmark theoretical game-tree architectures against engine mechanics.

All core strategic design decisions, state-vector feature engineering, mathematical trade-off analyses, and system debugging were guided and validated through rigorous local benchmarking by our team.