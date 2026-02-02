## Overview
This project implements and compares various search algorithms on the Taxi-v3 environment from OpenAI Gymnasium.

## Taxi-v3 Game
The Taxi environment is a classic reinforcement learning problem where:
- **Objective**: Pick up a passenger at one location and drop them off at their destination
- **Grid**: 5x5 grid with 4 designated locations (Red, Green, Yellow, Blue)
- **Actions**: 
  - Move: South (0), North (1), East (2), West (3)
  - Pickup passenger (4)
  - Dropoff passenger (5)
- **State Space**: 500 possible states encoding taxi position, passenger location, and destination
- **Goal**: Successfully deliver the passenger to their destination

## Implemented Algorithms

### 1. Breadth-First Search (BFS)
- **Type**: Uninformed search
- **Strategy**: Explores states level by level using a FIFO queue
- **Completeness**: Complete (guaranteed to find a solution if one exists)
- **Optimality**: Optimal for unit cost paths
- **Use Case**: Finding shortest path in terms of number of steps

### 2. Depth-First Search (DFS)
- **Type**: Uninformed search
- **Strategy**: Explores as deep as possible using a LIFO stack
- **Completeness**: Complete with depth limit
- **Optimality**: Not optimal
- **Use Case**: Quick exploration when path length is not critical

### 3. A* Search
- **Type**: Informed search with heuristic
- **Strategy**: Uses f(n) = g(n) + h(n) to prioritize states
  - g(n): Cost from start to current state
  - h(n): Estimated cost to goal (Manhattan distance heuristic)
- **Completeness**: Complete
- **Optimality**: Optimal with admissible heuristic
- **Heuristic**: Manhattan distance considering passenger pickup and destination
- **Use Case**: Finding optimal path efficiently