"""
Taxi-v3 Search Algorithm Runner
Demonstrates BFS, DFS, A*, and Hill Climbing on Taxi environment
"""

import gymnasium as gym
import time
from environment.ale_wrapper import TaxiWrapper
from search.algorithms import (
    BreadthFirstSearch,
    DepthFirstSearch,
    AStarSearch,
)
import random


def print_taxi_state(wrapped_env, state):
    """Pretty print the current taxi state"""
    if isinstance(state, int):
        state = wrapped_env.get_symbolic_state(state)
    
    taxi_row, taxi_col, passenger_loc, destination = state
    
    loc_names = {0: "Red", 1: "Green", 2: "Yellow", 3: "Blue"}
    
    print(f"  Taxi Position: ({taxi_row}, {taxi_col})")
    if passenger_loc == 4:
        print(f"  Passenger: In Taxi")
    else:
        print(f"  Passenger: At {loc_names[passenger_loc]}")
    print(f"  Destination: {loc_names[destination]}")


def execute_plan(wrapped_env, initial_state, action_plan, visualize=True):
    """
    Execute a plan in the environment
    
    Args:
        env: Gymnasium environment
        wrapped_env: Wrapped environment with helper methods
        initial_state: Starting state
        action_plan: List of actions to execute
        visualize: Whether to render the environment
    """
    if not action_plan:
        print("No plan to execute!")
        return False
    
    print("\n" + "="*60)
    print("EXECUTING PLAN")
    print("="*60)
    
    # Reset to initial state (we'll use a fresh environment)
    # state, _ = env.reset(seed=42)
    
    action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
    
    total_reward = 0
    success = False
    
    print(f"\nInitial State:")
    print_taxi_state(wrapped_env, initial_state)
    
    for step, action in enumerate(action_plan):
        print(f"\nStep {step + 1}: {action_names[action]} (action={action})")
        
        state, reward, terminated, truncated, info = wrapped_env.step(action)
        total_reward += reward
        
        print(f"  Reward: {reward:+.1f} | Total Reward: {total_reward:+.1f}")
        print_taxi_state(wrapped_env, state)
        
        if visualize:
            wrapped_env.render()
            time.sleep(0.5)
        
        if terminated:
            print("\n✓ GOAL REACHED! Passenger delivered successfully!")
            success = True
            break
        
        if truncated:
            print("\n✗ Episode truncated (time limit)")
            break
    
    print("\n" + "="*60)
    print(f"Plan execution complete!")
    print(f"Total Steps: {len(action_plan)}")
    print(f"Total Reward: {total_reward:+.1f}")
    print(f"Success: {'Yes' if success else 'No'}")
    print("="*60)
    
    return success


def run_search_algorithm(algorithm_name, search_algorithm, wrapped_env, initial_state, max_depth=50):
    """Run a single search algorithm and report results"""
    print("\n" + "="*60)
    print(f"Running {algorithm_name}")
    print("="*60)
    
    start_time = time.time()
    plan = search_algorithm.search(initial_state, max_depth=max_depth)
    end_time = time.time()
    
    print(f"Search Time: {end_time - start_time:.4f} seconds")
    
    if plan:
        print(f"Plan Length: {len(plan)} actions")
        print(f"Plan: {plan}")
        return plan, end_time - start_time
    else:
        print("No plan found!")
        return None, end_time - start_time


def main():
    """Main function to run Taxi-v3 search demonstrations"""
    seed = random.randint(0, 10000)
    print("="*60)
    print("TAXI-V3 SEARCH ALGORITHM DEMONSTRATION")
    print("="*60)
    print("\nTaxi-v3 Environment:")
    print("  - 5x5 grid world")
    print("  - 4 locations: R(Red), G(Green), Y(Yellow), B(Blue)")
    print("  - Goal: Pick up passenger and deliver to destination")
    print("  - Actions: South, North, East, West, Pickup, Dropoff")
    print("="*60)
    print(f"Random Seed: {seed}")
    # Create environment
    env = gym.make("Taxi-v3", render_mode="human")
    wrapped_env = TaxiWrapper(env)
    
    initial_state, info = wrapped_env.reset(seed=seed)
    symbolic_state = wrapped_env.get_symbolic_state(initial_state)
    
    # Choose which algorithms to run
    print("\n" + "="*60)
    print("Select Search Algorithm:")
    print("="*60)
    print("1. Breadth-First Search (BFS) - Uninformed, Complete, Optimal")
    print("2. Depth-First Search (DFS) - Uninformed, May get stuck")
    print("3. A* Search (Manhattan) - Informed, Optimal")
    print("4. Run All (Comparison)")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-4): ").strip()
    
    plan = None
    
    if choice == "1":
        bfs = BreadthFirstSearch(wrapped_env)
        plan, search_time = run_search_algorithm("BFS", bfs, wrapped_env, symbolic_state, max_depth=20)
    
    elif choice == "2":
        dfs = DepthFirstSearch(wrapped_env)
        plan, search_time = run_search_algorithm("DFS", dfs, wrapped_env, symbolic_state, max_depth=30)
    
    elif choice == "3":
        astar = AStarSearch(wrapped_env)
        plan, search_time = run_search_algorithm("A* (Manhattan)", astar, wrapped_env, symbolic_state, max_depth=20)
    
    elif choice == "4":
        # Run all algorithms for comparison
        algorithms = [
            ("BFS", BreadthFirstSearch(wrapped_env), 20),
            ("DFS", DepthFirstSearch(wrapped_env), 30),
            ("A* (Manhattan)", AStarSearch(wrapped_env), 20),
        ]
        
        results = []
        for name, algo, max_d in algorithms:
            p, t = run_search_algorithm(name, algo, wrapped_env, symbolic_state, max_depth=max_d)
            results.append((name, p, len(p) if p else None, t))
        
        # Summary
        print("\n" + "="*60)
        print("ALGORITHM COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Algorithm':<20} {'Plan Found':<15} {'Plan Length':<15} {'Plan:':<75} {'Search Time (s)':<25}")
        print("-"*60)
        for name, p, length, t in results:
            found = "Yes" if p else "No"
            length_str = str(length) if length else "N/A"
            print(f"{name:<20} {found:<15} {length_str:<15} {str(p):<75} {str(t):<25}")
        print("="*60)
        
        # Use the best plan for execution
        if results:
            best_result = min([r for r in results if r[1] is not None], 
                            key=lambda x: x[2], default=None)
            if best_result:
                plan = best_result[1]
                print(f"\nBest plan: {best_result[0]} with {best_result[2]} actions")
    
    elif choice == "0":
        print("Exiting...")
        env.close()
        return
    
    else:
        print("Invalid choice!")
        env.close()
        return
    
    # Execute the plan if found
    if plan:
        execute = input("\nExecute the plan? (y/n): ").strip().lower()
        if execute == 'y':
            execute_plan(wrapped_env, initial_state, plan, visualize=True)
    
    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
