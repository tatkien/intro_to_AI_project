"""
Search Algorithms for Boxing Environment
Implements: BFS, DFS, A*, and Hill Climbing
"""

import collections
import heapq
from typing import List, Tuple, Optional, Set


class SearchNode:
    """Node for search tree"""
    def __init__(self, state, parent=None, action=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost  # g(n): cost from start to this node
        self.heuristic = heuristic  # h(n): estimated cost to goal
        self.f_score = cost + heuristic  # f(n) = g(n) + h(n)
    
    def __lt__(self, other):
        """For priority queue comparison in A*"""
        return self.f_score < other.f_score
    
    def get_path(self):
        """Reconstruct path from start to this node"""
        path = []
        current = self
        while current.parent is not None:
            path.append(current.action)
            current = current.parent
        return list(reversed(path))


class BreadthFirstSearch:
    """Breadth-First Search (uninformed search)"""
    
    def __init__(self, env):
        self.env = env
        self.nodes_expanded = 0
    
    def search(self, start_state, max_depth=100):
        """
        BFS to find goal state
        
        Args:
            start_state: initial state tuple
            max_depth: maximum search depth
        
        Returns:
            list: action sequence to goal, or None if not found
        """
        self.nodes_expanded = 0
        
        # Initialize frontier as FIFO queue
        frontier = collections.deque([SearchNode(start_state)])
        visited = {start_state}
        
        while frontier:
            # Pop from front (FIFO)
            node = frontier.popleft()
            self.nodes_expanded += 1
            
            # Check if goal
            if self.env.is_goal_state(node.state):
                print(f"BFS: Goal found! Nodes expanded: {self.nodes_expanded}")
                return node.get_path()
            
            # Check depth limit
            if len(node.get_path()) >= max_depth:
                continue
            
            # Expand node
            for action, next_state, cost in self.env.get_successors(node.state):
                if next_state not in visited:
                    child = SearchNode(next_state, parent=node, action=action, cost=node.cost + cost)
                    frontier.append(child)
                    visited.add(next_state)
        
        print(f"BFS: Goal not found. Nodes expanded: {self.nodes_expanded}")
        return None


class DepthFirstSearch:
    """Depth-First Search (uninformed search)"""
    
    def __init__(self, env):
        self.env = env
        self.nodes_expanded = 0
    
    def search(self, start_state, max_depth=50):
        """
        DFS to find goal state
        
        Args:
            start_state: initial state tuple
            max_depth: maximum search depth
        
        Returns:
            list: action sequence to goal, or None if not found
        """
        self.nodes_expanded = 0
        
        # Initialize frontier as LIFO stack
        frontier = [SearchNode(start_state)]
        visited = {start_state}
        
        while frontier:
            # Pop from end (LIFO)
            node = frontier.pop()
            self.nodes_expanded += 1
            
            # Check if goal
            if self.env.is_goal_state(node.state):
                print(f"DFS: Goal found! Nodes expanded: {self.nodes_expanded}")
                return node.get_path()
            
            # Check depth limit
            if len(node.get_path()) >= max_depth:
                continue
            
            # Expand node (reverse order for more natural exploration)
            successors = self.env.get_successors(node.state)
            for action, next_state, cost in reversed(successors):
                if next_state not in visited:
                    child = SearchNode(next_state, parent=node, action=action, cost=node.cost + cost)
                    frontier.append(child)
                    visited.add(next_state)
        
        print(f"DFS: Goal not found. Nodes expanded: {self.nodes_expanded}")
        return None


class AStarSearch:
    """A* Search (informed search with heuristic)"""
    
    def __init__(self, env):
        self.env = env
        self.nodes_expanded = 0
        self.heuristic_func = env.heuristic
    
    def get_heuristic(self, state):
        """Get heuristic value based on selected function"""
        return self.heuristic_func(state)
    
    def search(self, start_state, max_depth=100):
        """
        A* search to find optimal path to goal
        
        Args:
            start_state: initial state tuple
            max_depth: maximum search depth
        
        Returns:
            list: optimal action sequence to goal, or None if not found
        """
        self.nodes_expanded = 0
        
        # Initialize frontier as priority queue
        start_h = self.get_heuristic(start_state)
        start_node = SearchNode(start_state, cost=0, heuristic=start_h)
        frontier = [start_node]
        
        # Track best cost to reach each state
        visited = {start_state: 0}
        
        while frontier:
            # Pop node with lowest f-score
            node = heapq.heappop(frontier)
            self.nodes_expanded += 1
            
            # Check if goal
            if self.env.is_goal_state(node.state):
                print(f"A*: Goal found! Nodes expanded: {self.nodes_expanded}, Path cost: {node.cost}")
                return node.get_path()
            
            # Check depth limit
            if len(node.get_path()) >= max_depth:
                continue
            
            # Expand node
            for action, next_state, step_cost in self.env.get_successors(node.state):
                new_cost = node.cost + step_cost
                
                # Only explore if we found a better path to this state
                if next_state not in visited or new_cost < visited[next_state]:
                    visited[next_state] = new_cost
                    h = self.get_heuristic(next_state)
                    child = SearchNode(next_state, parent=node, action=action, 
                                     cost=new_cost, heuristic=h)
                    heapq.heappush(frontier, child)
        
        print(f"A*: Goal not found. Nodes expanded: {self.nodes_expanded}")
        return None


