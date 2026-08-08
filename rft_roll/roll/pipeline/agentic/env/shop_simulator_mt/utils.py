# -*- coding: utf-8 -*-
"""
Utility functions for MTShopSimulatorEnv
"""

import random
from typing import List, Dict, Any


def generate_shop_task(task_type: str = "shopping", difficulty: str = "easy") -> Dict[str, Any]:
    """
    Generate a shop task configuration
    
    Args:
        task_type: Type of task ("shopping", "navigation", "help")
        difficulty: Difficulty level ("easy", "medium", "hard")
        
    Returns:
        Task configuration dictionary
    """
    task_configs = {
        "shopping": {
            "easy": {
                "description": "Find and purchase a specific product",
                "max_steps": 10,
                "required_actions": ["search", "click", "purchase"]
            },
            "medium": {
                "description": "Compare multiple products and make a choice",
                "max_steps": 15,
                "required_actions": ["search", "click", "compare", "purchase"]
            },
            "hard": {
                "description": "Complete a complex shopping task with multiple requirements",
                "max_steps": 20,
                "required_actions": ["search", "click", "compare", "navigate", "purchase"]
            }
        },
        "navigation": {
            "easy": {
                "description": "Navigate to a specific category",
                "max_steps": 8,
                "required_actions": ["navigate", "click"]
            },
            "medium": {
                "description": "Navigate through multiple categories",
                "max_steps": 12,
                "required_actions": ["navigate", "click", "search"]
            },
            "hard": {
                "description": "Complex navigation with multiple waypoints",
                "max_steps": 18,
                "required_actions": ["navigate", "click", "search", "back"]
            }
        },
        "help": {
            "easy": {
                "description": "Answer a simple customer question",
                "max_steps": 5,
                "required_actions": ["help", "respond"]
            },
            "medium": {
                "description": "Provide detailed product information",
                "max_steps": 10,
                "required_actions": ["help", "search", "respond"]
            },
            "hard": {
                "description": "Handle complex customer inquiry",
                "max_steps": 15,
                "required_actions": ["help", "search", "navigate", "respond"]
            }
        }
    }
    
    return task_configs.get(task_type, {}).get(difficulty, task_configs["shopping"]["easy"])


def create_shop_state(products: List[str], categories: List[str]) -> Dict[str, Any]:
    """
    Create a shop state representation
    
    Args:
        products: List of available products
        categories: List of product categories
        
    Returns:
        Shop state dictionary
    """
    return {
        "products": products,
        "categories": categories,
        "current_category": random.choice(categories) if categories else None,
        "search_results": [],
        "cart": [],
        "user_query": ""
    }


def validate_action(action: str, available_actions: List[str]) -> bool:
    """
    Validate if an action is available in the current state
    
    Args:
        action: Action to validate
        available_actions: List of currently available actions
        
    Returns:
        True if action is valid, False otherwise
    """
    # Extract action type from action string
    if "search[" in action:
        action_type = "search"
    elif "click[" in action:
        action_type = "click"
    elif action in ["help", "navigate", "back", "purchase"]:
        action_type = action
    else:
        return False
    
    return action_type in available_actions


def calculate_reward(action: str, task_completed: bool, steps_taken: int, max_steps: int) -> float:
    """
    Calculate reward for an action
    
    Args:
        action: Action taken
        task_completed: Whether the task was completed
        steps_taken: Number of steps taken
        max_steps: Maximum allowed steps
        
    Returns:
        Reward value
    """
    base_reward = 0.0
    
    # Task completion reward
    if task_completed:
        base_reward += 10.0
        # Bonus for completing quickly
        if steps_taken < max_steps * 0.5:
            base_reward += 5.0
        elif steps_taken < max_steps * 0.8:
            base_reward += 2.0
    
    # Step efficiency penalty
    if steps_taken > max_steps * 0.9:
        base_reward -= 2.0
    
    # Action-specific rewards
    if "search[" in action:
        base_reward += 0.5
    elif "click[" in action:
        base_reward += 0.3
    elif action == "help":
        base_reward += 0.2
    elif action == "navigate":
        base_reward += 0.4
    
    return base_reward


def format_observation(state: Dict[str, Any], available_actions: List[str]) -> str:
    """
    Format the current state as an observation string
    
    Args:
        state: Current shop state
        available_actions: List of available actions
        
    Returns:
        Formatted observation string
    """
    obs_parts = []
    
    # Current category
    if state.get("current_category"):
        obs_parts.append(f"Current category: {state['current_category']}")
    
    # Search results
    if state.get("search_results"):
        obs_parts.append(f"Search results: {', '.join(state['search_results'][:5])}")
    
    # Cart contents
    if state.get("cart"):
        obs_parts.append(f"Cart: {', '.join(state['cart'])}")
    
    # User query
    if state.get("user_query"):
        obs_parts.append(f"User query: {state['user_query']}")
    
    # Available actions
    if available_actions:
        obs_parts.append(f"Available actions: {', '.join(available_actions)}")
    
    return "\n".join(obs_parts) if obs_parts else "No information available"


def parse_search_query(action: str) -> str:
    """
    Parse search query from action string
    
    Args:
        action: Action string like "search[shoes]"
        
    Returns:
        Extracted search query
    """
    import re
    match = re.search(r"search\[(.*?)\]", action, re.IGNORECASE)
    return match.group(1) if match else ""


def parse_click_target(action: str) -> str:
    """
    Parse click target from action string
    
    Args:
        action: Action string like "click[item_name]"
        
    Returns:
        Extracted click target
    """
    import re
    match = re.search(r"click\[(.*?)\]", action, re.IGNORECASE)
    return match.group(1) if match else ""


def generate_product_catalog() -> Dict[str, List[str]]:
    """
    Generate a sample product catalog
    
    Returns:
        Dictionary mapping categories to product lists
    """
    return {
        "electronics": [
            "smartphone", "laptop", "tablet", "headphones", "camera",
            "smartwatch", "gaming_console", "speaker", "keyboard", "mouse"
        ],
        "clothing": [
            "shirt", "pants", "dress", "shoes", "hat", "jacket",
            "sweater", "jeans", "skirt", "sneakers", "boots"
        ],
        "books": [
            "fiction", "non_fiction", "science", "history", "biography",
            "cookbook", "travel", "self_help", "mystery", "romance"
        ],
        "home": [
            "furniture", "kitchen", "bathroom", "bedroom", "living_room",
            "garden", "tools", "decor", "lighting", "storage"
        ]
    }


def simulate_search_results(query: str, catalog: Dict[str, List[str]]) -> List[str]:
    """
    Simulate search results for a query
    
    Args:
        query: Search query
        catalog: Product catalog
        
    Returns:
        List of matching products
    """
    results = []
    query_lower = query.lower()
    
    for category, products in catalog.items():
        for product in products:
            if query_lower in product.lower() or product.lower() in query_lower:
                results.append(product)
    
    # Limit results and add some randomness
    if len(results) > 10:
        results = random.sample(results, 10)
    
    return results


def check_task_completion(state: Dict[str, Any], task_config: Dict[str, Any]) -> bool:
    """
    Check if the current task is completed
    
    Args:
        state: Current shop state
        task_config: Task configuration
        
    Returns:
        True if task is completed, False otherwise
    """
    required_actions = task_config.get("required_actions", [])
    
    # Check if all required action types have been performed
    performed_actions = set()
    
    # This is a simplified check - in practice, you'd track actual action history
    if state.get("search_results"):
        performed_actions.add("search")
    if state.get("cart"):
        performed_actions.add("click")
        performed_actions.add("purchase")
    if state.get("current_category"):
        performed_actions.add("navigate")
    
    return all(action in performed_actions for action in required_actions) 