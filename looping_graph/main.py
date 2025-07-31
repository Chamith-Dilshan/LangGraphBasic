"""
LangGraph Looping Graph Implementation - Number Guessing Game

This module demonstrates a LangGraph agent that implements a number guessing game
using looping logic. The agent repeatedly guesses numbers based on user feedback
until it finds the correct answer or exhausts available attempts.

The game flow:
1. Setup initial game parameters
2. Make a guess
3. Get user feedback (correct/higher/lower)
4. Evaluate and loop back or end

"""

import logging
import os
from typing import Literal, TypedDict, Dict, List, Any
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('looping_graph/langgraph_agent.log')
    ]
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    State structure for the number guessing game agent.
    
    This TypedDict defines the data structure that flows through the agent's
    looping processing pipeline for the number guessing game.
    
    Attributes:
        lower_bound (int): Lower boundary for number guessing range
        upper_bound (int): Upper boundary for number guessing range
        guess_num (int): Current number being guessed
        guessed_nums (List[int]): List of previously guessed numbers
        attempts (int): Maximum number of attempts allowed
        count (int): Current attempt counter
        phase (Literal): Current game state based on user feedback
        target_number (int): The secret number to guess (for simulation)
        game_over (bool): Flag indicating if game has ended
    """
    lower_bound: int 
    upper_bound: int 
    guess_num: int
    guessed_nums: List[int]
    attempts: int
    count: int
    phase: Literal["correct", "higher", "lower", "none"]
    target_number: int
    game_over: bool


def setup_node(state: AgentState) -> AgentState:
    """
    Initializes the number guessing game with default parameters.
    
    This node sets up the initial game state including boundaries,
    attempt limits, and resets all tracking variables.
    
    Args:
        state (AgentState): Current state (may be empty on first call)
        
    Returns:
        AgentState: Initialized state with game parameters
        
    Game Setup:
        - Sets number range (1-20)
        - Initializes attempt counter
        - Clears previous guesses
        - Sets initial phase to "none"
        
    Example:
        >>> state = {}
        >>> result = setup_node(state)
        >>> result["lower_bound"], result["upper_bound"]
        (1, 20)
    """
    logger.info("Setting up number guessing game")
    
    try:
        # Initialize the game parameters
        state["lower_bound"] = 1
        state["upper_bound"] = 20
        state["guess_num"] = 0
        state["guessed_nums"] = []
        state["attempts"] = 10  # Maximum attempts allowed
        state["count"] = 0  # Current attempt counter
        state["phase"] = "none"
        state["target_number"] = 0
        state["game_over"] = False
        
        logger.info(f"Game setup completed - Range: {state['lower_bound']}-{state['upper_bound']}")
        logger.info(f"Maximum attempts: {state['attempts']}")
        logger.debug(f"Secret target number: {state['target_number']}")  # For debugging only
       
    except Exception as e:
        logger.error(f"Error in setup_node: {str(e)}")
        # Set safe defaults on error
        state.update({
            "lower_bound": 1, "upper_bound": 20, "guess_num": 0,
            "guessed_nums": [], "attempts": 10, "count": 0,
            "phase": "none", "target_number": 0, "game_over": False
        })
    
    return state


def guess_node(state: AgentState) -> AgentState:
    """
    Generates intelligent guesses based on previous feedback and game state.
    
    This node implements a smart guessing strategy that:
    - Uses binary search approach for efficiency
    - Avoids previously guessed numbers
    - Adjusts search space based on "higher"/"lower" feedback
    
    Args:
        state (AgentState): Current game state with feedback and history
        
    Returns:
        AgentState: Updated state with new guess and incremented counter
        
    Guessing Strategy:
        - First guess: Middle of the range
        - Subsequent guesses: Adjusted based on feedback
        - Avoids duplicate guesses
        - Handles edge cases when no valid numbers remain
        
    Example:
        >>> state = {"lower_bound": 1, "upper_bound": 100, "guessed_nums": [], ...}
        >>> result = guess_node(state)
        >>> 1 <= result["guess_num"] <= 100
        True
    """
    logger.info(f"Making guess #{state['count'] + 1}")
    
    try:
        # Increment attempt counter
        state["count"] += 1
        
        if not state["guessed_nums"]:
            # First guess: Use middle of range (binary search strategy)
            state["guess_num"] = (state["lower_bound"] + state["upper_bound"]) // 2
            logger.debug("First guess: using middle of range")
        else:
            # Subsequent guesses: Adjust based on feedback
            if state["phase"] == "higher":
                # Previous guess was too low, adjust lower bound
                state["lower_bound"] = max(state["lower_bound"], state["guess_num"] + 1)
                logger.debug(f"Adjusting lower bound to {state['lower_bound']}")
            elif state["phase"] == "lower":
                # Previous guess was too high, adjust upper bound
                state["upper_bound"] = min(state["upper_bound"], state["guess_num"] - 1)
                logger.debug(f"Adjusting upper bound to {state['upper_bound']}")
            
            # Generate available numbers in the adjusted range
            available_numbers = [
                num for num in range(state["lower_bound"], state["upper_bound"] + 1) 
                if num not in state["guessed_nums"]
            ]
            
            if available_numbers:
                # Use binary search approach: pick middle of available range
                state["guess_num"] = available_numbers[len(available_numbers) // 2]
                logger.debug(f"New guess from {len(available_numbers)} available numbers")
            else:
                logger.warning("No more available numbers to guess - game should end")
                state["game_over"] = True
                return state
        
        # Record the guess
        state["guessed_nums"].append(state["guess_num"])
        
        logger.info(f"Guess #{state['count']}: {state['guess_num']}")
        logger.debug(f"Guessed numbers so far: {state['guessed_nums']}")
        logger.debug(f"Current range: {state['lower_bound']}-{state['upper_bound']}")
        
        # Check if maximum attempts reached
        if state["count"] >= state["attempts"]:
            logger.warning("Maximum attempts reached")
            state["game_over"] = True
            
    except Exception as e:
        logger.error(f"Error in guess_node: {str(e)}")
        state["game_over"] = True
    
    return state


def get_feedback_node(state: AgentState) -> AgentState:
    """
    Collects user feedback about the current guess.
    
    This node prompts the user for feedback about whether the guess
    is correct, too high, or too low, and updates the game state accordingly.
    
    Args:
        state (AgentState): Current state with the latest guess
        
    Returns:
        AgentState: Updated state with user feedback in the phase field
        
    User Feedback Options:
        - "correct": Guess is right, game ends
        - "higher": Guess too low, need higher number
        - "lower": Guess too high, need lower number
        
    Example:
        >>> state = {"guess_num": 50, "phase": "none", ...}
        >>> # User inputs "higher"
        >>> result = get_feedback_node(state)
        >>> result["phase"]
        "higher"
    """
    logger.info(f"Getting feedback for guess: {state['guess_num']}")
    
    try:
        # Display current guess to user
        print(f"\n🎯 My guess is: {state['guess_num']}")
        print(f"📊 Attempt {state['count']} of {state['attempts']}")
        print(f"📋 Previous guesses: {state['guessed_nums'][:-1] if len(state['guessed_nums']) > 1 else 'None'}")
        
        # Get user feedback
        while True:
            feedback = input("\n💭 Is my guess (c)orrect, too (h)igh, or too (l)ow? [c/h/l]: ").strip().lower()
            
            if feedback in ['c', 'correct']:
                state["phase"] = "correct"
                logger.info("User confirmed: guess is correct!")
                break
            elif feedback in ['h', 'high', 'higher']:
                state["phase"] = "lower"  # Guess was too high, need lower
                logger.info("User feedback: guess too high, need lower number")
                break
            elif feedback in ['l', 'low', 'lower']:
                state["phase"] = "higher"  # Guess was too low, need higher
                logger.info("User feedback: guess too low, need higher number")
                break
            else:
                print("❌ Invalid input. Please enter 'c' for correct, 'h' for high, or 'l' for low.")
        
        logger.debug(f"Phase updated to: {state['phase']}")
        
    except KeyboardInterrupt:
        logger.info("User interrupted the game")
        state["phase"] = "correct"  # End game gracefully
        state["game_over"] = True
    except Exception as e:
        logger.error(f"Error in get_feedback_node: {str(e)}")
        state["phase"] = "correct"  # End game on error
        state["game_over"] = True
    
    return state


def evaluation_function(state: AgentState) -> str:
    """
    Evaluates the current game state to determine next action.
    
    This function acts as a routing condition to determine whether
    the game should continue looping or end based on the current state.
    
    Args:
        state (AgentState): Current game state with feedback and counters
        
    Returns:
        str: Next node to route to ("continue" or "end")
        
    Routing Logic:
        - "end": If guess is correct, max attempts reached, or game_over flag set
        - "continue": If game should continue with more guesses
        
    End Conditions:
        1. Correct guess found
        2. Maximum attempts exhausted
        3. No more valid numbers to guess
        4. Game over flag set
        
    Example:
        >>> state = {"phase": "correct", ...}
        >>> evaluation_function(state)
        "end"
    """
    logger.info("Evaluating game state for continuation")
    
    try:
        # Check various end conditions
        if state["phase"] == "correct":
            logger.info("Game ending: Correct guess found!")
            return "end"
        
        if state["count"] >= state["attempts"]:
            logger.info("😞 Game ending: Maximum attempts reached")
            return "end"
        
        if state.get("game_over", False):
            logger.info("🛑 Game ending: Game over flag set")
            return "end"
        
        if state["lower_bound"] > state["upper_bound"]:
            logger.info("😵 Game ending: No valid numbers remaining")
            return "end"
        
        # Game continues
        logger.info(" Game continuing: Making another guess")
        return "continue"
       
    except Exception as e:
        logger.error(f"Error in evaluation_function: {str(e)}")
        return "end"  # End game on error


def create_agent_graph() -> StateGraph:
    """
    Creates and configures the looping LangGraph for the number guessing game.
    
    This function builds a graph with a loop structure that continues
    until the correct number is guessed or maximum attempts are reached.
    
    Returns:
        StateGraph: Configured graph ready for compilation
        
    Raises:
        Exception: If graph creation fails
        
    Graph Structure:
        START → setup_node → guess_node → get_feedback_node → evaluation_function
                                ↑                                      ↓
                                └── continue ← [end condition check] ←┘
                                                      ↓
                                                    END
        
    Loop Logic:
        - Setup initializes game parameters
        - Guess generates intelligent guesses
        - Feedback collects user input
        - Evaluation determines whether to loop or end
    """
    logger.info("Creating number guessing game graph with looping logic")
    
    try:
        # Initialize the state graph with AgentState type
        graph = StateGraph(AgentState)
        logger.debug("StateGraph initialized successfully")
        
        # Add game processing nodes
        graph.add_node("setup_node", setup_node)
        logger.debug("Setup node added to graph")
        
        graph.add_node("guess_node", guess_node)
        logger.debug("Guess node added to graph")
        
        graph.add_node("get_feedback_node", get_feedback_node)
        logger.debug("Feedback node added to graph")
        
        # Configure graph flow with looping structure
        graph.add_edge(START, "setup_node")
        graph.add_edge("setup_node", "guess_node")
        graph.add_edge("guess_node", "get_feedback_node")
        logger.debug("Linear flow configured: setup → guess → feedback")
        
        # Add conditional looping logic
        graph.add_conditional_edges(
            "get_feedback_node",
            evaluation_function,
            {
                "continue": "guess_node",  # Loop back to make another guess
                "end": END                 # End the game
            },
        )
        logger.debug("Conditional looping edges configured")

        logger.info("Number guessing game graph created successfully")
        return graph
        
    except Exception as e:
        logger.error(f"Failed to create agent graph: {str(e)}")
        raise


def save_graph_image(app, filename: str = "graph_visualization.png") -> str:
    """
    Saves the graph visualization to a file in the looping_graph directory.
    
    This function generates a Mermaid diagram representation of the looping
    graph structure and saves it as a PNG image for visualization purposes.
    
    Args:
        app: The compiled graph to visualize
        filename (str): Name of the file to save the image to
        
    Returns:
        str: Full path to the saved image file
        
    Raises:
        Exception: If image generation or saving fails
        
    File Location:
        Saves to: ./looping_graph/graph_visualization.png
    """
    logger.info("Generating and saving looping graph visualization")
    
    try:
        # Create the looping_graph directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "looping_graph")
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Output directory ensured: {output_dir}")
        
        # Generate the Mermaid diagram as PNG bytes
        mermaid_png = app.get_graph().draw_mermaid_png()
        logger.debug("Mermaid diagram generated successfully")
        
        # Save to file in the looping_graph directory
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(mermaid_png)
        
        logger.info(f"Graph visualization saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to save graph visualization: {str(e)}")
        raise


def run_agent(app) -> Dict[str, Any]:
    """
    Executes the number guessing game agent.
    
    This function starts the game by invoking the compiled graph
    with initial state and runs until completion.
    
    Args:
        app: The compiled LangGraph application
        
    Returns:
        Dict[str, Any]: Final game state with results
        
    Raises:
        Exception: If agent execution fails (caught and returned as error state)
        
    Game Flow:
        1. Initialize empty state
        2. Execute through setup → guess → feedback loop
        3. Return final state with game statistics
        
    Example:
        >>> app = create_agent_graph().compile()
        >>> result = run_agent(app)
        >>> result["count"]  # Number of attempts made
        5
    """
    logger.info("Starting number guessing game")
    
    try:
        # Prepare empty initial state (setup_node will initialize)
        initial_state = AgentState(
            lower_bound=1,
            upper_bound=20,
            guess_num=0,
            guessed_nums=[],
            attempts=10,
            count=0,
            phase="none",
            target_number=0,
            game_over=False
        )
        logger.debug("Initial empty state prepared")
        
        # Execute the game loop
        result = app.invoke(initial_state)
        logger.info("Number guessing game completed successfully")
        
        # Log game statistics
        logger.info(f"Game Statistics:")
        logger.info(f"  Total attempts: {result['count']}")
        logger.info(f"  Final guess: {result['guess_num']}")
        logger.info(f"  Game outcome: {'Won' if result['phase'] == 'correct' else 'Lost/Incomplete'}")
        logger.info(f"  All guesses: {result['guessed_nums']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error running number guessing agent: {str(e)}")
        # Return error state
        return {
            "error": str(e),
            "lower_bound": 0,
            "upper_bound": 0,
            "guess_num": 0,
            "guessed_nums": [],
            "attempts": 0,
            "count": 0,
            "phase": "error",
            "target_number": 0,
            "game_over": True
        }


def main() -> None:
    """
    Main function that orchestrates the entire number guessing game workflow.
    
    This function coordinates the following steps:
    1. Creates and compiles the looping game graph
    2. Saves the graph visualization
    3. Starts and runs the interactive number guessing game
    4. Displays comprehensive game results and statistics
    
    The function includes comprehensive error handling and logging
    for debugging and monitoring purposes.
    
    Game Overview:
        An interactive number guessing game where the AI tries to guess
        the user's secret number using intelligent binary search strategy.
    
    Raises:
        Exception: For any unhandled errors during execution
    """
    logger.info("="*60)
    logger.info("STARTING LANGGRAPH NUMBER GUESSING GAME")
    logger.info("="*60)
    
    try:
        # Step 1: Create and compile the game graph
        logger.info("Step 1: Creating number guessing game graph")
        graph = create_agent_graph()
        app = graph.compile()
        logger.info("Number guessing game graph compiled successfully")
        
        # Step 2: Save graph visualization
        logger.info("Step 2: Saving graph visualization")
        image_path = save_graph_image(app)
        logger.info(f"Graph visualization available at: {image_path}")
        
        # Step 3: Display game instructions
        print("\n" + "="*60)
        print("🎮 WELCOME TO THE AI NUMBER GUESSING GAME! 🎮")
        print("="*60)
        print("🎯 Think of a number between 1 and 100")
        print("🤖 I'll try to guess it using smart strategies")
        print("💡 Give me feedback: (c)orrect, (h)igh, or (l)ow")
        print("⏱️  You have 10 attempts to help me find it!")
        print("="*60)
        
        input("\n📝 Press Enter when you've thought of your number...")
        
        # Step 4: Execute the game
        logger.info("Step 4: Starting interactive game")
        result = run_agent(app)
        
        # Step 5: Display comprehensive results
        print("\n" + "="*60)
        print("🎯 GAME RESULTS")
        print("="*60)
        
        if result["phase"] == "correct":
            print(f" SUCCESS! I guessed your number: {result['guess_num']}")
            print(f"📊 It took me {result['count']} attempts")
        else:
            print(f"😞 Game ended without finding the number")
            print(f"📊 I made {result['count']} attempts")
            print(f"🔢 My last guess was: {result['guess_num']}")
        
        print(f"📋 All my guesses: {result['guessed_nums']}")
        print(f"🎯 Final search range: {result['lower_bound']}-{result['upper_bound']}")
        
        # Performance evaluation
        efficiency = (result['count'] / result['attempts']) * 100 if result['attempts'] > 0 else 0
        print(f"⚡ Efficiency: {efficiency:.1f}% of available attempts used")
        
        print("="*60)
        
        logger.info("Number guessing game application completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n\n👋 Thanks for playing! Game interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Game application failed: {str(e)}")
        print(f"\n❌ An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()