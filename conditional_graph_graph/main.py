"""
LangGraph Conditional Graph Agent Implementation

This module demonstrates a LangGraph agent that uses conditional routing to perform
mathematical operations on pairs of numbers. The agent processes two separate pairs
of numbers through conditional logic, allowing different operations for each pair.

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
        logging.FileHandler('conditional_graph_agent/langgraph_agent.log')
    ]
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    State structure for the conditional graph agent.
    
    This TypedDict defines the data structure that flows through the agent's
    conditional processing pipeline, supporting two separate mathematical operations
    on two pairs of numbers.
    
    Attributes:
        num1 (int): First number of the first pair
        num2 (int): Second number of the first pair
        num3 (int): First number of the second pair
        num4 (int): Second number of the second pair
        operation1 (Literal["add","multiply"]): Operation for first pair
        operation2 (Literal["add","multiply"]): Operation for second pair
        phase (Literal["first", "second"]): Current processing phase
        result1 (int): Result of the first operation
        result2 (int): Result of the second operation
    """
    num1: int
    num2: int
    num3: int
    num4: int
    operation1: Literal["add","multiply"]
    operation2: Literal["add","multiply"]
    phase: Literal["first", "second"]
    result1: int
    result2: int


def add_node(state: AgentState) -> AgentState:
    """
    Performs addition operation based on the current processing phase.
    
    This node handles addition for both phases of processing:
    - First phase: Adds num1 and num2, stores in result1
    - Second phase: Adds num3 and num4, stores in result2
    
    Args:
        state (AgentState): Current state containing numbers and phase information
        
    Returns:
        AgentState: Updated state with addition result stored
        
    Processing Logic:
        - Checks current phase to determine which numbers to add
        - Performs addition and stores result in appropriate field
        - Logs the operation for debugging purposes
        
    Example:
        >>> state = {"num1": 5, "num2": 3, "phase": "first", ...}
        >>> result = add_node(state)
        >>> result["result1"]
        8
    """
    try:
        if state["phase"] == "first":
            logger.info(f"Phase 1: Adding numbers {state['num1']} and {state['num2']}")
            state["result1"] = state["num1"] + state["num2"]
            logger.debug(f"First addition result: {state['result1']}")
        elif state["phase"] == "second":
            logger.info(f"Phase 2: Adding numbers {state['num3']} and {state['num4']}")
            state["result2"] = state["num3"] + state["num4"]
            logger.debug(f"Second addition result: {state['result2']}")
    except Exception as e:
        logger.error(f"Error in add_node: {str(e)}")
        if state["phase"] == "first":
            state["result1"] = 0
        else:
            state["result2"] = 0
    
    return state


def multiply_node(state: AgentState) -> AgentState:
    """
    Performs multiplication operation based on the current processing phase.
    
    This node handles multiplication for both phases of processing:
    - First phase: Multiplies num1 and num2, stores in result1
    - Second phase: Multiplies num3 and num4, stores in result2
    
    Args:
        state (AgentState): Current state containing numbers and phase information
        
    Returns:
        AgentState: Updated state with multiplication result stored
        
    Processing Logic:
        - Checks current phase to determine which numbers to multiply
        - Performs multiplication and stores result in appropriate field
        - Logs the operation for debugging purposes
        
    Example:
        >>> state = {"num1": 5, "num2": 3, "phase": "first", ...}
        >>> result = multiply_node(state)
        >>> result["result1"]
        15
    """
    try:
        if state["phase"] == "first":
            logger.info(f"Phase 1: Multiplying numbers {state['num1']} and {state['num2']}")
            state["result1"] = state["num1"] * state["num2"]
            logger.debug(f"First multiplication result: {state['result1']}")
        elif state["phase"] == "second":
            logger.info(f"Phase 2: Multiplying numbers {state['num3']} and {state['num4']}")
            state["result2"] = state["num3"] * state["num4"]
            logger.debug(f"Second multiplication result: {state['result2']}")
    except Exception as e:
        logger.error(f"Error in multiply_node: {str(e)}")
        if state["phase"] == "first":
            state["result1"] = 0
        else:
            state["result2"] = 0
    
    return state


def conditional_node1(state: AgentState) -> str:
    """
    First conditional routing node that determines the operation for the first pair.
    
    This node sets the processing phase to "first" and routes to the appropriate
    operation node based on operation1 parameter.
    
    Args:
        state (AgentState): Current state containing operation1 choice
        
    Returns:
        str: Node name to route to ("add_node" or "multiply_node")
        
    Routing Logic:
        - Sets phase to "first" for first pair processing
        - Routes to "add_node" if operation1 is "add"
        - Routes to "multiply_node" if operation1 is "multiply"
        
    Example:
        >>> state = {"operation1": "add", ...}
        >>> route = conditional_node1(state)
        >>> route
        "add_node"
    """
    logger.info("Processing conditional node 1 (first pair)")
    state["phase"] = "first"
    logger.debug(f"Phase set to: {state['phase']}")
    
    if state["operation1"] == "add":
        logger.debug("Routing to addition operation for first pair")
        return "add_node_operation"
    elif state["operation1"] == "multiply":
        logger.debug("Routing to multiplication operation for first pair")
        return "multiply_node_operation"
    else:
        logger.warning(f"Unknown operation1: {state['operation1']}, defaulting to add")
        return "add_node_operation"


def conditional_node2(state: AgentState) -> str:
    """
    Second conditional routing node that determines the operation for the second pair.
    
    This node sets the processing phase to "second" and routes to the appropriate
    operation node based on operation2 parameter.
    
    Args:
        state (AgentState): Current state containing operation2 choice
        
    Returns:
        str: Node name to route to ("add_node" or "multiply_node")
        
    Routing Logic:
        - Sets phase to "second" for second pair processing
        - Routes to "add_node" if operation2 is "add"
        - Routes to "multiply_node" if operation2 is "multiply"
        
    Example:
        >>> state = {"operation2": "multiply", ...}
        >>> route = conditional_node2(state)
        >>> route
        "multiply_node"
    """
    logger.info("Processing conditional node 2 (second pair)")
    state["phase"] = "second"
    logger.debug(f"Phase set to: {state['phase']}")
    
    if state["operation2"] == "add":
        logger.debug("Routing to addition operation for second pair")
        return "add_node_operation"
    elif state["operation2"] == "multiply":
        logger.debug("Routing to multiplication operation for second pair")
        return "multiply_node_operation"
    else:
        logger.warning(f"Unknown operation2: {state['operation2']}, defaulting to add")
        return "add_node_operation"


def create_agent_graph() -> StateGraph:
    """
    Creates and configures the conditional LangGraph for mathematical operations.
    
    This function builds a conditional graph that processes two pairs of numbers
    through different mathematical operations based on user choices.
    
    Returns:
        StateGraph: Configured graph ready for compilation
        
    Raises:
        Exception: If graph creation fails
        
    Graph Structure:
        START → conditional_node1 → [add_node|multiply_node] → conditional_node2 → [add_node|multiply_node] → END
        
    Conditional Flow:
        1. conditional_node1: Routes first pair based on operation1
        2. Operation node: Processes first pair (add or multiply)
        3. conditional_node2: Routes second pair based on operation2
        4. Operation node: Processes second pair (add or multiply)
        5. END: Completes with both results
    """
    logger.info("Creating conditional mathematical operations agent graph")
    
    try:
        # Initialize the state graph with AgentState type
        graph = StateGraph(AgentState)
        logger.debug("StateGraph initialized successfully")
        
        # Add operation processing nodes
        graph.add_node("add_node", add_node)
        logger.debug("Addition node added to graph")
        
        graph.add_node("multiply_node", multiply_node)
        logger.debug("Multiplication node added to graph")

        # Add conditional routing nodes
        set_phase_first = lambda state: {**state, "phase": "first"}
        graph.add_node("conditional_node1", set_phase_first)
        logger.debug("First conditional node added to graph")
        
        set_phase_second = lambda state: {**state, "phase": "second"}
        graph.add_node("conditional_node2", set_phase_second)
        logger.debug("Second conditional node added to graph")
        
        # Configure conditional graph flow
        graph.add_edge(START, "conditional_node1")
        logger.debug("Entry point set to conditional_node1")
        
        # First conditional routing
        graph.add_conditional_edges(
            "conditional_node1",
            conditional_node1,
            # Edge : Node
            {
                "add_node_operation": "add_node",
                "multiply_node_operation": "multiply_node"
            },
        )
        logger.debug("First conditional edges configured")
        
        # Connect first operation to second conditional
        graph.add_edge("add_node", "conditional_node2")
        graph.add_edge("multiply_node", "conditional_node2")
        logger.debug("Edges to second conditional node configured")

        # Add second operation processing nodes
        graph.add_node("add_node_2", add_node)
        logger.debug("Addition node2 added to graph")
        
        graph.add_node("multiply_node_2", multiply_node)
        logger.debug("Multiplication node2 added to graph")
        
        # Second conditional routing
        graph.add_conditional_edges(
            "conditional_node2",
            conditional_node2,
            # Edge : Node
            {
                "add_node_operation": "add_node_2",
                "multiply_node_operation": "multiply_node_2"
            },
        )
        logger.debug("Second conditional edges configured")
        
        # Set final endpoints
        graph.add_edge("add_node_2", END)
        graph.add_edge("multiply_node_2", END)
        logger.debug("End points configured")
        
        logger.info("Conditional agent graph created successfully")
        return graph
        
    except Exception as e:
        logger.error(f"Failed to create conditional agent graph: {str(e)}")
        raise


def save_graph_image(app, filename: str = "graph_visualization.png") -> str:
    """
    Saves the graph visualization to a file in the conditional_graph_agent directory.
    
    This function generates a Mermaid diagram representation of the conditional
    graph structure and saves it as a PNG image for visualization purposes.
    
    Args:
        app: The compiled graph to visualize
        filename (str): Name of the file to save the image to
        
    Returns:
        str: Full path to the saved image file
        
    Raises:
        Exception: If image generation or saving fails
        
    File Location:
        Saves to: ./conditional_graph_agent/graph_visualization.png
    """
    logger.info("Generating and saving conditional graph visualization")
    
    try:
        # Create the conditional_graph_agent directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "conditional_graph_agent")
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Output directory ensured: {output_dir}")
        
        # Generate the Mermaid diagram as PNG bytes
        mermaid_png = app.get_graph().draw_mermaid_png()
        logger.debug("Mermaid diagram generated successfully")
        
        # Save to file in the conditional_graph_agent directory
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(mermaid_png)
        
        logger.info(f"Graph visualization saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to save graph visualization: {str(e)}")
        raise


def run_agent(app, num1: int, num2: int, num3: int, num4: int, 
              operation1: Literal["add","multiply"], 
              operation2: Literal["add","multiply"]) -> Dict[str, Any]:
    """
    Executes the conditional graph agent with provided number pairs and operations.
    
    This function invokes the compiled graph with user inputs and processes
    them through the conditional routing logic.
    
    Args:
        app: The compiled LangGraph application
        num1 (int): First number of the first pair
        num2 (int): Second number of the first pair
        num3 (int): First number of the second pair
        num4 (int): Second number of the second pair
        operation1 (Literal["add","multiply"]): Operation for first pair
        operation2 (Literal["add","multiply"]): Operation for second pair
        
    Returns:
        Dict[str, Any]: Agent state containing both operation results
        
    Raises:
        Exception: If agent execution fails (caught and returned as error state)
        
    Processing Flow:
        1. Creates initial state with all inputs
        2. Executes through conditional routing
        3. Returns final state with both results
        
    Example:
        >>> app = create_agent_graph().compile()
        >>> result = run_agent(app, 5, 3, 7, 2, "add", "multiply")
        >>> result["result1"], result["result2"]
        (8, 14)
    """
    logger.info("Running conditional agent")
    logger.debug(f"Input parameters - Pair 1: ({num1}, {num2}) {operation1}, Pair 2: ({num3}, {num4}) {operation2}")
    
    try:
        # Prepare the initial state with user inputs
        initial_state = {
            "num1": num1,
            "num2": num2,
            "num3": num3,
            "num4": num4,
            "operation1": operation1,
            "operation2": operation2,
            "phase": "first",
            "result1": 0,
            "result2": 0
        }
        logger.debug(f"Initial state prepared: {initial_state}")
        
        # Execute the agent through conditional routing
        result = app.invoke(initial_state)
        logger.info(f"Conditional agent execution completed successfully")
        logger.info(f"Results - First: {result['result1']}, Second: {result['result2']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error running conditional agent: {str(e)}")
        # Return error state instead of raising
        return {
            "num1": num1,
            "num2": num2,
            "num3": num3,
            "num4": num4,
            "operation1": operation1,
            "operation2": operation2,
            "phase": "error",
            "result1": 0,
            "result2": 0,
        }


def get_user_inputs() -> Dict[str, Any]:
    """
    Collects and validates user inputs for the conditional graph agent.
    
    This function prompts the user for:
    1. Four integers (two pairs of numbers)
    2. Two operations (one for each pair)
    
    Returns:
        Dict[str, Any]: Dictionary containing validated user inputs
        
    Raises:
        ValueError: If user provides invalid input format
        KeyboardInterrupt: If user interrupts the input process
        
    Input Format:
        - Numbers: Four separate integer inputs
        - Operations: "add" or "multiply" for each pair
        
    Validation:
        - All numbers must be valid integers
        - Operations default to "add" if invalid input provided
        
    Example:
        >>> inputs = get_user_inputs()
        Enter the first integer: 5
        Enter the second integer: 3
        Enter operation for first pair (add/multiply): add
        Enter the third integer: 7
        Enter the fourth integer: 2
        Enter operation for second pair (add/multiply): multiply
        >>> inputs
        {'num1': 5, 'num2': 3, 'num3': 7, 'num4': 2, 'operation1': 'add', 'operation2': 'multiply'}
    """
    logger.info("Collecting user inputs for conditional agent")
    
    try:
        # Collect integer values for first pair
        logger.info("Collecting first pair of numbers")
        num1 = int(input("Enter the first integer: ").strip())
        num2 = int(input("Enter the second integer: ").strip())
        logger.debug(f"First pair: {num1}, {num2}")
        
        # Collect operation for first pair
        operation1 = input("Enter operation for first pair (add/multiply): ").strip().lower()
        if operation1 not in ['add', 'multiply']:
            logger.warning(f"Invalid operation '{operation1}', defaulting to 'add'")
            operation1 = 'add'
        logger.debug(f"First operation: '{operation1}'")
        
        # Collect integer values for second pair
        logger.info("Collecting second pair of numbers")
        num3 = int(input("Enter the third integer: ").strip())
        num4 = int(input("Enter the fourth integer: ").strip())
        logger.debug(f"Second pair: {num3}, {num4}")
        
        # Collect operation for second pair
        operation2 = input("Enter operation for second pair (add/multiply): ").strip().lower()
        if operation2 not in ['add', 'multiply']:
            logger.warning(f"Invalid operation '{operation2}', defaulting to 'add'")
            operation2 = 'add'
        logger.debug(f"Second operation: '{operation2}'")
        
        # Prepare inputs dictionary
        inputs = {
            "num1": num1,
            "num2": num2,
            "num3": num3,
            "num4": num4,
            "operation1": operation1,
            "operation2": operation2
        }
        
        logger.info(f"User inputs collected successfully: 2 pairs, operations: {operation1}, {operation2}")
        return inputs
        
    except KeyboardInterrupt:
        logger.warning("User interrupted input collection")
        raise
    except ValueError as e:
        logger.error(f"Invalid input provided: {str(e)}")
        raise ValueError("Please enter valid integers and operations.") from e


def main() -> None:
    """
    Main function that orchestrates the entire conditional graph workflow.
    
    This function coordinates the following steps:
    1. Creates and compiles the conditional agent graph
    2. Saves the graph visualization
    3. Collects user inputs (two number pairs and operations)
    4. Executes the agent through conditional routing
    5. Displays the comprehensive results
    
    The function includes comprehensive error handling and logging
    for debugging and monitoring purposes.
    
    Conditional Processing Flow:
        Input → conditional_node1 → operation → conditional_node2 → operation → Results
    
    Raises:
        Exception: For any unhandled errors during execution
    """
    logger.info("="*60)
    logger.info("STARTING LANGGRAPH CONDITIONAL GRAPH AGENT")
    logger.info("="*60)
    
    try:
        # Step 1: Create and compile the conditional agent graph
        logger.info("Step 1: Creating conditional agent graph")
        graph = create_agent_graph()
        app = graph.compile()
        logger.info("Conditional agent graph compiled successfully")
        
        # Step 2: Save graph visualization
        logger.info("Step 2: Saving graph visualization")
        image_path = save_graph_image(app)
        logger.info(f"Graph visualization available at: {image_path}")
        
        # Step 3: Collect user inputs
        logger.info("Step 3: Collecting user inputs")
        user_inputs = get_user_inputs()
        
        # Step 4: Execute the conditional agent
        logger.info("Step 4: Executing conditional agent")
        result = run_agent(app, **user_inputs)
        
        # Step 5: Display comprehensive results
        logger.info("="*60)
        logger.info("CONDITIONAL AGENT EXECUTION SUMMARY")
        logger.info("="*60)
        logger.info(f"First pair: {user_inputs['num1']} and {user_inputs['num2']}")
        logger.info(f"First operation: {user_inputs['operation1']}")
        logger.info(f"First result: {result['result1']}")
        logger.info("-"*30)
        logger.info(f"Second pair: {user_inputs['num3']} and {user_inputs['num4']}")
        logger.info(f"Second operation: {user_inputs['operation2']}")
        logger.info(f"Second result: {result['result2']}")
        logger.info("="*60)
        
        logger.info("Conditional graph agent completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Conditional agent application failed: {str(e)}")
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()