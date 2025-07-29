"""
LangGraph Multiple Inputs Agent Implementation

This module demonstrates a LangGraph agent that performs mathematical operations
(addition or multiplication) on a list of integers provided by the user.
It includes proper logging, state management, and graph visualization.

"""

import logging
import os
from typing import TypedDict, Dict, List, Any
from langgraph.graph import StateGraph
from IPython.display import Image, display
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('multiple_inputs_agent/langgraph_agent.log')
    ]
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    State structure for the mathematical operations agent.
    
    This TypedDict defines the data structure that flows through the agent's
    processing pipeline, containing user inputs and computation results.
    
    Attributes:
        values (List[int]): List of integers to perform operations on
        name (str): User's name for personalized responses
        operation (str): Mathematical operation to perform ('add' or 'multiply')
        result (str): Final formatted result message
    """
    values: List[int]
    name: str
    operation: str
    result: str


def operation_node(state: AgentState) -> AgentState:
    """
    Processes mathematical operations on the provided list of integers.
    
    This node performs either addition or multiplication based on the user's
    choice and formats a personalized response message.
    
    Args:
        state (AgentState): Current state containing values, name, and operation
        
    Returns:
        AgentState: Updated state with the computed result
        
    Supported Operations:
        - 'add': Computes the sum of all values in the list
        - 'multiply': Computes the product of all values in the list
        
    Example:
        >>> state = {
        ...     "values": [2, 3, 4],
        ...     "name": "Alice",
        ...     "operation": "add",
        ...     "result": ""
        ... }
        >>> result = operation_node(state)
        >>> result["result"]
        "Hello Alice, the sum is 9."
    """
    logger.info(f"Processing operation node for user: {state['name']}")
    logger.debug(f"Values: {state['values']}, Operation: {state['operation']}")
    
    try:
        if state['operation'].lower() == 'add':
            # Calculate sum of all values
            total = sum(state['values'])
            state['result'] = f"Hello {state['name']}, the sum is {total}."
            logger.info(f"Addition completed: {total}")
            
        elif state['operation'].lower() == 'multiply':
            # Calculate product of all values
            product = 1
            for value in state['values']:
                product *= value
            state['result'] = f"Hello {state['name']}, the product is {product}."
            logger.info(f"Multiplication completed: {product}")
            
        else:
            # Handle unsupported operations
            error_msg = f"Are you stupid?, Unsupported operation idiot...'{state['operation']}'. Use 'add' or 'multiply'."
            state['result'] = f"Hello {state['name']}, {error_msg}"
            logger.warning(f"Invalid operation attempted: {state['operation']}")
    
    except Exception as e:
        logger.error(f"Error in operation_node: {str(e)}")
        state['result'] = f"Hello {state['name']}, an error occurred during calculation."
    
    logger.debug(f"Operation result: {state['result']}")
    return state


def create_agent_graph() -> StateGraph:
    """
    Creates and configures the LangGraph state graph for mathematical operations.
    
    This function initializes a simple graph with a single operation node
    that processes mathematical calculations based on user input.
    
    Returns:
        StateGraph: Configured graph ready for compilation
        
    Raises:
        Exception: If graph creation fails
        
    Graph Structure:
        START → operation_node → END
    """
    logger.info("Creating mathematical operations agent graph")
    
    try:
        # Initialize the state graph with AgentState type
        graph = StateGraph(AgentState)
        logger.debug("StateGraph initialized successfully")
        
        # Add the operation processing node
        graph.add_node("operation", operation_node)
        logger.debug("Operation node added to graph")
        
        # Configure graph flow: direct entry and exit through operation node
        graph.set_entry_point("operation")
        graph.set_finish_point("operation")
        logger.debug("Graph entry and finish points configured")
        
        logger.info("Agent graph created successfully")
        return graph
        
    except Exception as e:
        logger.error(f"Failed to create agent graph: {str(e)}")
        raise


def save_graph_image(app, filename: str = "graph_visualization.png") -> str:
    """
    Saves the graph visualization to a file in the multiple_inputs_agent directory.
    
    This function generates a Mermaid diagram representation of the graph
    structure and saves it as a PNG image for visualization purposes.
    
    Args:
        app: The compiled graph to visualize
        filename (str): Name of the file to save the image to
        
    Returns:
        str: Full path to the saved image file
        
    Raises:
        Exception: If image generation or saving fails
        
    File Location:
        Saves to: ./multiple_inputs_agent/graph_visualization.png
    """
    logger.info("Generating and saving graph visualization")
    
    try:
        # Create the multiple_inputs_agent directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "multiple_inputs_agent")
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Output directory ensured: {output_dir}")
        
        # Generate the Mermaid diagram as PNG bytes
        mermaid_png = app.get_graph().draw_mermaid_png()
        logger.debug("Mermaid diagram generated successfully")
        
        # Save to file in the multiple_inputs_agent directory
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(mermaid_png)
        
        logger.info(f"Graph visualization saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to save graph visualization: {str(e)}")
        raise


def run_agent(app, inputsList: List[int], name: str, operation: str) -> Dict[str, Any]:
    """
    Executes the mathematical operations agent with provided inputs.
    
    This function invokes the compiled graph with user inputs and returns
    the processed result containing the mathematical computation.
    
    Args:
        app: The compiled LangGraph application
        inputsList (List[int]): List of integers for mathematical operations
        name (str): User's name for personalized responses
        operation (str): Mathematical operation to perform ('add' or 'multiply')
        
    Returns:
        Dict[str, Any]: Agent state containing the computation result
        
    Raises:
        Exception: If agent execution fails
        
    Example:
        >>> app = create_agent_graph().compile()
        >>> result = run_agent(app, [1, 2, 3], "Alice", "add")
        >>> result["result"]
        "Hello Alice, the sum is 6."
    """
    logger.info(f"Running agent for user: {name}")
    logger.debug(f"Input parameters - Values: {inputsList}, Operation: {operation}")
    
    try:
        # Prepare the initial state with user inputs
        initial_state = {
            "values": inputsList, 
            "name": name, 
            "operation": operation,
            "result": ""
        }
        logger.debug(f"Initial state prepared: {initial_state}")
        
        # Execute the agent
        result = app.invoke(initial_state)
        logger.info(f"Agent execution completed successfully")
        logger.info(f"Final result: {result['result']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        # Return error state instead of raising
        return {
            "values": inputsList,
            "name": name,
            "operation": operation,
            "result": f"Hello {name}, an error occurred: {str(e)}"
        }


def get_user_inputs() -> Dict[str, Any]:
    """
    Collects and validates user inputs for the mathematical operations agent.
    
    This function prompts the user for:
    1. A comma-separated list of integers
    2. Their name for personalized responses
    3. The mathematical operation to perform
    
    Returns:
        Dict[str, Any]: Dictionary containing validated user inputs
        
    Raises:
        ValueError: If user provides invalid input format
        KeyboardInterrupt: If user interrupts the input process
        
    Input Format:
        - Values: "1,2,3,4" (comma-separated integers)
        - Name: Any string
        - Operation: "add" or "multiply"
        
    Example:
        >>> inputs = get_user_inputs()
        Enter a list of integers (comma-separated): 1,2,3
        Enter your name: Alice
        Enter an operation (add/multiply): add
        >>> inputs
        {'values': [1, 2, 3], 'name': 'Alice', 'operation': 'add'}
    """
    logger.info("Collecting user inputs")
    
    try:
        # Collect integer values
        values_input = input("Enter a list of integers (comma-separated): ").strip()
        logger.debug(f"Raw values input: '{values_input}'")
        
        # Parse and validate integers
        try:
            values = [int(x.strip()) for x in values_input.split(",") if x.strip()]
            if not values:
                raise ValueError("No valid integers provided")
            logger.debug(f"Parsed values: {values}")
        except ValueError as e:
            logger.error(f"Invalid integer input: {values_input}")
            raise ValueError("Please enter valid integers separated by commas") from e
        
        # Collect user name
        name = input("Enter your name: ").strip()
        if not name:
            logger.warning("Empty name provided, using default")
            name = "Guest"
        logger.debug(f"User name: '{name}'")
        
        # Collect operation
        operation = input("Enter an operation (add/multiply): ").strip().lower()
        if operation not in ['add', 'multiply']:
            logger.warning(f"Invalid operation '{operation}', defaulting to 'add'")
            operation = 'add'
        logger.debug(f"Operation: '{operation}'")
        
        # Prepare inputs dictionary
        inputs = {
            "inputsList": values,
            "name": name,
            "operation": operation
        }
        
        logger.info(f"User inputs collected successfully: {len(values)} values, operation: {operation}")
        return inputs
        
    except KeyboardInterrupt:
        logger.warning("User interrupted input collection")
        raise
    except ValueError as e:
        logger.error(f"Invalid input provided: {str(e)}")
        raise ValueError("Please enter valid inputs.") from e


def main() -> None:
    """
    Main function that orchestrates the entire mathematical operations workflow.
    
    This function coordinates the following steps:
    1. Creates and compiles the agent graph
    2. Saves the graph visualization
    3. Collects user inputs
    4. Executes the agent with user inputs
    5. Displays the results
    
    The function includes comprehensive error handling and logging
    for debugging and monitoring purposes.
    
    Raises:
        Exception: For any unhandled errors during execution
    """
    logger.info("="*60)
    logger.info("STARTING LANGGRAPH MATHEMATICAL OPERATIONS AGENT")
    logger.info("="*60)
    
    try:
        # Step 1: Create and compile the agent graph
        logger.info("Step 1: Creating agent graph")
        graph = create_agent_graph()
        app = graph.compile()
        logger.info("Agent graph compiled successfully")
        
        # Step 2: Save graph visualization
        logger.info("Step 2: Saving graph visualization")
        image_path = save_graph_image(app)
        logger.info(f"Graph visualization available at: {image_path}")
        
        # Step 3: Collect user inputs
        logger.info("Step 3: Collecting user inputs")
        user_inputs = get_user_inputs()
        
        # Step 4: Execute the agent
        logger.info("Step 4: Executing agent")
        result = run_agent(app, **user_inputs)
        
        # Step 5: Display results
        logger.info("="*60)
        logger.info("AGENT EXECUTION SUMMARY")
        logger.info("="*60)
        logger.info(f"User: {user_inputs['name']}")
        logger.info(f"Values: {user_inputs['inputsList']}")
        logger.info(f"Operation: {user_inputs['operation']}")
        logger.info(f"Result: {result['result']}")
        logger.info("="*60)
        
        logger.info("Mathematical operations agent completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Application failed with error: {str(e)}")
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()