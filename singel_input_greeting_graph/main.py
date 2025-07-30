"""
LangGraph Agent Implementation

This module demonstrates a basic LangGraph agent with a greeting node.
It includes proper logging, state management, and graph visualization.

"""

import logging
import os
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph
from IPython.display import Image, display
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('singel_input_greeting_graph/langgraph_agent.log')
    ]
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    Shared data structure that tracks the agent's state throughout the workflow.
    
    Attributes:
        message (str): The current message being processed by the agent
    """
    message: str


def greeting_node(state: AgentState) -> AgentState:
    """
    A greeting node that processes and modifies the agent's state with a greeting message.
    
    This node takes the current state, extracts the message, and formats it
    with a greeting and an offer to help.
    
    Args:
        state (AgentState): The current state containing the message
        
    Returns:
        AgentState: Updated state with the formatted greeting message
        
    Example:
        >>> state = {"message": "John"}
        >>> result = greeting_node(state)
        >>> result["message"]
        "Hey John! How can I help you?"
    """
    logger.info("Processing greeting node")
    
    original_message = state.get('message', '')
    logger.debug(f"Original message: '{original_message}'")
    
    # Format the greeting message
    formatted_message = f"Hey {original_message}! How can I help you?"
    state['message'] = formatted_message
    
    logger.info(f"Greeting processed successfully: '{formatted_message}'")
    return state


def create_agent_graph() -> StateGraph:
    """
    Creates and configures the LangGraph state graph for the greeting agent.
    
    This function sets up a simple graph with a single greeting node,
    configures the entry and finish points, and returns the compiled graph.
    
    Returns:
        StateGraph: Compiled graph ready for execution
        
    Raises:
        Exception: If graph creation or compilation fails
    """
    logger.info("Creating agent graph")
    
    try:
        # Initialize the state graph
        graph = StateGraph(AgentState)
        logger.debug("StateGraph initialized successfully")
        
        # Add nodes to the graph
        graph.add_node("greeting", greeting_node)
        logger.debug("Greeting node added to graph")
        
        # Configure graph flow
        graph.set_entry_point("greeting")
        graph.set_finish_point("greeting")
        logger.debug("Graph entry and finish points configured")
        
        return graph
        
    except Exception as e:
        logger.error(f"Failed to create agent graph: {str(e)}")
        raise


def save_graph_image(app, filename: str = "graph_visualization.png") -> str:
    """
    Saves the graph visualization to a file in the singel_input_greeting_graph directory.
    
    Args:
        app: The compiled graph to visualize
        filename: Name of the file to save the image to
        
    Returns:
        str: Path to the saved image file
    """
    try:
        # Create the singel_input_greeting_graph directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "singel_input_greeting_graph")
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Output directory ensured: {output_dir}")
        
        # Generate the Mermaid diagram
        mermaid_png = app.get_graph().draw_mermaid_png()
        logger.debug("Mermaid diagram generated successfully")
        
        # Save to file in the singel_input_greeting_graph directory
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(mermaid_png)
        
        logger.info(f"Graph visualization saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to save graph visualization: {str(e)}")
        raise

def run_agent(app, input_message: str) -> Dict[str, Any]:
    """
    Executes the agent with the provided input message.
    
    Args:
        app (StateGraph): The compiled graph to execute
        input_message (str): The input message to process
        
    Returns:
        Dict[str, Any]: The result state after processing
        
    Raises:
        Exception: If agent execution fails
    """
    logger.info(f"Running agent with input: '{input_message}'")
    
    try:
        # Prepare the initial state
        initial_state = {"message": input_message}
        logger.debug(f"Initial state prepared: {initial_state}")
        
        # Execute the agent
        result = app.invoke(initial_state)
        logger.info(f"Agent execution completed successfully")
        logger.info(f"Final result: '{result['message']}'")
        
        return result
        
    except Exception as e:
        logger.error(f"Agent execution failed: {str(e)}")
        raise


def get_user_input() -> str:
    """
    Prompts the user for input and validates the response.
    
    Returns:
        str: The user's input message
        
    Raises:
        KeyboardInterrupt: If user interrupts the input process
        EOFError: If EOF is encountered during input
    """
    logger.info("Prompting user for input")
    
    try:
        # Prompt user for input
        user_input = input("Please enter your name or message: ").strip()
        
        # Validate input
        if not user_input:
            logger.warning("Empty input received, using default value")
            user_input = "Guest"
        
        logger.info(f"User input received: '{user_input}'")
        return user_input
        
    except KeyboardInterrupt:
        logger.warning("User interrupted input process")
        raise
    except EOFError:
        logger.error("EOF encountered during input")
        raise


def main() -> None:
    """
    Main function that orchestrates the entire workflow.
    
    This function creates the graph, visualizes it, gets user input,
    and runs the agent with the provided input message.
    """
    logger.info("Starting LangGraph Agent application")
    
    try:
        # Create the agent graph
        graph = create_agent_graph()

        # Compile the graph
        app = graph.compile()
        logger.info("Graph compiled successfully")
        
        # Display the graph visualization
        logger.info("Displaying graph visualization")
        save_graph_image(app)
        
        # Get input from user
        user_input = get_user_input()
        
        # Run the agent with user input
        result = run_agent(app, user_input)
        
        # Log the final output
        logger.info("="*50)
        logger.info("AGENT EXECUTION SUMMARY")
        logger.info("="*50)
        logger.info(f"Input: '{user_input}'")
        logger.info(f"Output: '{result['message']}'")
        logger.info("="*50)
        
        logger.info("LangGraph Agent application completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Application failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()




## Try multiple visualization options
# try:
#     from IPython.display import Image, display
#     IPYTHON_AVAILABLE = True
# except ImportError:
#     IPYTHON_AVAILABLE = False
#     logger.info("IPython not available, will use alternative visualization methods")

# try:
#     import matplotlib.pyplot as plt
#     import matplotlib.image as mpimg
#     MATPLOTLIB_AVAILABLE = True
# except ImportError:
#     MATPLOTLIB_AVAILABLE = False

# try:
#     from PIL import Image as PILImage
#     PIL_AVAILABLE = True
# except ImportError:
#     PIL_AVAILABLE = False


# def visualize_graph(app: StateGraph) -> None:
#     """
#     Displays the graph structure using multiple visualization methods.
    
#     This function tries different approaches to display the graph:
#     1. IPython display (for Jupyter notebooks)
#     2. Matplotlib (opens in a window)
#     3. PIL (shows image)
#     4. Save to file and open with default system viewer
    
#     Args:
#         app: The compiled graph to visualize
        
#     Raises:
#         Exception: If all visualization methods fail
#     """
#     logger.info("Generating graph visualization")
    
#     try:
#         # Generate the Mermaid diagram
#         mermaid_png = app.get_graph().draw_mermaid_png()
#         logger.debug("Mermaid diagram generated successfully")
        
#         # Method 1: Try IPython display (works in Jupyter)
#         if IPYTHON_AVAILABLE:
#             try:
#                 display(Image(mermaid_png))
#                 logger.info("Graph visualization displayed using IPython")
#                 return
#             except Exception as e:
#                 logger.debug(f"IPython display failed: {e}")
        
#         # Method 2: Save to file first
#         filepath = save_graph_image(app)
        
#         # Method 3: Try matplotlib
#         if MATPLOTLIB_AVAILABLE:
#             try:
#                 img = mpimg.imread(filepath)
#                 plt.figure(figsize=(12, 8))
#                 plt.imshow(img)
#                 plt.axis('off')
#                 plt.title('LangGraph Agent Structure')
#                 plt.tight_layout()
#                 plt.show()
#                 logger.info("Graph visualization displayed using matplotlib")
#                 return
#             except Exception as e:
#                 logger.debug(f"Matplotlib display failed: {e}")
        
#         # Method 4: Try PIL
#         if PIL_AVAILABLE:
#             try:
#                 img = PILImage.open(filepath)
#                 img.show()
#                 logger.info("Graph visualization displayed using PIL")
#                 return
#             except Exception as e:
#                 logger.debug(f"PIL display failed: {e}")
        
#         # Method 5: Open with system default viewer
#         try:
#             if sys.platform.startswith('win'):
#                 os.system(f'start {filepath}')
#             elif sys.platform.startswith('darwin'):
#                 os.system(f'open {filepath}')
#             elif sys.platform.startswith('linux'):
#                 os.system(f'xdg-open {filepath}')
            
#             logger.info(f"Graph visualization saved and opened with system viewer: {filepath}")
            
#         except Exception as e:
#             logger.warning(f"Failed to open with system viewer: {e}")
#             logger.info(f"Graph visualization saved to: {filepath}")
#             logger.info("Please open the file manually to view the graph")
        
#     except Exception as e:
#         logger.error(f"Failed to generate graph visualization: {str(e)}")
#         raise