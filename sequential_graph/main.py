"""
LangGraph Sequential Graph Agent Implementation

This module demonstrates a LangGraph agent that uses several nodes connected
through edges to create a sequential processing workflow. The agent processes
user information (name, age, skills) through a pipeline of connected nodes.

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
        logging.FileHandler('sequential_graph/langgraph_agent.log')
    ]
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    State structure for the sequential graph agent.
    
    This TypedDict defines the data structure that flows through the agent's
    sequential processing pipeline, containing user information and results.
    
    Attributes:
        skills (List[str]): List of user's skills/abilities
        name (str): User's name for personalized responses
        age (int): User's age in years
        result (str): Accumulated result message built through the pipeline
    """
    skills: List[str]
    name: str
    age: int
    result: str


def name_node(state: AgentState) -> AgentState:
    """
    First node in the pipeline that processes the user's name.
    
    This node initiates the result message with a personalized greeting
    using the user's name. It serves as the entry point for the sequential
    processing workflow.
    
    Args:
        state (AgentState): Current state containing user information
        
    Returns:
        AgentState: Updated state with initial greeting message
        
    Processing:
        - Creates initial greeting with user's name
        - Starts building the result message
        
    Example:
        >>> state = {"name": "Alice", "age": 25, "skills": ["Python"], "result": ""}
        >>> result = name_node(state)
        >>> result["result"]
        "Hello Alice,"
    """
    logger.info(f"Processing name node for user: {state['name']}")
    logger.debug(f"Name: {state['name']}, Age: {state['age']}, Skills: {state['skills']}")

    try:
        # Initialize the result message with personalized greeting
        state['result'] = f"Hello {state['name']},"
        logger.info(f"Name processing completed: greeting initialized")
    
    except Exception as e:
        logger.error(f"Error in name_node: {str(e)}")
        state['result'] = f"Hello {state.get('name', 'Guest')}, an error occurred during name processing."
    
    logger.debug(f"Name node result: {state['result']}")
    return state


def age_node(state: AgentState) -> AgentState:
    """
    Second node in the pipeline that processes the user's age.
    
    This node appends age information to the existing result message,
    building upon the greeting created by the name node.
    
    Args:
        state (AgentState): Current state with existing result from name node
        
    Returns:
        AgentState: Updated state with age information added to result
        
    Processing:
        - Appends age information to existing result message
        - Maintains the sequential flow of information
        
    Example:
        >>> state = {"name": "Alice", "age": 25, "skills": ["Python"], "result": "Hello Alice,"}
        >>> result = age_node(state)
        >>> result["result"]
        "Hello Alice, you are 25 years old."
    """
    logger.info(f"Processing age node for user: {state['name']}")
    logger.debug(f"Name: {state['name']}, Age: {state['age']}, Skills: {state['skills']}")
    
    try:
        # Append age information to existing result
        state["result"] += f" you are {state['age']} years old."
        logger.info(f"Age processing completed: age {state['age']} added to result")
    
    except Exception as e:
        logger.error(f"Error in age_node: {str(e)}")
        state['result'] = f"Hello {state.get('name', 'Guest')}, an error occurred during age processing."
    
    logger.debug(f"Age node result: {state['result']}")
    return state


def skills_node(state: AgentState) -> AgentState:
    """
    Final node in the pipeline that processes the user's skills.
    
    This node completes the result message by adding the user's skills
    information, creating a comprehensive profile summary.
    
    Args:
        state (AgentState): Current state with existing result from previous nodes
        
    Returns:
        AgentState: Updated state with complete profile information
        
    Processing:
        - Formats and appends skills list to the result message
        - Handles empty skills list gracefully
        - Completes the sequential processing workflow
        
    Example:
        >>> state = {
        ...     "name": "Alice", 
        ...     "age": 25, 
        ...     "skills": ["Python", "JavaScript"], 
        ...     "result": "Hello Alice, you are 25 years old."
        ... }
        >>> result = skills_node(state)
        >>> "Python" in result["result"] and "JavaScript" in result["result"]
        True
    """
    logger.info(f"Processing skills node for user: {state['name']}")
    logger.debug(f"Name: {state['name']}, Age: {state['age']}, Skills: {state['skills']}")
    
    try:
        if state['skills']:
            # Format skills as a bulleted list
            skills_list = '\n\t- '.join(state['skills'])
            state["result"] += f"\nYour skills include:\n\t- {skills_list}"
            logger.info(f"Skills processing completed: {len(state['skills'])} skills added")
        else:
            # Handle empty skills list
            state["result"] += "\nYou have no specified skills."
            logger.info("Skills processing completed: no skills specified")
    
    except Exception as e:
        logger.error(f"Error in skills_node: {str(e)}")
        state['result'] = f"Hello {state.get('name', 'Guest')}, an error occurred during skills processing."
    
    logger.debug(f"Skills node result: {state['result']}")
    return state


def create_agent_graph() -> StateGraph:
    """
    Creates and configures the LangGraph state graph for sequential processing.
    
    This function builds a sequential graph with three connected nodes:
    name → age → skills, creating a pipeline for processing user information.
    
    Returns:
        StateGraph: Configured graph ready for compilation
        
    Raises:
        Exception: If graph creation fails
        
    Graph Structure:
        START → name_node → age_node → skills_node → END
        
    Node Flow:
        1. name_node: Processes user's name, creates initial greeting
        2. age_node: Adds age information to the result
        3. skills_node: Completes with skills information
    """
    logger.info("Creating sequential graph agent")
    
    try:
        # Initialize the state graph with AgentState type
        graph = StateGraph(AgentState)
        logger.debug("StateGraph initialized successfully")
        
        # Add processing nodes to the graph
        graph.add_node("name", name_node)
        logger.debug("Name node added to graph")
        graph.add_node("age", age_node)
        logger.debug("Age node added to graph")
        graph.add_node("skills", skills_node)
        logger.debug("Skills node added to graph")
        
        # Configure sequential graph flow
        graph.set_entry_point("name")
        graph.add_edge("name", "age")
        graph.add_edge("age", "skills")
        graph.set_finish_point("skills")
        logger.debug("Sequential graph flow configured: name → age → skills")
        
        logger.info("Sequential agent graph created successfully")
        return graph
        
    except Exception as e:
        logger.error(f"Failed to create agent graph: {str(e)}")
        raise


def save_graph_image(app, filename: str = "graph_visualization.png") -> str:
    """
    Saves the graph visualization to a file in the sequential_graph directory.
    
    This function generates a Mermaid diagram representation of the sequential
    graph structure and saves it as a PNG image for visualization purposes.
    
    Args:
        app: The compiled graph to visualize
        filename (str): Name of the file to save the image to
        
    Returns:
        str: Full path to the saved image file
        
    Raises:
        Exception: If image generation or saving fails
        
    File Location:
        Saves to: ./sequential_graph/graph_visualization.png
    """
    logger.info("Generating and saving graph visualization")
    
    try:
        # Create the sequential_graph directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "sequential_graph")
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Output directory ensured: {output_dir}")
        
        # Generate the Mermaid diagram as PNG bytes
        mermaid_png = app.get_graph().draw_mermaid_png()
        logger.debug("Mermaid diagram generated successfully")
        
        # Save to file in the sequential_graph directory
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(mermaid_png)
        
        logger.info(f"Graph visualization saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to save graph visualization: {str(e)}")
        raise


def run_agent(app, skills: List[str], name: str, age: int) -> Dict[str, Any]:
    """
    Executes the sequential graph agent with provided user information.
    
    This function invokes the compiled graph with user inputs and processes
    them through the sequential pipeline of nodes.
    
    Args:
        app: The compiled LangGraph application
        skills (List[str]): List of user's skills/abilities
        name (str): User's name for personalized responses
        age (int): User's age in years
        
    Returns:
        Dict[str, Any]: Agent state containing the complete profile summary
        
    Raises:
        Exception: If agent execution fails (caught and returned as error state)
        
    Processing Flow:
        1. Creates initial state with user inputs
        2. Executes through name → age → skills pipeline
        3. Returns final state with complete profile
        
    Example:
        >>> app = create_agent_graph().compile()
        >>> result = run_agent(app, ["Python", "AI"], "Alice", 25)
        >>> "Alice" in result["result"] and "25" in result["result"]
        True
    """
    logger.info(f"Running sequential agent for user: {name}")
    logger.debug(f"Input parameters - Skills: {skills}, Age: {age}")
    
    try:
        # Prepare the initial state with user inputs
        initial_state = {
            "skills": skills, 
            "name": name, 
            "age": age,
            "result": ""
        }
        logger.debug(f"Initial state prepared: {initial_state}")
        
        # Execute the agent through sequential pipeline
        result = app.invoke(initial_state)
        logger.info(f"Sequential agent execution completed successfully")
        logger.info(f"Final result length: {len(result['result'])} characters")
        
        return result
        
    except Exception as e:
        logger.error(f"Error running sequential agent: {str(e)}")
        # Return error state instead of raising
        return {
            "skills": skills,
            "name": name,
            "age": age,
            "result": f"Hello {name}, an error occurred during processing: {str(e)}"
        }


def get_user_inputs() -> Dict[str, Any]:
    """
    Collects and validates user inputs for the sequential graph agent.
    
    This function prompts the user for:
    1. A comma-separated list of skills (strings)
    2. Their name for personalized responses
    3. Their age as an integer
    
    Returns:
        Dict[str, Any]: Dictionary containing validated user inputs
        
    Raises:
        ValueError: If user provides invalid input format
        KeyboardInterrupt: If user interrupts the input process
        
    Input Format:
        - Skills: "Python,JavaScript,AI" (comma-separated strings)
        - Name: Any non-empty string
        - Age: Integer value
        
    Validation:
        - Skills list can be empty but strings are stripped of whitespace
        - Name defaults to "Guest" if empty
        - Age must be a valid integer, defaults to 0 if invalid
        
    Example:
        >>> inputs = get_user_inputs()
        Enter your skills (comma-separated): Python,AI,Machine Learning
        Enter your name: Alice
        Enter your age(int): 25
        >>> inputs
        {'skills': ['Python', 'AI', 'Machine Learning'], 'name': 'Alice', 'age': 25}
    """
    logger.info("Collecting user inputs for sequential agent")
    
    try:
        # Collect skills as strings (not integers like in the original)
        skills_input = input("Enter your skills (comma-separated): ").strip()
        logger.debug(f"Raw skills input: '{skills_input}'")
        
        # Parse and validate skills as strings
        try:
            skills = [x.strip() for x in skills_input.split(",") if x.strip()]
            # Skills can be empty, so no validation error needed
            logger.debug(f"Parsed skills: {skills}")
        except Exception as e:
            logger.warning(f"Error parsing skills input: {e}")
            skills = []  # Default to empty list
        
        # Collect user name
        name = input("Enter your name: ").strip()
        if not name:
            logger.warning("Empty name provided, using default")
            name = "Guest"
        logger.debug(f"User name: '{name}'")
        
        # Collect and validate age
        age_input = input("Enter your age (integer): ").strip()
        try:
            age = int(age_input)
            if age < 0:
                logger.warning(f"Negative age provided: {age}, using absolute value")
                age = abs(age)
        except ValueError:
            logger.warning(f"Invalid age input '{age_input}', defaulting to 0")
            age = 0
        logger.debug(f"User age: {age}")
        
        # Prepare inputs dictionary
        inputs = {
            "skills": skills,
            "name": name,
            "age": age
        }

        logger.info(f"User inputs collected successfully: {len(skills)} skills, age: {age}")
        return inputs
        
    except KeyboardInterrupt:
        logger.warning("User interrupted input collection")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during input collection: {str(e)}")
        raise ValueError("Failed to collect user inputs") from e


def main() -> None:
    """
    Main function that orchestrates the entire sequential graph workflow.
    
    This function coordinates the following steps:
    1. Creates and compiles the sequential agent graph
    2. Saves the graph visualization
    3. Collects user inputs (name, age, skills)
    4. Executes the agent through the sequential pipeline
    5. Displays the comprehensive results
    
    The function includes comprehensive error handling and logging
    for debugging and monitoring purposes.
    
    Sequential Processing Flow:
        User Input → Name Node → Age Node → Skills Node → Final Result
    
    Raises:
        Exception: For any unhandled errors during execution
    """
    logger.info("="*60)
    logger.info("STARTING LANGGRAPH SEQUENTIAL GRAPH AGENT")
    logger.info("="*60)
    
    try:
        # Step 1: Create and compile the sequential agent graph
        logger.info("Step 1: Creating sequential agent graph")
        graph = create_agent_graph()
        app = graph.compile()
        logger.info("Sequential agent graph compiled successfully")
        
        # Step 2: Save graph visualization
        logger.info("Step 2: Saving graph visualization")
        image_path = save_graph_image(app)
        logger.info(f"Graph visualization available at: {image_path}")
        
        # Step 3: Collect user inputs
        logger.info("Step 3: Collecting user inputs")
        user_inputs = get_user_inputs()
        
        # Step 4: Execute the sequential agent
        logger.info("Step 4: Executing sequential agent")
        result = run_agent(app, user_inputs["skills"], user_inputs["name"], user_inputs["age"])
        
        # Step 5: Display comprehensive results
        logger.info("="*60)
        logger.info("SEQUENTIAL AGENT EXECUTION SUMMARY")
        logger.info("="*60)
        logger.info(f"User: {user_inputs['name']}")
        logger.info(f"Age: {user_inputs['age']}")
        logger.info(f"Skills: {user_inputs['skills']}")
        logger.info("-"*60)
        logger.info("GENERATED PROFILE:")
        logger.info(f"{result['result']}")
        logger.info("="*60)
        
        logger.info("Sequential graph agent completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Sequential agent application failed: {str(e)}")
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()