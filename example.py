import asyncio
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from computeruse.agent.service import Agent
from computeruse.uia.windows import Windows, WindowsConfig
from computeruse.uia.context import WindowsContextConfig

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_agent(
    task: str,
    model: str = "gpt-4o",
    temperature: float = 0,
    max_steps: int = 10,
    vision: bool = True,
    use_anthropic: bool = False,
):
    """
    Run the agent with the given task.
    
    Args:
        task: The task to perform
        model: The model to use
        temperature: The temperature to use for the model
        max_steps: The maximum number of steps to take
        vision: Whether to use vision
        use_anthropic: Whether to use Anthropic Claude instead of OpenAI
    """
    # Create language model
    if use_anthropic:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        llm = ChatAnthropic(
            model="claude-3-opus-20240229",
            temperature=temperature,
            api_key=api_key,
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )
    
    # Create Windows UIA session
    windows_config = WindowsConfig(
        take_screenshots=vision,
    )
    
    # Create context config with appropriate settings
    context_config = WindowsContextConfig(
        minimum_wait_time=0.5,
        wait_for_idle_timeout=1.0,
        maximum_wait_time=5.0,
        wait_between_actions=0.5,
        highlight_elements=True,
    )
    
    # Create Windows instance
    windows = Windows(config=windows_config)
    
    # Create agent with all required components
    agent = Agent(
        task=task,
        llm=llm,
        windows=windows,
        use_vision=vision,
        max_actions_per_step=5,
        windows_context_config=context_config,
    )
    
    try:
        # Run the agent
        history = await agent.run(max_steps=max_steps)
        
        # Print final result
        logger.info("-" * 50)
        logger.info("Task completed")
        logger.info(f"Success: {history.is_successful()}")
        logger.info(f"Steps taken: {history.number_of_steps()}")
        logger.info(f"Final result: {history.final_result()}")
        
        return history
    finally:
        # Clean up resources
        await windows.close()


if __name__ == "__main__":
    # Define the task for the agent
    task = "Open Notepad, type 'Hello from ComputerUse UIA', save the file to desktop, and close Notepad."
    
    # Run the agent
    asyncio.run(
        run_agent(
            task=task,
            model="gpt-4o",
            max_steps=10,
            vision=True,
        )
    )