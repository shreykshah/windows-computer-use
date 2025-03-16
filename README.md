# ComputerUse UIA: Windows Automation with LLMs

ComputerUse UIA provides AI-driven control of Windows desktop applications using Microsoft's UI Automation framework. It allows large language models (LLMs) to interact with Windows applications by capturing the UI element tree, making decisions, and taking actions.

This project was adapted from [browser-use](https://github.com/browser-use/browser-use), which provides a similar capability for web browsers. ComputerUse UIA extends the same architecture to work with native Windows applications.

## Features

- 🖥️ **Windows UI Automation**: Interact with any Windows desktop application
- 🔍 **Element Detection**: Automatically identifies interactive UI elements
- 📸 **Screenshot Vision**: Optional visual understanding of the UI (for vision-capable models)
- 🧠 **LLM Integration**: Works with OpenAI, Azure OpenAI, and Anthropic models
- 🤖 **Autonomous Agent**: Complete complex tasks with minimal human intervention

## Requirements

- Windows 10/11
- Python 3.10+
- Access to OpenAI API (GPT-4o recommended), Azure OpenAI API, or Anthropic API

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/computeruse-uia.git
   cd computeruse-uia
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   # OpenAI API (Direct)
   OPENAI_API_KEY=your_openai_api_key
   
   # Anthropic API (Optional)
   ANTHROPIC_API_KEY=your_anthropic_api_key
   
   # Azure OpenAI (Optional)
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   ```

## Usage

Here's a simple example of how to use ComputerUse UIA to control Windows applications:

```python
import asyncio
from computeruse.agent.service import Agent
from computeruse.uia.windows import Windows, WindowsConfig
from langchain_openai import ChatOpenAI

async def main():
    # Create LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Create Windows UIA instance
    windows = Windows(config=WindowsConfig(take_screenshots=True))
    
    # Create agent
    agent = Agent(
        task="Open Notepad, type 'Hello World', save to desktop, then close",
        llm=llm,
        windows=windows,
        use_vision=True,
    )
    
    # Run the agent
    try:
        history = await agent.run(max_steps=10)
        print(f"Success: {history.is_successful()}")
        print(f"Result: {history.final_result()}")
    finally:
        await windows.close()

if __name__ == "__main__":
    asyncio.run(main())
```

See the `example.py` file for a more complete example.

### Using Azure OpenAI with DefaultAzureCredential

ComputerUse UIA also supports Azure OpenAI with Azure Active Directory authentication via DefaultAzureCredential:

```python
import asyncio
from computeruse.agent.service import Agent
from computeruse.uia.windows import Windows, WindowsConfig
from computeruse.llm.azure import create_azure_openai

async def main():
    # Create Azure OpenAI LLM with AAD auth
    llm = create_azure_openai(
        deployment_name="gpt4o",  # Your Azure OpenAI deployment name
        azure_endpoint="https://your-resource.openai.azure.com/",
        api_version="2024-02-15-preview",
        temperature=0,
    )
    
    # Create Windows UIA instance
    windows = Windows(config=WindowsConfig(take_screenshots=True))
    
    # Create agent
    agent = Agent(
        task="Open Notepad, type 'Hello World', save to desktop, then close",
        llm=llm,
        windows=windows,
        use_vision=True,
    )
    
    # Run the agent
    try:
        history = await agent.run(max_steps=10)
        print(f"Success: {history.is_successful()}")
        print(f"Result: {history.final_result()}")
    finally:
        await windows.close()

if __name__ == "__main__":
    asyncio.run(main())
```

See the `examples/azure_example.py` file for a complete example with Azure OpenAI.

## How It Works

1. **UI Capture**: The system captures the UI tree and screenshots of the active Windows application
2. **Element Processing**: Interactive elements are identified and indexed
3. **LLM Decision**: The LLM receives the UI state and decides what actions to take
4. **Action Execution**: The system executes actions on the Windows application
5. **Feedback Loop**: Results are sent back to the LLM for the next decision

## Dependencies

- `uiautomation`: Windows UI Automation interface
- `pywinauto`: Additional automation capabilities
- `langchain`: LLM framework integration
- `pillow`: Image processing for screenshots
- `azure-identity`: Azure Active Directory authentication
- OpenAI/Azure OpenAI/Anthropic Python libraries

## Acknowledgements

This project is based on the architecture of [browser-use](https://github.com/browser-use/browser-use), adapted for Windows UI Automation.