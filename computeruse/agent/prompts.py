import datetime
import importlib
import os
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from computeruse.agent.views import ActionResult, AgentStepInfo
    from computeruse.uia.views import WindowState


class SystemPrompt:
    def __init__(
        self,
        action_description: str,
        max_actions_per_step: int = 10,
        override_system_message: Optional[str] = None,
        extend_system_message: Optional[str] = None,
    ):
        self.default_action_description = action_description
        self.max_actions_per_step = max_actions_per_step
        
        # Default system prompt
        default_prompt = """You are an AI agent designed to control Windows applications. Your goal is to accomplish tasks by interacting with UI elements.

# Input Format
Task
Previous steps
Current Active Window
Available Windows
Interactive Elements
[index]<type>text</type>
- index: Numeric identifier for interaction
- type: UI element type (button, textbox, etc.)
- text: Element description
Example:
[33]<Button>Submit Form</Button>

- Only elements with numeric indexes in [] are interactive
- Elements without [] provide only context

# Response Rules
1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
{{"current_state": {{"evaluation_previous_goal": "Success|Failed|Unknown - Analyze the current UI and the image to check if the previous goals/actions are successful like intended by the task. Mention if something unexpected happened. Briefly state why/why not",
"memory": "Description of what has been done and what you need to remember. Be very specific. Count here ALWAYS how many times you have done something and how many remain. E.g. 0 out of 10 applications analyzed. Continue with abc and xyz",
"next_goal": "What needs to be done with the next immediate action"}},
"action":[{{"one_action_name": {{// action-specific parameter}}}}, // ... more actions in sequence]}}

2. ACTIONS: You can specify multiple actions in the list to be executed in sequence. But always specify only one action name per item. Use maximum {max_actions} actions per sequence.
Common action sequences:
- Form filling: [{{"input_text": {{"index": 1, "text": "username"}}}}, {{"input_text": {{"index": 2, "text": "password"}}}}, {{"click_element": {{"index": 3}}}}]
- Application launching: [{{"launch_application": {{"app_name": "Notepad"}}}}, {{"input_text": {{"index": 1, "text": "Hello world"}}}}]
- Actions are executed in the given order
- If the window changes after an action, the sequence is interrupted and you get the new state.
- Only provide the action sequence until an action which changes the window state significantly.
- Try to be efficient, but always prioritize accuracy over efficiency.

3. ELEMENT INTERACTION:
- Only use indexes of the interactive elements shown in the UI
- Elements marked with "[]Non-interactive text" are non-interactive

4. NAVIGATION & ERROR HANDLING:
- If no suitable elements exist, use other functions like launch_application or send_keys
- If stuck, try alternative approaches
- Use send_keys for keyboard shortcuts (e.g., Alt+F4 to close windows)
- When using launch_application, be specific with the application name

5. TASK COMPLETION:
- Use the done action as the last action as soon as the ultimate task is complete
- Don't use "done" before you are done with everything the user asked you, except you reach the last step of max_steps. 
- If you reach your last step, use the done action even if the task is not fully finished. Provide all the information you have gathered so far. If the ultimate task is completely finished set success to true. If not everything the user asked for is completed set success in done to false!
- If you have to do something repeatedly for example the task says for "each", or "for all", or "x times", count always inside "memory" how many times you have done it and how many remain. Don't stop until you have completed like the task asked you. Only call done after the last step.
- Don't hallucinate actions or elements
- Make sure you include everything you found out for the ultimate task in the done text parameter. Do not just say you are done, but include the requested information of the task. 

6. VISUAL CONTEXT:
- When an image is provided, use it to understand the window layout
- Indexed elements in the image correspond to the interactive elements in the text description

7. WINDOWS-SPECIFIC TIPS:
- Use launch_application to start applications
- Use run_process for command-line operations
- Use send_keys for keyboard shortcuts
- Use switch_window to move between different application windows

Your responses must be always JSON with the specified format.
"""

        prompt = ''
        if override_system_message:
            prompt = override_system_message
        else:
            # Use default prompt with formatted max_actions
            prompt = default_prompt.format(max_actions=self.max_actions_per_step)

        if extend_system_message:
            prompt += f'\n{extend_system_message}'

        self.system_message = SystemMessage(content=prompt)

    def get_system_message(self) -> SystemMessage:
        """
        Get the system prompt for the agent.

        Returns:
            SystemMessage: Formatted system prompt
        """
        return self.system_message


class AgentMessagePrompt:
    def __init__(
        self,
        state: 'WindowState',
        result: Optional[List['ActionResult']] = None,
        include_attributes: list[str] = [],
        step_info: Optional['AgentStepInfo'] = None,
    ):
        self.state = state
        self.result = result
        self.include_attributes = include_attributes
        self.step_info = step_info

    def get_user_message(self, use_vision: bool = True) -> HumanMessage:
        """
        Generate a user message with the current state information.
        
        Args:
            use_vision: Whether to include screenshots in the message
            
        Returns:
            A HumanMessage with the current state
        """
        # Convert the element tree to a string representation
        elements_text = self.state.element_tree.clickable_elements_to_string(include_attributes=self.include_attributes)

        if elements_text != '':
            elements_text = f'[Start of window]\n{elements_text}\n[End of window]'
        else:
            elements_text = 'empty window'

        if self.step_info:
            step_info_description = f'Current step: {self.step_info.step_number + 1}/{self.step_info.max_steps}'
        else:
            step_info_description = ''
            
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        step_info_description += f'\nCurrent date and time: {time_str}'

        state_description = f"""
[Task history memory ends]
[Current state starts here]
The following is one-time information - if you need to remember it write it to memory:
Current active window: {self.state.title}
Current process: {self.state.process_name}
Available windows:
{self.state.windows}
Interactive elements from the current window:
{elements_text}
{step_info_description}
"""

        if self.result:
            for i, result in enumerate(self.result):
                if result.extracted_content:
                    state_description += f'\nAction result {i + 1}/{len(self.result)}: {result.extracted_content}'
                if result.error:
                    # only use last line of error
                    error = result.error.split('\n')[-1]
                    state_description += f'\nAction error {i + 1}/{len(self.result)}: ...{error}'

        if self.state.screenshot and use_vision == True:
            # Format message for vision model
            return HumanMessage(
                content=[
                    {'type': 'text', 'text': state_description},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/png;base64,{self.state.screenshot}'},
                    },
                ]
            )

        return HumanMessage(content=state_description)