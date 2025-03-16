from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, TypeVar

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, ValidationError

from computeruse.agent.message_manager.service import MessageManager, MessageManagerSettings
from computeruse.agent.message_manager.utils import convert_input_messages, extract_json_from_model_output, save_conversation
from computeruse.agent.prompts import AgentMessagePrompt, SystemPrompt
from computeruse.agent.views import (
    ActionResult,
    AgentError,
    AgentHistory,
    AgentHistoryList,
    AgentOutput,
    AgentSettings,
    AgentState,
    AgentStepInfo,
    StepMetadata,
    ToolCallingMethod,
    WindowStateHistory,
)
from computeruse.controller.registry.views import ActionModel
from computeruse.controller.service import Controller
from computeruse.dom.history_tree_processor.service import (
    DOMHistoryElement,
)
from computeruse.uia.context import WindowsContext, WindowsContextConfig
from computeruse.uia.windows import Windows
from computeruse.utils import time_execution_async, time_execution_sync

load_dotenv()
logger = logging.getLogger(__name__)


Context = TypeVar('Context')


class Agent(Generic[Context]):
    @time_execution_sync('--init (agent)')
    def __init__(
        self,
        task: str,
        llm: BaseChatModel,
        # Optional parameters
        windows: Windows | None = None,
        windows_context: WindowsContext | None = None,
        controller: Controller[Context] = Controller(),
        # Initial agent run parameters
        sensitive_data: Optional[Dict[str, str]] = None,
        initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
        # Callbacks
        register_new_step_callback: Callable[['WindowState', 'AgentOutput', int], Awaitable[None]] | None = None,
        register_done_callback: Callable[['AgentHistoryList'], Awaitable[None]] | None = None,
        register_external_agent_status_raise_error_callback: Callable[[], Awaitable[bool]] | None = None,
        # Agent settings
        use_vision: bool = True,
        use_vision_for_planner: bool = False,
        save_conversation_path: Optional[str] = None,
        save_conversation_path_encoding: Optional[str] = 'utf-8',
        max_failures: int = 3,
        retry_delay: int = 10,
        override_system_message: Optional[str] = None,
        extend_system_message: Optional[str] = None,
        max_input_tokens: int = 128000,
        validate_output: bool = False,
        message_context: Optional[str] = None,
        generate_gif: bool | str = False,
        available_file_paths: Optional[list[str]] = None,
        include_attributes: list[str] = [
            'Name',
            'AutomationId',
            'ClassName',
            'ControlType',
            'Value',
            'IsEnabled',
            'role',
        ],
        max_actions_per_step: int = 10,
        tool_calling_method: Optional[ToolCallingMethod] = 'auto',
        page_extraction_llm: Optional[BaseChatModel] = None,
        planner_llm: Optional[BaseChatModel] = None,
        planner_interval: int = 1,  # Run planner every N steps
        # Windows UIA specific settings
        windows_context_config: Optional[WindowsContextConfig] = None,
        # Inject state
        injected_agent_state: Optional[AgentState] = None,
        #
        context: Context | None = None,
    ):
        if page_extraction_llm is None:
            page_extraction_llm = llm

        # Core components
        self.task = task
        self.llm = llm
        self.controller = controller
        self.sensitive_data = sensitive_data

        self.settings = AgentSettings(
            use_vision=use_vision,
            use_vision_for_planner=use_vision_for_planner,
            save_conversation_path=save_conversation_path,
            save_conversation_path_encoding=save_conversation_path_encoding,
            max_failures=max_failures,
            retry_delay=retry_delay,
            override_system_message=override_system_message,
            extend_system_message=extend_system_message,
            max_input_tokens=max_input_tokens,
            validate_output=validate_output,
            message_context=message_context,
            generate_gif=generate_gif,
            available_file_paths=available_file_paths,
            include_attributes=include_attributes,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method,
            page_extraction_llm=page_extraction_llm,
            planner_llm=planner_llm,
            planner_interval=planner_interval,
            windows_context_config=windows_context_config,
        )

        # Initialize state
        self.state = injected_agent_state or AgentState()

        # Action setup
        self._setup_action_models()
        self.initial_actions = self._convert_initial_actions(initial_actions) if initial_actions else None

        # Model setup
        self._set_model_names()

        # for models without tool calling, add available actions to context
        self.available_actions = self.controller.registry.get_prompt_description()

        self.tool_calling_method = self._set_tool_calling_method()
        self.settings.message_context = self._set_message_context()

        # Initialize message manager with state
        message_manager_state = self.state.message_manager_state
        self._message_manager = MessageManager(
            task=task,
            system_message=SystemPrompt(
                action_description=self.available_actions,
                max_actions_per_step=self.settings.max_actions_per_step,
                override_system_message=override_system_message,
                extend_system_message=extend_system_message,
            ).get_system_message(),
            settings=MessageManagerSettings(
                max_input_tokens=self.settings.max_input_tokens,
                include_attributes=self.settings.include_attributes,
                message_context=self.settings.message_context,
                sensitive_data=sensitive_data,
                available_file_paths=self.settings.available_file_paths,
            ),
            state=message_manager_state or MessageManagerSettings(),
        )

        # Windows UIA setup
        self.injected_windows = windows is not None
        self.injected_windows_context = windows_context is not None
        self.windows = windows if windows is not None else (None if windows_context else Windows())
        if windows_context:
            self.windows_context = windows_context
        elif self.windows:
            context_config = self.settings.windows_context_config or WindowsContextConfig()
            self.windows_context = WindowsContext(windows=self.windows, config=context_config)
        else:
            self.windows = Windows()
            context_config = self.settings.windows_context_config or WindowsContextConfig()
            self.windows_context = WindowsContext(windows=self.windows, config=context_config)

        # Callbacks
        self.register_new_step_callback = register_new_step_callback
        self.register_done_callback = register_done_callback
        self.register_external_agent_status_raise_error_callback = register_external_agent_status_raise_error_callback

        # Context
        self.context = context

    def _set_message_context(self) -> str | None:
        if self.tool_calling_method == 'raw':
            if self.settings.message_context:
                self.settings.message_context += f'\n\nAvailable actions: {self.available_actions}'
            else:
                self.settings.message_context = f'Available actions: {self.available_actions}'
        return self.settings.message_context

    def _set_model_names(self) -> None:
        self.chat_model_library = self.llm.__class__.__name__
        self.model_name = 'Unknown'
        if hasattr(self.llm, 'model_name'):
            model = self.llm.model_name  # type: ignore
            self.model_name = model if model is not None else 'Unknown'
        elif hasattr(self.llm, 'model'):
            model = self.llm.model  # type: ignore
            self.model_name = model if model is not None else 'Unknown'

        if self.settings.planner_llm:
            if hasattr(self.settings.planner_llm, 'model_name'):
                self.planner_model_name = self.settings.planner_llm.model_name  # type: ignore
            elif hasattr(self.settings.planner_llm, 'model'):
                self.planner_model_name = self.settings.planner_llm.model  # type: ignore
            else:
                self.planner_model_name = 'Unknown'
        else:
            self.planner_model_name = None

    def _setup_action_models(self) -> None:
        """Setup dynamic action models from controller's registry"""
        self.ActionModel = self.controller.registry.create_action_model()
        # Create output model with the dynamic actions
        self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)

        # used to force the done action when max_steps is reached
        self.DoneActionModel = self.controller.registry.create_action_model(include_actions=['done'])
        self.DoneAgentOutput = AgentOutput.type_with_custom_actions(self.DoneActionModel)

    def _set_tool_calling_method(self) -> Optional[ToolCallingMethod]:
        tool_calling_method = self.settings.tool_calling_method
        if tool_calling_method == 'auto':
            if 'deepseek-reasoner' in self.model_name or 'deepseek-r1' in self.model_name:
                return 'raw'
            elif self.chat_model_library == 'ChatGoogleGenerativeAI':
                return None
            elif self.chat_model_library == 'ChatOpenAI':
                return 'function_calling'
            elif self.chat_model_library == 'AzureChatOpenAI':
                return 'function_calling'
            else:
                return None
        else:
            return tool_calling_method

    def add_new_task(self, new_task: str) -> None:
        self._message_manager.add_new_task(new_task)

    async def _raise_if_stopped_or_paused(self) -> None:
        """Utility function that raises an InterruptedError if the agent is stopped or paused."""

        if self.register_external_agent_status_raise_error_callback:
            if await self.register_external_agent_status_raise_error_callback():
                raise InterruptedError

        if self.state.stopped or self.state.paused:
            logger.debug('Agent paused after getting state')
            raise InterruptedError

    @time_execution_async('--step (agent)')
    async def step(self, step_info: Optional[AgentStepInfo] = None) -> None:
        """Execute one step of the task"""
        logger.info(f'📍 Step {self.state.n_steps}')
        state = None
        model_output = None
        result: list[ActionResult] = []
        step_start_time = time.time()
        tokens = 0

        try:
            state = await self.windows_context.get_state()

            await self._raise_if_stopped_or_paused()

            self._message_manager.add_state_message(state, self.state.last_result, step_info, self.settings.use_vision)

            # Run planner at specified intervals if planner is configured
            if self.settings.planner_llm and self.state.n_steps % self.settings.planner_interval == 0:
                plan = await self._run_planner()
                # add plan before last state message
                self._message_manager.add_plan(plan, position=-1)

            if step_info and step_info.is_last_step():
                # Add last step warning if needed
                msg = 'Now comes your last step. Use only the "done" action now. No other actions - so here your action sequence must have length 1.'
                msg += '\nIf the task is not yet fully finished as requested by the user, set success in "done" to false! E.g. if not all steps are fully completed.'
                msg += '\nIf the task is fully finished, set success in "done" to true.'
                msg += '\nInclude everything you found out for the ultimate task in the done text.'
                logger.info('Last step finishing up')
                self._message_manager._add_message_with_tokens(HumanMessage(content=msg))
                self.AgentOutput = self.DoneAgentOutput

            input_messages = self._message_manager.get_messages()
            tokens = self._message_manager.state.history.current_tokens

            try:
                model_output = await self.get_next_action(input_messages)

                self.state.n_steps += 1

                if self.register_new_step_callback:
                    await self.register_new_step_callback(state, model_output, self.state.n_steps)

                if self.settings.save_conversation_path:
                    target = self.settings.save_conversation_path + f'_{self.state.n_steps}.txt'
                    save_conversation(input_messages, model_output, target, self.settings.save_conversation_path_encoding)

                self._message_manager._remove_last_state_message()  # we dont want the whole state in the chat history

                await self._raise_if_stopped_or_paused()

                self._message_manager.add_model_output(model_output)
            except Exception as e:
                # model call failed, remove last state message from history
                self._message_manager._remove_last_state_message()
                raise e

            result: list[ActionResult] = await self.multi_act(model_output.action)

            self.state.last_result = result

            if len(result) > 0 and result[-1].is_done:
                logger.info(f'📄 Result: {result[-1].extracted_content}')

            self.state.consecutive_failures = 0

        except InterruptedError:
            logger.debug('Agent paused')
            self.state.last_result = [
                ActionResult(
                    error='The agent was paused - now continuing actions might need to be repeated', include_in_memory=True
                )
            ]
            return
        except Exception as e:
            result = await self._handle_step_error(e)
            self.state.last_result = result

        finally:
            step_end_time = time.time()
            
            if not result:
                return

            if state:
                metadata = StepMetadata(
                    step_number=self.state.n_steps,
                    step_start_time=step_start_time,
                    step_end_time=step_end_time,
                    input_tokens=tokens,
                )
                self._make_history_item(model_output, state, result, metadata)

    @time_execution_async('--handle_step_error (agent)')
    async def _handle_step_error(self, error: Exception) -> list[ActionResult]:
        """Handle all types of errors that can occur during a step"""
        include_trace = logger.isEnabledFor(logging.DEBUG)
        error_msg = AgentError.format_error(error, include_trace=include_trace)
        prefix = f'❌ Result failed {self.state.consecutive_failures + 1}/{self.settings.max_failures} times:\n '

        if isinstance(error, (ValidationError, ValueError)):
            logger.error(f'{prefix}{error_msg}')
            if 'Max token limit reached' in error_msg:
                # cut tokens from history
                self._message_manager.settings.max_input_tokens = self.settings.max_input_tokens - 500
                logger.info(
                    f'Cutting tokens from history - new max input tokens: {self._message_manager.settings.max_input_tokens}'
                )
                self._message_manager.cut_messages()
            elif 'Could not parse response' in error_msg:
                # give model a hint how output should look like
                error_msg += '\n\nReturn a valid JSON object with the required fields.'

            self.state.consecutive_failures += 1
        else:
            from google.api_core.exceptions import ResourceExhausted
            from openai import RateLimitError

            if isinstance(error, RateLimitError) or isinstance(error, ResourceExhausted):
                logger.warning(f'{prefix}{error_msg}')
                await asyncio.sleep(self.settings.retry_delay)
                self.state.consecutive_failures += 1
            else:
                logger.error(f'{prefix}{error_msg}')
                self.state.consecutive_failures += 1

        return [ActionResult(error=error_msg, include_in_memory=True)]

    def _make_history_item(
        self,
        model_output: AgentOutput | None,
        state: 'WindowState',
        result: list[ActionResult],
        metadata: Optional[StepMetadata] = None,
    ) -> None:
        """Create and store history item"""

        if model_output:
            interacted_elements = AgentHistory.get_interacted_element(model_output, state.selector_map)
        else:
            interacted_elements = [None]

        state_history = WindowStateHistory(
            title=state.title,
            process_name=state.process_name,
            interacted_element=interacted_elements,
            screenshot=state.screenshot,
        )

        history_item = AgentHistory(model_output=model_output, result=result, state=state_history, metadata=metadata)

        self.state.history.history.append(history_item)

    THINK_TAGS = re.compile(r'<think>.*?</think>', re.DOTALL)
    STRAY_CLOSE_TAG = re.compile(r'.*?</think>', re.DOTALL)

    def _remove_think_tags(self, text: str) -> str:
        # Step 1: Remove well-formed <think>...</think>
        text = re.sub(self.THINK_TAGS, '', text)
        # Step 2: If there's an unmatched closing tag </think>,
        #         remove everything up to and including that.
        text = re.sub(self.STRAY_CLOSE_TAG, '', text)
        return text.strip()

    def _convert_input_messages(self, input_messages: list[BaseMessage]) -> list[BaseMessage]:
        """Convert input messages to the correct format"""
        if self.model_name == 'deepseek-reasoner' or 'deepseek-r1' in self.model_name:
            return convert_input_messages(input_messages, self.model_name)
        else:
            return input_messages

    @time_execution_async('--get_next_action (agent)')
    async def get_next_action(self, input_messages: list[BaseMessage]) -> AgentOutput:
        """Get next action from LLM based on current state"""
        input_messages = self._convert_input_messages(input_messages)

        if self.tool_calling_method == 'raw':
            output = self.llm.invoke(input_messages)
            # TODO: currently invoke does not return reasoning_content, we should override invoke
            output.content = self._remove_think_tags(str(output.content))
            try:
                parsed_json = extract_json_from_model_output(output.content)
                parsed = self.AgentOutput(**parsed_json)
            except (ValueError, ValidationError) as e:
                logger.warning(f'Failed to parse model output: {output} {str(e)}')
                raise ValueError('Could not parse response.')

        elif self.tool_calling_method is None:
            structured_llm = self.llm.with_structured_output(self.AgentOutput, include_raw=True)
            response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore
            parsed: AgentOutput | None = response['parsed']
        else:
            structured_llm = self.llm.with_structured_output(self.AgentOutput, include_raw=True, method=self.tool_calling_method)
            response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore
            parsed: AgentOutput | None = response['parsed']

        if parsed is None:
            raise ValueError('Could not parse response.')

        # cut the number of actions to max_actions_per_step if needed
        if len(parsed.action) > self.settings.max_actions_per_step:
            parsed.action = parsed.action[: self.settings.max_actions_per_step]

        self._log_response(parsed)

        return parsed
    
    def _log_response(self, response: AgentOutput) -> None:
        """Utility function to log the model's response."""

        if 'Success' in response.current_state.evaluation_previous_goal:
            emoji = '👍'
        elif 'Failed' in response.current_state.evaluation_previous_goal:
            emoji = '⚠'
        else:
            emoji = '🤷'

        logger.info(f'{emoji} Eval: {response.current_state.evaluation_previous_goal}')
        logger.info(f'🧠 Memory: {response.current_state.memory}')
        logger.info(f'🎯 Next goal: {response.current_state.next_goal}')
        for i, action in enumerate(response.action):
            logger.info(f'🛠️  Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}')

    @time_execution_async('--run (agent)')
    async def run(self, max_steps: int = 100) -> AgentHistoryList:
        """Execute the task with maximum number of steps"""
        try:
            logger.info(f'🚀 Starting task: {self.task}')

            # Execute initial actions if provided
            if self.initial_actions:
                result = await self.multi_act(self.initial_actions, check_for_new_elements=False)
                self.state.last_result = result

            for step in range(max_steps):
                # Check if we should stop due to too many failures
                if self.state.consecutive_failures >= self.settings.max_failures:
                    logger.error(f'❌ Stopping due to {self.settings.max_failures} consecutive failures')
                    break

                # Check control flags before each step
                if self.state.stopped:
                    logger.info('Agent stopped')
                    break

                while self.state.paused:
                    await asyncio.sleep(0.2)  # Small delay to prevent CPU spinning
                    if self.state.stopped:  # Allow stopping while paused
                        break

                step_info = AgentStepInfo(step_number=step, max_steps=max_steps)
                await self.step(step_info)

                if self.state.history.is_done():
                    if self.settings.validate_output and step < max_steps - 1:
                        if not await self._validate_output():
                            continue

                    await self.log_completion()
                    break
            else:
                logger.info('❌ Failed to complete task in maximum steps')

            return self.state.history
        finally:
            if not self.injected_windows_context:
                await self.windows_context.close()

            if not self.injected_windows and self.windows:
                await self.windows.close()

    async def take_step(self) -> tuple[bool, bool]:
        """Take a step

        Returns:
            Tuple[bool, bool]: (is_done, is_valid)
        """
        await self.step()

        if self.state.history.is_done():
            if self.settings.validate_output:
                if not await self._validate_output():
                    return True, False

            await self.log_completion()
            if self.register_done_callback:
                await self.register_done_callback(self.state.history)

            return True, True

        return False, False

    async def _validate_output(self) -> bool:
        """Validate the output of the last action is what the user wanted"""
        system_msg = (
            f'You are a validator for an Windows desktop automation agent. '
            f'Validate if the output of last action is what the user wanted and if the task is completed. '
            f'If the task is unclear defined, you can let it pass. But if something is missing or the image does not show what was requested dont let it pass. '
            f'Try to understand the window contents and help the model with suggestions like launch application X, do Y, etc. to get the solution right. '
            f'Task to validate: {self.task}. Return a JSON object with 2 keys: is_valid and reason. '
            f'is_valid is a boolean that indicates if the output is correct. '
            f'reason is a string that explains why it is valid or not.'
            f' example: {{"is_valid": false, "reason": "The user wanted to open Notepad, but the agent opened Calculator instead."}}'
        )

        state = await self.windows_context.get_state()
        content = AgentMessagePrompt(
            state=state,
            result=self.state.last_result,
            include_attributes=self.settings.include_attributes,
        )
        msg = [SystemMessage(content=system_msg), content.get_user_message(self.settings.use_vision)]

        class ValidationResult(BaseModel):
            """
            Validation results.
            """

            is_valid: bool
            reason: str

        validator = self.llm.with_structured_output(ValidationResult, include_raw=True)
        response: dict[str, Any] = await validator.ainvoke(msg)  # type: ignore
        parsed: ValidationResult = response['parsed']
        is_valid = parsed.is_valid
        if not is_valid:
            logger.info(f'❌ Validator decision: {parsed.reason}')
            msg = f'The output is not yet correct. {parsed.reason}.'
            self.state.last_result = [ActionResult(extracted_content=msg, include_in_memory=True)]
        else:
            logger.info(f'✅ Validator decision: {parsed.reason}')
        return is_valid

    async def log_completion(self) -> None:
        """Log the completion of the task"""
        logger.info('✅ Task completed')
        if self.state.history.is_successful():
            logger.info('✅ Successfully')
        else:
            logger.info('❌ Unfinished')

        if self.register_done_callback:
            await self.register_done_callback(self.state.history)

    async def multi_act(
        self,
        actions: list[ActionModel],
        check_for_new_elements: bool = True,
    ) -> list[ActionResult]:
        """Execute multiple actions"""
        results = []

        cached_selector_map = await self.windows_context.get_selector_map()
        
        for i, action in enumerate(actions):
            # Before executing action, check if we need to wait
            if i > 0:
                await asyncio.sleep(self.windows_context.config.wait_between_actions)

            await self._raise_if_stopped_or_paused()

            result = await self.controller.act(
                action,
                self.windows_context,
                self.settings.page_extraction_llm,
                self.sensitive_data,
                self.settings.available_file_paths,
                context=self.context,
            )

            results.append(result)

            logger.debug(f'Executed action {i + 1} / {len(actions)}')
            if results[-1].is_done or results[-1].error or i == len(actions) - 1:
                break

        return results

    async def _run_planner(self) -> Optional[str]:
        """Run the planner to analyze state and suggest next steps"""
        # Skip planning if no planner_llm is set
        if not self.settings.planner_llm:
            return None

        # Create planner message history using full message history
        planner_prompt = f"""You are a planning agent that helps break down Windows desktop automation tasks into smaller steps and reason about the current state.
        Your role is to:
        1. Analyze the current state of the Windows desktop and application windows
        2. Evaluate progress towards the ultimate goal: {self.task}
        3. Identify potential challenges or roadblocks
        4. Suggest the next high-level steps to take
        
        Your output format should be always a JSON object with the following fields:
        {{
            "state_analysis": "Brief analysis of the current desktop state and what has been done so far",
            "progress_evaluation": "Evaluation of progress towards the ultimate goal (as percentage and description)",
            "challenges": "List any potential challenges or roadblocks",
            "next_steps": "List 2-3 concrete next steps to take",
            "reasoning": "Explain your reasoning for the suggested next steps"
        }}
        
        Keep your responses concise and focused on actionable insights."""
        
        planner_messages = [
            SystemMessage(content=planner_prompt),
            *self._message_manager.get_messages()[1:],  # Use full message history except the first
        ]

        if not self.settings.use_vision_for_planner and self.settings.use_vision:
            last_state_message: HumanMessage = planner_messages[-1]
            # remove image from last state message
            new_msg = ''
            if isinstance(last_state_message.content, list):
                for msg in last_state_message.content:
                    if msg['type'] == 'text':  # type: ignore
                        new_msg += msg['text']  # type: ignore
                    elif msg['type'] == 'image_url':  # type: ignore
                        continue  # type: ignore
            else:
                new_msg = last_state_message.content

            planner_messages[-1] = HumanMessage(content=new_msg)

        planner_messages = convert_input_messages(planner_messages, self.planner_model_name)

        # Get planner output
        response = await self.settings.planner_llm.ainvoke(planner_messages)
        plan = str(response.content)
        # if deepseek-reasoner, remove think tags
        if self.planner_model_name and ('deepseek-r1' in self.planner_model_name or 'deepseek-reasoner' in self.planner_model_name):
            plan = self._remove_think_tags(plan)
        try:
            plan_json = json.loads(plan)
            logger.info(f'Planning Analysis:\n{json.dumps(plan_json, indent=4)}')
        except json.JSONDecodeError:
            logger.info(f'Planning Analysis:\n{plan}')
        except Exception as e:
            logger.debug(f'Error parsing planning analysis: {e}')
            logger.info(f'Plan: {plan}')

        return plan

    def _convert_initial_actions(self, actions: List[Dict[str, Dict[str, Any]]]) -> List[ActionModel]:
        """Convert dictionary-based actions to ActionModel instances"""
        converted_actions = []
        action_model = self.ActionModel
        for action_dict in actions:
            # Each action_dict should have a single key-value pair
            action_name = next(iter(action_dict))
            params = action_dict[action_name]

            # Get the parameter model for this action from registry
            action_info = self.controller.registry.registry.actions[action_name]
            param_model = action_info.param_model

            # Create validated parameters using the appropriate param model
            validated_params = param_model(**params)

            # Create ActionModel instance with the validated parameters
            action_model = self.ActionModel(**{action_name: validated_params})
            converted_actions.append(action_model)

        return converted_actions
    
    @property
    def message_manager(self) -> MessageManager:
        return self._message_manager