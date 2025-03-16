import asyncio
import enum
import json
import logging
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from computeruse.agent.views import ActionModel, ActionResult
from computeruse.controller.registry.service import Registry
from computeruse.controller.views import (
    ClickElementAction,
    CloseWindowAction,
    DoneAction,
    DoubleClickAction,
    InputTextAction,
    LaunchApplicationAction,
    NoParamsAction,
    RightClickAction,
    RunProcessAction,
    ScreenshotAction,
    ScrollAction,
    SelectMenuItemAction,
    SendKeysAction,
    SwitchWindowAction,
    WaitAction,
)
from computeruse.uia.context import WindowsContext
from computeruse.utils import time_execution_sync

logger = logging.getLogger(__name__)


Context = TypeVar('Context')


class Controller(Generic[Context]):
    def __init__(
        self,
        exclude_actions: list[str] = [],
        output_model: Optional[Type[BaseModel]] = None,
    ):
        self.registry = Registry[Context](exclude_actions)

        """Register all default Windows UIA actions"""

        if output_model is not None:
            # Create a new model that extends the output model with success parameter
            class ExtendedOutputModel(BaseModel):  # type: ignore
                success: bool = True
                data: output_model

            @self.registry.action(
                'Complete task - return text and indicate if the task is finished (success=True) or not yet completely finished (success=False)',
                param_model=ExtendedOutputModel,
            )
            async def done(params: ExtendedOutputModel):
                # Exclude success from the output JSON since it's an internal parameter
                output_dict = params.data.model_dump()

                # Enums are not serializable, convert to string
                for key, value in output_dict.items():
                    if isinstance(value, enum.Enum):
                        output_dict[key] = value.value

                return ActionResult(is_done=True, success=params.success, extracted_content=json.dumps(output_dict))
        else:
            @self.registry.action(
                'Complete task - with return text and if the task is finished (success=True) or not yet completely finished (success=False)',
                param_model=DoneAction,
            )
            async def done(params: DoneAction):
                return ActionResult(is_done=True, success=params.success, extracted_content=params.text)

        # Application Control Actions
        @self.registry.action(
            'Launch a Windows application by name or path',
            param_model=LaunchApplicationAction,
        )
        async def launch_application(params: LaunchApplicationAction, windows_context: WindowsContext):
            try:
                if not windows_context.windows.auto:
                    return ActionResult(error="UI Automation not available")

                app_name = params.app_name
                auto = windows_context.windows.auto
                
                # Try to launch application using UI Automation
                if auto:
                    # Try using Start Menu first
                    try:
                        # Note: This is a simplified implementation. Actual implementation
                        # would require more complex logic for Start Menu navigation.
                        start_button = auto.ButtonControl(Name="Start")
                        start_button.Click()
                        
                        # Wait for Start Menu to appear
                        await asyncio.sleep(0.5)
                        
                        # Type application name
                        auto.SendKeys(app_name)
                        
                        # Wait for search results
                        await asyncio.sleep(1)
                        
                        # Click on top result
                        result = auto.ListItemControl(Name=app_name)
                        if result.Exists():
                            result.Click()
                            msg = f"💻 Launched application '{app_name}' via Start Menu"
                            logger.info(msg)
                            return ActionResult(extracted_content=msg, include_in_memory=True)
                    except Exception as e:
                        logger.debug(f"Error launching via Start Menu: {e}")
                
                # Try using Run dialog as fallback
                try:
                    # Press Win+R to open Run dialog
                    auto.SendKeys('{Win}r')
                    await asyncio.sleep(0.5)
                    
                    # Type app name and press Enter
                    auto.SendKeys(app_name + '{Enter}')
                    
                    msg = f"💻 Launched application '{app_name}' via Run dialog"
                    logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True)
                except Exception as e:
                    logger.error(f"Failed to launch application: {e}")
                    return ActionResult(error=f"Failed to launch application '{app_name}': {str(e)}")
            except Exception as e:
                logger.error(f"Error launching application: {e}")
                return ActionResult(error=f"Error launching application: {str(e)}")

        @self.registry.action(
            'Run a command line process',
            param_model=RunProcessAction,
        )
        async def run_process(params: RunProcessAction):
            try:
                import subprocess
                
                # Execute the command
                process = subprocess.Popen(
                    params.command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                stdout, stderr = process.communicate(timeout=params.timeout if params.timeout else 30)
                
                if process.returncode == 0:
                    output = stdout.strip()
                    msg = f"🖥️ Command executed successfully: {params.command}\nOutput: {output}"
                    logger.info(f"Command executed successfully: {params.command}")
                    return ActionResult(extracted_content=msg, include_in_memory=True)
                else:
                    error_msg = f"Command failed: {params.command}\nError: {stderr.strip()}"
                    logger.error(error_msg)
                    return ActionResult(error=error_msg, include_in_memory=True)
            except subprocess.TimeoutExpired:
                logger.error(f"Command timed out: {params.command}")
                return ActionResult(error=f"Command timed out: {params.command}", include_in_memory=True)
            except Exception as e:
                logger.error(f"Error running command: {e}")
                return ActionResult(error=f"Error running command: {str(e)}", include_in_memory=True)

        # Element Interaction Actions
        @self.registry.action('Click UI element', param_model=ClickElementAction)
        async def click_element(params: ClickElementAction, windows_context: WindowsContext):
            session = await windows_context.get_session()

            if params.index not in await windows_context.get_selector_map():
                raise Exception(f'Element with index {params.index} does not exist - retry or use alternative actions')

            element_node = await windows_context.get_selector_map()[params.index]
            
            # Find the actual UIA element
            auto = windows_context.windows.auto
            if not auto:
                return ActionResult(error="UI Automation not available")
            
            try:
                # Use the DOM element's xpath to find the UIA element
                # This is a simplification - real implementation would need a more robust matching method
                control_type = element_node.tag_name
                name = element_node.attributes.get("Name", "")
                automation_id = element_node.attributes.get("AutomationId", "")
                
                # Attempt to find element using different properties
                element = None
                if automation_id:
                    # Try by AutomationId first (most reliable)
                    element = auto.FindControl(lambda c: c.AutomationId == automation_id)
                
                if not element and name:
                    # Try by Name
                    element = auto.FindControl(lambda c: c.Name == name)
                
                if not element and control_type:
                    # Try by control type and other properties
                    element = auto.FindControl(
                        lambda c: c.GetControlTypeName() == control_type and 
                                  (not name or c.Name == name)
                    )
                
                if not element:
                    return ActionResult(error=f"Could not find UI element with index {params.index}")
                
                # Click the element
                success = await windows_context.click_element(element)
                
                if success:
                    element_text = element_node.get_all_text_till_next_clickable_element(max_depth=2)
                    msg = f'🖱️ Clicked element with index {params.index}: {element_text}'
                    logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True)
                else:
                    return ActionResult(error=f"Failed to click element with index {params.index}")
                
            except Exception as e:
                logger.warning(f'Element not clickable with index {params.index}: {str(e)}')
                return ActionResult(error=str(e))

        @self.registry.action(
            'Input text into a UI element',
            param_model=InputTextAction,
        )
        async def input_text(params: InputTextAction, windows_context: WindowsContext):
            if params.index not in await windows_context.get_selector_map():
                raise Exception(f'Element index {params.index} does not exist - retry or use alternative actions')

            element_node = await windows_context.get_selector_map()[params.index]
            
            # Find the actual UIA element
            auto = windows_context.windows.auto
            if not auto:
                return ActionResult(error="UI Automation not available")
            
            try:
                # Similar to click_element, find the UIA control
                automation_id = element_node.attributes.get("AutomationId", "")
                name = element_node.attributes.get("Name", "")
                control_type = element_node.tag_name
                
                element = None
                if automation_id:
                    element = auto.FindControl(lambda c: c.AutomationId == automation_id)
                
                if not element and name:
                    element = auto.FindControl(lambda c: c.Name == name)
                
                if not element and control_type:
                    element = auto.FindControl(
                        lambda c: c.GetControlTypeName() == control_type and 
                                  (not name or c.Name == name)
                    )
                
                if not element:
                    return ActionResult(error=f"Could not find UI element with index {params.index}")
                
                # Input text into the element
                success = await windows_context.input_text(element, params.text)
                
                if success:
                    msg = f'⌨️ Input text "{params.text}" into element with index {params.index}'
                    logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True)
                else:
                    return ActionResult(error=f"Failed to input text to element with index {params.index}")
                
            except Exception as e:
                logger.error(f'Failed to input text: {str(e)}')
                return ActionResult(error=str(e))

        # Window Management Actions
        @self.registry.action('Switch to window', param_model=SwitchWindowAction)
        async def switch_window(params: SwitchWindowAction, windows_context: WindowsContext):
            success = await windows_context.switch_to_window(params.window_id)
            
            if success:
                msg = f'🔄 Switched to window {params.window_id}'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            else:
                return ActionResult(error=f"Failed to switch to window {params.window_id}")

        # Special Key Actions
        @self.registry.action(
            'Send keyboard keys like Escape, Enter, Tab or keyboard shortcuts like Alt+Tab, Ctrl+C',
            param_model=SendKeysAction,
        )
        async def send_keys(params: SendKeysAction, windows_context: WindowsContext):
            try:
                auto = windows_context.windows.auto
                if not auto:
                    return ActionResult(error="UI Automation not available")
                
                # Send the keys
                auto.SendKeys(params.keys)
                
                msg = f'⌨️ Sent keys: {params.keys}'
                logger.info(msg)
                await asyncio.sleep(windows_context.config.wait_between_actions)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                logger.error(f"Error sending keys: {e}")
                return ActionResult(error=f"Error sending keys: {str(e)}")
        
        @self.registry.action(
            'Right-click on a UI element to open context menu',
            param_model=RightClickAction,
        )
        async def right_click(params: RightClickAction, windows_context: WindowsContext):
            session = await windows_context.get_session()

            if params.index not in await windows_context.get_selector_map():
                raise Exception(f'Element with index {params.index} does not exist - retry or use alternative actions')

            element_node = await windows_context.get_selector_map()[params.index]
            
            # Find the actual UIA element
            auto = windows_context.windows.auto
            if not auto:
                return ActionResult(error="UI Automation not available")
            
            try:
                # Find the element to right-click
                automation_id = element_node.attributes.get("AutomationId", "")
                name = element_node.attributes.get("Name", "")
                control_type = element_node.tag_name
                
                element = None
                if automation_id:
                    element = auto.FindControl(lambda c: c.AutomationId == automation_id)
                
                if not element and name:
                    element = auto.FindControl(lambda c: c.Name == name)
                
                if not element and control_type:
                    element = auto.FindControl(
                        lambda c: c.GetControlTypeName() == control_type and 
                                  (not name or c.Name == name)
                    )
                
                if not element:
                    return ActionResult(error=f"Could not find UI element with index {params.index}")
                
                # Get element position
                rect = element.BoundingRectangle
                if not rect:
                    return ActionResult(error=f"Element with index {params.index} has no bounding rectangle")
                
                # Calculate center point
                x = rect.left + rect.width() // 2
                y = rect.top + rect.height() // 2
                
                # Right-click at the element's center
                auto.RightClick(x, y)
                
                msg = f'🖱️ Right-clicked on element with index {params.index}'
                logger.info(msg)
                
                # Wait for context menu to appear
                await asyncio.sleep(windows_context.config.wait_between_actions)
                
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                logger.error(f"Failed to right-click element: {e}")
                return ActionResult(error=f"Failed to right-click element: {str(e)}")
        
        @self.registry.action(
            'Double-click on a UI element',
            param_model=DoubleClickAction,
        )
        async def double_click(params: DoubleClickAction, windows_context: WindowsContext):
            session = await windows_context.get_session()

            if params.index not in await windows_context.get_selector_map():
                raise Exception(f'Element with index {params.index} does not exist - retry or use alternative actions')

            element_node = await windows_context.get_selector_map()[params.index]
            
            # Find the actual UIA element
            auto = windows_context.windows.auto
            if not auto:
                return ActionResult(error="UI Automation not available")
            
            try:
                # Find the element to double-click
                automation_id = element_node.attributes.get("AutomationId", "")
                name = element_node.attributes.get("Name", "")
                control_type = element_node.tag_name
                
                element = None
                if automation_id:
                    element = auto.FindControl(lambda c: c.AutomationId == automation_id)
                
                if not element and name:
                    element = auto.FindControl(lambda c: c.Name == name)
                
                if not element and control_type:
                    element = auto.FindControl(
                        lambda c: c.GetControlTypeName() == control_type and 
                                  (not name or c.Name == name)
                    )
                
                if not element:
                    return ActionResult(error=f"Could not find UI element with index {params.index}")
                
                # Get element position
                rect = element.BoundingRectangle
                if not rect:
                    return ActionResult(error=f"Element with index {params.index} has no bounding rectangle")
                
                # Calculate center point
                x = rect.left + rect.width() // 2
                y = rect.top + rect.height() // 2
                
                # Double-click at the element's center
                auto.DoubleClick(x, y)
                
                msg = f'🖱️ Double-clicked on element with index {params.index}'
                logger.info(msg)
                
                # Wait for action effects
                await asyncio.sleep(windows_context.config.wait_between_actions)
                
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                logger.error(f"Failed to double-click element: {e}")
                return ActionResult(error=f"Failed to double-click element: {str(e)}")
        
        @self.registry.action(
            'Scroll in specified direction (up, down, left, right)',
            param_model=ScrollAction,
        )
        async def scroll(params: ScrollAction, windows_context: WindowsContext):
            auto = windows_context.windows.auto
            if not auto:
                return ActionResult(error="UI Automation not available")
            
            try:
                # Get active window
                window = await windows_context.get_active_window()
                if not window:
                    return ActionResult(error="No active window to scroll")
                
                # Find scrollable element
                scrollable = auto.FindControl(
                    lambda c: c.IsScrollPatternAvailable() and c.IsVisible, 
                    searchFromControl=window
                )
                
                if not scrollable:
                    # If no scrollable element found, try to use mouse wheel
                    direction = params.direction.lower()
                    amount = params.amount if params.amount else 10  # Default scroll amount
                    
                    if direction == 'down':
                        auto.MouseWheel(-amount)
                    elif direction == 'up':
                        auto.MouseWheel(amount)
                    elif direction == 'right':
                        auto.MouseHorizontalWheel(-amount)
                    elif direction == 'left':
                        auto.MouseHorizontalWheel(amount)
                    else:
                        return ActionResult(error=f"Invalid scroll direction: {direction}")
                    
                    msg = f'🔄 Scrolled {direction} using mouse wheel'
                else:
                    # Use scroll pattern
                    direction = params.direction.lower()
                    amount = params.amount if params.amount else 10  # Default scroll amount
                    
                    if direction == 'down':
                        scrollable.Scroll(horizontalPercent=None, verticalPercent=amount)
                    elif direction == 'up':
                        scrollable.Scroll(horizontalPercent=None, verticalPercent=-amount)
                    elif direction == 'right':
                        scrollable.Scroll(horizontalPercent=amount, verticalPercent=None)
                    elif direction == 'left':
                        scrollable.Scroll(horizontalPercent=-amount, verticalPercent=None)
                    else:
                        return ActionResult(error=f"Invalid scroll direction: {direction}")
                    
                    msg = f'🔄 Scrolled {direction} using scroll pattern'
                
                logger.info(msg)
                await asyncio.sleep(windows_context.config.wait_between_actions)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                logger.error(f"Failed to scroll: {e}")
                return ActionResult(error=f"Failed to scroll: {str(e)}")
        
        @self.registry.action(
            'Take a screenshot of the active window, with optional save to file',
            param_model=ScreenshotAction,
        )
        async def take_screenshot(params: ScreenshotAction, windows_context: WindowsContext):
            try:
                # Take screenshot of active window
                active_window = await windows_context.get_active_window()
                if not active_window:
                    return ActionResult(error="No active window for screenshot")
                
                screenshot = await windows_context.windows.take_screenshot(
                    control=active_window,
                    save_path=params.save_path
                )
                
                if screenshot:
                    save_info = f" and saved to {params.save_path}" if params.save_path else ""
                    msg = f'📸 Screenshot taken{save_info}'
                    logger.info(msg)
                    
                    # If no save path, include small version of screenshot in result
                    if not params.save_path:
                        truncated = screenshot[:50] + '...' if len(screenshot) > 50 else screenshot
                        msg += f"\nBase64 data: {truncated}"
                    
                    return ActionResult(extracted_content=msg, include_in_memory=True)
                else:
                    return ActionResult(error="Failed to take screenshot")
            except Exception as e:
                logger.error(f"Failed to take screenshot: {e}")
                return ActionResult(error=f"Failed to take screenshot: {str(e)}")
        
        @self.registry.action(
            'Wait for specified number of seconds',
            param_model=WaitAction,
        )
        async def wait(params: WaitAction):
            try:
                seconds = min(params.seconds, 30)  # Cap at 30 seconds for safety
                msg = f'⏱️ Waiting for {seconds} seconds'
                logger.info(msg)
                
                await asyncio.sleep(seconds)
                
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                logger.error(f"Error during wait: {e}")
                return ActionResult(error=f"Error during wait: {str(e)}")
        
        @self.registry.action(
            'Close the specified window or active window if no ID provided',
            param_model=CloseWindowAction,
        )
        async def close_window(params: CloseWindowAction, windows_context: WindowsContext):
            auto = windows_context.windows.auto
            if not auto:
                return ActionResult(error="UI Automation not available")
            
            try:
                if params.window_id is not None:
                    # Get windows list
                    windows_info = await windows_context._get_all_windows_info()
                    
                    if 0 <= params.window_id < len(windows_info):
                        window_title = windows_info[params.window_id]["title"]
                        window = await windows_context.windows.get_window_by_title(window_title)
                        
                        if window:
                            # Try to close window
                            if hasattr(window, 'Close'):
                                window.Close()
                            else:
                                # Fallback to Alt+F4
                                window.SetFocus()
                                auto.SendKeys('{Alt}F4')
                            
                            msg = f'🚪 Closed window with ID {params.window_id}: {window_title}'
                            logger.info(msg)
                            
                            # Wait for window to close
                            await asyncio.sleep(windows_context.config.wait_between_actions)
                            
                            return ActionResult(extracted_content=msg, include_in_memory=True)
                        else:
                            return ActionResult(error=f"Could not find window with ID {params.window_id}")
                    else:
                        return ActionResult(error=f"Window ID {params.window_id} is out of range")
                else:
                    # Close active window
                    active_window = await windows_context.get_active_window()
                    if active_window:
                        window_title = active_window.Name
                        
                        if hasattr(active_window, 'Close'):
                            active_window.Close()
                        else:
                            # Fallback to Alt+F4
                            auto.SendKeys('{Alt}F4')
                        
                        msg = f'🚪 Closed active window: {window_title}'
                        logger.info(msg)
                        
                        # Wait for window to close
                        await asyncio.sleep(windows_context.config.wait_between_actions)
                        
                        return ActionResult(extracted_content=msg, include_in_memory=True)
                    else:
                        return ActionResult(error="No active window to close")
            except Exception as e:
                logger.error(f"Failed to close window: {e}")
                return ActionResult(error=f"Failed to close window: {str(e)}")
        
        @self.registry.action(
            'Select menu item by path (e.g., "File>Open", "Edit>Preferences>Settings")',
            param_model=SelectMenuItemAction,
        )
        async def select_menu_item(params: SelectMenuItemAction, windows_context: WindowsContext):
            auto = windows_context.windows.auto
            if not auto:
                return ActionResult(error="UI Automation not available")
            
            try:
                # Get active window
                window = await windows_context.get_active_window()
                if not window:
                    return ActionResult(error="No active window to access menu")
                
                # Split the menu path
                menu_parts = params.menu_path.split('>')
                if not menu_parts:
                    return ActionResult(error="Invalid menu path format. Use 'Menu>Submenu>Command'")
                
                # First, find the menu bar
                menu_bar = auto.FindControl(
                    lambda c: c.ControlType == auto.ControlType.MenuBar and c.IsVisible,
                    searchFromControl=window
                )
                
                if not menu_bar:
                    # Try alternative approach - just click on the first menu item
                    # This may work for ribbon interfaces where the menu bar isn't a standard control
                    main_menu_name = menu_parts[0].strip()
                    main_menu = auto.FindControl(
                        lambda c: c.Name == main_menu_name and c.IsVisible,
                        searchFromControl=window
                    )
                    
                    if not main_menu:
                        # As last resort, try Alt key activation
                        auto.SendKeys('{Alt}')
                        await asyncio.sleep(0.5)  # Wait for menu to activate
                        
                        # Try to find the menu item by name again
                        main_menu = auto.FindControl(
                            lambda c: c.Name == main_menu_name and c.IsVisible
                        )
                        
                        if not main_menu:
                            return ActionResult(error=f"Could not find menu '{main_menu_name}'")
                    
                    # Click on the main menu to open it
                    main_menu.Click()
                    await asyncio.sleep(0.5)  # Wait for menu to open
                    
                    # Navigate through submenus
                    current_menu = main_menu
                    for i in range(1, len(menu_parts)):
                        item_name = menu_parts[i].strip()
                        
                        # Find submenu item
                        menu_item = auto.FindControl(
                            lambda c: c.Name == item_name and c.IsVisible
                        )
                        
                        if not menu_item:
                            return ActionResult(error=f"Could not find menu item '{item_name}'")
                        
                        # If this is the last item, click it to execute
                        if i == len(menu_parts) - 1:
                            menu_item.Click()
                        else:
                            # Otherwise click to open submenu
                            menu_item.Click()
                            await asyncio.sleep(0.5)  # Wait for submenu to open
                            current_menu = menu_item
                else:
                    # Standard menu navigation
                    # First menu item
                    main_menu_name = menu_parts[0].strip()
                    main_menu = auto.FindControl(
                        lambda c: c.Name == main_menu_name and c.IsVisible,
                        searchFromControl=menu_bar
                    )
                    
                    if not main_menu:
                        return ActionResult(error=f"Could not find menu '{main_menu_name}'")
                    
                    # Click on the main menu to open it
                    main_menu.Click()
                    await asyncio.sleep(0.5)  # Wait for menu to open
                    
                    # Navigate through submenus
                    current_menu = main_menu
                    for i in range(1, len(menu_parts)):
                        item_name = menu_parts[i].strip()
                        
                        # Find submenu item
                        menu_item = auto.FindControl(
                            lambda c: c.Name == item_name and c.IsVisible,
                            searchFromControl=window
                        )
                        
                        if not menu_item:
                            return ActionResult(error=f"Could not find menu item '{item_name}'")
                        
                        # If this is the last item, click it to execute
                        if i == len(menu_parts) - 1:
                            menu_item.Click()
                        else:
                            # Otherwise click to open submenu
                            menu_item.Click()
                            await asyncio.sleep(0.5)  # Wait for submenu to open
                            current_menu = menu_item
                
                msg = f'📋 Selected menu item: {params.menu_path}'
                logger.info(msg)
                
                # Wait for menu action to complete
                await asyncio.sleep(windows_context.config.wait_between_actions)
                
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                logger.error(f"Failed to select menu item: {e}")
                return ActionResult(error=f"Failed to select menu item: {str(e)}")

        # Extraction Actions
        @self.registry.action(
            'Extract content from the current window to retrieve specific information',
        )
        async def extract_content(goal: str, windows_context: WindowsContext, page_extraction_llm: BaseChatModel):
            try:
                state = await windows_context.get_state()
                if not state:
                    return ActionResult(error="Failed to get window state")
                
                # Convert element tree to text format for extraction
                elements_text = state.element_tree.clickable_elements_to_string()
                
                prompt = '''Your task is to extract content from the current window. 
                You will be given a window description and a goal, and you should extract all 
                relevant information related to this goal. If the goal is vague, summarize the window content.
                Respond in a structured, helpful format.
                
                Extraction goal: {goal}
                
                Window content:
                {content}
                '''
                
                template = PromptTemplate(input_variables=['goal', 'content'], template=prompt)
                
                try:
                    output = page_extraction_llm.invoke(template.format(goal=goal, content=elements_text))
                    msg = f'📄 Extracted from window\n: {output.content}\n'
                    logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True)
                except Exception as e:
                    logger.debug(f'Error extracting content with LLM: {e}')
                    msg = f'📄 Raw window content:\n{elements_text}\n'
                    logger.info(msg)
                    return ActionResult(extracted_content=msg)
            except Exception as e:
                logger.error(f"Error extracting content: {e}")
                return ActionResult(error=f"Error extracting content: {str(e)}")

    # Register ---------------------------------------------------------------

    def action(self, description: str, **kwargs):
        """Decorator for registering custom actions

        @param description: Describe the LLM what the function does (better description == better function calling)
        """
        return self.registry.action(description, **kwargs)

    # Act --------------------------------------------------------------------

    @time_execution_sync('--act')
    async def act(
        self,
        action: ActionModel,
        windows_context: WindowsContext,
        #
        page_extraction_llm: Optional[BaseChatModel] = None,
        sensitive_data: Optional[Dict[str, str]] = None,
        available_file_paths: Optional[list[str]] = None,
        #
        context: Context | None = None,
    ) -> ActionResult:
        """Execute an action"""

        try:
            for action_name, params in action.model_dump(exclude_unset=True).items():
                if params is not None:
                    result = await self.registry.execute_action(
                        action_name,
                        params,
                        windows_context=windows_context,
                        page_extraction_llm=page_extraction_llm,
                        sensitive_data=sensitive_data,
                        available_file_paths=available_file_paths,
                        context=context,
                    )

                    if isinstance(result, str):
                        return ActionResult(extracted_content=result)
                    elif isinstance(result, ActionResult):
                        return result
                    elif result is None:
                        return ActionResult()
                    else:
                        raise ValueError(f'Invalid action result type: {type(result)} of {result}')
            return ActionResult()
        except Exception as e:
            raise e