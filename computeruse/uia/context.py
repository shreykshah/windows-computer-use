"""
Windows UIA Context.
"""

import asyncio
import gc
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar

from computeruse.uia.windows import Windows
from computeruse.utils import time_execution_async

logger = logging.getLogger(__name__)

@dataclass
class WindowsContextConfig:
    """
    Configuration for the Windows context session.

    Default values:
        minimum_wait_time: 0.5
            Minimum time to wait before getting window state for LLM input

        wait_for_idle_timeout: 1.0
            Time to wait for UI to stabilize before getting window state.

        maximum_wait_time: 5.0
            Maximum time to wait for UI to stabilize before proceeding anyway

        wait_between_actions: 1.0
            Time to wait between multiple actions per step

        highlight_elements: True
            Highlight elements in the UI on the screen

        viewport_expansion: 500
            Viewport expansion in pixels. This is used to include elements slightly outside the visible area.
    """

    minimum_wait_time: float = 0.25
    wait_for_idle_timeout: float = 1.0
    maximum_wait_time: float = 5.0
    wait_between_actions: float = 0.5
    highlight_elements: bool = True
    viewport_expansion: int = 500

    _force_keep_context_alive: bool = False

@dataclass
class WindowsSession:
    """Holds session data for a Windows context"""
    active_window: Any  # UIAutomation window object
    cached_state: Optional[Any] = None  # WindowState object

@dataclass
class WindowsContextState:
    """
    State of the Windows context
    """
    target_id: Optional[str] = None  # Window identifier
    last_active_window: Optional[str] = None  # Title of last active window

class WindowsContext:
    """
    Context for interacting with a Windows desktop session.
    Manages interactions with the active window.
    """
    
    def __init__(
        self,
        windows: Windows,
        config: WindowsContextConfig = WindowsContextConfig(),
        state: Optional[WindowsContextState] = None,
    ):
        self.windows = windows
        self.config = config
        self.state = state or WindowsContextState()
        
        # Will be set up when needed
        self.session: Optional[WindowsSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._initialize_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    @time_execution_async('--close')
    async def close(self):
        """Close the Windows context"""
        logger.debug('Closing Windows context')
        
        try:
            if self.session is None:
                return
            
            if not self.config._force_keep_context_alive:
                self.session = None
                
                # Force garbage collection
                gc.collect()
        
        except Exception as e:
            logger.debug(f'Failed to close Windows context: {e}')
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if not self.config._force_keep_context_alive and self.session is not None:
            logger.debug('WindowsContext was not properly closed before destruction')
            try:
                self.session = None
                gc.collect()
            except Exception as e:
                logger.warning(f'Failed to force close Windows context: {e}')
    
    @time_execution_async('--initialize_session')
    async def _initialize_session(self):
        """Initialize the Windows session"""
        logger.debug('Initializing Windows context')
        
        # Initialize the Windows UIA if needed
        if not self.windows._initialized:
            await self.windows.initialize()
        
        # Get active window
        active_window = await self.windows.get_active_window()
        
        self.session = WindowsSession(
            active_window=active_window,
            cached_state=None,
        )
        
        if active_window:
            window_title = active_window.Name
            logger.debug(f'Active window: {window_title}')
            self.state.last_active_window = window_title
        else:
            logger.warning('No active window found')
        
        return self.session
    
    async def get_session(self) -> WindowsSession:
        """Lazy initialization of the Windows session and related components"""
        if self.session is None:
            return await self._initialize_session()
        return self.session
    
    async def get_active_window(self):
        """Get the currently active window"""
        session = await self.get_session()
        active_window = await self.windows.get_active_window()
        
        # Update session if active window has changed
        if active_window and (not session.active_window or active_window.Name != session.active_window.Name):
            session.active_window = active_window
            self.state.last_active_window = active_window.Name
            
        return session.active_window
    
    @time_execution_async('--get_state')
    async def get_state(self):
        """Get the current state of the Windows desktop and active window"""
        try:
            await self._wait_for_ui_idle()
            session = await self.get_session()
            
            # Update with current active window
            active_window = await self.get_active_window()
            if not active_window:
                logger.warning("No active window to get state from")
                if session.cached_state:
                    return session.cached_state
                
                # Return a minimal fallback state if no cached state
                from computeruse.uia.views import WindowState
                from computeruse.dom.views import DOMElementNode, DOMTextNode
                
                root_element = DOMElementNode(
                    tag_name="window",
                    xpath="/window",
                    attributes={"Name": "No active window", "ControlType": "Window"},
                    children=[
                        DOMTextNode(
                            text="No active window detected. Try clicking on the application window you want to control.",
                            xpath="/window/text",
                            is_visible=True,
                            parent=None
                        )
                    ],
                    is_visible=True,
                    is_interactive=False,
                    is_top_element=True,
                    is_in_viewport=True,
                    highlight_index=None,
                    parent=None,
                )
                
                fallback_state = WindowState(
                    title="No active window",
                    process_name="",
                    element_tree=root_element,
                    selector_map={},
                    screenshot=None,
                    windows=[],
                )
                
                session.cached_state = fallback_state
                return fallback_state
            
            # Import here to avoid circular imports
            from computeruse.dom.service import DomService
            from computeruse.uia.views import WindowState
            
            # Process UI tree to get interactive elements
            try:
                dom_service = DomService(active_window)
                state = await dom_service.get_clickable_elements(
                    focus_element=-1,
                    highlight_elements=self.config.highlight_elements,
                    viewport_expansion=self.config.viewport_expansion,
                )
            except Exception as dom_error:
                logger.error(f"Error processing DOM: {dom_error}")
                if session.cached_state:
                    return session.cached_state
                
                # Create a minimal DOM if processing fails
                from computeruse.dom.views import DOMElementNode, DOMTextNode, DOMState
                
                root_element = DOMElementNode(
                    tag_name="window",
                    xpath="/window",
                    attributes={"Name": active_window.Name if hasattr(active_window, "Name") else "Unknown", 
                               "ControlType": "Window"},
                    children=[
                        DOMTextNode(
                            text=f"Error processing UI elements: {str(dom_error)}",
                            xpath="/window/text",
                            is_visible=True,
                            parent=None
                        )
                    ],
                    is_visible=True,
                    is_interactive=False,
                    is_top_element=True,
                    is_in_viewport=True,
                    highlight_index=None,
                    parent=None,
                )
                
                state = DOMState(element_tree=root_element, selector_map={})
            
            # Take screenshot with better error handling
            try:
                screenshot = await self.windows.take_screenshot(active_window)
            except Exception as screenshot_error:
                logger.error(f"Error taking screenshot: {screenshot_error}")
                screenshot = None
            
            # Get window information with error handling
            try:
                windows_info = await self._get_all_windows_info()
            except Exception as windows_info_error:
                logger.error(f"Error getting windows info: {windows_info_error}")
                windows_info = []
            
            # Create window state with best available information
            try:
                window_title = active_window.Name if hasattr(active_window, "Name") else "Unknown"
                process_name = active_window.ProcessName if hasattr(active_window, "ProcessName") else ''
            except Exception as attr_error:
                logger.error(f"Error reading window attributes: {attr_error}")
                window_title = "Unknown"
                process_name = ""
            
            window_state = WindowState(
                title=window_title,
                process_name=process_name,
                element_tree=state.element_tree,
                selector_map=state.selector_map,
                screenshot=screenshot,
                windows=windows_info,
            )
            
            session.cached_state = window_state
            return window_state
            
        except Exception as e:
            logger.error(f"Error getting Windows state: {e}")
            # Return cached state if available
            if hasattr(self, 'session') and self.session and self.session.cached_state:
                return self.session.cached_state
            
            # Create minimal fallback state as last resort
            from computeruse.uia.views import WindowState
            from computeruse.dom.views import DOMElementNode, DOMTextNode
            
            root_element = DOMElementNode(
                tag_name="window",
                xpath="/window",
                attributes={"Name": "Error", "ControlType": "Window"},
                children=[
                    DOMTextNode(
                        text=f"Error getting UI state: {str(e)}",
                        xpath="/window/text",
                        is_visible=True,
                        parent=None
                    )
                ],
                is_visible=True,
                is_interactive=False,
                is_top_element=True,
                is_in_viewport=True,
                highlight_index=None,
                parent=None,
            )
            
            return WindowState(
                title="Error state",
                process_name="",
                element_tree=root_element,
                selector_map={},
                screenshot=None,
                windows=[],
            )
    
    async def _wait_for_ui_idle(self):
        """Wait for UI to stabilize"""
        # Start timing
        start_time = time.time()
        
        # Basic wait for minimum time
        await asyncio.sleep(self.config.minimum_wait_time)
        
        try:
            # Additional wait logic for UI idle detection could be added here
            # For now we'll use a simple timeout approach
            remaining = max(0, self.config.wait_for_idle_timeout - (time.time() - start_time))
            if remaining > 0:
                await asyncio.sleep(remaining)
        except Exception as e:
            logger.debug(f"Error during idle wait: {e}")
        
        # Calculate time spent waiting
        elapsed = time.time() - start_time
        logger.debug(f"--UI wait time: {elapsed:.2f} seconds")
    
    async def _get_all_windows_info(self) -> List[Dict[str, Any]]:
        """Get information about all windows, similar to tabs in a browser"""
        try:
            if not self.windows.auto:
                return []
            
            windows_info = []
            desktop = self.windows.desktop
            
            if not desktop:
                return []
            
            # Get all top-level windows
            windows = desktop.GetChildren()
            for i, window in enumerate(windows):
                # Skip invisible windows using helper method
                if not self.windows.is_control_visible(window):
                    continue
                
                # Skip windows from ignored processes
                process_name = window.ProcessName if hasattr(window, 'ProcessName') else ''
                if process_name in self.windows.config.ignore_processes:
                    continue
                
                windows_info.append({
                    "id": i,
                    "title": window.Name,
                    "process_name": process_name
                })
            
            return windows_info
            
        except Exception as e:
            logger.error(f"Error getting windows info: {e}")
            return []
    
    async def get_selector_map(self):
        """Get the selector map for the current active window"""
        session = await self.get_session()
        if session.cached_state is None:
            return {}
        return session.cached_state.selector_map
    
    async def remove_highlights(self):
        """Remove highlights from UI elements"""
        # No direct implementation for UIA - this may require custom highlighting
        pass
    
    async def switch_to_window(self, window_id: int):
        """Switch to a window by its ID"""
        windows_info = await self._get_all_windows_info()
        
        if 0 <= window_id < len(windows_info):
            try:
                window_title = windows_info[window_id]["title"]
                target_window = await self.windows.get_window_by_title(window_title)
                
                if target_window:
                    if hasattr(target_window, 'SetFocus'):
                        target_window.SetFocus()
                    else:
                        # Fall back to alternative method
                        target_window.SwitchToThisWindow()
                    
                    # Update session
                    session = await self.get_session() 
                    session.active_window = target_window
                    self.state.last_active_window = window_title
                    
                    # Wait for window to become active
                    await asyncio.sleep(self.config.wait_between_actions)
                    
                    logger.info(f"Switched to window: {window_title}")
                    return True
                else:
                    logger.error(f"Could not find window with title: {window_title}")
                    return False
            except Exception as e:
                logger.error(f"Error switching to window {window_id}: {e}")
                return False
        else:
            logger.error(f"Window ID {window_id} out of range")
            return False
    
    async def click_element(self, element):
        """Click on a UI element"""
        try:
            # Ensure element is valid
            if not element:
                logger.error("Cannot click on None element")
                return False
            
            # Wait briefly before interacting with the element
            await asyncio.sleep(0.1)
            
            # Scroll if needed to bring element into view
            try:
                if hasattr(element, 'ScrollIntoView'):
                    element.ScrollIntoView()
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Failed to scroll element into view: {e}")
            
            # First try to click using the element's Click method
            success = False
            try:
                if hasattr(element, 'Click'):
                    element.Click()
                    success = True
                    logger.debug("Clicked element using element.Click()")
            except Exception as e:
                logger.warning(f"Failed to click element using Click method: {e}")
            
            # If the first method failed, try getting clickable point
            if not success:
                try:
                    if hasattr(element, 'GetClickablePoint'):
                        try:
                            x, y = element.GetClickablePoint()
                            if x > 0 and y > 0:
                                self.windows.auto.Click(x, y)
                                success = True
                                logger.debug(f"Clicked element at coordinates ({x}, {y})")
                            else:
                                logger.warning(f"Invalid clickable point: ({x}, {y})")
                        except Exception as e:
                            logger.warning(f"Failed to get clickable point: {e}")
                except Exception as e:
                    logger.warning(f"Failed to click using clickable point: {e}")
            
            # If the second method failed, try getting the bounding rectangle and clicking the center
            if not success:
                try:
                    if hasattr(element, 'BoundingRectangle'):
                        rect = element.BoundingRectangle
                        if rect and hasattr(rect, 'left') and hasattr(rect, 'top') and \
                           hasattr(rect, 'right') and hasattr(rect, 'bottom'):
                            # Calculate center point
                            x = int((rect.left + rect.right) / 2)
                            y = int((rect.top + rect.bottom) / 2)
                            if x > 0 and y > 0:
                                self.windows.auto.Click(x, y)
                                success = True
                                logger.debug(f"Clicked element center at ({x}, {y})")
                            else:
                                logger.warning(f"Invalid center point: ({x}, {y})")
                        else:
                            logger.warning("Invalid bounding rectangle")
                except Exception as e:
                    logger.warning(f"Failed to click using bounding rectangle: {e}")
            
            # As a last resort, try using Invoke pattern
            if not success:
                try:
                    if hasattr(element, 'Invoke') and callable(element.Invoke):
                        element.Invoke()
                        success = True
                        logger.debug("Invoked element using Invoke pattern")
                    elif hasattr(element, 'IsInvokePatternAvailable') and element.IsInvokePatternAvailable():
                        element.Invoke()
                        success = True
                        logger.debug("Invoked element using Invoke pattern after availability check")
                except Exception as e:
                    logger.warning(f"Failed to invoke element: {e}")
            
            if not success:
                logger.error("All click methods failed")
                return False
                
            # Wait for UI to stabilize
            await asyncio.sleep(self.config.wait_between_actions)
            
            return True
        except Exception as e:
            logger.error(f"Error clicking element: {e}")
            return False
    
    async def input_text(self, element, text: str):
        """Input text into a UI element"""
        try:
            # Ensure element is valid
            if not element:
                logger.error("Cannot input text to None element")
                return False
            
            # Make sure text is a string
            if not isinstance(text, str):
                text = str(text)
                
            # Wait briefly before interacting with the element
            await asyncio.sleep(0.1)
            
            # Check if element is suitable for text input
            is_text_input = False
            if hasattr(element, 'ControlType'):
                # Check control type
                is_edit_control = (hasattr(element, 'GetControlTypeName') and 
                                  'Edit' in element.GetControlTypeName())
                is_text_input = is_edit_control
            
            # Scroll if needed to bring element into view
            try:
                if hasattr(element, 'ScrollIntoView'):
                    element.ScrollIntoView()
            except Exception as e:
                logger.warning(f"Failed to scroll element into view: {e}")
            
            # First try to focus the element
            try:
                # Clear existing text if applicable
                if hasattr(element, 'SetFocus'):
                    element.SetFocus()
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Failed to set focus: {e}")
            
            # Try different methods to clear existing text
            try:
                if hasattr(element, 'Clear'):
                    # Direct clear method
                    element.Clear()
                elif hasattr(element, 'SetValue') and is_text_input:
                    # Try setting to empty value first
                    element.SetValue("")
                elif hasattr(element, 'SendKeys'):
                    # Send Ctrl+A and Delete to clear text
                    element.SendKeys('{Ctrl}a{Delete}')
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Failed to clear text: {e}")
            
            # Input new text using multiple fallback methods
            success = False
            
            # Method 1: SetValue if available
            if hasattr(element, 'SetValue') and is_text_input and not success:
                try:
                    element.SetValue(text)
                    success = True
                except Exception as e:
                    logger.warning(f"Failed to set value: {e}")
            
            # Method 2: SendKeys on element if available
            if hasattr(element, 'SendKeys') and not success:
                try:
                    element.SendKeys(text)
                    success = True
                except Exception as e:
                    logger.warning(f"Failed to send keys to element: {e}")
            
            # Method 3: Fall back to global SendKeys
            if not success:
                try:
                    if self.windows.auto:
                        self.windows.auto.SendKeys(text)
                        success = True
                except Exception as e:
                    logger.warning(f"Failed to send global keys: {e}")
            
            # Method 4: Virtual keyboard simulation as last resort
            if not success:
                try:
                    # Simulate individual keypresses
                    for char in text:
                        self.windows.auto.SendKeys(char)
                        await asyncio.sleep(0.01)
                    success = True
                except Exception as e:
                    logger.error(f"All text input methods failed, last error: {e}")
                    return False
            
            # Wait for UI to stabilize
            await asyncio.sleep(self.config.wait_between_actions)
            
            logger.info(f"Input text: '{text}'")
            return True
        except Exception as e:
            logger.error(f"Error inputting text: {e}")
            return False