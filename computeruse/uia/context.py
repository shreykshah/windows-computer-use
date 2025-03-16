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
        await self._wait_for_ui_idle()
        session = await self.get_session()
        
        # Update with current active window
        active_window = await self.get_active_window()
        if not active_window:
            logger.warning("No active window to get state from")
            if session.cached_state:
                return session.cached_state
            return None
        
        try:
            # Import here to avoid circular imports
            from computeruse.dom.service import DomService
            
            # Process UI tree to get interactive elements
            dom_service = DomService(active_window)
            state = await dom_service.get_clickable_elements(
                focus_element=-1,
                highlight_elements=self.config.highlight_elements,
                viewport_expansion=self.config.viewport_expansion,
            )
            
            # Take screenshot
            screenshot = await self.windows.take_screenshot(active_window)
            
            # Create the state object
            from computeruse.uia.views import WindowState, TabInfo
            
            # Get window information including tabs/child windows
            windows_info = await self._get_all_windows_info()
            
            window_state = WindowState(
                title=active_window.Name,
                process_name=active_window.ProcessName if hasattr(active_window, 'ProcessName') else '',
                element_tree=state.element_tree,
                selector_map=state.selector_map,
                screenshot=screenshot,
                windows=windows_info,
            )
            
            session.cached_state = window_state
            return window_state
            
        except Exception as e:
            logger.error(f"Error getting Windows state: {e}")
            return session.cached_state if session.cached_state else None
    
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
                # Skip invisible windows
                if not window.IsVisible:
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
            
            # Scroll if needed to bring element into view
            if hasattr(element, 'ScrollIntoView'):
                element.ScrollIntoView()
            
            # Click the element
            if hasattr(element, 'Click'):
                element.Click()
            else:
                x, y = element.GetClickablePoint()
                if x > 0 and y > 0:
                    self.windows.auto.Click(x, y)
                else:
                    logger.error("Element has no clickable point")
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
            
            # Scroll if needed to bring element into view
            if hasattr(element, 'ScrollIntoView'):
                element.ScrollIntoView()
            
            # Clear existing text if applicable
            if hasattr(element, 'SetFocus'):
                element.SetFocus()
            
            if hasattr(element, 'Clear'):
                element.Clear()
            elif hasattr(element, 'SendKeys'):
                # Send Ctrl+A and Delete to clear text
                element.SendKeys('{Ctrl}a{Delete}')
            
            # Input new text
            if hasattr(element, 'SendKeys'):
                element.SendKeys(text)
            else:
                # Fall back to global input
                self.windows.auto.SendKeys(text)
            
            # Wait for UI to stabilize
            await asyncio.sleep(self.config.wait_between_actions)
            
            return True
        except Exception as e:
            logger.error(f"Error inputting text: {e}")
            return False