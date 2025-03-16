"""
Windows UIA Wrapper.
"""

import asyncio
import gc
import logging
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class WindowsConfig:
    """
    Configuration for the Windows UIA.

    Default values:
        take_screenshots: True
            Whether to take screenshots of the desktop

        extra_args: []
            Extra arguments to pass to the UIA

        ignore_processes: []
            List of process names to ignore when searching for elements
    """

    take_screenshots: bool = True
    extra_args: list[str] = field(default_factory=list)
    ignore_processes: list[str] = field(default_factory=lambda: ["explorer.exe", "SystemSettings.exe"])
    
    _force_keep_windows_alive: bool = False


class Windows:
    """
    Windows UIA wrapper.

    This is a persistent Windows UIA factory that can control Windows desktop applications.
    It uses the UI Automation framework to interact with desktop applications.
    """

    def __init__(
        self,
        config: WindowsConfig = WindowsConfig(),
    ):
        logger.debug('Initializing Windows UIA')
        self.config = config
        
        if platform.system() != "Windows":
            logger.warning("Windows UIA can only be used on Windows systems.")
            self.is_windows = False
        else:
            self.is_windows = True
            try:
                # Only import Windows-specific modules when running on Windows
                import comtypes.client
                import pywinauto
                from pywinauto.application import Application
                
                self.comtypes = comtypes
                self.pywinauto = pywinauto
                self.Application = Application
                
                # Import UIA specific modules
                try:
                    import uiautomation as auto
                    self.auto = auto
                except ImportError:
                    logger.error("uiautomation package not found. Install with: pip install uiautomation")
                    self.auto = None
            except ImportError as e:
                logger.error(f"Failed to import Windows UIA dependencies: {e}")
                logger.error("Install required packages with: pip install pywinauto comtypes")
                self.is_windows = False
        
        self.desktop = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the Windows UIA with required components"""
        if not self.is_windows:
            logger.error("Cannot initialize Windows UIA on non-Windows system")
            return False
        
        try:
            if self.auto:
                # Initialize UIAutomation
                self.desktop = self.auto.GetRootControl()
                logger.debug(f"Desktop control obtained: {self.desktop}")
                self._initialized = True
                return True
            else:
                logger.error("UIAutomation package not available")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize Windows UIA: {e}")
            return False
    
    async def get_active_window(self):
        """Get the currently active window"""
        if not self._initialized:
            await self.initialize()
        
        if self.auto:
            return self.auto.GetForegroundControl()
        return None
    
    async def get_window_by_title(self, title: str, partial_match: bool = True):
        """Get a window by its title"""
        if not self._initialized:
            await self.initialize()
        
        if self.auto:
            if partial_match:
                return self.auto.WindowControl(SubName=title)
            else:
                return self.auto.WindowControl(Name=title)
        return None
    
    async def get_window_by_process(self, process_name: str):
        """Get a window by its process name"""
        if not self._initialized:
            await self.initialize()
        
        if self.auto:
            return self.auto.WindowControl(ProcessName=process_name)
        return None
    
    async def take_screenshot(self, control=None, save_path: Optional[str] = None) -> Optional[str]:
        """
        Take a screenshot of the desktop or specific control.
        Returns base64 encoded image.
        """
        if not self.config.take_screenshots:
            return None
        
        if not self._initialized:
            await self.initialize()
        
        if not self.auto:
            return None
        
        try:
            import base64
            from io import BytesIO
            
            # Take the screenshot using UIAutomation
            if control is None:
                control = self.desktop
            
            # Get the control's rectangle
            rect = control.BoundingRectangle
            
            # Capture the screen
            try:
                image = self.auto.CaptureImage(rect)
                
                # Convert to base64
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                # Optionally save to file
                if save_path:
                    image.save(save_path)
                
                return base64_image
            except Exception as e:
                logger.error(f"Failed to capture screenshot: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None
    
    async def close(self):
        """Close the Windows UIA instance"""
        try:
            if not self.config._force_keep_windows_alive:
                # Clean up resources
                self.desktop = None
                self._initialized = False
                
                # Force garbage collection to release COM objects
                gc.collect()
        except Exception as e:
            logger.debug(f'Failed to close Windows UIA properly: {e}')

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if self._initialized and not self.config._force_keep_windows_alive:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    asyncio.run(self.close())
        except Exception as e:
            logger.debug(f'Failed to cleanup Windows UIA in destructor: {e}')