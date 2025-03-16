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
        logger.info('Initializing Windows UIA')
        self.config = config
        
        # Check if we're on Windows
        if platform.system() != "Windows":
            logger.error("Windows UIA can only be used on Windows systems.")
            logger.error(f"Current platform: {platform.system()}")
            self.is_windows = False
        else:
            self.is_windows = True
            try:
                # Log Python version and implementation details
                logger.info(f"Python version: {platform.python_version()} ({platform.python_implementation()})")
                logger.info(f"Windows version: {platform.version()}")
                
                # Only import Windows-specific modules when running on Windows
                logger.debug("Importing Windows-specific modules...")
                
                try:
                    import comtypes.client
                    logger.info(f"comtypes version: {comtypes.__version__ if hasattr(comtypes, '__version__') else 'unknown'}")
                    self.comtypes = comtypes
                except ImportError as e:
                    logger.error(f"Failed to import comtypes: {e}")
                    logger.error("Install with: pip install comtypes")
                    self.comtypes = None
                
                try:
                    import pywinauto
                    logger.info(f"pywinauto version: {pywinauto.__version__ if hasattr(pywinauto, '__version__') else 'unknown'}")
                    from pywinauto.application import Application
                    self.pywinauto = pywinauto
                    self.Application = Application
                except ImportError as e:
                    logger.error(f"Failed to import pywinauto: {e}")
                    logger.error("Install with: pip install pywinauto")
                    self.pywinauto = None
                    self.Application = None
                
                # Import UIA specific modules
                try:
                    import uiautomation as auto
                    logger.info(f"uiautomation version: {auto.VERSION if hasattr(auto, 'VERSION') else 'unknown'}")
                    self.auto = auto
                except ImportError as e:
                    logger.error(f"Failed to import uiautomation: {e}")
                    logger.error("Install with: pip install uiautomation")
                    self.auto = None
                    
                # Log COM status
                try:
                    if self.comtypes:
                        logger.info("Testing COM initialization...")
                        self.comtypes.CoInitialize()
                        logger.info("COM initialized successfully")
                except Exception as e:
                    logger.error(f"COM initialization test failed: {e}")
                
            except Exception as e:
                logger.error(f"Unexpected error during Windows UIA initialization: {e}")
                import traceback
                logger.error(f"Stack trace: {traceback.format_exc()}")
                self.is_windows = False
        
        # Initialize state variables
        self.desktop = None
        self._initialized = False
        
        # Log initialization status
        if self.is_windows and self.auto:
            logger.info("Windows UIA components imported successfully")
        else:
            logger.warning("Windows UIA initialization incomplete - issues detected")

    async def initialize(self) -> bool:
        """Initialize the Windows UIA with required components"""
        if not self.is_windows:
            logger.error("Cannot initialize Windows UIA on non-Windows system")
            return False
        
        # If already initialized successfully, don't reinitialize
        if self._initialized and self.desktop:
            logger.debug("Windows UIA already initialized")
            return True
            
        # Multiple initialization attempts with different strategies
        initialization_attempts = 0
        max_attempts = 3
        
        while initialization_attempts < max_attempts:
            initialization_attempts += 1
            logger.debug(f"Initialization attempt {initialization_attempts}/{max_attempts}")
            
            try:
                if not self.auto:
                    logger.error("UIAutomation package not available")
                    return False
                
                # Strategy 1: Direct GetRootControl
                try:
                    self.desktop = self.auto.GetRootControl()
                    if self.desktop:
                        logger.debug(f"Desktop control obtained: {self.desktop}")
                        self._initialized = True
                        return True
                except Exception as e:
                    logger.warning(f"Standard initialization failed: {e}")
                
                # Strategy 2: Initialize and then GetRootControl
                try:
                    if hasattr(self.auto, 'InitializeUIAutomation'):
                        self.auto.InitializeUIAutomation()
                        self.desktop = self.auto.GetRootControl()
                        if self.desktop:
                            logger.info("Desktop control obtained through explicit initialization")
                            self._initialized = True
                            return True
                except Exception as e:
                    logger.warning(f"Explicit initialization failed: {e}")
                
                # Strategy 3: Try to get active window as fallback
                try:
                    if hasattr(self.auto, 'GetForegroundControl'):
                        active_window = self.auto.GetForegroundControl()
                        if active_window:
                            # If we can get foreground control, we have a working UIA
                            self.desktop = self.auto.GetRootControl()
                            if self.desktop:
                                logger.info("Desktop control obtained after getting foreground control")
                                self._initialized = True
                                return True
                except Exception as e:
                    logger.warning(f"Foreground control fallback failed: {e}")
                
                # If all attempts failed, wait and retry
                if initialization_attempts < max_attempts:
                    wait_time = 1.0 * initialization_attempts  # Increasing wait time
                    logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Initialization attempt {initialization_attempts} failed: {e}")
                if initialization_attempts < max_attempts:
                    await asyncio.sleep(1.0)
        
        logger.error(f"Failed to initialize Windows UIA after {max_attempts} attempts")
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
            from PIL import ImageGrab, Image
            
            # Take screenshot of the desktop as a fallback method
            # This doesn't require specific APIs that might not be available in different uiautomation versions
            try:
                # Get coordinates if control is provided
                if control is not None and hasattr(control, 'BoundingRectangle'):
                    try:
                        rect = control.BoundingRectangle
                        # Validate rectangle coordinates
                        if (hasattr(rect, 'left') and hasattr(rect, 'top') and 
                            hasattr(rect, 'right') and hasattr(rect, 'bottom')):
                            # Ensure valid coordinates
                            left = max(0, rect.left)
                            top = max(0, rect.top)
                            right = max(left + 1, rect.right)
                            bottom = max(top + 1, rect.bottom)
                            bbox = (left, top, right, bottom)
                            try:
                                image = ImageGrab.grab(bbox=bbox)
                            except Exception as e:
                                logger.warning(f"Failed to grab bounded screenshot: {e}, falling back to full screen")
                                image = ImageGrab.grab()
                        else:
                            logger.warning("Invalid bounding rectangle, falling back to full screen")
                            image = ImageGrab.grab()
                    except Exception as e:
                        logger.warning(f"Failed to get bounding rectangle: {e}, falling back to full screen")
                        image = ImageGrab.grab()
                else:
                    # Capture the entire screen
                    image = ImageGrab.grab()
                
                # Convert to base64
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                # Optionally save to file
                if save_path:
                    try:
                        image.save(save_path)
                    except Exception as e:
                        logger.error(f"Failed to save screenshot to {save_path}: {e}")
                
                return base64_image
            
            except Exception as e:
                logger.error(f"Failed to capture screenshot with PIL: {e}")
                # Return a dummy image or placeholder instead of None
                # This prevents cascading errors in the rest of the code
                try:
                    # Create a small dummy image
                    dummy_img = Image.new('RGB', (100, 100), color = 'gray')
                    dummy_text = "Screenshot failed"
                    
                    # Add text to the image if PIL has ImageDraw
                    try:
                        from PIL import ImageDraw
                        draw = ImageDraw.Draw(dummy_img)
                        draw.text((10, 40), dummy_text, fill=(255, 255, 255))
                    except Exception as draw_error:
                        logger.debug(f"Could not add text to dummy image: {draw_error}")
                    
                    # Convert to base64
                    buffer = BytesIO()
                    dummy_img.save(buffer, format="PNG")
                    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    
                    return base64_image
                except Exception as dummy_error:
                    logger.error(f"Failed to create dummy image: {dummy_error}")
                    # Generate a minimal 1x1 pixel dummy image
                    try:
                        minimal_img = Image.new('RGB', (1, 1), color='black')
                        buffer = BytesIO()
                        minimal_img.save(buffer, format="PNG")
                        return base64.b64encode(buffer.getvalue()).decode("utf-8")
                    except:
                        # If everything fails, return None
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
            
    def is_control_visible(self, control: Any) -> bool:
        """
        Safely check if a UI Automation control is visible
        
        Args:
            control: A UIAutomation control
            
        Returns:
            True if the control is visible, False otherwise
        """
        try:
            # First check if IsVisible attribute exists
            if hasattr(control, 'IsVisible'):
                return bool(control.IsVisible)
            
            # Alternative check if control has BoundingRectangle
            if hasattr(control, 'BoundingRectangle'):
                rect = control.BoundingRectangle
                if rect:
                    # Safely check dimensions with proper error handling
                    try:
                        # If rectangle has non-zero dimensions, consider it visible
                        width = rect.right - rect.left if hasattr(rect, 'right') and hasattr(rect, 'left') else 0
                        height = rect.bottom - rect.top if hasattr(rect, 'bottom') and hasattr(rect, 'top') else 0
                        return width > 0 and height > 0
                    except Exception as e:
                        logger.debug(f"Error calculating rectangle dimensions: {e}")
            
            # Additional check - try getting ControlType if available
            if hasattr(control, 'ControlType') and control.ControlType is not None:
                return True
                
            # Default assumption - if we can't determine visibility, assume it's visible
            # to avoid skipping potentially important elements
            return True
        except Exception as e:
            logger.debug(f"Error checking control visibility: {e}")
            # On exception, assume visible to avoid skipping elements
            return True