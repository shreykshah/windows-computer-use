import gc
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from computeruse.dom.views import (
    DOMBaseNode,
    DOMElementNode,
    DOMState,
    DOMTextNode,
    SelectorMap,
)
from computeruse.utils import time_execution_async, time_execution_sync

logger = logging.getLogger(__name__)


@dataclass
class ViewportInfo:
    width: int
    height: int


class DomService:
    def __init__(self, window_control: Any):
        """
        Initialize the DOM service with a UI Automation window control
        
        Args:
            window_control: A UIAutomation window control
        """
        self.window = window_control
        self.xpath_cache = {}

    @time_execution_async('--get_clickable_elements')
    async def get_clickable_elements(
        self,
        highlight_elements: bool = True,
        focus_element: int = -1,
        viewport_expansion: int = 0,
    ) -> DOMState:
        """
        Get all clickable elements from the UIA tree
        
        Args:
            highlight_elements: Whether to highlight elements
            focus_element: Index of the element to focus on, or -1 for all
            viewport_expansion: Expansion of viewport in pixels
            
        Returns:
            DOMState object containing the element tree and selector map
        """
        element_tree, selector_map = await self._build_uia_tree(highlight_elements, focus_element, viewport_expansion)
        return DOMState(element_tree=element_tree, selector_map=selector_map)

    @time_execution_async('--build_uia_tree')
    async def _build_uia_tree(
        self,
        highlight_elements: bool,
        focus_element: int,
        viewport_expansion: int,
    ) -> Tuple[DOMElementNode, SelectorMap]:
        """
        Build the UIA tree for the current window
        
        Args:
            highlight_elements: Whether to highlight elements
            focus_element: Index of the element to focus on, or -1 for all
            viewport_expansion: Expansion of viewport in pixels
            
        Returns:
            Tuple of (element_tree, selector_map)
        """
        if not self.window:
            raise ValueError("Window control is not available")
        
        # Process the UIA tree using Python instead of JavaScript
        try:
            root_element = await self._process_window_to_dom_tree(self.window)
            selector_map = await self._build_selector_map(root_element)
            
            # Optional: Implement highlighting for Windows UIA elements if needed
            # This would require a platform-specific implementation
            
            return root_element, selector_map
        except Exception as e:
            logger.error(f"Error building UIA tree: {e}")
            raise
    
    async def _process_window_to_dom_tree(self, window_control: Any) -> DOMElementNode:
        """
        Convert a UIA window control and its children to a DOM tree
        
        Args:
            window_control: A UIAutomation window control
            
        Returns:
            DOMElementNode representing the window
        """
        # Create the root element for the window
        highlight_index = 0
        processed_elements = {}
        
        # Process the window's control tree
        try:
            # Create root element
            root_element = DOMElementNode(
                tag_name="window",
                xpath="/window",
                attributes=self._get_control_attributes(window_control),
                children=[],
                is_visible=True,
                is_interactive=True,
                is_top_element=True,
                is_in_viewport=True,
                highlight_index=None,  # Root window shouldn't be interactable directly
                parent=None,
            )
            
            # Process all controls recursively
            await self._process_control_children(window_control, root_element, processed_elements, highlight_index)
            
            return root_element
        
        except Exception as e:
            logger.error(f"Error processing window to DOM tree: {e}")
            raise
    
    async def _process_control_children(
        self, 
        control: Any, 
        parent_element: DOMElementNode,
        processed_elements: Dict[int, DOMElementNode],
        highlight_index: int,
    ) -> int:
        """
        Process all children of a UIA control recursively
        
        Args:
            control: A UIAutomation control
            parent_element: Parent DOMElementNode
            processed_elements: Dict of processed elements to avoid duplicates
            highlight_index: Current highlight index
            
        Returns:
            Updated highlight index
        """
        try:
            # Get all children of the control
            children = []
            if hasattr(control, 'GetChildren'):
                children = control.GetChildren()
            
            for child in children:
                # Skip controls that are not relevant or visible
                if not self._is_relevant_control(child):
                    continue
                
                # Process the child control
                element, new_highlight_index = await self._process_control(
                    child, 
                    parent_element,
                    processed_elements,
                    highlight_index
                )
                
                if element:
                    parent_element.children.append(element)
                    highlight_index = new_highlight_index
            
            return highlight_index
        
        except Exception as e:
            logger.error(f"Error processing control children: {e}")
            return highlight_index
    
    async def _process_control(
        self,
        control: Any,
        parent_element: DOMElementNode,
        processed_elements: Dict[int, DOMElementNode],
        highlight_index: int,
    ) -> Tuple[Optional[Union[DOMElementNode, DOMTextNode]], int]:
        """
        Process a single UIA control
        
        Args:
            control: A UIAutomation control
            parent_element: Parent DOMElementNode
            processed_elements: Dict of processed elements to avoid duplicates
            highlight_index: Current highlight index
            
        Returns:
            Tuple of (element, new_highlight_index)
        """
        try:
            # Check if we've already processed this control
            control_id = id(control)
            if control_id in processed_elements:
                return processed_elements[control_id], highlight_index
            
            # Get attributes and determine if interactive
            attributes = self._get_control_attributes(control)
            is_interactive = self._is_interactive_control(control)
            
            # Check if this is a text-only element
            if self._is_text_only_control(control):
                text = control.Name if hasattr(control, 'Name') and control.Name else ""
                if not text:
                    return None, highlight_index
                
                text_node = DOMTextNode(
                    text=text,
                    is_visible=True if hasattr(control, 'IsVisible') and control.IsVisible else False,
                    parent=parent_element,
                )
                return text_node, highlight_index
            
            # Create element for this control
            current_highlight_index = None
            if is_interactive:
                current_highlight_index = highlight_index
                highlight_index += 1
            
            # Build xpath
            xpath = self._build_xpath_for_control(control, parent_element)
            
            # Create the element
            element = DOMElementNode(
                tag_name=attributes.get('ControlType', 'unknown'),
                xpath=xpath,
                attributes=attributes,
                children=[],
                is_visible=True if hasattr(control, 'IsVisible') and control.IsVisible else False,
                is_interactive=is_interactive,
                is_top_element=True,  # Assuming all direct UIA controls are top elements
                is_in_viewport=True,  # Assuming all accessible controls are in viewport
                highlight_index=current_highlight_index,
                parent=parent_element,
            )
            
            # Store in processed elements to avoid duplicates
            processed_elements[control_id] = element
            
            # Process children recursively
            highlight_index = await self._process_control_children(
                control, element, processed_elements, highlight_index
            )
            
            return element, highlight_index
        
        except Exception as e:
            logger.error(f"Error processing control: {e}")
            return None, highlight_index
    
    def _get_control_attributes(self, control: Any) -> Dict[str, str]:
        """
        Get attributes of a UIA control
        
        Args:
            control: A UIAutomation control
            
        Returns:
            Dictionary of attributes
        """
        attributes = {}
        
        # Common attributes to extract from UIA controls
        attribute_names = [
            'Name', 'AutomationId', 'ClassName', 'ControlType', 
            'ProcessId', 'ProcessName', 'Value', 'IsEnabled',
            'AccessKey', 'AcceleratorKey', 'HelpText'
        ]
        
        for attr in attribute_names:
            if hasattr(control, attr):
                value = getattr(control, attr)
                if value is not None:
                    attributes[attr] = str(value)
        
        # Add role information if available
        if hasattr(control, 'GetControlTypeName'):
            attributes['role'] = control.GetControlTypeName()
        
        # Add pattern-specific properties
        if hasattr(control, 'GetToggleState') and hasattr(control, 'IsTogglePatternAvailable') and control.IsTogglePatternAvailable():
            try:
                attributes['ToggleState'] = str(control.GetToggleState())
            except Exception:
                pass
                
        if hasattr(control, 'GetRangeValue') and hasattr(control, 'IsRangeValuePatternAvailable') and control.IsRangeValuePatternAvailable():
            try:
                attributes['RangeValue'] = str(control.GetRangeValue())
            except Exception:
                pass
                
        if hasattr(control, 'IsSelected') and hasattr(control, 'IsSelectionItemPatternAvailable') and control.IsSelectionItemPatternAvailable():
            try:
                attributes['IsSelected'] = str(control.IsSelected())
            except Exception:
                pass
                
        if hasattr(control, 'IsExpanded') and hasattr(control, 'IsExpandCollapsePatternAvailable') and control.IsExpandCollapsePatternAvailable():
            try:
                attributes['IsExpanded'] = str(control.IsExpanded())
            except Exception:
                pass
        
        # Add position information if available
        if hasattr(control, 'BoundingRectangle'):
            try:
                rect = control.BoundingRectangle
                if rect:
                    attributes['X'] = str(rect.left)
                    attributes['Y'] = str(rect.top)
                    attributes['Width'] = str(rect.width())
                    attributes['Height'] = str(rect.height())
            except Exception:
                pass
        
        return attributes
    
    def _is_relevant_control(self, control: Any) -> bool:
        """
        Check if a control is relevant for processing
        
        Args:
            control: A UIAutomation control
            
        Returns:
            True if the control is relevant, False otherwise
        """
        # Skip invisible controls
        if hasattr(control, 'IsVisible') and not control.IsVisible:
            return False
        
        # Filter out certain control types that aren't useful for interaction
        if hasattr(control, 'ControlType'):
            # Example: filter out separator controls, etc.
            # This would need customization based on the UIA library used
            pass
        
        return True
    
    def _is_interactive_control(self, control: Any) -> bool:
        """
        Check if a control is interactive
        
        Args:
            control: A UIAutomation control
            
        Returns:
            True if the control is interactive, False otherwise
        """
        # Check for common interactive control types
        interactive_types = [
            'Button', 'CheckBox', 'ComboBox', 'Edit', 'Hyperlink',
            'ListItem', 'MenuItem', 'RadioButton', 'Slider', 'Spinner',
            'Tab', 'TextBox', 'TreeItem', 'ListBox', 'Menu', 'ToolBar',
            'SplitButton', 'Calendar', 'DataGrid', 'DataItem', 'Document',
            'ScrollBar', 'TitleBar', 'ProgressBar'
        ]
        
        # Check control type
        control_type = None
        if hasattr(control, 'GetControlTypeName'):
            control_type = control.GetControlTypeName()
        elif hasattr(control, 'ControlType'):
            control_type = control.ControlType
        
        if control_type and any(t in str(control_type) for t in interactive_types):
            return True
        
        # Check patterns that indicate interactivity
        interactive_patterns = [
            'Invoke', 'Toggle', 'Value', 'RangeValue', 
            'Selection', 'SelectionItem', 'ExpandCollapse', 
            'Transform', 'Scroll', 'GridItem', 'MultipleView',
            'Window', 'ItemContainer', 'VirtualizedItem', 'Dock'
        ]
        
        # Check if control supports any interactive patterns
        if hasattr(control, 'GetSupportedPatterns'):
            patterns = control.GetSupportedPatterns()
            if any(p in patterns for p in interactive_patterns):
                return True
        
        # Check for specific pattern availability methods
        pattern_methods = [
            'IsInvokePatternAvailable',
            'IsTogglePatternAvailable',
            'IsExpandCollapsePatternAvailable',
            'IsValuePatternAvailable',
            'IsScrollPatternAvailable',
            'IsSelectionItemPatternAvailable'
        ]
        
        for method in pattern_methods:
            if hasattr(control, method) and getattr(control, method)():
                return True
        
        # Check if control is enabled
        if hasattr(control, 'IsEnabled') and control.IsEnabled:
            # Check for Name or AutomationId to identify potentially interactive controls
            if ((hasattr(control, 'Name') and control.Name) or 
                (hasattr(control, 'AutomationId') and control.AutomationId)):
                # Additional heuristic: check if it has a non-empty bounding rectangle
                if hasattr(control, 'BoundingRectangle'):
                    rect = control.BoundingRectangle
                    if rect and rect.width() > 0 and rect.height() > 0:
                        return True
        
        return False
    
    def _is_text_only_control(self, control: Any) -> bool:
        """
        Check if a control is a text-only element
        
        Args:
            control: A UIAutomation control
            
        Returns:
            True if the control is text-only, False otherwise
        """
        # Identify text-only controls (like TextBlock in WPF)
        text_types = ['Text', 'Label', 'StaticText']
        
        if hasattr(control, 'GetControlTypeName'):
            control_type = control.GetControlTypeName()
            if control_type and any(t in control_type for t in text_types):
                return True
        
        # If it has a name but no interactive patterns, it might be just text
        if (hasattr(control, 'Name') and control.Name and 
            hasattr(control, 'GetSupportedPatterns') and not control.GetSupportedPatterns()):
            return True
        
        return False
    
    def _build_xpath_for_control(self, control: Any, parent_element: DOMElementNode) -> str:
        """
        Build XPath for a UIA control
        
        Args:
            control: A UIAutomation control
            parent_element: Parent DOMElementNode
            
        Returns:
            XPath string
        """
        # Build simple XPath based on control type and parent XPath
        control_type = "unknown"
        if hasattr(control, 'GetControlTypeName'):
            control_type = control.GetControlTypeName()
        elif hasattr(control, 'ControlType'):
            control_type = str(control.ControlType)
        
        # Sanitize control type for XPath (remove spaces, special chars)
        control_type = ''.join(c if c.isalnum() else '_' for c in control_type)
        
        # Build XPath
        if parent_element.xpath == "/window":
            return f"/window/{control_type}"
        else:
            return f"{parent_element.xpath}/{control_type}"
    
    async def _build_selector_map(self, root_element: DOMElementNode) -> SelectorMap:
        """
        Build a selector map from the DOM tree
        
        Args:
            root_element: Root DOMElementNode
            
        Returns:
            SelectorMap mapping highlight indices to elements
        """
        selector_map = {}
        
        def process_element(element: DOMElementNode):
            if element.highlight_index is not None:
                selector_map[element.highlight_index] = element
            
            for child in element.children:
                if isinstance(child, DOMElementNode):
                    process_element(child)
        
        process_element(root_element)
        return selector_map