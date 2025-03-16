from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=False)
class DOMBaseNode:
    """Base class for all DOM nodes"""
    is_visible: bool
    # Use None as default and set parent later to avoid circular reference issues
    parent: Optional['DOMElementNode']


@dataclass(frozen=False)
class DOMTextNode(DOMBaseNode):
    """Represents a text node in the DOM tree"""
    text: str
    type: str = 'TEXT_NODE'

    def has_parent_with_highlight_index(self) -> bool:
        """Check if any parent has a highlight index"""
        current = self.parent
        while current is not None:
            # stop if the element has a highlight index (will be handled separately)
            if current.highlight_index is not None:
                return True

            current = current.parent
        return False

    def is_parent_in_viewport(self) -> bool:
        """Check if the parent is in the viewport"""
        if self.parent is None:
            return False
        return self.parent.is_in_viewport

    def is_parent_top_element(self) -> bool:
        """Check if the parent is a top element"""
        if self.parent is None:
            return False
        return self.parent.is_top_element


@dataclass(frozen=False)
class DOMElementNode(DOMBaseNode):
    """
    Represents an element node in the DOM/UIA tree
    
    xpath: The xpath/path of the element in the UIA tree
    """
    tag_name: str
    xpath: str
    attributes: Dict[str, str]
    children: List[DOMBaseNode]
    is_interactive: bool = False
    is_top_element: bool = False
    is_in_viewport: bool = False
    shadow_root: bool = False
    highlight_index: Optional[int] = None
    viewport_coordinates: Optional[Any] = None
    page_coordinates: Optional[Any] = None
    viewport_info: Optional[Any] = None

    def __repr__(self) -> str:
        """Get a string representation of the element"""
        tag_str = f'<{self.tag_name}'

        # Add attributes
        for key, value in self.attributes.items():
            tag_str += f' {key}="{value}"'
        tag_str += '>'

        # Add extra info
        extras = []
        if self.is_interactive:
            extras.append('interactive')
        if self.is_top_element:
            extras.append('top')
        if self.shadow_root:
            extras.append('shadow-root')
        if self.highlight_index is not None:
            extras.append(f'highlight:{self.highlight_index}')
        if self.is_in_viewport:
            extras.append('in-viewport')

        if extras:
            tag_str += f' [{", ".join(extras)}]'

        return tag_str

    @cached_property
    def hash(self) -> 'HashedDomElement':
        """Get a hash of this element for identification"""
        from computeruse.dom.history_tree_processor.service import HistoryTreeProcessor
        return HistoryTreeProcessor._hash_dom_element(self)

    def get_all_text_till_next_clickable_element(self, max_depth: int = -1) -> str:
        """Get all text content from this element and its children until the next clickable element"""
        text_parts = []

        def collect_text(node: DOMBaseNode, current_depth: int) -> None:
            if max_depth != -1 and current_depth > max_depth:
                return

            # Skip this branch if we hit a highlighted element (except for the current node)
            if isinstance(node, DOMElementNode) and node != self and node.highlight_index is not None:
                return

            if isinstance(node, DOMTextNode):
                text_parts.append(node.text)
            elif isinstance(node, DOMElementNode):
                for child in node.children:
                    collect_text(child, current_depth + 1)

        collect_text(self, 0)
        return '\n'.join(text_parts).strip()

    def clickable_elements_to_string(self, include_attributes: list[str] | None = None) -> str:
        """Convert the processed DOM content to a string representation."""
        formatted_text = []

        def process_node(node: DOMBaseNode, depth: int) -> None:
            if isinstance(node, DOMElementNode):
                # Add element with highlight_index
                if node.highlight_index is not None:
                    attributes_str = ''
                    text = node.get_all_text_till_next_clickable_element()
                    if include_attributes:
                        attributes = list(
                            set(
                                [
                                    str(value)
                                    for key, value in node.attributes.items()
                                    if key in include_attributes and value != node.tag_name
                                ]
                            )
                        )
                        if text in attributes:
                            attributes.remove(text)
                        attributes_str = ';'.join(attributes)
                    line = f'[{node.highlight_index}]<{node.tag_name} '
                    if attributes_str:
                        line += f'{attributes_str}'
                    if text:
                        if attributes_str:
                            line += f'>{text}'
                        else:
                            line += f'{text}'
                    line += '/>'
                    formatted_text.append(line)

                # Process children regardless
                for child in node.children:
                    process_node(child, depth + 1)

            elif isinstance(node, DOMTextNode):
                # Add text only if it doesn't have a highlighted parent
                if not node.has_parent_with_highlight_index() and node.is_visible:
                    formatted_text.append(f'{node.text}')

        process_node(self, 0)
        return '\n'.join(formatted_text)


# Type alias for the selector map
SelectorMap = Dict[int, DOMElementNode]


@dataclass
class DOMState:
    """State of the DOM/UIA tree"""
    element_tree: DOMElementNode
    selector_map: SelectorMap


# Add history tree processor classes
class HashedDomElement(BaseModel):
    """Hash representation of a DOM element to track across states"""
    branch_path_hash: str
    attributes_hash: str
    xpath_hash: str

    model_config = ConfigDict(frozen=True)


class DOMHistoryElement(BaseModel):
    """Snapshot of a DOM element for history tracking"""
    tag_name: str
    xpath: str
    highlight_index: Optional[int]
    entire_parent_branch_path: list[str]
    attributes: dict[str, str]
    shadow_root: bool = False
    css_selector: Optional[str] = None
    page_coordinates: Optional[Any] = None
    viewport_coordinates: Optional[Any] = None
    viewport_info: Optional[Any] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            'tag_name': self.tag_name,
            'xpath': self.xpath,
            'highlight_index': self.highlight_index,
            'entire_parent_branch_path': self.entire_parent_branch_path,
            'attributes': self.attributes,
            'shadow_root': self.shadow_root,
            'css_selector': self.css_selector,
        }