import hashlib
from typing import List, Optional

from computeruse.dom.views import DOMElementNode, DOMHistoryElement, HashedDomElement


class HistoryTreeProcessor:
    """
    Provides operations on DOM elements to track elements across different states
    """

    @staticmethod
    def convert_dom_element_to_history_element(dom_element: DOMElementNode) -> DOMHistoryElement:
        """
        Convert a DOM element to a history element for tracking across states
        
        Args:
            dom_element: DOM element to convert
            
        Returns:
            History element representation
        """
        parent_branch_path = HistoryTreeProcessor._get_parent_branch_path(dom_element)
        return DOMHistoryElement(
            tag_name=dom_element.tag_name,
            xpath=dom_element.xpath,
            highlight_index=dom_element.highlight_index,
            entire_parent_branch_path=parent_branch_path,
            attributes=dom_element.attributes,
            shadow_root=dom_element.shadow_root,
            viewport_coordinates=dom_element.viewport_coordinates,
            page_coordinates=dom_element.page_coordinates,
            viewport_info=dom_element.viewport_info,
        )

    @staticmethod
    def find_history_element_in_tree(
        dom_history_element: DOMHistoryElement, 
        tree: DOMElementNode
    ) -> Optional[DOMElementNode]:
        """
        Find a history element in the current DOM tree
        
        Args:
            dom_history_element: History element to find
            tree: Current DOM tree to search in
            
        Returns:
            Matching DOM element if found, None otherwise
        """
        hashed_dom_history_element = HistoryTreeProcessor._hash_dom_history_element(dom_history_element)

        def process_node(node: DOMElementNode):
            if node.highlight_index is not None:
                hashed_node = HistoryTreeProcessor._hash_dom_element(node)
                if hashed_node == hashed_dom_history_element:
                    return node
            for child in node.children:
                if isinstance(child, DOMElementNode):
                    result = process_node(child)
                    if result is not None:
                        return result
            return None

        return process_node(tree)

    @staticmethod
    def compare_history_element_and_dom_element(
        dom_history_element: DOMHistoryElement, 
        dom_element: DOMElementNode
    ) -> bool:
        """
        Compare a history element and a DOM element
        
        Args:
            dom_history_element: History element
            dom_element: DOM element
            
        Returns:
            True if elements match, False otherwise
        """
        hashed_dom_history_element = HistoryTreeProcessor._hash_dom_history_element(dom_history_element)
        hashed_dom_element = HistoryTreeProcessor._hash_dom_element(dom_element)

        return hashed_dom_history_element == hashed_dom_element

    @staticmethod
    def _hash_dom_history_element(dom_history_element: DOMHistoryElement) -> HashedDomElement:
        """
        Generate a hash for a history element
        
        Args:
            dom_history_element: History element
            
        Returns:
            Hashed element
        """
        branch_path_hash = HistoryTreeProcessor._parent_branch_path_hash(dom_history_element.entire_parent_branch_path)
        attributes_hash = HistoryTreeProcessor._attributes_hash(dom_history_element.attributes)
        xpath_hash = HistoryTreeProcessor._xpath_hash(dom_history_element.xpath)

        return HashedDomElement(branch_path_hash=branch_path_hash, attributes_hash=attributes_hash, xpath_hash=xpath_hash)

    @staticmethod
    def _hash_dom_element(dom_element: DOMElementNode) -> HashedDomElement:
        """
        Generate a hash for a DOM element
        
        Args:
            dom_element: DOM element
            
        Returns:
            Hashed element
        """
        parent_branch_path = HistoryTreeProcessor._get_parent_branch_path(dom_element)
        branch_path_hash = HistoryTreeProcessor._parent_branch_path_hash(parent_branch_path)
        attributes_hash = HistoryTreeProcessor._attributes_hash(dom_element.attributes)
        xpath_hash = HistoryTreeProcessor._xpath_hash(dom_element.xpath)

        return HashedDomElement(branch_path_hash=branch_path_hash, attributes_hash=attributes_hash, xpath_hash=xpath_hash)

    @staticmethod
    def _get_parent_branch_path(dom_element: DOMElementNode) -> List[str]:
        """
        Get the path of tag names from root to element
        
        Args:
            dom_element: DOM element
            
        Returns:
            List of tag names
        """
        parents: List[DOMElementNode] = []
        current_element: DOMElementNode = dom_element
        while current_element.parent is not None:
            parents.append(current_element)
            current_element = current_element.parent

        parents.reverse()

        return [parent.tag_name for parent in parents]

    @staticmethod
    def _parent_branch_path_hash(parent_branch_path: List[str]) -> str:
        """
        Hash a parent branch path
        
        Args:
            parent_branch_path: Parent branch path
            
        Returns:
            Hash of the path
        """
        parent_branch_path_string = '/'.join(parent_branch_path)
        return hashlib.sha256(parent_branch_path_string.encode()).hexdigest()

    @staticmethod
    def _attributes_hash(attributes: dict[str, str]) -> str:
        """
        Hash element attributes
        
        Args:
            attributes: Element attributes
            
        Returns:
            Hash of attributes
        """
        attributes_string = ''.join(f'{key}={value}' for key, value in attributes.items())
        return hashlib.sha256(attributes_string.encode()).hexdigest()

    @staticmethod
    def _xpath_hash(xpath: str) -> str:
        """
        Hash an XPath
        
        Args:
            xpath: XPath string
            
        Returns:
            Hash of the XPath
        """
        return hashlib.sha256(xpath.encode()).hexdigest()