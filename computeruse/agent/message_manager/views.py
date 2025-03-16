from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from langchain_core.load import dumpd, load
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

if TYPE_CHECKING:
    from computeruse.agent.views import AgentOutput


class MessageMetadata(BaseModel):
    """Metadata for a message"""
    tokens: int = 0


class ManagedMessage(BaseModel):
    """A message with its metadata"""
    message: BaseMessage
    metadata: MessageMetadata = Field(default_factory=MessageMetadata)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # https://github.com/pydantic/pydantic/discussions/7558
    @model_serializer(mode='wrap')
    def to_json(self, original_dump):
        """
        Returns the JSON representation of the model.

        It uses langchain's `dumps` function to serialize the `message`
        property before encoding the overall dict with json.dumps.
        """
        data = original_dump(self)

        # NOTE: We override the message field to use langchain JSON serialization.
        data['message'] = dumpd(self.message)

        return data

    @model_validator(mode='before')
    @classmethod
    def validate(
        cls,
        value: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
    ) -> Any:
        """
        Custom validator that uses langchain's `loads` function
        to parse the message if it is provided as a JSON string.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            if isinstance(value, dict) and 'message' in value:
                # NOTE: We use langchain's load to convert the JSON string back into a BaseMessage object.
                msg_value = value['message']
                if isinstance(msg_value, (dict, str)):
                    try:
                        value['message'] = load(msg_value)
                    except Exception as load_err:
                        logger.error(f"Error loading message with langchain load: {load_err}")
                        # Handle dict conversion directly for common message types
                        if isinstance(msg_value, dict) and 'type' in msg_value:
                            msg_type = msg_value.get('type')
                            content = msg_value.get('content', '')
                            if msg_type == 'human':
                                value['message'] = HumanMessage(content=content)
                            elif msg_type == 'ai':
                                value['message'] = AIMessage(content=content)
                            elif msg_type == 'system':
                                value['message'] = SystemMessage(content=content)
                            elif msg_type == 'tool':
                                value['message'] = ToolMessage(
                                    content=content, 
                                    tool_call_id=msg_value.get('tool_call_id', '1')
                                )
                            else:
                                value['message'] = HumanMessage(content=f"[Conversion error for {msg_type} message]")
                        else:
                            value['message'] = HumanMessage(content="[Error loading serialized message]")
                # If it's already a BaseMessage object, no need to load it
                elif isinstance(msg_value, BaseMessage):
                    pass
                else:
                    logger.warning(f"Unexpected message type: {type(msg_value)}")
                    value['message'] = HumanMessage(content=f"[Error: unexpected message type {type(msg_value)}]")
        except Exception as e:
            logger.error(f"Error in message validation: {e}")
            # If message loading fails, create a fresh message to avoid breaking the agent
            if isinstance(value, dict):
                value['message'] = HumanMessage(content="[Error loading previous message]")
        return value


class MessageHistory(BaseModel):
    """History of messages with metadata"""
    messages: list[ManagedMessage] = Field(default_factory=list)
    current_tokens: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_message(self, message: BaseMessage, metadata: MessageMetadata, position: int | None = None) -> None:
        """Add message with metadata to history"""
        if position is None:
            self.messages.append(ManagedMessage(message=message, metadata=metadata))
        else:
            self.messages.insert(position, ManagedMessage(message=message, metadata=metadata))
        self.current_tokens += metadata.tokens

    def add_model_output(self, output: 'AgentOutput') -> None:
        """Add model output as AI message"""
        tool_calls = [
            {
                'name': 'AgentOutput',
                'args': output.model_dump(mode='json', exclude_unset=True),
                'id': '1',
                'type': 'tool_call',
            }
        ]

        msg = AIMessage(
            content='',
            tool_calls=tool_calls,
        )
        self.add_message(msg, MessageMetadata(tokens=100))  # Estimate tokens for tool calls

        # Empty tool response
        tool_message = ToolMessage(content='', tool_call_id='1')
        self.add_message(tool_message, MessageMetadata(tokens=10))  # Estimate tokens for empty response

    def get_messages(self) -> list[BaseMessage]:
        """Get all messages"""
        return [m.message for m in self.messages]

    def get_total_tokens(self) -> int:
        """Get total tokens in history"""
        return self.current_tokens

    def remove_oldest_message(self) -> None:
        """Remove oldest non-system message"""
        for i, msg in enumerate(self.messages):
            if not isinstance(msg.message, SystemMessage):
                self.current_tokens -= msg.metadata.tokens
                self.messages.pop(i)
                break

    def remove_last_state_message(self) -> None:
        """Remove last state message from history"""
        if len(self.messages) > 2 and isinstance(self.messages[-1].message, HumanMessage):
            self.current_tokens -= self.messages[-1].metadata.tokens
            self.messages.pop()


class MessageManagerState(BaseModel):
    """Holds the state for MessageManager"""
    history: MessageHistory = Field(default_factory=MessageHistory)
    tool_id: int = 1

    model_config = ConfigDict(arbitrary_types_allowed=True)