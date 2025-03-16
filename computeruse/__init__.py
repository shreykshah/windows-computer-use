from computeruse.logging_config import setup_logging

setup_logging()

from computeruse.agent.prompts import SystemPrompt as SystemPrompt
from computeruse.agent.service import Agent as Agent
from computeruse.controller.registry.views import ActionModel as ActionModel
from computeruse.agent.views import ActionResult as ActionResult
from computeruse.agent.views import AgentHistoryList as AgentHistoryList
from computeruse.uia.windows import Windows as Windows
from computeruse.uia.windows import WindowsConfig as WindowsConfig
from computeruse.uia.context import WindowsContextConfig
from computeruse.controller.service import Controller as Controller
from computeruse.dom.service import DomService as DomService

__all__ = [
    'Agent',
    'Windows',
    'WindowsConfig',
    'Controller',
    'DomService',
    'SystemPrompt',
    'ActionResult',
    'ActionModel',
    'AgentHistoryList',
    'WindowsContextConfig',
]