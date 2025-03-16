from typing import Optional

from pydantic import BaseModel, model_validator


# Action Input Models
class ClickElementAction(BaseModel):
    index: int
    xpath: Optional[str] = None


class InputTextAction(BaseModel):
    index: int
    text: str
    xpath: Optional[str] = None


class DoneAction(BaseModel):
    text: str
    success: bool


class SwitchWindowAction(BaseModel):
    window_id: int


class SendKeysAction(BaseModel):
    keys: str


class LaunchApplicationAction(BaseModel):
    app_name: str
    arguments: Optional[str] = None


class RunProcessAction(BaseModel):
    command: str
    timeout: Optional[int] = None


class ScrollAction(BaseModel):
    direction: str  # 'up', 'down', 'left', 'right'
    amount: Optional[int] = None  # pixels to scroll, default to a page scroll if None


class RightClickAction(BaseModel):
    index: int
    xpath: Optional[str] = None


class DoubleClickAction(BaseModel):
    index: int
    xpath: Optional[str] = None


class ScreenshotAction(BaseModel):
    save_path: Optional[str] = None  # if provided, save to file


class WaitAction(BaseModel):
    seconds: int


class CloseWindowAction(BaseModel):
    window_id: Optional[int] = None  # if None, close active window


class SelectMenuItemAction(BaseModel):
    menu_path: str  # format: "File>Open" or "Edit>Preferences>Settings"


class NoParamsAction(BaseModel):
    """
    Accepts absolutely anything in the incoming data
    and discards it, so the final parsed model is empty.
    """

    @model_validator(mode='before')
    def ignore_all_inputs(cls, values):
        # No matter what the user sends, discard it and return empty.
        return {}

    class Config:
        # If you want to silently allow unknown fields at top-level,
        # set extra = 'allow' as well:
        extra = 'allow'