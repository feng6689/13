import time
from typing import Dict, Callable, Optional

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

try:
    from pynput.keyboard import Controller, Key
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


class KeyboardController:
    def __init__(self):
        self.pyautogui_controller = None
        self.pynput_controller = None
        
        if PYAUTOGUI_AVAILABLE:
            pyautogui.FAILSAFE = False
        elif PYNPUT_AVAILABLE:
            self.pynput_controller = Controller()
        
        self.gesture_map: Dict[int, Callable] = {
            1: self._type_i,
            2: self._type_love,
            3: self._type_you,
        }
        
        self.last_gesture: Optional[int] = None
        self.last_time: float = 0
        self.cooldown: float = 1.5
        
    def _type_i(self) -> None:
        text = "I"
        if PYAUTOGUI_AVAILABLE:
            pyautogui.typewrite(text, interval=0.05)
        elif PYNPUT_AVAILABLE:
            for char in text:
                self.pynput_controller.type(char)
                time.sleep(0.05)
        print(f"[手势识别] 1根手指 -> 输入: {text}")
        
    def _type_love(self) -> None:
        text = "love"
        if PYAUTOGUI_AVAILABLE:
            pyautogui.typewrite(text, interval=0.05)
        elif PYNPUT_AVAILABLE:
            for char in text:
                self.pynput_controller.type(char)
                time.sleep(0.05)
        print(f"[手势识别] 2根手指 -> 输入: {text}")
        
    def _type_you(self) -> None:
        text = "you"
        if PYAUTOGUI_AVAILABLE:
            pyautogui.typewrite(text, interval=0.05)
        elif PYNPUT_AVAILABLE:
            for char in text:
                self.pynput_controller.type(char)
                time.sleep(0.05)
        print(f"[手势识别] 3根手指 -> 输入: {text}")
        
    def execute_gesture(self, finger_count: int) -> bool:
        if finger_count not in self.gesture_map:
            return False
            
        current_time = time.time()
        
        if (self.last_gesture == finger_count and 
            current_time - self.last_time < self.cooldown):
            return False
            
        self.gesture_map[finger_count]()
        self.last_gesture = finger_count
        self.last_time = current_time
        return True
        
    def get_available_library(self) -> str:
        if PYAUTOGUI_AVAILABLE:
            return "pyautogui"
        elif PYNPUT_AVAILABLE:
            return "pynput"
        else:
            return "none"
            
    def print_library_info(self) -> None:
        lib = self.get_available_library()
        if lib == "pyautogui":
            print("[键盘控制] 使用 pyautogui 库")
        elif lib == "pynput":
            print("[键盘控制] 使用 pynput 库")
        else:
            print("[警告] 未检测到 pyautogui 或 pynput 库")
            print("[提示] 请运行: pip install pyautogui 或 pip install pynput")
