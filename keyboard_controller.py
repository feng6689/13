import time
from typing import Dict, Callable, Optional

PYAUTOGUI_AVAILABLE = False
PYNPUT_AVAILABLE = False

pyautogui = None
Controller = None
Key = None

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
    print("[键盘控制] pyautogui 库已加载")
except ImportError as e:
    print(f"[键盘控制] pyautogui 导入失败: {e}")

try:
    from pynput.keyboard import Controller, Key
    PYNPUT_AVAILABLE = True
    print("[键盘控制] pynput 库已加载")
except ImportError as e:
    print(f"[键盘控制] pynput 导入失败: {e}")


class KeyboardController:
    def __init__(self):
        self.pynput_controller = None
        
        if PYAUTOGUI_AVAILABLE:
            try:
                pyautogui.FAILSAFE = False
                pyautogui.PAUSE = 0.05
                print("[键盘控制] pyautogui 已初始化，FAILSAFE=False, PAUSE=0.05")
            except Exception as e:
                print(f"[键盘控制] pyautogui 初始化警告: {e}")
        elif PYNPUT_AVAILABLE and Controller:
            try:
                self.pynput_controller = Controller()
                print("[键盘控制] pynput 控制器已创建")
            except Exception as e:
                print(f"[键盘控制] pynput 初始化失败: {e}")
        
        self.gesture_map: Dict[int, Callable] = {
            1: self._type_i,
            2: self._type_love,
            3: self._type_you,
        }
        
        self.last_gesture: Optional[int] = None
        self.last_time: float = 0
        self.cooldown: float = 2.0
        
    def _type_with_pyautogui(self, text: str) -> bool:
        if not PYAUTOGUI_AVAILABLE or pyautogui is None:
            return False
            
        try:
            pyautogui.typewrite(text, interval=0.1)
            return True
        except Exception as e:
            print(f"[错误] pyautogui 输入失败: {e}")
            return False
    
    def _type_with_pynput(self, text: str) -> bool:
        if not PYNPUT_AVAILABLE or self.pynput_controller is None:
            return False
            
        try:
            for char in text:
                self.pynput_controller.type(char)
                time.sleep(0.1)
            return True
        except Exception as e:
            print(f"[错误] pynput 输入失败: {e}")
            return False
    
    def _type_text(self, text: str) -> bool:
        print(f"[手势识别] 准备输入文本: '{text}'")
        print(f"[提示] 请确保光标在可输入的位置（如记事本、聊天框等）")
        
        if PYAUTOGUI_AVAILABLE:
            print(f"[键盘控制] 使用 pyautogui 输入...")
            success = self._type_with_pyautogui(text)
            if success:
                print(f"[键盘控制] pyautogui 输入完成")
                return True
            else:
                print(f"[键盘控制] pyautogui 输入失败，尝试 pynput...")
        
        if PYNPUT_AVAILABLE:
            print(f"[键盘控制] 使用 pynput 输入...")
            success = self._type_with_pynput(text)
            if success:
                print(f"[键盘控制] pynput 输入完成")
                return True
        
        print(f"[警告] 没有可用的键盘输入库！")
        print(f"[提示] 请安装: pip install pyautogui 或 pip install pynput")
        return False
        
    def _type_i(self) -> None:
        text = "I"
        self._type_text(text)
        print(f"[手势识别] 1根手指 -> 已尝试输入: '{text}'")
        
    def _type_love(self) -> None:
        text = "love"
        self._type_text(text)
        print(f"[手势识别] 2根手指 -> 已尝试输入: '{text}'")
        
    def _type_you(self) -> None:
        text = "you"
        self._type_text(text)
        print(f"[手势识别] 3根手指 -> 已尝试输入: '{text}'")
        
    def execute_gesture(self, finger_count: int) -> bool:
        if finger_count not in self.gesture_map:
            return False
            
        current_time = time.time()
        
        if (self.last_gesture == finger_count and 
            current_time - self.last_time < self.cooldown):
            return False
            
        print(f"\n{'='*50}")
        print(f"[手势触发] 检测到 {finger_count} 根手指")
        
        self.gesture_map[finger_count]()
        self.last_gesture = finger_count
        self.last_time = current_time
        print(f"{'='*50}\n")
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
        print("\n" + "-"*50)
        print("键盘输入库状态:")
        print("-"*50)
        
        if lib == "pyautogui":
            print("[✓] pyautogui: 已安装并可用")
            print("    - 自动控制键盘输入")
            print("    - 支持中英文输入")
        elif lib == "pynput":
            print("[✓] pynput: 已安装并可用")
            print("    - 自动控制键盘输入")
        else:
            print("[✗] pyautogui: 未安装")
            print("[✗] pynput: 未安装")
            print("\n[重要] 没有检测到键盘输入库！")
            print("       手势识别后只会打印信息，不会真正输入文字。")
            print("\n请安装其中一个库:")
            print("  pip install pyautogui")
            print("  或")
            print("  pip install pynput")
        
        print("-"*50 + "\n")
