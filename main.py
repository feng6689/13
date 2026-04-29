import cv2
import numpy as np
import os

from hand_detection import HandDetector
from keyboard_controller import KeyboardController
from statistics_tracker import StatisticsTracker


class GestureRecognizerApp:
    def __init__(self):
        self.cap = None
        self.hand_detector = HandDetector()
        self.keyboard_controller = KeyboardController()
        self.statistics_tracker = StatisticsTracker()
        
        self.is_recognition_mode = False
        self.running = True
        
        self.window_name = "Gesture Recognizer"
        self.frame_width = 640
        self.frame_height = 480
        
        self.last_finger_count = 0
        self.finger_count_stability = 0
        self.stability_threshold = 5
        
    def initialize_camera(self, camera_index: int = 0) -> bool:
        print("[系统] 正在初始化摄像头...")
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            print("[错误] 无法打开摄像头!")
            return False
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        print(f"[系统] 摄像头已就绪，分辨率: {int(actual_width)}x{int(actual_height)}")
        return True
        
    def draw_mode_indicator(self, frame: np.ndarray) -> np.ndarray:
        output = frame.copy()
        height, width = output.shape[:2]
        
        if self.is_recognition_mode:
            mode_text = "识别模式 (按 's' 保存, 'q' 退出)"
            text_color = (0, 255, 0)
            bg_color = (0, 100, 0)
        else:
            mode_text = "单视频模式 (按 'e' 开始识别, 'q' 退出)"
            text_color = (255, 255, 255)
            bg_color = (50, 50, 50)
        
        text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 40
        
        cv2.rectangle(output, (text_x - 10, text_y - 30), 
                      (text_x + text_size[0] + 10, text_y + 10), 
                      bg_color, -1)
        cv2.putText(output, mode_text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        return output
    
    def draw_finger_count(self, frame: np.ndarray, finger_count: int) -> np.ndarray:
        output = frame.copy()
        height, width = output.shape[:2]
        
        if finger_count > 0:
            count_text = f"手指数量: {finger_count}"
            text_color = (0, 255, 255)
            
            text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 30
            
            cv2.rectangle(output, (text_x - 10, text_y - 40),
                          (text_x + text_size[0] + 10, text_y + 15),
                          (0, 0, 0), -1)
            cv2.putText(output, count_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 3)
            
            if finger_count == 1:
                hint_text = "即将输入: I"
            elif finger_count == 2:
                hint_text = "即将输入: love"
            elif finger_count == 3:
                hint_text = "即将输入: you"
            else:
                hint_text = ""
                
            if hint_text:
                hint_size = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                hint_x = (width - hint_size[0]) // 2
                hint_y = height - 70
                
                cv2.rectangle(output, (hint_x - 10, hint_y - 30),
                              (hint_x + hint_size[0] + 10, hint_y + 10),
                              (0, 0, 128), -1)
                cv2.putText(output, hint_text, (hint_x, hint_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output
    
    def save_frame(self, frame: np.ndarray) -> bool:
        try:
            filename = "gesture.jpg"
            filepath = os.path.join(os.getcwd(), filename)
            cv2.imwrite(filepath, frame)
            print(f"[系统] 帧已保存到: {filepath}")
            return True
        except Exception as e:
            print(f"[错误] 保存帧失败: {e}")
            return False
    
    def process_keyboard_input(self, key: int, frame: np.ndarray) -> None:
        if key == ord('e') or key == ord('E'):
            self.is_recognition_mode = not self.is_recognition_mode
            if self.is_recognition_mode:
                print("[系统] 已切换到: 手指识别模式")
            else:
                print("[系统] 已切换到: 单视频模式")
                
        elif key == ord('s') or key == ord('S'):
            self.save_frame(frame)
            
        elif key == ord('q') or key == ord('Q') or key == 27:
            print("[系统] 收到退出信号...")
            self.running = False
    
    def run(self) -> None:
        print("="*60)
        print("                手势识别系统")
        print("="*60)
        print("\n使用说明:")
        print("  - 'e' 键: 切换识别模式/单视频模式")
        print("  - 's' 键: 保存当前帧为 gesture.jpg")
        print("  - 'q' 键: 退出程序")
        print("\n手势映射:")
        print("  - 1根手指 -> 输入 'I'")
        print("  - 2根手指 -> 输入 'love'")
        print("  - 3根手指 -> 输入 'you'")
        print("="*60 + "\n")
        
        self.keyboard_controller.print_library_info()
        print()
        
        if not self.initialize_camera():
            return
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        print("[系统] 程序已启动，初始模式: 单视频模式")
        print("[提示] 请将手放在摄像头前，按 'e' 开始识别\n")
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("[错误] 无法读取帧!")
                break
            
            frame = cv2.flip(frame, 1)
            
            display_frame = frame.copy()
            
            if self.is_recognition_mode:
                (processed_frame, mask, contour, hull, 
                 finger_count, finger_tips) = self.hand_detector.process_frame(frame)
                
                if contour is not None:
                    display_frame = self.hand_detector.draw_visualizations(
                        display_frame, contour, finger_tips, finger_count
                    )
                    
                    if finger_count != self.last_finger_count:
                        self.last_finger_count = finger_count
                        self.finger_count_stability = 1
                    else:
                        self.finger_count_stability += 1
                    
                    if self.finger_count_stability >= self.stability_threshold:
                        if finger_count in [1, 2, 3]:
                            executed = self.keyboard_controller.execute_gesture(finger_count)
                            if executed:
                                self.statistics_tracker.record_gesture(finger_count)
                else:
                    self.finger_count_stability = 0
                    self.last_finger_count = 0
                
                display_frame = self.draw_finger_count(display_frame, finger_count)
                display_frame = self.draw_mode_indicator(display_frame)
            else:
                display_frame = self.draw_mode_indicator(display_frame)
            
            cv2.imshow(self.window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self.process_keyboard_input(key, display_frame)
        
        self.cleanup()
    
    def cleanup(self) -> None:
        print("\n[系统] 正在清理资源...")
        
        if self.cap is not None:
            self.cap.release()
            print("[系统] 摄像头已释放")
        
        cv2.destroyAllWindows()
        print("[系统] 窗口已关闭")
        
        self.statistics_tracker.print_statistics()
        
        print("[系统] 程序已安全退出")


if __name__ == "__main__":
    app = GestureRecognizerApp()
    app.run()
