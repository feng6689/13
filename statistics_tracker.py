import time
from typing import Dict, List, Optional
from collections import defaultdict


class StatisticsTracker:
    def __init__(self):
        self.start_time: float = time.time()
        self.gesture_counts: Dict[int, int] = defaultdict(int)
        self.total_gestures: int = 0
        self.valid_gestures: List[int] = [1, 2, 3, 4, 5]
        
        self.gesture_names: Dict[int, str] = {
            1: "1根手指 (输入 'I')",
            2: "2根手指 (输入 'love')",
            3: "3根手指 (输入 'you')",
            4: "4根手指",
            5: "5根手指 (拳头/手掌)"
        }
        
    def record_gesture(self, finger_count: int) -> None:
        if finger_count in self.valid_gestures:
            self.gesture_counts[finger_count] += 1
            self.total_gestures += 1
            
    def get_gesture_count(self, finger_count: int) -> int:
        return self.gesture_counts.get(finger_count, 0)
    
    def get_elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    def get_average_per_second(self) -> float:
        elapsed = self.get_elapsed_time()
        if elapsed > 0 and self.total_gestures > 0:
            return self.total_gestures / elapsed
        return 0.0
    
    def print_statistics(self) -> None:
        elapsed = self.get_elapsed_time()
        avg_per_sec = self.get_average_per_second()
        
        print("\n" + "="*60)
        print("                    手势识别统计报告")
        print("="*60)
        print(f"\n程序运行时间: {elapsed:.2f} 秒")
        print(f"总识别手势次数: {self.total_gestures} 次")
        print(f"平均每秒识别: {avg_per_sec:.3f} 次/秒")
        print("\n" + "-"*60)
        print("各手势识别详情:")
        print("-"*60)
        
        for fingers in sorted(self.valid_gestures):
            count = self.gesture_counts.get(fingers, 0)
            if self.total_gestures > 0:
                percentage = (count / self.total_gestures) * 100
            else:
                percentage = 0.0
            name = self.gesture_names.get(fingers, f"{fingers}根手指")
            print(f"  {name}: {count} 次 ({percentage:.1f}%)")
        
        print("\n" + "="*60)
        print("                    统计报告结束")
        print("="*60 + "\n")
        
    def get_statistics_summary(self) -> Dict:
        return {
            "elapsed_time": self.get_elapsed_time(),
            "total_gestures": self.total_gestures,
            "average_per_second": self.get_average_per_second(),
            "gesture_counts": dict(self.gesture_counts)
        }
