import cv2
import numpy as np
from typing import Tuple, List, Optional

class HandDetector:
    def __init__(self):
        self.skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        self.kernel = np.ones((3, 3), np.uint8)
        
    def adaptive_skin_detection(self, hsv_frame: np.ndarray) -> np.ndarray:
        h, s, v = cv2.split(hsv_frame)
        
        v_mean = np.mean(v)
        v_std = np.std(v)
        
        adaptive_lower = self.skin_lower.copy()
        adaptive_upper = self.skin_upper.copy()
        
        if v_mean < 100:
            adaptive_lower[2] = max(40, adaptive_lower[2] - 30)
            adaptive_upper[2] = min(255, adaptive_upper[2] + 30)
        elif v_mean > 180:
            adaptive_lower[2] = min(255, adaptive_lower[2] + 20)
            adaptive_upper[1] = min(255, adaptive_upper[1] - 20)
            
        if v_std > 60:
            adaptive_lower[2] = max(30, adaptive_lower[2] - 20)
            adaptive_upper[2] = min(255, adaptive_upper[2] + 20)
            
        mask = cv2.inRange(hsv_frame, adaptive_lower, adaptive_upper)
        return mask
    
    def preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return mask
    
    def find_hand_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        max_area = 0
        max_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area and area > 2000:
                max_area = area
                max_contour = contour
                
        return max_contour
    
    def approx_contour(self, contour: np.ndarray) -> np.ndarray:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx
    
    def get_convex_hull(self, contour: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hull_points = cv2.convexHull(contour)
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        return hull_points, hull_indices
    
    def get_convexity_defects(self, contour: np.ndarray, hull_indices: np.ndarray) -> Optional[np.ndarray]:
        if len(hull_indices) < 3:
            return None
        
        try:
            defects = cv2.convexityDefects(contour, hull_indices)
            return defects
        except:
            return None
    
    def count_fingers(self, contour: np.ndarray, defects: Optional[np.ndarray],
                       frame_height: int) -> Tuple[int, List[Tuple[int, int]]]:
        if defects is None or len(defects) == 0:
            return 0, []
        
        finger_count = 0
        finger_tips = []
        
        hull = cv2.convexHull(contour)
        if len(hull) < 3:
            return 0, []
        
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0, []
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        y_coords = [point[0][1] for point in hull]
        min_y = min(y_coords)
        top_points = [point[0] for point in hull if point[0][1] <= min_y + 30]
        
        if top_points:
            avg_top_x = int(np.mean([p[0] for p in top_points]))
            avg_top_y = int(np.mean([p[1] for p in top_points]))
            
            dist_to_center = np.sqrt((avg_top_x - cx) ** 2 + (avg_top_y - cy) ** 2)
            if dist_to_center > 30:
                finger_count += 1
                finger_tips.append((avg_top_x, avg_top_y))
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            
            if d < 5000:
                continue
                
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c + 1e-6))
            angle_deg = np.degrees(angle)
            
            if angle_deg < 120:
                finger_count += 1
                
                if start[1] < cy:
                    finger_tips.append(start)
                if end[1] < cy:
                    finger_tips.append(end)
        
        finger_tips = self._filter_duplicate_points(finger_tips, 50)
        
        finger_count = min(max(finger_count, 0), 5)
        
        return finger_count, finger_tips
    
    def _filter_duplicate_points(self, points: List[Tuple[int, int]], 
                                  min_distance: int) -> List[Tuple[int, int]]:
        if not points:
            return []
            
        filtered = []
        for point in points:
            duplicate = False
            for existing in filtered:
                dist = np.sqrt((point[0] - existing[0]) ** 2 + (point[1] - existing[1]) ** 2)
                if dist < min_distance:
                    duplicate = True
                    break
            if not duplicate:
                filtered.append(point)
                
        return filtered
    
    def draw_visualizations(self, frame: np.ndarray, contour: np.ndarray,
                            hull: np.ndarray, finger_tips: List[Tuple[int, int]],
                            finger_count: int) -> np.ndarray:
        output = frame.copy()
        
        if contour is not None:
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
            
        if hull is not None and len(hull) > 0:
            cv2.drawContours(output, [hull], -1, (255, 0, 0), 2)
        
        for tip in finger_tips:
            cv2.circle(output, tip, 10, (0, 0, 255), -1)
            cv2.circle(output, tip, 5, (255, 255, 255), -1)
        
        if contour is not None:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        return output
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], 
                                                         Optional[np.ndarray], Optional[np.ndarray],
                                                         int, List[Tuple[int, int]]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = self.adaptive_skin_detection(hsv)
        mask = self.preprocess_mask(mask)
        
        contour = self.find_hand_contour(mask)
        
        if contour is None:
            return frame, mask, None, None, 0, []
        
        approx = self.approx_contour(contour)
        hull_points, hull_indices = self.get_convex_hull(contour)
        defects = self.get_convexity_defects(contour, hull_indices)
        
        finger_count, finger_tips = self.count_fingers(contour, defects, frame.shape[0])
        
        return frame, mask, contour, hull_points, finger_count, finger_tips
