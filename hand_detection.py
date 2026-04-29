import cv2
import numpy as np
from typing import Tuple, List, Optional

class HandDetector:
    def __init__(self):
        self.skin_lower_hsv = np.array([0, 40, 80], dtype=np.uint8)
        self.skin_upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
        
        self.skin_lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        self.skin_upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        
        self.kernel = np.ones((5, 5), np.uint8)
        self.small_kernel = np.ones((3, 3), np.uint8)
        
    def skin_detection_hsv(self, hsv_frame: np.ndarray) -> np.ndarray:
        h, s, v = cv2.split(hsv_frame)
        
        v_mean = np.mean(v)
        
        lower = self.skin_lower_hsv.copy()
        upper = self.skin_upper_hsv.copy()
        
        if v_mean < 100:
            lower[2] = max(50, lower[2] - 20)
            lower[1] = max(30, lower[1] - 10)
        elif v_mean > 180:
            upper[1] = min(255, upper[1] + 20)
            lower[2] = min(255, lower[2] + 20)
        
        mask1 = cv2.inRange(hsv_frame, lower, upper)
        
        lower2 = np.array([160, 40, 80], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv_frame, lower2, upper2)
        
        mask = cv2.bitwise_or(mask1, mask2)
        return mask
    
    def skin_detection_ycrcb(self, frame: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.skin_lower_ycrcb, self.skin_upper_ycrcb)
        return mask
    
    def combined_skin_detection(self, frame: np.ndarray, hsv_frame: np.ndarray) -> np.ndarray:
        mask_hsv = self.skin_detection_hsv(hsv_frame)
        mask_ycrcb = self.skin_detection_ycrcb(frame)
        
        mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        
        return mask
    
    def preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.small_kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.small_kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        return mask
    
    def filter_contour_by_shape(self, contour: np.ndarray, frame_area: float) -> bool:
        if contour is None:
            return False
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        if aspect_ratio > 3.0 or aspect_ratio < 0.3:
            return False
        
        area = cv2.contourArea(contour)
        if area < 5000 or area > frame_area * 0.7:
            return False
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area > 0:
            solidity = float(area) / hull_area
        else:
            solidity = 0
        
        if solidity < 0.3 or solidity > 0.95:
            return False
        
        return True
    
    def find_hand_contour(self, mask: np.ndarray, frame_area: float) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        valid_contours = []
        
        for contour in contours:
            if self.filter_contour_by_shape(contour, frame_area):
                area = cv2.contourArea(contour)
                valid_contours.append((area, contour))
        
        if not valid_contours:
            return None
        
        valid_contours.sort(key=lambda x: x[0], reverse=True)
        
        return valid_contours[0][1]
    
    def approx_contour(self, contour: np.ndarray) -> np.ndarray:
        epsilon = 0.015 * cv2.arcLength(contour, True)
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
        finger_count = 0
        finger_tips = []
        
        if defects is None or len(defects) == 0:
            return 0, []
        
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0, []
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        hull = cv2.convexHull(contour)
        if len(hull) < 3:
            return 0, []
        
        y_coords = [point[0][1] for point in hull]
        if not y_coords:
            return 0, []
            
        min_y = min(y_coords)
        top_points = [point[0] for point in hull if point[0][1] <= min_y + 40]
        
        if top_points:
            avg_top_x = int(np.mean([p[0] for p in top_points]))
            avg_top_y = int(np.mean([p[1] for p in top_points]))
            
            dist_to_center = np.sqrt((avg_top_x - cx) ** 2 + (avg_top_y - cy) ** 2)
            if dist_to_center > 40:
                finger_count += 1
                finger_tips.append((avg_top_x, avg_top_y))
        
        valid_defects = []
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            
            if d < 8000:
                continue
                
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c + 1e-6))
            angle_deg = np.degrees(angle)
            
            if angle_deg < 100:
                valid_defects.append({
                    'start': start,
                    'end': end,
                    'far': far,
                    'angle': angle_deg
                })
                
                if start[1] < cy:
                    finger_count += 1
                    finger_tips.append(start)
                    finger_tips.append(end)
        
        finger_tips = self._filter_duplicate_points(finger_tips, 60)
        
        if len(valid_defects) == 0 and finger_count == 1:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            if aspect_ratio < 0.8 and len(finger_tips) <= 1:
                pass
        
        finger_count = len(finger_tips)
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
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 3)
            
        if hull is not None and len(hull) > 0:
            cv2.drawContours(output, [hull], -1, (255, 0, 0), 2)
        
        for tip in finger_tips:
            cv2.circle(output, tip, 12, (0, 0, 255), -1)
            cv2.circle(output, tip, 6, (255, 255, 255), -1)
        
        if contour is not None:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(output, (cx, cy), 8, (255, 0, 255), -1)
        
        return output
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], 
                                                         Optional[np.ndarray], Optional[np.ndarray],
                                                         int, List[Tuple[int, int]]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = self.combined_skin_detection(frame, hsv)
        mask = self.preprocess_mask(mask)
        
        frame_area = frame.shape[0] * frame.shape[1]
        contour = self.find_hand_contour(mask, frame_area)
        
        if contour is None:
            return frame, mask, None, None, 0, []
        
        approx = self.approx_contour(contour)
        hull_points, hull_indices = self.get_convex_hull(contour)
        defects = self.get_convexity_defects(contour, hull_indices)
        
        finger_count, finger_tips = self.count_fingers(contour, defects, frame.shape[0])
        
        return frame, mask, contour, hull_points, finger_count, finger_tips
