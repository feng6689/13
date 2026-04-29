import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict

class HandDetector:
    def __init__(self):
        self.skin_lower_hsv1 = np.array([0, 50, 70], dtype=np.uint8)
        self.skin_upper_hsv1 = np.array([20, 255, 255], dtype=np.uint8)
        self.skin_lower_hsv2 = np.array([160, 50, 70], dtype=np.uint8)
        self.skin_upper_hsv2 = np.array([180, 255, 255], dtype=np.uint8)
        
        self.skin_lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        self.skin_upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        
        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)
        self.kernel_large = np.ones((7, 7), np.uint8)
        
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        
        self.prev_frame_gray = None
        self.frame_count = 0
        
        self.roi_x_ratio = 0.2
        self.roi_y_ratio = 0.15
        self.roi_w_ratio = 0.6
        self.roi_h_ratio = 0.7
        
        self.history_finger_counts: List[int] = []
        self.max_history = 10
        
    def get_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        roi_x = int(w * self.roi_x_ratio)
        roi_y = int(h * self.roi_y_ratio)
        roi_w = int(w * self.roi_w_ratio)
        roi_h = int(h * self.roi_h_ratio)
        
        mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = 255
        return mask
    
    def draw_roi_rectangle(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        roi_x = int(w * self.roi_x_ratio)
        roi_y = int(h * self.roi_y_ratio)
        roi_w = int(w * self.roi_w_ratio)
        roi_h = int(h * self.roi_h_ratio)
        
        output = frame.copy()
        cv2.rectangle(output, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 255), 2)
        cv2.putText(output, "Place your hand inside", (roi_x, roi_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return output
    
    def skin_detection_hsv(self, hsv_frame: np.ndarray) -> np.ndarray:
        mask1 = cv2.inRange(hsv_frame, self.skin_lower_hsv1, self.skin_upper_hsv1)
        mask2 = cv2.inRange(hsv_frame, self.skin_lower_hsv2, self.skin_upper_hsv2)
        mask = cv2.bitwise_or(mask1, mask2)
        return mask
    
    def skin_detection_ycrcb(self, frame: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.skin_lower_ycrcb, self.skin_upper_ycrcb)
        return mask
    
    def background_subtraction(self, frame: np.ndarray) -> np.ndarray:
        fg_mask = self.background_subtractor.apply(frame)
        
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        return fg_mask
    
    def motion_detection(self, frame_gray: np.ndarray) -> np.ndarray:
        motion_mask = np.zeros(frame_gray.shape, dtype=np.uint8)
        
        if self.prev_frame_gray is not None:
            frame_diff = cv2.absdiff(self.prev_frame_gray, frame_gray)
            
            _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            
            motion_mask = cv2.dilate(motion_mask, self.kernel_medium, iterations=2)
        
        self.prev_frame_gray = frame_gray.copy()
        
        return motion_mask
    
    def combined_skin_detection(self, frame: np.ndarray, hsv_frame: np.ndarray) -> np.ndarray:
        mask_hsv = self.skin_detection_hsv(hsv_frame)
        mask_ycrcb = self.skin_detection_ycrcb(frame)
        
        mask_skin = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        
        return mask_skin
    
    def preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_small, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_large, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel_medium, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask
    
    def calculate_hand_features(self, contour: np.ndarray, frame_area: float) -> Dict:
        features = {}
        
        x, y, w, h = cv2.boundingRect(contour)
        features['x'] = x
        features['y'] = y
        features['width'] = w
        features['height'] = h
        features['aspect_ratio'] = float(w) / h if h > 0 else 0
        
        area = cv2.contourArea(contour)
        features['area'] = area
        features['area_ratio'] = area / frame_area if frame_area > 0 else 0
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        features['solidity'] = float(area) / hull_area if hull_area > 0 else 0
        
        perimeter = cv2.arcLength(contour, True)
        features['perimeter'] = perimeter
        
        if area > 0:
            features['compactness'] = (perimeter ** 2) / (4 * np.pi * area)
        else:
            features['compactness'] = 0
        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            features['center'] = (cx, cy)
        else:
            features['center'] = (0, 0)
        
        try:
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            if len(hull_indices) >= 3:
                defects = cv2.convexityDefects(contour, hull_indices)
                if defects is not None:
                    features['defect_count'] = len(defects)
                else:
                    features['defect_count'] = 0
            else:
                features['defect_count'] = 0
        except:
            features['defect_count'] = 0
        
        return features
    
    def is_valid_hand_contour(self, features: Dict, frame_area: float, frame_shape: Tuple) -> bool:
        h, w = frame_shape[:2]
        
        if features['area'] < 8000 or features['area'] > frame_area * 0.6:
            return False
        
        if features['aspect_ratio'] < 0.4 or features['aspect_ratio'] > 2.5:
            return False
        
        if features['solidity'] < 0.35 or features['solidity'] > 0.92:
            return False
        
        if features['defect_count'] < 0 or features['defect_count'] > 10:
            return False
        
        roi_x = int(w * self.roi_x_ratio)
        roi_y = int(h * self.roi_y_ratio)
        roi_w = int(w * self.roi_w_ratio)
        roi_h = int(h * self.roi_h_ratio)
        
        cx, cy = features['center']
        if cx < roi_x or cx > roi_x + roi_w or cy < roi_y or cy > roi_y + roi_h:
            return False
        
        return True
    
    def find_hand_contour(self, mask: np.ndarray, frame_area: float, frame_shape: Tuple) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 8000:
                continue
            
            features = self.calculate_hand_features(contour, frame_area)
            
            if self.is_valid_hand_contour(features, frame_area, frame_shape):
                valid_contours.append((area, contour, features))
        
        if not valid_contours:
            return None
        
        valid_contours.sort(key=lambda x: x[0], reverse=True)
        
        return valid_contours[0][1]
    
    def count_fingers(self, contour: np.ndarray, frame_shape: Tuple) -> Tuple[int, List[Tuple[int, int]]]:
        finger_count = 0
        finger_tips = []
        
        if contour is None or len(contour) < 3:
            return 0, []
        
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0, []
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        hull = cv2.convexHull(contour)
        if len(hull) < 3:
            return 0, []
        
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        
        try:
            defects = cv2.convexityDefects(contour, hull_indices)
        except:
            defects = None
        
        finger_count = 0
        finger_tips = []
        
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                
                if d < 10000:
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
                    if start[1] < cy + 20:
                        finger_tips.append(start)
                    if end[1] < cy + 20:
                        finger_tips.append(end)
        
        y_coords = [point[0][1] for point in hull]
        if y_coords:
            min_y = min(y_coords)
            top_points = [point[0] for point in hull if point[0][1] <= min_y + 30]
            
            if top_points:
                avg_top_x = int(np.mean([p[0] for p in top_points]))
                avg_top_y = int(np.mean([p[1] for p in top_points]))
                
                dist_to_center = np.sqrt((avg_top_x - cx) ** 2 + (avg_top_y - cy) ** 2)
                if dist_to_center > 30:
                    finger_tips.append((avg_top_x, avg_top_y))
        
        finger_tips = self._filter_duplicate_points(finger_tips, 70)
        
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
    
    def smooth_finger_count(self, finger_count: int) -> int:
        self.history_finger_counts.append(finger_count)
        
        if len(self.history_finger_counts) > self.max_history:
            self.history_finger_counts.pop(0)
        
        if len(self.history_finger_counts) < self.max_history:
            return finger_count
        
        from collections import Counter
        count_counter = Counter(self.history_finger_counts)
        most_common = count_counter.most_common(1)
        
        if most_common:
            return most_common[0][0]
        return finger_count
    
    def draw_visualizations(self, frame: np.ndarray, contour: np.ndarray,
                            finger_tips: List[Tuple[int, int]],
                            finger_count: int) -> np.ndarray:
        output = frame.copy()
        
        output = self.draw_roi_rectangle(output)
        
        if contour is not None:
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 3)
            
            hull = cv2.convexHull(contour)
            cv2.drawContours(output, [hull], -1, (255, 0, 0), 2)
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(output, (cx, cy), 8, (255, 0, 255), -1)
        
        for tip in finger_tips:
            cv2.circle(output, tip, 12, (0, 0, 255), -1)
            cv2.circle(output, tip, 6, (255, 255, 255), -1)
        
        return output
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], 
                                                         Optional[np.ndarray], Optional[np.ndarray],
                                                         int, List[Tuple[int, int]]]:
        self.frame_count += 1
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        mask_skin = self.combined_skin_detection(frame, hsv)
        
        mask_bg = self.background_subtraction(frame)
        
        mask_motion = self.motion_detection(gray)
        
        mask_roi = self.get_roi_mask(frame)
        
        mask = cv2.bitwise_and(mask_skin, mask_roi)
        
        if self.frame_count > 10:
            mask_bg_motion = cv2.bitwise_or(mask_bg, mask_motion)
            mask = cv2.bitwise_and(mask, mask_bg_motion)
        
        mask = self.preprocess_mask(mask)
        
        frame_area = frame.shape[0] * frame.shape[1]
        contour = self.find_hand_contour(mask, frame_area, frame.shape)
        
        if contour is None:
            self.history_finger_counts = []
            return frame, mask, None, None, 0, []
        
        finger_count, finger_tips = self.count_fingers(contour, frame.shape)
        
        smoothed_count = self.smooth_finger_count(finger_count)
        
        return frame, mask, contour, None, smoothed_count, finger_tips
