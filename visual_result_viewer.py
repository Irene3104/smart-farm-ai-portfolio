#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import os
import time
import tempfile
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import colorsys

# 환경변수 로드
load_dotenv()

class VisualCattleDetector:
    def __init__(self):
        """시각화 소 감지기 초기화"""
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=os.getenv("ROBOFLOW_API_KEY", "tIxsJ20iZQPBw9gMLTkq")
        )
        # 최고 성능 모델
        self.primary_model = "yolov8m-coco/1"
        self.backup_models = ["yolov8s-coco/1", "microsoft-coco/9"]
    
    def detect_and_visualize(self, image_path):
        """감지하고 시각화"""
        
        print(f"🔍 Starting image analysis and visualization: {image_path}")
        
        # 이미지 로드
        original_image = cv2.imread(image_path)
        if original_image is None:
            print("❌ Unable to load image!")
            return None
        
        h, w = original_image.shape[:2]
        print(f"📏 Image size: {w}x{h}")
        
        all_detections = []
        
        try:
            # 1. 전체 이미지 분석
            print("📍 Analyzing full image...")
            full_detections = self._detect_in_region(image_path, 0, 0, "full")
            all_detections.extend(full_detections)
            
            # 2. 9등분 격자 분석 (가장 효과적)
            print("📍 Analyzing 9-grid sections...")
            for row in range(3):
                for col in range(3):
                    rx = col * (w // 3)
                    ry = row * (h // 3)
                    rw = w // 3
                    rh = h // 3
                    
                    # 겹침 처리
                    overlap = 20
                    roi_x1 = max(0, rx - overlap)
                    roi_y1 = max(0, ry - overlap)
                    roi_x2 = min(w, rx + rw + overlap)
                    roi_y2 = min(h, ry + rh + overlap)
                    
                    roi = original_image[roi_y1:roi_y2, roi_x1:roi_x2]
                    
                    temp_path = f"temp_visual_{row}_{col}.jpg"
                    cv2.imwrite(temp_path, roi)
                    grid_detections = self._detect_in_region(temp_path, roi_x1, roi_y1, f"grid{row+1}-{col+1}")
                    all_detections.extend(grid_detections)
                    try:
                        os.remove(temp_path)
                    except:
                        pass
            
            print(f"📊 Raw detections: {len(all_detections)} items")
            
            # 3. 스마트 중복 제거
            final_detections = self._smart_nms_with_regions(all_detections)
            print(f"🎯 Final detections: {len(final_detections)} cattle")
            
            # 4. 시각화
            result_image = self._create_visualization(original_image, final_detections)
            
            return result_image, final_detections
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return original_image, []
    
    def _detect_in_region(self, image_path, offset_x=0, offset_y=0, region_name=""):
        """특정 영역에서 소 감지"""
        detections = []
        
        # 주 모델 시도
        try:
            result = self.client.infer(image_path, model_id=self.primary_model)
            predictions = result.get("predictions", [])
            
            for pred in predictions:
                if pred.get('class', '').lower() == 'cow':
                    pred['x'] += offset_x
                    pred['y'] += offset_y
                    pred['region'] = region_name
                    detections.append(pred)
                    
        except Exception as e:
            # 백업 모델 시도
            for backup_model in self.backup_models:
                try:
                    result = self.client.infer(image_path, model_id=backup_model)
                    predictions = result.get("predictions", [])
                    
                    for pred in predictions:
                        if pred.get('class', '').lower() == 'cow':
                            pred['x'] += offset_x
                            pred['y'] += offset_y
                            pred['region'] = f"{region_name}(backup)"
                            detections.append(pred)
                    break
                except:
                    continue
        
        return detections
    
    def _smart_nms_with_regions(self, detections):
        """스마트 NMS"""
        if not detections:
            return []
        
        def calculate_iou(box1, box2):
            x1, y1, w1, h1 = box1['x'], box1['y'], box1['width'], box1['height']
            x2, y2, w2, h2 = box2['x'], box2['y'], box2['width'], box2['height']
            
            x1_min, y1_min = x1 - w1/2, y1 - h1/2
            x1_max, y1_max = x1 + w1/2, y1 + h1/2
            x2_min, y2_min = x2 - w2/2, y2 - h2/2
            x2_max, y2_max = x2 + w2/2, y2 + h2/2
            
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0
            
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            box1_area = w1 * h1
            box2_area = w2 * h2
            union_area = box1_area + box2_area - inter_area
            
            return inter_area / union_area if union_area > 0 else 0
        
        # 신뢰도 순 정렬
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        final_detections = []
        for detection in detections:
            is_duplicate = False
            
            for existing in final_detections:
                iou = calculate_iou(detection, existing)
                
                # 동적 IoU 임계값
                if detection.get('region', '') == existing.get('region', ''):
                    threshold = 0.3
                else:
                    threshold = 0.15
                
                if iou > threshold:
                    is_duplicate = True
                    if (detection['confidence'] > existing['confidence'] * 1.2 or
                        'full' in detection.get('region', '') and 'grid' in existing.get('region', '')):
                        final_detections.remove(existing)
                        final_detections.append(detection)
                    break
            
            if not is_duplicate:
                final_detections.append(detection)
        
        return final_detections
    
    def _estimate_behavior(self, detection):
        """행동 추정"""
        aspect_ratio = detection['width'] / detection['height']
        
        if aspect_ratio > 2.5:
            return "lying"
        elif aspect_ratio > 1.8:
            return "sitting"
        elif detection['y'] > 400:  # 이미지 하단
            return "eating"
        else:
            return "standing"
    
    def _create_visualization(self, original_image, detections):
        """시각화 이미지 생성"""
        result_image = original_image.copy()
        h, w = result_image.shape[:2]
        
        # 배경 어둡게
        overlay = result_image.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        result_image = cv2.addWeighted(result_image, 0.7, overlay, 0.3, 0)
        
        # 색상 생성
        colors = []
        for i in range(len(detections)):
            hue = i / max(len(detections), 1)
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
            color = tuple(int(c * 255) for c in rgb[::-1])  # BGR 순서
            colors.append(color)
        
        # 작은 헤더 정보 (우측 상단)
        header_w, header_h = 250, 60
        header_x = w - header_w - 10
        cv2.rectangle(result_image, (header_x, 10), (header_x + header_w, 10 + header_h), (0, 0, 0), -1)
        cv2.putText(result_image, "Cow Detection", (header_x + 5, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result_image, f"Found: {len(detections)} cows", (header_x + 5, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 감지된 소들 그리기
        for i, detection in enumerate(detections):
            x1 = int(detection['x'] - detection['width']/2)
            y1 = int(detection['y'] - detection['height']/2)
            x2 = int(detection['x'] + detection['width']/2)
            y2 = int(detection['y'] + detection['height']/2)
            
            color = colors[i]
            
            # 바운딩 박스
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
            
            # 중심점과 번호
            center_x, center_y = int(detection['x']), int(detection['y'])
            cv2.circle(result_image, (center_x, center_y), 15, color, -1)
            cv2.putText(result_image, str(i+1), (center_x-8, center_y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Simple label (English)
            behavior_map = {
                "lying": "Lying",
                "sitting": "Sitting", 
                "eating": "Eating",
                "standing": "Standing"
            }
            behavior = self._estimate_behavior(detection)
            behavior_en = behavior_map.get(behavior, "Unknown")
            confidence = detection['confidence']
            
            # 작은 라벨
            label = f"#{i+1} {behavior_en}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(result_image, (x1, y1-20), (x1+label_size[0]+5, y1), color, -1)
            cv2.putText(result_image, label, (x1+2, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Small statistics info (bottom right)
        behavior_map = {"lying": "Lying", "sitting": "Sitting", "eating": "Eating", "standing": "Standing"}
        behaviors = [self._estimate_behavior(d) for d in detections]
        behavior_counts = {}
        for behavior in behaviors:
            behavior_en = behavior_map.get(behavior, behavior)
            behavior_counts[behavior_en] = behavior_counts.get(behavior_en, 0) + 1
        
        stats_y = h - 100
        stats_w = 180
        cv2.rectangle(result_image, (w-stats_w-10, stats_y-20), (w-10, h-10), (0, 0, 0), -1)
        cv2.putText(result_image, "Behaviors", (w-stats_w-5, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        y_offset = 20
        for behavior_en, count in behavior_counts.items():
            cv2.putText(result_image, f"{behavior_en}: {count}", 
                       (w-stats_w-5, stats_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15
        
        return result_image

def main():
    print("🐄 Cattle Detection Visualization Program")
    print("=" * 50)
    
    detector = VisualCattleDetector()
    image_path = "images/cow_test.jpg"
    
    print("🔄 Detecting and visualizing...")
    start_time = time.time()
    
    result_image, detections = detector.detect_and_visualize(image_path)
    
    end_time = time.time()
    print(f"⏱️ Processing time: {end_time - start_time:.2f} seconds")
    
    if result_image is not None:
        # 결과 저장
        output_path = "cattle_detection_visualization.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"💾 Result saved: {output_path}")
        
        # 화면에 표시
        print(f"\n🎉 {len(detections)} cattle detected successfully!")
        print("🖼️ Visualization window opening...")
        print("📝 Controls:")
        print("  - ESC: Exit program")
        print("  - SPACE: View next")
        print("  - Any key: Close window")
        
        # 창 크기 조정 (큰 이미지인 경우)
        h, w = result_image.shape[:2]
        if w > 1200 or h > 800:
            scale = min(1200/w, 800/h)
            new_w, new_h = int(w*scale), int(h*scale)
            result_image = cv2.resize(result_image, (new_w, new_h))
        
        cv2.imshow('Cow Detection Result - 16 cows found!', result_image)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                # 원본과 비교 보기
                original = cv2.imread(image_path)
                if original is not None:
                    h_orig, w_orig = original.shape[:2]
                    if w_orig > 1200 or h_orig > 800:
                        scale = min(1200/w_orig, 800/h_orig)
                        new_w, new_h = int(w_orig*scale), int(h_orig*scale)
                        original = cv2.resize(original, (new_w, new_h))
                    cv2.imshow('Original Image', original)
                    cv2.waitKey(0)
                    cv2.destroyWindow('Original Image')
            else:
                break
        
        cv2.destroyAllWindows()
        print("✅ Program terminated")
    else:
        print("❌ Visualization failed")

if __name__ == "__main__":
    main()
