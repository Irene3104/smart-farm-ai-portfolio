#!/usr/bin/env python3
import requests
import base64
import json
import time

def test_cam1_frame():
    print("🎯 API를 통한 cam1 프레임 상세 분석")
    print("=" * 50)
    
    # 이미지 읽기 및 Base64 인코딩
    with open('cam1_frame.jpg', 'rb') as img_file:
        img_data = img_file.read()
        b64_image = base64.b64encode(img_data).decode()
    
    # API 요청
    try:
        response = requests.post(
            'http://localhost:5000/detect/image', 
            json={'image': b64_image},
            timeout=60
        )
        
        result = response.json()
        
        print(f"📊 감지 결과:")
        print(f"  - 총 소: {result['total_cows']}마리")
        print(f"  - 알람: {len(result['alarms'])}개")
        print(f"  - 정상: {result['summary']['normal_count']}마리") 
        print(f"  - 이상: {result['summary']['abnormal_count']}마리")
        print(f"  - 처리시간: {time.strftime('%H:%M:%S', time.localtime(result['timestamp']))}")
        
        print(f"\n🐄 개별 소 정보:")
        for cow in result['cows']:
            print(f"  소 #{cow['id']}: {cow['behavior']} (신뢰도: {cow['confidence']:.3f})")
            print(f"    위치: ({cow['x']:.0f}, {cow['y']:.0f}) 크기: {cow['width']:.0f}x{cow['height']:.0f}")
        
        if result['alarms']:
            print(f"\n🚨 알람 정보:")
            for alarm in result['alarms']:
                print(f"  - {alarm['message']} (심각도: {alarm['severity']})")
        
        print(f"\n📈 감지 통계:")
        info = result['detection_info']
        print(f"  - 원시 감지: {info['raw_detections']}개")
        print(f"  - NMS 후: {info['after_nms']}개")
        print(f"  - 최종 확정: {info['total_found']}개")
        
    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    test_cam1_frame()


