import cv2
from lane_detector import LaneDetector
from car_detector import CarDetector

vidcap = cv2.VideoCapture("sample.mp4")

def nothing(x):
    pass

# Controls setup
cv2.namedWindow("Image Thresholding")
cv2.createTrackbar("L - H", "Image Thresholding", 0, 255, nothing)
cv2.createTrackbar("L - S", "Image Thresholding", 0, 255, nothing)
cv2.createTrackbar("L - V", "Image Thresholding", 145, 255, nothing)
cv2.createTrackbar("U - H", "Image Thresholding", 255, 255, nothing)
cv2.createTrackbar("U - S", "Image Thresholding", 50, 255, nothing)
cv2.createTrackbar("U - V", "Image Thresholding", 255, 255, nothing)

cv2.namedWindow("Original")
#adjust based on camera positioning
cv2.createTrackbar("Center X", "Original", 335, 640, nothing)
cv2.createTrackbar("Top Y", "Original", 330, 480, nothing)
cv2.createTrackbar("Top Width", "Original", 175, 640, nothing)
cv2.createTrackbar("Bottom Width", "Original", 450, 640, nothing)

lane_detector = LaneDetector()
car_detector = CarDetector()
paused = False

while vidcap.isOpened():
    if not paused:
        success, image = vidcap.read()
        
        if not success or image is None:
            print("Fișierul video s-a terminat.")
            break
            
        frame = cv2.resize(image, (640, 480))
        
        result, bird_view, mask, sliding_windows = lane_detector.process_frame(frame)
        
        cx = cv2.getTrackbarPos("Center X", "Original")
        ty = 255 # ~305 for high speed / 255 normal
        tw = cv2.getTrackbarPos("Top Width", "Original")
        bw = cv2.getTrackbarPos("Bottom Width", "Original")
        result = car_detector.detect_vehicles(result, cx, ty, tw, bw)
    else:
        result, bird_view, mask, sliding_windows = lane_detector.process_frame(frame)
        cx = cv2.getTrackbarPos("Center X", "Original")
        ty = 255 # ~305 for high speed / 255 normal
        tw = cv2.getTrackbarPos("Top Width", "Original")
        bw = cv2.getTrackbarPos("Bottom Width", "Original")
        result = car_detector.detect_vehicles(result, cx, ty, tw, bw)
        cv2.putText(result, "Paused", (30, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    cv2.imshow("Bird's Eye View", bird_view)
    cv2.imshow("Image Thresholding", mask)
    cv2.imshow("Lane Detection - Sliding Windows", sliding_windows)
    cv2.imshow('Original', result)

    key = cv2.waitKey(1) & 0xFF
    
    if key == 27: # ESC to exit
        break
    elif key == 32: # Spacebar to pause/unpause
        paused = not paused

vidcap.release()
cv2.destroyAllWindows()