import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        self.prevLx = []
        self.prevRx = []
        
        # Spatial points for frame to frame tracking stability
        self.last_stable_left = None
        self.last_stable_right = None
        
        # Anomaly counters to manage noise (shadows, missing dashes..)
        self.left_anomaly_frames = 0
        self.right_anomaly_frames = 0

    def process_frame(self, frame):
        # Fetch interactive perspective boundaries from UI Trackbars
        cx = cv2.getTrackbarPos("Center X", "Original")
        ty = cv2.getTrackbarPos("Top Y", "Original")
        tw = cv2.getTrackbarPos("Top Width", "Original")
        bw = cv2.getTrackbarPos("Bottom Width", "Original")

        # Generate mapping points
        tl = (cx - tw // 2, ty)
        tr = (cx + tw // 2, ty)
        bl = (cx - bw // 2, 472)
        br = (cx + bw // 2, 472)

        points = np.array([tl, tr, br, bl], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 255), thickness=2)

        # Execute Bird's Eye View Warp Transformation
        pts1 = np.float32([tl, bl, tr, br]) 
        pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]]) 
        matrix = cv2.getPerspectiveTransform(pts1, pts2) 
        transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

        # HSV Color space thresholding for lane extraction
        hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
        l_h = cv2.getTrackbarPos("L - H", "Image Thresholding")
        l_s = cv2.getTrackbarPos("L - S", "Image Thresholding")
        l_v = cv2.getTrackbarPos("L - V", "Image Thresholding")
        u_h = cv2.getTrackbarPos("U - H", "Image Thresholding")
        u_s = cv2.getTrackbarPos("U - S", "Image Thresholding")
        u_v = cv2.getTrackbarPos("U - V", "Image Thresholding")
        
        lower = np.array([l_h, l_s, l_v])
        upper = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv_transformed_frame, lower, upper)

        # Generate pixel density histogram along vertical columns (bottom half of image)
        histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
        midpoint = int(histogram.shape[0]/2)
        
        current_left_base = np.argmax(histogram[:midpoint])
        current_right_base = np.argmax(histogram[midpoint:]) + midpoint

        if self.last_stable_left is None:
            self.last_stable_left = current_left_base
        if self.last_stable_right is None:
            self.last_stable_right = current_right_base

        # Lane Validation
        # Rejects high-frequency structural jitter. If a lane base shifts >20px instantly, 
        # it is flagged as noise (e.g. shadow artifact) and held at its previous stable 
        # position, unless the drift persists for consecutive validation frames.
        
        # Left Lane Check
        if abs(current_left_base - self.last_stable_left) > 20:
            self.left_anomaly_frames += 1
            if self.left_anomaly_frames >= 3:
                self.last_stable_left = current_left_base
                self.left_anomaly_frames = 0
            else:
                current_left_base = self.last_stable_left
        else:
            self.last_stable_left = current_left_base
            self.left_anomaly_frames = 0

        # Right Lane Check
        if abs(current_right_base - self.last_stable_right) > 20:
            self.right_anomaly_frames += 1
            if self.right_anomaly_frames >= 6:
                self.last_stable_right = current_right_base
                self.right_anomaly_frames = 0
            else:
                current_right_base = self.last_stable_right
        else:
            self.last_stable_right = current_right_base
            self.right_anomaly_frames = 0

        left_base = current_left_base
        right_base = current_right_base
        
        init_left = left_base
        init_right = right_base

        # Sliding Window - extract vertical path pixels
        y = 472
        window_height = 40
        margin = 50
        min_pix = 50  

        lx = []
        rx = []
        msk = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 

        while y > 0:
            # Processing left windows
            win_y_low = y - window_height
            win_y_high = y
            win_xleft_low = max(0, left_base - margin)
            win_xleft_high = min(mask.shape[1], left_base + margin)
            
            cv2.rectangle(msk, (win_xleft_low, win_y_high), (win_xleft_high, win_y_low), (0, 255, 0), 2)
            
            good_left_inds = np.argwhere((mask[win_y_low:win_y_high, win_xleft_low:win_xleft_high] > 0))
            if len(good_left_inds) > min_pix:
                left_base = int(np.mean(good_left_inds[:, 1])) + win_xleft_low
                lx.append(left_base)
            elif len(lx) > 0:
                lx.append(lx[-1]) 
            else:
                lx.append(left_base)

            # Processing right windows
            win_xright_low = max(0, right_base - margin)
            win_xright_high = min(mask.shape[1], right_base + margin)
            
            cv2.rectangle(msk, (win_xright_low, win_y_high), (win_xright_high, win_y_low), (0, 0, 255), 2)
            
            good_right_inds = np.argwhere((mask[win_y_low:win_y_high, win_xright_low:win_xright_high] > 0))
            if len(good_right_inds) > min_pix:
                right_base = int(np.mean(good_right_inds[:, 1])) + win_xright_low
                rx.append(right_base)
            elif len(rx) > 0:
                rx.append(rx[-1])
            else:
                rx.append(right_base)

            y -= window_height
            
        if len(lx) == 0: lx = self.prevLx
        else: self.prevLx = lx

        if len(rx) == 0: rx = self.prevRx
        else: self.prevRx = rx

        min_length = min(len(lx), len(rx))

        if min_length > 2:
            left_points = [(lx[i], 472 - i * 40) for i in range(min_length)]
            right_points = [(rx[i], 472 - i * 40) for i in range(min_length)]

            # Fit 2nd-degree mathematical polynomials to model road curves
            try:
                left_fit = np.polyfit([p[1] for p in left_points], [p[0] for p in left_points], 2)
                right_fit = np.polyfit([p[1] for p in right_points], [p[0] for p in right_points], 2)
            except (np.linalg.LinAlgError, ValueError):
                pass

            ploty = np.linspace(0, 479, 480)
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            # Radius of curvature calculation
            y_eval = 480
            left_curvature = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.abs(2*left_fit[0])
            right_curvature = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.abs(2*right_fit[0])
            curvature = (left_curvature + right_curvature) / 2
            
            # Steering geometry and lane offset tracking
            lane_offset = ((init_left + init_right) / 2 - cx) * 3.7 / 640
            steering_angle = np.arctan(lane_offset / curvature) * 180 / np.pi

            if curvature > 6000:
                curvature_text = "Straight"
            else:
                direction = "Right" if steering_angle > 0.3 else "Left" if steering_angle < -0.3 else ""
                curvature_text = f"{curvature:.0f} m ({direction})".strip(" ()")

            line_length = 100  
            end_x = int(cx + line_length * np.sin(np.radians(steering_angle)))
            end_y = int(480 - line_length * np.cos(np.radians(steering_angle)))

            # Construct polynomial polygon overlays
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            
            overlay = transformed_frame.copy()
            cv2.fillPoly(overlay, np.int32([pts]), (0, 255, 0))
            cv2.addWeighted(overlay, 0.25, transformed_frame, 0.75, 0, transformed_frame)
            # Map the warped bird's-eye view lane mask back onto original camera perspective
            inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
            original_perspective_lane_image = cv2.warpPerspective(transformed_frame, inv_matrix, (640, 480))
            result = cv2.addWeighted(frame, 1, original_perspective_lane_image, 0.5, 0)

            cv2.line(result, (cx, 480), (end_x, end_y), (255, 0, 0), 2)
            cv2.putText(result, f'Curvature: {curvature_text}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(result, f'Offset: {lane_offset:.2f} m', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(result, f'Angle: {steering_angle:.2f} deg', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            result = frame.copy()
            cv2.putText(result, "CALIBRATING ADAS...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return result, transformed_frame, mask, msk