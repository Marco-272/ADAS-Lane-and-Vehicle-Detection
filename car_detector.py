from ultralytics import YOLO
import cv2

class CarDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.traffic_classes = [2, 3, 5, 7]
        self.frame_counter = 0

        # Target tracking states
        self.target_box = None
        self.smoothed_distance = None

        # Target confirmation filtering
        self.CONFIRM_FRAMES = 4
        self.candidate_box = None
        self.candidate_score = 0

        self.prev_distance = None
        self.aeb_status = "SAFE"
        self._no_target_frames = 0

        self.current_frame_cars = []

    def _max_lateral_deviation(self, y2, ty, tw, bw):
        """
        Calculates dynamic lane boundary at a given Y coordinate.
        Uses linear interpolation across the camera's perspective trapezoid.
        """

        TOLERANCE_FACTOR = 0.22 # Extra margin to accommodate slight body roll/curves   

        if y2 <= ty:
            half_lane = tw // 2
            return half_lane + half_lane * TOLERANCE_FACTOR
        if y2 >= 472:
            half_lane = bw // 2
            return half_lane + half_lane * TOLERANCE_FACTOR

        denom = (472 - ty) if (472 - ty) != 0 else 1
        pct = (472 - y2) / denom
        pct = max(0.0, min(1.0, pct))

        half_lane = (bw // 2) - pct * ((bw // 2) - (tw // 2))
        return half_lane + half_lane * TOLERANCE_FACTOR

    def _select_best_match(self, results, cx, ty, tw, bw):
        """
        Filters raw YOLO detections to identify the primary lead vehicle in our lane.
        Prioritizes the closest valid target based on perspective Y coordinate.
        """
        best_match = None
        self.current_frame_cars = []

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls not in self.traffic_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results.names[cls]
            car_cx = (x1 + x2) // 2
            car_cy_bottom = y2

            # Check if vehicle center falls inside our dynamic lane projection
            max_dev = self._max_lateral_deviation(car_cy_bottom, ty, tw, bw)
            lateral_deviation = abs(cx - car_cx)
            is_in_lane = lateral_deviation <= max_dev

            if car_cy_bottom < (ty + 15):
                is_in_lane = False

            self.current_frame_cars.append((x1, y1, x2, y2, label, is_in_lane))

            if not is_in_lane:
                continue

            # Track the closest vehicle (maximum Y base coordinate and minimum lateral deviation)
            if best_match is None:
                best_match = (x1, y1, x2, y2, label, lateral_deviation, car_cy_bottom)
            else:
                _, _, _, _, _, b_dev, b_cy_bottom = best_match
                if abs(lateral_deviation - b_dev) <= 20:
                    if car_cy_bottom > b_cy_bottom:
                        best_match = (x1, y1, x2, y2, label, lateral_deviation, car_cy_bottom)
                else:
                    if lateral_deviation < b_dev:
                        best_match = (x1, y1, x2, y2, label, lateral_deviation, car_cy_bottom)

        if best_match is not None:
            return (best_match[0], best_match[1], best_match[2], best_match[3], best_match[4])
        return None
    
    def _is_same_target(self, box_a, box_b, iou_threshold=0.3):
        """
        Intersection over Union tracker to match targets across consecutive frames.
        """
        if box_a is None or box_b is None:
            return False
        ax1, ay1, ax2, ay2, _ = box_a
        bx1, by1, bx2, by2, _ = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union_area = area_a + area_b - inter_area

        if union_area == 0:
            return False
        return (inter_area / union_area) >= iou_threshold

    def _update_target(self, new_best):
        """
        State machine handling target acquisition and lost-target timeouts.
        Prevents rapid target-swapping due to intermittent YOLO misclassifications.
        """
        TARGET_CLEAR_FRAMES = 8

        if new_best is not None:
            if self._is_same_target(new_best, self.target_box):
                self.target_box = new_best
                self.candidate_box = None
                self.candidate_score = 0
            else:
                if self._is_same_target(new_best, self.candidate_box):
                    self.candidate_score += 1
                    self.candidate_box = new_best
                else:
                    self.candidate_box = new_best
                    self.candidate_score = 1

                if self.candidate_score >= self.CONFIRM_FRAMES:
                    self.target_box = self.candidate_box
                    self.candidate_box = None
                    self.candidate_score = 0
                    self.smoothed_distance = None
                    self.prev_distance = None

            self._no_target_frames = 0

        else:
            self.candidate_box = None
            self.candidate_score = 0
            self._no_target_frames += 1

            if self._no_target_frames >= TARGET_CLEAR_FRAMES:
                self.target_box = None
                self.smoothed_distance = None
                self.prev_distance = None
                self.aeb_status = "SAFE"

    def detect_vehicles(self, frame, cx, ty, tw, bw):
        """
        Main execution thread. Inference runs every 2 frames to reduce CPU/GPU overhead.
        """
        self.frame_counter += 1

        if self.frame_counter % 2 == 0:
            results = self.model(frame, verbose=False)[0]
            new_best = self._select_best_match(results, cx, ty, tw, bw)
            self._update_target(new_best)

            if self.target_box is not None:
                x1, y1, x2, y2, label = self.target_box
                width = x2 - x1

                if width > 0:
                    # Focal length scaling proxy for distance estimation
                    distance_constant = 1400.0
                    if label in ["truck", "bus"]:
                        distance_constant = 2400.0
                    raw_distance = distance_constant / width

                    # First-order Low-Pass Filter for distance smoothing
                    if self.smoothed_distance is None:
                        self.smoothed_distance = raw_distance
                    else:
                        self.smoothed_distance = 0.7 * self.smoothed_distance + 0.3 * raw_distance

                    # Autonomous Emergency Braking Logic
                    if self.prev_distance is not None:
                        scadere_distanta = self.prev_distance - self.smoothed_distance

                        if self.aeb_status == "BRAKE! EMERGENCY":
                            if scadere_distanta >= -0.1 or self.smoothed_distance < 4.5:
                                self.aeb_status = "BRAKE! EMERGENCY"
                            else:
                                self.aeb_status = "SAFE"
                        else:
                            if scadere_distanta > 2.5 or self.smoothed_distance < 4.5:
                                self.aeb_status = "BRAKE! EMERGENCY"
                            elif self.smoothed_distance < 11.0:
                                self.aeb_status = "WARNING: CLOSING IN"
                            else:
                                self.aeb_status = "SAFE"

                    self.prev_distance = self.smoothed_distance

        # UI telemetry readings
        tol_bottom = int(bw // 2 * (1 + 0.35))-60
        tol_top    = int(tw // 2 * (1 + 0.35))-50
        c_tl = (cx - tol_top,ty)
        c_tr = (cx + tol_top,ty)
        c_bl = (cx - tol_bottom, 472)
        c_br = (cx + tol_bottom, 472)
        cv2.line(frame, c_bl, c_tl, (0, 80, 0), 1)
        cv2.line(frame, c_br, c_tr, (0, 80, 0), 1)

        for x1, y1, x2, y2, label, is_in_lane in self.current_frame_cars:
            if not is_in_lane:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 0), 1)
                cv2.putText(frame, f"{label.upper()} (OTHER)", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 80, 0), 1)

        if self.candidate_box is not None and self.candidate_score > 0:
            cx1, cy1, cx2, cy2, _ = self.candidate_box
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 165, 255), 1)
            cv2.putText(frame, f"CANDIDATE ({self.candidate_score}/{self.CONFIRM_FRAMES})",
                        (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        if self.target_box is not None:
            x1, y1, x2, y2, label = self.target_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = (f"LEAD VEHICLE: {self.smoothed_distance:.1f} m"
                    if self.smoothed_distance is not None
                    else f"LEAD VEHICLE ({label.upper()})")
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 5, (0, 0, 255), -1)

        # AEB Alerts
        if self.aeb_status == "BRAKE! EMERGENCY":
            cv2.rectangle(frame, (20, 400), (450, 460), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, "!!! EMERGENCY BRAKING !!!", (30, 440),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        elif self.aeb_status == "WARNING: CLOSING IN":
            cv2.rectangle(frame, (20, 420), (350, 460), (0, 255, 255), cv2.FILLED)
            cv2.putText(frame, "WARNING: DISTANCE CLOSING", (30, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            status_text = (f"ACC: FOLLOWING {self.smoothed_distance:.1f}m"
                           if self.smoothed_distance is not None
                           else "ACC STATUS: NO TARGET")
            cv2.putText(frame, status_text, (30, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame