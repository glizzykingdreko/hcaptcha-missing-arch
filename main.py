import cv2
import numpy as np
import math
import os

class DonutChallengeSolver:
    def __init__(self, debug=False, debug_prefix="debug"):
        """
        Initialize the solver.
        :param debug: If True, intermediate debug images will be saved.
        :param debug_prefix: File prefix for saved debug images.
        """
        self.debug = debug
        self.debug_prefix = debug_prefix
        self.debug_images = {}

    @staticmethod
    def is_curved(contour, threshold=0.1):
        """Check if a contour segment is curved."""
        if len(contour) < 3:
            return False
        start = contour[0][0]
        end = contour[-1][0]
        straight_dist = np.linalg.norm(end - start)
        contour_length = cv2.arcLength(contour, False)
        return (contour_length / straight_dist) > (1 + threshold)

    @staticmethod
    def merge_arcs(arcs, gap_threshold=5):
        """
        Merge arcs that overlap or are very close (<= gap_threshold).
        Each arc is (start_angle, end_angle). Returns a sorted list of disjoint arcs.
        """
        if not arcs:
            return []
        arcs = sorted(arcs, key=lambda x: x[0])
        merged = [arcs[0]]
        for i in range(1, len(arcs)):
            curr_start, curr_end = arcs[i]
            last_start, last_end = merged[-1]
            if curr_start <= last_end + gap_threshold:
                merged[-1] = (last_start, max(last_end, curr_end))
            else:
                merged.append((curr_start, curr_end))
        final = []
        for s, e in merged:
            s_mod = s % 360
            e_mod = e % 360
            if e_mod < s_mod:
                e_mod += 360
            final.append((s_mod, e_mod))
        return sorted(final, key=lambda x: x[0])

    @staticmethod
    def find_missing_arcs(arcs):
        """
        Given a sorted list of disjoint arcs in [0,360), return the complement (missing arcs).
        """
        missing = []
        if not arcs:
            return [(0, 360)]
        for i in range(len(arcs) - 1):
            end_current = arcs[i][1]
            start_next = arcs[i+1][0]
            if start_next > end_current:
                missing.append((end_current, start_next))
        last_end = arcs[-1][1]
        first_start = arcs[0][0] + 360
        if first_start > last_end:
            missing.append((last_end, first_start))
        return missing

    def save_debug_image(self, name, img):
        """Save debug image if debug flag is True."""
        if self.debug:
            os.makedirs("debug_output", exist_ok=True)
            cv2.imwrite(os.path.join("debug_output", f"{self.debug_prefix}_{name}.png"), img)

    def preprocess_donut(self, image):
        """
        Process the input image (BGR) to detect the donut shape,
        compute the missing arc, and return the drop-out coordinates.
        Also collects and saves debug images if enabled.
        """
        # 1. Preprocess: grayscale, blur, and edge detection.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.debug_images["1_gray"] = gray
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self.debug_images["2_blurred"] = blurred
        edges = cv2.Canny(blurred, 50, 150)
        self.debug_images["3_edges"] = edges

        # 2. Detect circle using Hough Circle Transform.
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                   param1=50, param2=30, minRadius=20, maxRadius=100)
        if circles is None:
            return None  # No circle found.
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0], key=lambda x: x[2])
        x_center, y_center, r = largest_circle
        circle_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.circle(circle_vis, (x_center, y_center), r, (0, 255, 0), 2)
        cv2.circle(circle_vis, (x_center, y_center), r+5, (255, 0, 0), 2)
        cv2.circle(circle_vis, (x_center, y_center), r-5, (255, 0, 255), 2)
        self.debug_images["4_circle_detection"] = circle_vis

        # 3. Isolate donut region: create outer and inner masks.
        outer_mask = np.zeros_like(edges)
        inner_mask = np.zeros_like(edges)
        cv2.circle(outer_mask, (x_center, y_center), r+5, 255, -1)
        cv2.circle(inner_mask, (x_center, y_center), r-5, 255, -1)
        donut_region = cv2.bitwise_and(edges, cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask)))
        # Keep only the largest contour.
        contours, _ = cv2.findContours(donut_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cleaned = np.zeros_like(donut_region)
            cv2.drawContours(cleaned, [largest_contour], -1, 255, thickness=cv2.FILLED)
            donut_region = cleaned
        # 4. Extract valid edge points.
        white_pixels = np.argwhere(donut_region > 0)
        white_pixels = [(p[1], p[0]) for p in white_pixels]  # convert (y,x) to (x,y)

        # 5. Compute angles for valid points and group into arcs.
        margin = 5
        angle_points = {}
        for (px, py) in white_pixels:
            dist = math.hypot(px - x_center, py - y_center)
            if (r - margin) <= dist <= (r + margin):
                angle = math.degrees(math.atan2(py - y_center, px - x_center))
                if angle < 0:
                    angle += 360
                angle = int(round(angle))
                angle_points.setdefault(angle, []).append((px, py))
        sorted_angles = sorted(angle_points.keys())
        raw_arcs = []
        current_arc = []
        gap_thresh = 5
        min_arc_len = 5
        if sorted_angles:
            current_arc = [sorted_angles[0]]
            for i in range(1, len(sorted_angles)):
                curr = sorted_angles[i]
                prev = sorted_angles[i - 1]
                if (curr - prev) <= gap_thresh:
                    current_arc.append(curr)
                else:
                    if len(current_arc) >= min_arc_len:
                        raw_arcs.append((min(current_arc), max(current_arc)))
                    current_arc = [curr]
            if len(current_arc) >= min_arc_len:
                raw_arcs.append((min(current_arc), max(current_arc)))
        present_arcs = self.merge_arcs(raw_arcs, gap_threshold=5)
        missing_arcs = self.find_missing_arcs(present_arcs)

        # Optionally, save a basic Cartesian visualization debug image.
        basic_cart = self.create_basic_cartesian_visualization(edges, x_center, y_center, r)
        self.debug_images["X_basic_cartesian"] = basic_cart

        # 6. Compute the drop-out point: the midpoint of the largest missing arc.
        if not missing_arcs:
            return None
        largest_missing_arc = max(missing_arcs, key=lambda arc: arc[1] - arc[0])
        arc_start, arc_end = largest_missing_arc
        mid_angle = (arc_start + arc_end) / 2.0
        mid_rad = math.radians(mid_angle)
        # We mark on an "inset" circle with radius (r - 7)
        drop_x = int(x_center + (r - 7) * math.cos(mid_rad))
        drop_y = int(y_center + (r - 7) * math.sin(mid_rad))

        # Optionally, save an original image debug with the drop-out dot.
        original_with_dot = image.copy()
        cv2.circle(original_with_dot, (drop_x, drop_y), 5, (0, 255, 0), -1)
        cv2.putText(original_with_dot, f"({drop_x},{drop_y})", (drop_x+5, drop_y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        self.debug_images["7_original_with_missing_dot"] = original_with_dot

        return (drop_x, drop_y)

    def create_basic_cartesian_visualization(self, edges, center_x, center_y, radius, size=400):
        """
        Create a basic Cartesian visualization that displays:
          - Red dots for valid edge points.
          - Three reference circles (green: detected, blue: outer, purple: inner).
        """
        projection = np.ones((size, size, 3), dtype=np.uint8) * 255
        cv2.line(projection, (0, size//2), (size, size//2), (0, 0, 0), 1)
        cv2.line(projection, (size//2, 0), (size//2, size), (0, 0, 0), 1)
        scale = (size * 0.8) / (2 * radius)
        center_pt = (size // 2, size // 2)

        outer_mask = np.zeros_like(edges)
        inner_mask = np.zeros_like(edges)
        cv2.circle(outer_mask, (center_x, center_y), radius + 5, 255, -1)
        cv2.circle(inner_mask, (center_x, center_y), radius - 5, 255, -1)
        donut_region = cv2.bitwise_and(edges, cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask)))
        contours, _ = cv2.findContours(donut_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cleaned = np.zeros_like(donut_region)
            cv2.drawContours(cleaned, [largest_contour], -1, 255, thickness=cv2.FILLED)
            donut_region = cleaned
        white_pixels = np.argwhere(donut_region > 0)
        white_pixels = [(p[1], p[0]) for p in white_pixels]
        for (px, py) in white_pixels:
            dx = px - center_x
            dy = py - center_y
            proj_x = int(center_pt[0] + dx * scale)
            proj_y = int(center_pt[1] + dy * scale)
            cv2.circle(projection, (proj_x, proj_y), 1, (0, 0, 255), -1)
        cv2.circle(projection, center_pt, int(radius * scale), (0, 255, 0), 1)
        cv2.circle(projection, center_pt, int((radius + 5) * scale), (255, 0, 0), 1)
        cv2.circle(projection, center_pt, int((radius - 5) * scale), (255, 0, 255), 1)
        return projection

    def solve(self, image_bytes):
        """
        Given image bytes (e.g. from a PNG file), decode the image, process it,
        and return the coordinates (x, y) of the missing-arch drop-out point.
        If debug is enabled, intermediate images will be saved.
        """
        # Decode image bytes to numpy array.
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image bytes.")
        coords = self.preprocess_donut(image)
        # If debug flag is on, save all debug images.
        if self.debug:
            for name, img in self.debug_images.items():
                self.save_debug_image(name, img)
        return coords

# Example usage:
if __name__ == "__main__":
    # For demonstration, read an image file as bytes.
    with open("./images/image-2.png", "rb") as f:
        image_bytes = f.read()
    solver = DonutChallengeSolver(debug=True)
    result = solver.solve(image_bytes)
    if result is not None:
        print(f"Missing arch drop-out coordinates: {result}")
    else:
        print("Failed to find a matching donut shape.")
