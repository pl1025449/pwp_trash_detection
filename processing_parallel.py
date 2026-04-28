import cv2
import numpy as np
import martian_detection as finder
def ptl(point:tuple,line:list,linPoint):
    x=point[0]
    y=point[1]
    z=line[0]*line[1][0]+line[1][1]

def process_frame(frame):
    # Convert to HSV and boost saturation
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_channel = hsv_img[:, :, 0]
    s_channel = hsv_img[:, :, 1]
    v_channel = hsv_img[:, :, 2]

    new_s = cv2.multiply(s_channel, 1.6)
    new_hsv = cv2.merge([h_channel, new_s, v_channel])
    #face processing
    face_found,face_frm=finder.detect_face(frame)
#    face_found,face_frm=True,True





    # Convert to grayscale
    gray = cv2.cvtColor(new_hsv, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.GaussianBlur(gray, (9, 9), 10)

    # Preprocessing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    dilated = cv2.dilate(blur, kernel, iterations=1)

    binary = cv2.adaptiveThreshold(
        dilated,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        13,
        3
    )

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_big)

    # Canny
    edges = cv2.Canny(binary, 100, 200)

    h, w = frame.shape[:2]
    frame_center_x = w // 2

    # ROI / segmentation for lane line
    roi_pts = np.array([[
        (int(0.2 * w), 0),
        (int(0.8 * w), 0),
        (int(1.0 * w), int(1.0 * h)),
        (int(0.0*w), int(1.0 * h))
    ]], dtype=np.int32)
    # cv2.polylines(out,roi_pts,True,(0,0,255),5)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, roi_pts, 255)
    roi = cv2.bitwise_and(edges, edges, mask=mask)
    # Detect line segments for lane following
    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180,
        80,
        minLineLength=80,
        maxLineGap=40
    )

    out = frame.copy()

    left_lines = []
    right_lines = []
    horizontal_lines = []

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                continue

            slope = dy / dx

            # horizontal stop line candidates
            if abs(slope) < 0.3:
                line_length = np.hypot(dx, dy)
                # only keep strong horizontal lines near the bottom half
                if line_length > 100 and max(y1, y2) > int(0.6 * h):
                    horizontal_lines.append((x1, y1, x2, y2))
                continue

            # keep only steeper lines for left/right lane lines
            if abs(slope) < 0.5:
                continue

            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

    def average_line(line_list):
        if len(line_list) == 0:
            return None
        xs = []
        ys = []

        for x1, y1, x2, y2 in line_list:
            xs += [x1, x2]
            ys += [y1, y2]

        fit = np.polyfit(ys, xs, 1)

        y1 = h
        y2 = int(h * 0.5)

        x1 = int(fit[0] * y1 + fit[1])
        x2 = int(fit[0] * y2 + fit[1])

        return (x1, y1, x2, y2)

    left = average_line(left_lines)
    right = average_line(right_lines)
    lft=False
    rght=False
    if left is not None:
        cv2.line(out, (left[0], left[1]), (left[2], left[3]), (0, 255, 0), 4)
        lft=True
    if right is not None:
        cv2.line(out, (right[0], right[1]), (right[2], right[3]), (0, 255, 0), 4)
        rght=True
    steering_value = 0

    # draw robot/frame center line
    cv2.line(out, (frame_center_x, 0), (frame_center_x, h), (0, 0, 255), 2)

    if left is not None and right is not None:
        cx1 = int((left[0] + right[0]) / 2)
        cy1 = int((left[1] + right[1]) / 2)

        cx2 = int((left[2] + right[2]) / 2)
        cy2 = int((left[3] + right[3]) / 2)

        # blue center line
        cv2.line(out, (cx1, cy1), (cx2, cy2), (255, 0, 0), 4)
        center_line=True
        # use bottom point of lane center
        lane_center_x = cx1

        # positive = lane center right of frame center
        # negative = lane center left of frame center
        steering_value = (np.pi/2)-np.arctan((y2-y1)/(1+x2-x1))
    else:
        center_line=False
    # stop line detection
    stop_line_detected = False
   # determining_point=frame_center_x
    if len(horizontal_lines) > 0:
        longest_line = max(
            horizontal_lines,
            key=lambda line: np.hypot(line[2] - line[0], line[3] - line[1])
        )

        x1, y1, x2, y2 = longest_line
    #    determining_point=(x1+x2)//2
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 255), 4)
        stop_line_detected = True
    #point to line
    cv2.circle(out,(out.shape[1]//2,9*out.shape[0]//10),5,(255,0,0),-1)
    cv2.polylines(out,roi_pts,True,(0,0,255),5)
#    out =cv2.imread('wanted.jpg')
    return out, steering_value, stop_line_detected,center_line,lft,rght,face_frm
