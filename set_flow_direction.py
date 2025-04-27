import cv2
import math
import numpy as np
from tkinter import simpledialog, filedialog

class FlowDirectionDrawer:
    def __init__(self):
        self.drawing = False
        self.ix = -1
        self.iy = -1
        self.flow_direction_vector = None
        self.flow_direction_rad = None
        self.grid_img = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left button pressed
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:  # Mouse movement
            if self.drawing:
                img_copy = self.grid_img.copy()
                cv2.arrowedLine(img_copy, (self.ix, self.iy), (x, y), (0, 0, 255), 2)  # Draw arrow
                cv2.putText(img_copy, f"({x}, {y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow('Draw Flow Direction', img_copy)

        elif event == cv2.EVENT_LBUTTONUP:  # Left button released
            self.drawing = False
            cv2.arrowedLine(self.grid_img, (self.ix, self.iy), (x, y), (0, 0, 255), 2)
            self.flow_direction_vector = (x - self.ix, y - self.iy)
            cv2.imshow('Draw Flow Direction', self.grid_img)
            cv2.destroyWindow('Draw Flow Direction')

    def save_flow_direction(self):
        if not self.flow_direction_rad:
            print("流向が存在しません。")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".txt",
                                                filetypes=[("Text files", "*.txt")],
                                                title="流向を保存")
        if filename:
            with open(filename, 'w') as f:
                f.write(f"{self.flow_direction_rad}\n")
            print(f"流向を {filename} に保存しました。")
            print("Flow direction degree set to", np.rad2deg(self.flow_direction_rad))
            print("Flow direction radian set to", self.flow_direction_rad)

    def load_flow_direction(self):
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")],
                                              title="流向を読み込み")
        if filename:
            with open(filename, 'r') as f:
                self.flow_direction_rad = float(f.readline())
            print(f"流向を {filename} から読み込みました。")
            print("Flow direction degree set to", np.rad2deg(self.flow_direction_rad))
            print("Flow direction radian set to", self.flow_direction_rad)

    def input_flow_direction(self):
        self.flow_direction_vector = simpledialog.askstring("ベクトル入力", "原点からのベクトル(x, y)を入力してください。例: 0, 1")
        self.flow_direction_vector = tuple(map(float, self.flow_direction_vector.split(",")))
        self.flow_direction_rad = math.atan2(self.flow_direction_vector[1], self.flow_direction_vector[0])
        print("Flow direction degree set to", np.rad2deg(self.flow_direction_rad))
        print("Flow direction radian set to", self.flow_direction_rad)
        return self.flow_direction_rad


    def set_flow_direction_manually(self, input_video_path):
        user_input = simpledialog.askstring("モード選択", "sかl, もしくはwキーを押すか、キャンセルを押してください。sを入力すると今回作成した流向を保存します。lを入力するとセーブした流向を読み込みます。wを入力するとベクトルの数値を直接入力します。キャンセルを押すとどちらも行いません。")
        if user_input == "l":
            self.load_flow_direction()
            return self.flow_direction_rad
        
        elif user_input == "w":
            self.input_flow_direction()       
            
        cap = cv2.VideoCapture(input_video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Draw grid on the image
            self.grid_img = frame.copy()
            step_size = 50  # Grid spacing
            color = (200, 200, 200)  # Grid color (light gray)
            thickness = 1

            # Draw horizontal grid lines
            for y in range(0, frame.shape[0], step_size):
                cv2.line(self.grid_img, (0, y), (frame.shape[1], y), color, thickness)

            # Draw vertical grid lines
            for x in range(0, frame.shape[1], step_size):
                cv2.line(self.grid_img, (x, 0), (x, frame.shape[0]), color, thickness)

            cv2.namedWindow('Draw Flow Direction')
            cv2.setMouseCallback('Draw Flow Direction', self.mouse_callback)
            cv2.imshow('Draw Flow Direction', self.grid_img)
            cv2.waitKey(1)
            if self.flow_direction_vector is not None:
                break
            
        if self.flow_direction_vector is None:
            print("Flow direction not set.")
            cap.release()
            return None
        
        # Calculate angle from the vector
        self.flow_direction_rad = math.atan2(self.flow_direction_vector[1], self.flow_direction_vector[0])
        print("Flow direction degree set to", np.rad2deg(self.flow_direction_rad))
        print("Flow direction radian set to", self.flow_direction_rad)
        cap.release()
        if user_input == "s":
            self.save_flow_direction()
        return self.flow_direction_rad


def normalize_angle(angle):
    """ Normalize angle to be in range -pi to pi """
    return (angle + math.pi) % (2 * math.pi) - math.pi

def determine_direction_thres(direction_thres, flow_direction_rad):
    direction_thres = {key: np.deg2rad(value) for key, value in direction_thres.items()}
    absolute_direction_thres = {}
    for class_name, angle_ranges in direction_thres.items():
        converted_ranges = []
        for angle_range in angle_ranges:
            min_angle, max_angle = angle_range
            absolute_min_angle = normalize_angle(flow_direction_rad + min_angle)
            absolute_max_angle = normalize_angle(flow_direction_rad + max_angle)

            if absolute_min_angle > absolute_max_angle:
                converted_ranges.append((absolute_min_angle, math.pi))
                converted_ranges.append((-math.pi, absolute_max_angle))
            else:
                converted_ranges.append((absolute_min_angle, absolute_max_angle))
        absolute_direction_thres[class_name] = merged_ranges(converted_ranges)

    return absolute_direction_thres

def merged_ranges(ranges):
    if not ranges:
        return []
    ranges.sort(key=lambda x: x[0])
    merged = [ranges[0]]
    for current in ranges[1:]:
        previous = merged[-1]
        if current[0] <= previous[1] or (previous[1] == math.pi and current[0] == -math.pi):
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:
            merged.append(current)
    return merged

if __name__ == '__main__':
    input_video_path = "../1_data_raw/ayu_clipped.mp4"
    direction_thres = {"fish": [(-180, -90), (90, 180)], "debris": [(-90, 90)]}
    drawer = FlowDirectionDrawer()
    flow_direction_rad = drawer.set_flow_direction_manually(input_video_path)

    if flow_direction_rad is not None:
        direction_thres = determine_direction_thres(direction_thres, flow_direction_rad)
        print(direction_thres)
        for class_name, angle_ranges in direction_thres.items():
            print(class_name)
            for angle_range in angle_ranges:
                print(np.rad2deg(angle_range[0]), np.rad2deg(angle_range[1]))


# import cv2
# import math
# import numpy as np

# def draw_flow_direction(event, x, y, flags, param):
#     state, grid_img = param

#     if event == cv2.EVENT_LBUTTONDOWN:  # Left button pressed
#         state['drawing'] = True
#         state['ix'], state['iy'] = x, y

#     elif event == cv2.EVENT_MOUSEMOVE:  # Mouse movement
#         img_copy = grid_img.copy()
#         if state['drawing']:
#             cv2.arrowedLine(img_copy, (state['ix'], state['iy']), (x, y), (0, 0, 255), 2)  # Draw arrow
#         # Display current coordinates
#         cv2.putText(img_copy, f"({x}, {y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#         cv2.imshow('Draw Flow Direction', img_copy)

#     elif event == cv2.EVENT_LBUTTONUP:  # Left button released
#         state['drawing'] = False
#         cv2.arrowedLine(grid_img, (state['ix'], state['iy']), (x, y), (0, 0, 255), 2)
#         state['flow_direction_vector'] = (x - state['ix'], y - state['iy'])
#         cv2.imshow('Draw Flow Direction', grid_img)
#         cv2.destroyWindow('Draw Flow Direction')

# def set_flow_direction_manually(cap):
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         state = {
#             'drawing': False,
#             'ix': -1,
#             'iy': -1,
#             'flow_direction_vector': None
#         }

#         # Draw grid on the image
#         grid_img = frame.copy()
#         step_size = 50  # Grid spacing
#         color = (200, 200, 200)  # Grid color (light gray)
#         thickness = 1

#         # Draw horizontal grid lines
#         for y in range(0, frame.shape[0], step_size):
#             cv2.line(grid_img, (0, y), (frame.shape[1], y), color, thickness)

#         # Draw vertical grid lines
#         for x in range(0, frame.shape[1], step_size):
#             cv2.line(grid_img, (x, 0), (x, frame.shape[0]), color, thickness)

#         cv2.namedWindow('Draw Flow Direction')
#         cv2.setMouseCallback('Draw Flow Direction', draw_flow_direction, param=(state, grid_img))
#         cv2.imshow('Draw Flow Direction', grid_img)
#         cv2.waitKey(1)
        
#     if state['flow_direction_vector'] is None:
#         print("Flow direction not set.")
#         return None
#     # Calculate angle from the vector
#     flow_direction_rad = math.atan2(state['flow_direction_vector'][1], state['flow_direction_vector'][0])
#     return flow_direction_rad

# # change degree to radian
# def normalize_angle(angle):
#     """ Normalize angle to be in range -pi to pi """
#     return (angle + math.pi) % (2 * math.pi) - math.pi

# def determine_direction_thres(direction_thres, flow_direction_rad):
#     direction_thres = {key: np.deg2rad(value) for key, value in direction_thres.items()}
#     # Calculate the range of angles for each object type
#     absolute_direction_thres = {}
#     for class_name, angle_ranges in direction_thres.items():
#         converted_ranges = []
#         for angle_range in angle_ranges:
#             min_angle, max_angle = angle_range
#             # Convert relative angles to absolute image coordinates and normalize
#             absolute_min_angle = normalize_angle(flow_direction_rad + min_angle)
#             absolute_max_angle = normalize_angle(flow_direction_rad + max_angle)

#             # Handle cases where the range crosses -pi to pi boundary
#             if absolute_min_angle > absolute_max_angle:
#                 converted_ranges.append((absolute_min_angle, math.pi))
#                 converted_ranges.append((-math.pi, absolute_max_angle))
#             else:
#                 converted_ranges.append((absolute_min_angle, absolute_max_angle))
#         absolute_direction_thres[class_name] = merged_ranges(converted_ranges)

#     return absolute_direction_thres


# def merged_ranges(ranges):
#     """ Merge overlapping or adjacent ranges. """
#     if not ranges:
#         return []

#     # Sort ranges by their start value
#     ranges.sort(key=lambda x: x[0])
#     merged = [ranges[0]]
#     for current in ranges[1:]:
#         previous = merged[-1]
#         # If the current range overlaps or is adjacent to the previous, merge them
#         if current[0] <= previous[1] or (previous[1] == math.pi and current[0] == -math.pi):
#             merged[-1] = (previous[0], max(previous[1], current[1]))
#         else:
#             merged.append(current)
#     return merged

# if __name__ == '__main__':
#     input_video_path = "../1_data_raw/ayu_clipped.mp4"
#     cap = cv2.VideoCapture(input_video_path)
#     direction_thres={"fish": [(-180, -90), (90, 180)], "debris": [(-90, 90)]}
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         direction_thres = determine_direction_thres(direction_thres, set_flow_direction_manually(cap))