import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog, filedialog
import screeninfo

# 動画ファイルのパス
input_video = "../1_data_raw/ayu_clipped.mp4"
output_video = "test_roi_polygon.mp4"

class PolygonDrawer:
    def __init__(self, window_name, first_frame):
        self.window_name = window_name
        self.original_frame = first_frame  # オリジナルのフレームを保持
        self.first_frame = first_frame.copy()  # 操作用のフレーム
        self.frame_height, self.frame_width = first_frame.shape[:2]
        self.points = []  # ポリゴンの頂点座標を格納
        self.current_mouse_pos = (0, 0)  # マウスの現在位置を格納
        self.is_drawing = False  # ズーム領域を描画中かどうか
        self.zoom_rect = None  # ズーム領域
        self.is_zoomed = False  # ズーム状態かどうか
        self.offset = [0, 0]  # パンのオフセット
        self.drag_start_pos = None  # ドラッグ開始位置
        self.zoom_size = [1,1]
        self.output_frame = first_frame
        self.frame_height, self.frame_width = self.first_frame.shape[:2]
        # 現在のモニターの解像度を取得
        screen = screeninfo.get_monitors()[0]
        self.screen_width, self.screen_height = screen.width, screen.height

        cv2.namedWindow(self.window_name)
        # ウィンドウ設定
        cv2.resizeWindow(self.window_name, self.screen_width, self.screen_height)
        cv2.setMouseCallback(self.window_name, self.mouse_events)

    def mouse_events(self, event, x, y, flags, param):
        self.current_mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((int(x*self.zoom_size[0] + self.offset[0]), int(y*self.zoom_size[1] + self.offset[1])))
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.drag_start_pos = (x, y)
        elif event == cv2.EVENT_MBUTTONUP:
            self.drag_start_pos = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_start_pos:
                dx = x - self.drag_start_pos[0]
                dy = y - self.drag_start_pos[1]
                self.offset[0] -= dx
                self.offset[1] -= dy
                self.drag_start_pos = (x, y)
            elif self.is_drawing:
                # ズーム矩形の終点を更新
                self.zoom_rect[2] = x + self.offset[0]
                self.zoom_rect[3] = y + self.offset[1]
        elif event == cv2.EVENT_RBUTTONDOWN:
            if not self.is_drawing:
                self.is_drawing = True
                self.zoom_rect = [x + self.offset[0], y + self.offset[1], x + self.offset[0], y + self.offset[1]]
        elif event == cv2.EVENT_RBUTTONUP:
            if self.is_drawing:
                self.is_drawing = False
                x1, y1, x2, y2 = self.zoom_rect
                x1, x2 = sorted([int(x1), int(x2)])
                y1, y2 = sorted([int(y1), int(y2)])
                x1 = max(0, min(x1, self.frame_width - 1))
                x2 = max(0, min(x2, self.frame_width))
                y1 = max(0, min(y1, self.frame_height - 1))
                y2 = max(0, min(y2, self.frame_height))
                self.offset = [x1, y1]
                if x2 - x1 > 1 and y2 - y1 > 1:
                    # ズーム領域のアスペクト比を元のフレームに合わせる
                    zoom_width = x2 - x1
                    zoom_height = y2 - y1
                    aspect_ratio = self.frame_width / self.frame_height

                    if zoom_width / zoom_height > aspect_ratio:
                        # 幅に基づいて高さを調整
                        new_zoom_height = int(zoom_width / aspect_ratio)
                        y_center = (y1 + y2) // 2
                        y1 = max(0, y_center - new_zoom_height // 2)
                        y2 = min(self.frame_height, y_center + new_zoom_height // 2)
                    else:
                        # 高さに基づいて幅を調整
                        new_zoom_width = int(zoom_height * aspect_ratio)
                        x_center = (x1 + x2) // 2
                        x1 = max(0, x_center - new_zoom_width // 2)
                        x2 = min(self.frame_width, x_center + new_zoom_width // 2)

                    # 調整後のズーム領域を適用
                    self.zoomed_frame = self.first_frame[y1:y2, x1:x2]
                    self.zoomed_frame = cv2.resize(self.zoomed_frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)
                    self.zoom_size = [(x2 - x1) / self.frame_width, (y2 - y1) / self.frame_height]
                    self.output_frame = self.zoomed_frame
                    self.is_zoomed = True
                    # # ズームを適用
                    # print("x1, x2, y1, y2", x1, x2, y1, y2)
                    # self.zoomed_frame = self.first_frame[y1:y2, x1:x2]
                    # print(self.first_frame.shape[:2])
                    # self.frame_height, self.frame_width = self.first_frame.shape[:2]
                    # self.zoomed_frame = cv2.resize(self.zoomed_frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)
                    # self.offset = [x1, y1]
                    # self.zoom_size = [(x2-x1)/self.frame_width, (y2-y1)/self.frame_height]
                    # # self.offset = [0, 0]
                    # self.output_frame = self.zoomed_frame
                    # self.is_zoomed = True
                else:
                    print("ズーム領域が無効です。")
                    self.output_frame = self.first_frame
                    self.zoom_size = [1, 1]
                    self.is_zoomed = False

    def input_coordinates(self):
        # TkinterのGUIを作成
        root = tk.Tk()
        root.withdraw()  # メインウィンドウを非表示

        # 座標を数値で入力
        while True:
            user_input = simpledialog.askstring("座標入力", "座標をカンマ区切りで入力してください (例: 100,200):")
            if user_input is None:  # キャンセルを押された場合
                break
            try:
                x, y = map(int, user_input.split(","))
                if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                    self.points.append((x, y))
                    print(f"座標 ({x}, {y}) を追加しました。")
                    break
                else:
                    print(f"座標は画像サイズ ({self.frame_width}, {self.frame_height}) の範囲内で指定してください。")
            except ValueError:
                print("入力が無効です。カンマ区切りで数値を入力してください（例: 100,200）。")

    def save_polygon(self):
        if not self.points:
            print("ポリゴンが存在しません。")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".txt",
                                                filetypes=[("Text files", "*.txt")],
                                                title="ポリゴンを保存")
        if filename:
            with open(filename, 'w') as f:
                for point in self.points:
                    f.write(f"{point[0]},{point[1]}\n")
            print(f"ポリゴンを {filename} に保存しました。")

    def load_polygon(self):
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")],
                                              title="ポリゴンを読み込み")
        if filename:
            with open(filename, 'r') as f:
                self.points = []
                for line in f:
                    x_str, y_str = line.strip().split(',')
                    x, y = int(x_str), int(y_str)
                    self.points.append((x, y))
            print(f"ポリゴンを {filename} から読み込みました。")

    def run(self):
        user_input = simpledialog.askstring("モード選択", "sもしくはlを押すか、キャンセルを押してください。sを入力すると今回作成したポリゴンを保存します。lを入力するとセーブしたポリゴンを読み込みます。lsまたはslを入力すると、両方行います。キャンセルを押すとどちらも行いません。")
        if user_input == "l":
            self.load_polygon()
        elif user_input == "sl" or user_input == "ls":
            self.load_polygon()
        

        print("マウス左クリックでポリゴンの頂点を指定してください。")
        print("Enterキーで数値入力モードに切り替えます。")
        print("Backspaceキーで最後の点を削除します。")
        print("Escapeキーで終了します。")

        while True:
            # オフセットを適用してフレームを取得
            # temp_frame = np.zeros_like(self.first_frame)
            x_offset = self.offset[0]
            y_offset = self.offset[1]
            # print("x_offset, y_offset", x_offset, y_offset)

            x_start = max(0, -x_offset)
            y_start = max(0, -y_offset)
            x_end = min(self.frame_width, self.frame_width - x_offset)
            y_end = min(self.frame_height, self.frame_height - y_offset)
            # print("x_start, x_end, y_start, y_end", x_start, x_end, y_start, y_end)

            dst_x_start = max(0, x_offset)
            dst_y_start = max(0, y_offset)
            dst_x_end = dst_x_start + (x_end - x_start)
            dst_y_end = dst_y_start + (y_end - y_start)
            # print("dst_x_start, dst_x_end, dst_y_start, dst_y_end", dst_x_start, dst_x_end, dst_y_start, dst_y_end)

            temp_frame=self.output_frame.copy()#[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = self.output_frame#[y_start:y_end, x_start:x_end]

            # マウス位置を表示
            cv2.putText(temp_frame, f"Mouse: {self.current_mouse_pos}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ポリゴンを描画
            if len(self.points) > 1:
                if not self.is_zoomed:
                    adjusted_points = [((x+self.offset[0])/self.zoom_size[0] , (y+self.offset[1])/self.zoom_size[1]) for x, y in self.points]
                    cv2.polylines(temp_frame, [np.array(adjusted_points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            for point in self.points:
                if not self.is_zoomed:
                    cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)
                else:
                    # print(point)
                    adjusted_point = (int((point[0] - self.offset[0])/self.zoom_size[0]), int(point[1] - self.offset[1])) #((self.offset[0] + point[0]/self.zoom_size[0]), (self.offset[1] + point[1]/self.zoom_size[1]))
                    # print(adjusted_point)
                    if adjusted_point[0] >= 0 and adjusted_point[0] < self.frame_width and adjusted_point[1] >= 0 and adjusted_point[1] < self.frame_height:
                        cv2.circle(temp_frame, adjusted_point, 5, (0, 0, 255), -1)

            # ズーム領域を描画
            if self.is_drawing and self.zoom_rect:
                x1 = self.zoom_rect[0] - self.offset[0]
                y1 = self.zoom_rect[1] - self.offset[1]
                x2 = self.zoom_rect[2] - self.offset[0]
                y2 = self.zoom_rect[3] - self.offset[1]
                cv2.rectangle(temp_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            cv2.imshow(self.window_name, temp_frame)

            key = cv2.waitKey(1)

            if key == 27:  # Escapeキーで終了
                print("ポリゴンの描画を終了します。")
                break
            elif key == 13:  # Enterキーで数値入力
                self.input_coordinates()
            elif key == ord('r'):  # 'r'キーでリセット
                self.points = []
                print("ポリゴンをリセットしました。")
            elif key == ord('s'):  # 's'キーで保存
                self.save_polygon()
            elif key == ord('l'):  # 'l'キーで読み込み
                self.load_polygon()
            elif key == 8:  # Backspaceキーで最後の点を削除
                if self.points:
                    removed_point = self.points.pop()
                    print(f"座標 {removed_point} を削除しました。")

        cv2.destroyWindow(self.window_name)

        # ポリゴンが指定されていない場合は終了
        if len(self.points) < 3:
            print("ポリゴンが描画されていません。プログラムを終了します。")
            return None
        
        if user_input == "s":
            self.save_polygon()
        elif user_input == "sl" or user_input == "ls":
            self.save_polygon()

        return self.points, self.offset, self.first_frame

def set_roi_polygon(input_video):
    # 動画を読み込む
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("動画を開けませんでした")
        exit()

    # 動画の情報を取得
    ret, first_frame = cap.read()
    if not ret:
        print("動画の最初のフレームを取得できませんでした")
        exit()

    drawer = PolygonDrawer("Draw Polygon", first_frame)
    result = drawer.run()

    if result is None:
        exit()
    
    cap.release()

    points, offset, processed_frame = result
    frame_height, frame_width = processed_frame.shape[:2]
    
    return points, offset, frame_height, frame_width

if __name__ == '__main__':
    # 動画を読み込む
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("動画を開けませんでした")
        exit()

    # 動画の情報を取得
    ret, first_frame = cap.read()
    if not ret:
        print("動画の最初のフレームを取得できませんでした")
        exit()

    drawer = PolygonDrawer("Draw Polygon", first_frame)
    result = drawer.run()

    if result is None:
        cap.release()
        exit()

    points, offset, processed_frame = result
    frame_height, frame_width = processed_frame.shape[:2]

    # 動画の各フレームに対して領域を切り出す処理
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 動画の先頭に戻す
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ズームとオフセットを適用
        y1 = offset[1]
        y2 = y1 + frame_height
        x1 = offset[0]
        x2 = x1 + frame_width
        frame = frame[y1:y2, x1:x2]
        # print(frame_height, frame_width)

        # マスクを作成
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        adjusted_points = [np.array(points, np.int32) - np.array(offset)]
        cv2.fillPoly(mask, adjusted_points, 255)

        # マスクを適用
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # 結果を保存
        out.write(masked_frame)

        # プレビュー
        cv2.imshow("Masked Frame", masked_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソースを解放
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import tkinter as tk
# from tkinter import simpledialog

# # 動画ファイルのパス
# input_video = "../1_data_raw/ayu_clipped.mp4"
# output_video = "test_roi_polygon.mp4"

# # グローバル変数
# points = []  # ポリゴンの頂点座標を格納
# current_mouse_pos = (0, 0)  # マウスの現在位置を格納

# # マウスコールバック関数
# def draw_polygon(event, x, y, flags, param):
#     global points, current_mouse_pos
#     current_mouse_pos = (x, y)  # 現在のマウス位置を更新
#     if event == cv2.EVENT_LBUTTONDOWN:  # 左クリックで頂点を追加
#         points.append((x, y))

# # 数値入力用ウィンドウを表示する関数
# def input_coordinates():
#     global points
#     # TkinterのGUIを作成
#     root = tk.Tk()
#     root.withdraw()  # メインウィンドウを非表示

#     # 座標を数値で入力
#     while True:
#         user_input = simpledialog.askstring("座標入力", "座標をカンマ区切りで入力してください (例: 100,200):")
#         if user_input is None:  # キャンセルを押された場合
#             break
#         try:
#             x, y = map(int, user_input.split(","))
#             if 0 <= x < frame_width and 0 <= y < frame_height:
#                 points.append((x, y))
#                 print(f"座標 ({x}, {y}) を追加しました。")
#                 break
#             else:
#                 print(f"座標は画像サイズ ({frame_width}, {frame_height}) の範囲内で指定してください。")
#         except ValueError:
#             print("入力が無効です。カンマ区切りで数値を入力してください（例: 100,200）。")

# # 動画を読み込む
# cap = cv2.VideoCapture(input_video)
# if not cap.isOpened():
#     print("動画を開けませんでした")
#     exit()

# # 動画の情報を取得
# ret, first_frame = cap.read()
# if not ret:
#     print("動画の最初のフレームを取得できませんでした")
#     exit()

# frame_height, frame_width = first_frame.shape[:2]

# # ウィンドウを作成し、マウスコールバックを設定
# cv2.namedWindow("Draw Polygon")
# cv2.setMouseCallback("Draw Polygon", draw_polygon)

# print("マウス左クリックでポリゴンの頂点を指定してください。")
# print("Enterキーで数値入力モードに切り替えます。")
# print("Escapeキーで終了します。")

# while True:
#     temp_frame = first_frame.copy()

#     # マウス位置を表示
#     cv2.putText(temp_frame, f"Mouse: {current_mouse_pos}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     # ポリゴンを描画
#     if len(points) > 1:
#         cv2.polylines(temp_frame, [np.array(points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
#     for point in points:
#         cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)

#     cv2.imshow("Draw Polygon", temp_frame)

#     key = cv2.waitKey(1)

#     if key == 27:  # Escapeキーで終了
#         print("プログラムを終了します。")
#         break
#     elif key == 13:  # Enterキーで数値入力
#         input_coordinates()
#     elif key == ord('r'):  # 'r'キーでリセット
#         points = []
#         print("ポリゴンをリセットしました。")

# cv2.destroyWindow("Draw Polygon")

# # ポリゴンが指定されていない場合は終了
# if len(points) < 3:
#     print("ポリゴンが描画されていません。プログラムを終了します。")
#     cap.release()
#     exit()

# # 動画の各フレームに対して領域を切り出す処理
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 動画の先頭に戻す
# out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # マスクを作成
#     mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
#     cv2.fillPoly(mask, [np.array(points, np.int32)], 255)

#     # マスクを適用
#     masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

#     # 結果を保存
#     out.write(masked_frame)

#     # プレビュー
#     cv2.imshow("Masked Frame", masked_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # リソースを解放
# cap.release()
# out.release()
# cv2.destroyAllWindows()
