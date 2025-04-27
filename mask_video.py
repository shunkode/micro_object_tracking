import cv2
import numpy as np
import pandas as pd
from bresenham import bresenham

def get_heatmap_mog2(video_path, calc_heatmap_time):
    cap = cv2.VideoCapture(video_path)

    # MOG2 の初期化
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # 初期設定
    ret, frame = cap.read()
    if not ret:
        print("動画を読み込めませんでした。")
        cap.release()
        exit()

    height, width, _ = frame.shape
    accumulated_movement = np.zeros((height, width), dtype=np.float32)

    # ヒートマップの生成（最初の5秒間）
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(fps * calc_heatmap_time)

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # 前景の抽出
        fgmask = fgbg.apply(frame)
        # 前景マスクを移動量に加算
        accumulated_movement += fgmask.astype(np.float32)

    # ヒートマップを正規化
    norm_movement = cv2.normalize(accumulated_movement, None, 0, 255, cv2.NORM_MINMAX)
    norm_movement = norm_movement.astype(np.uint8)

    cap.release()

    return width, height, norm_movement

def get_heatmap_gray_diff(video_path, calc_heatmap_time):
    # 動画の読み込み
    cap = cv2.VideoCapture(video_path)

    # フレームレートの取得
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 最初の5秒間のフレーム数を計算
    num_frames = int(fps * calc_heatmap_time)

    # 初期設定
    ret, frame1 = cap.read()
    if not ret:
        print("動画を読み込めませんでした。")
        cap.release()
        exit()

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    height, width = prvs.shape
    accumulated_movement = np.zeros((height, width), dtype=np.float32)

    # ヒートマップの生成
    for i in range(num_frames):
        ret, frame2 = cap.read()
        if not ret:
            break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # フレーム間の差分を計算
        flow = cv2.absdiff(prvs, next)
        accumulated_movement += flow
        prvs = next.copy()

    # ヒートマップを正規化
    norm_movement = cv2.normalize(accumulated_movement, None, 0, 255, cv2.NORM_MINMAX)
    norm_movement = norm_movement.astype(np.uint8)

    cap.release()
    
    return width, height, norm_movement
# 中央値探索の関数
def find_peaks(bottom_points, window_size, width, DIRECTION="LEFT"):
    decreasing = False
    peaks = []
    # bottom_pointsをX座標でソート
    bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
    # 間を補完したリスト
    filled_bottom_points = []

    # Xの最小値と最大値を取得
    x_min = bottom_points[0][0]
    x_max = bottom_points[-1][0]

    true_min_list = []
    true_max_list = []

    # 左端、右端までのデータを補完する処理
    for i in range(0, x_min):
        true_min_list.append([i, bottom_points[0][1]])
    for i in range(x_max+1, width):
        true_max_list.append([i, bottom_points[-1][1]])

    # X座標の飛びデータを補完する処理
    current_y = bottom_points[0][1]  # 最初のYを初期値とする
    peak_dict = {p[0]: p[1] for p in bottom_points}  # Xをキーとした辞書を作成

    for x in range(x_min, x_max + 1):
        if x in peak_dict:
            current_y = peak_dict[x]  # 既存データがあればそのYを使用
        filled_bottom_points.append(np.array([x, current_y]))

    # リストを結合する
    filled_bottom_points = true_min_list + filled_bottom_points + true_max_list
    filled_bottom_points = np.array(filled_bottom_points)
    x_coords = filled_bottom_points[:, 0]
    y_coords = filled_bottom_points[:, 1]

    for i in range(window_size, len(x_coords) - window_size - 1):
        # 左側の中央値を計算
        left_median = np.median(y_coords[i - window_size:i])
        # 右側の中央値を計算
        right_median = np.median(y_coords[i + 1:i + window_size + 1])
        # 判定
        if DIRECTION == "LEFT":
            
            if left_median - right_median <= 0:
                decreasing = True
                # peaks.append((x_coords[i], y_coords[i]))
            if left_median - right_median > 0:
                if decreasing:
                    peaks.append((x_coords[i], y_coords[i]))
                decreasing = False

    # if DIRECTION == "RIGHT":
    #     peaks = peaks[::-1]
    return peaks
    
def specify_convex(video_path, WAY="MOG2", calc_heatmap_time=5):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline

    if WAY == "MOG2":
        width, height, norm_movement = get_heatmap_mog2(video_path, calc_heatmap_time)
    elif WAY == "GRAY_DIFF":
        width, height, norm_movement = get_heatmap_gray_diff(video_path, calc_heatmap_time)

    # cv2.imshow("norm_movement", norm_movement)
    # cv2.imwrite("../2_data/ayu/auto_heatmap/norm_movement_mog2.jpg", norm_movement)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()

    # ヒートマップから高移動量の領域を二値化
    _, thresh = cv2.threshold(norm_movement, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imwrite("../2_data/ayu/auto_heatmap/thresh_otsu.jpg", thresh)
    # cv2.imshow('Thresh0', thresh)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()

    # モルフォロジー変換でノイズを除去し、堰の輪郭を明確にする
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # クロージング処理により、白い部分をつなげ、堰の輪郭を明確にする
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # オープニング処理により、ノイズを除去する
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite("../2_data/ayu/auto_heatmap/morphology_close_open.jpg", thresh)
    # cv2.imshow('Thresh', thresh)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()

    # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(norm_movement,(5,5),0)
    # ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow('Thresh3', th3)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    image = np.zeros_like(thresh)

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            cv2.drawContours(image, [contour], -1, 255, -1)
    # cv2.imwrite("../2_data/ayu/auto_heatmap/denoise_with_contours.jpg", image)
    # cv2.imshow('Contours', image)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()

    # 白い領域の座標を取得
    height, width = image.shape
    bottom_points = []

    for x in range(width):
        column = image[:, x]
        white_pixels = np.where(column == 255)[0]  # 白いピクセルのインデックス
        if len(white_pixels) > 0:
            bottom_y = white_pixels[-1]  # 最下端のy座標
            bottom_points.append((x, bottom_y))

    # 左から探索
    bottom_points = np.array(bottom_points)
    # x_coords = bottom_points[:, 0]
    # y_coords = bottom_points[:, 1]
    window_size = 50

    left_peaks = find_peaks(bottom_points, window_size, width, DIRECTION="LEFT")

    # # 右から探索（逆順）
    # right_peaks = find_peaks(x_coords, y_coords, window_size, DIRECTION="RIGHT")

    # 左右の結果を統合
    peaks = left_peaks# + right_peaks
    # Xの値でソート
    peaks = sorted(peaks, key=lambda p: p[0])
    # peaksの最小のX座標とその点におけるY座標を取得
    min_x = peaks[0][0]
    min_x_y = peaks[0][1]
    # peaksの最大のX座標とその点におけるY座標を取得
    max_x = peaks[-1][0]
    max_x_y = peaks[-1][1] # Xが最大の点におけるY座標。最大のY座標ではない。
    # bottom_pointsのmin_xより小さいX座標を持つもののうち、最大のY座標を取得
    left_bottoms_max_y = max([p[1] for p in bottom_points if p[0] < min_x])
    # left_bottoms_max_yが、min_x_yよりも大きい場合、peaksに追加
    if left_bottoms_max_y > min_x_y:
        # そのY座標と等しいY座標を持つ座標を取得
        left_bottoms = [b for b in bottom_points if b[1] == left_bottoms_max_y]
        for l in reversed(left_bottoms):
            peaks.insert(0, l)
        # peaks.insert(0, left_bottoms)
    
    # bottom_pointsのmax_xより大きいX座標を持つもののうち、最大のY座標とその点におけるX座標を取得
    right_bottoms_max_y = max([p[1] for p in bottom_points if p[0] > max_x])
    if right_bottoms_max_y > max_x_y:
        # そのY座標と等しいY座標を持つ座標を取得
        right_bottoms = [b for b in bottom_points if b[1] == right_bottoms_max_y]
        for r in (right_bottoms):
            peaks.append(r)
        # peaks.append(right_bottoms)
    # 最小のX座標のY座標を左端まで延長
    ymin_left = peaks[0][1]
    peaks.insert(0, (0, ymin_left))
    # 最大のX座標のY座標を右端まで延長
    ymin_right = peaks[-1][1]
    peaks.append((width - 1, ymin_right))

    peak_lines = bresenham(int(peaks[0][0]), int(peaks[0][1]), int(peaks[-1][0]), int(peaks[-1][1]))
    true_peaks = []
    for (x, y) in peak_lines:
        # bottom_pointsに存在し、線分のY座標よりも大きい場合は挿入 
        matching_points = [(bx, by) for (bx, by) in bottom_points if bx == x and by > y]
        if matching_points:
            # Yが最大の点を選択
            true_peaks.append(max(matching_points, key=lambda p: p[1]))
        else:
            # 一致する点がない場合はBresenhamの点をそのまま挿入
            true_peaks.append((x, y))







    # # 重複チェックとフィルタリング
    # filtered_peaks = []
    # seen_x = {}
    # for peak in bottom_points:
    #     x, y = peak
    #     # Xが初めて出現するか、または現在のYの方が大きければ更新
    #     if x not in seen_x or y > seen_x[x]:
    #         seen_x[x] = y

    # # 最終結果を取得
    # filtered_peaks = [np.array([x, y]) for x, y in seen_x.items()]

    # # Xで再ソート（必要に応じて）
    # filtered_peaks = sorted(filtered_peaks, key=lambda p: p[0])


    # # 出っ張りを画像に描画
    # output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # for peak in peaks:
    #     cv2.circle(output_image, peak, 10, (0, 0, 255), -1)  # 赤い点を描画

    # for i in range(len(true_peaks) - 1):
    #     x1, y1 = true_peaks[i]
    #     x2, y2 = true_peaks[i + 1]
    #     cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # # 結果を描画
    # cv2.imshow("Final Peaks", output_image)
    
    # cv2.waitKey(1)

    # 結果をプロット
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Original Binary Image")
    # plt.imshow(image, cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.title("Detected Peaks (Final Result)")
    # plt.imshow(output_image[..., ::-1])  # OpenCVのBGRをRGBに変換
    # plt.tight_layout()
    # plt.show()

    
    # cv2.destroyAllWindows()

    return width, height, true_peaks

# true_peaksを使用して解析範囲を絞る
def mask_analysis_region(video_path):
    width, height, true_peaks = specify_convex(video_path)
    # 1. true_peaksの最小のY座標以上の領域のみを解析
    min_y = min([y for _, y in true_peaks])
    # 白いframeを作成
    mask = np.full((height, width, 3), [255, 255, 255], dtype=np.uint8)

    # 2. true_peaks座標を結んで得られた領域より上をマスク
    true_peaks.insert(0, (0, 0))
    true_peaks.append((width - 1, 0))
    true_peaks_polygon = np.array([true_peaks], dtype=np.int32)  # ポリゴン形式に変換
    cv2.fillPoly(mask, true_peaks_polygon, (0, 0, 0))  # true_peaksで囲まれた領域を黒くする
    mask = mask[min_y:, :, :]  # 最小のY座標以上の領域のみを残す
    
    return min_y, mask




# def test(video_path, WAY="MOG2"):
#     import cv2
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy.interpolate import make_interp_spline

#     if WAY == "MOG2":
#         width, height, norm_movement = get_heatmap_mog2(video_path)
#     elif WAY == "GRAY_DIFF":
#         width, height, norm_movement = get_heatmap_gray_diff(video_path)

#     cv2.imshow("norm_movement", norm_movement)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # ヒートマップから高移動量の領域を二値化
#     _, thresh = cv2.threshold(norm_movement, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     cv2.imshow('Thresh0', thresh)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # モルフォロジー変換でノイズを除去し、堰の輪郭を明確にする
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     # クロージング処理により、白い部分をつなげ、堰の輪郭を明確にする
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     # オープニング処理により、ノイズを除去する
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#     cv2.imshow('Thresh', thresh)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Otsu's thresholding after Gaussian filtering
#     blur = cv2.GaussianBlur(norm_movement,(5,5),0)
#     ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     cv2.imshow('Thresh3', th3)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#     image = np.zeros_like(thresh)

#     for contour in contours:
#         if cv2.contourArea(contour) > 100:
#             cv2.drawContours(image, [contour], -1, 255, -1)

#     cv2.imshow('Contours', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     from scipy.ndimage import gaussian_filter1d

#     # 白い領域の最下端を取得
#     height, width = image.shape
#     bottom_points = []

#     for x in range(width):
#         column = image[:, x]
#         white_pixels = np.where(column == 255)[0]  # 白いピクセルのインデックス
#         if len(white_pixels) > 0:
#             bottom_y = white_pixels[-1]  # 最下端のy座標
#             bottom_points.append((x, bottom_y))

#     # 中央値の計算
#     bottom_points = np.array(bottom_points)
#     x_coords = bottom_points[:, 0]
#     y_coords = bottom_points[:, 1]

#     window_size = 50  # 左右10ピクセル分を使用
#     peaks = []  # 出っ張り部分を格納

#     for i in range(window_size, len(x_coords) - window_size - 1):
#         # 左側の中央値を計算
#         left_median = np.median(y_coords[i - window_size:i + 1])
#         # 右側の中央値を計算
#         right_median = np.median(y_coords[i + 1:i + window_size + 1])
#         # 判定
#         if left_median - right_median > 0 and y_coords[i - 1] - y_coords[i] >= 0:
#             peaks.append((x_coords[i], y_coords[i]))

#     # 出っ張りを画像に描画
#     output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     for peak in peaks:
#         cv2.circle(output_image, peak, 5, (0, 0, 255), -1)  # 赤い点を描画

#     # 出っ張り間を線で結ぶ
#     for i in range(1, len(peaks)):
#         cv2.line(output_image, peaks[i - 1], peaks[i], (0, 255, 0), 2)  # 緑の線で結ぶ

#     # 結果をプロット
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Original Binary Image")
#     plt.imshow(image, cmap='gray')
#     plt.subplot(1, 2, 2)
#     plt.title("Detected Peaks with Median Comparison")
#     plt.imshow(output_image[..., ::-1])  # OpenCVのBGRをRGBに変換
#     plt.tight_layout()
#     plt.show()








# def mask_with_weir(video_path, WAY="MOG2"):
#     if WAY == "MOG2":
#         width, height, norm_movement = get_heatmap_mog2(video_path)
#     elif WAY == "GRAY_DIFF":
#         width, height, norm_movement = get_heatmap_gray_diff(video_path)

#     cv2.imshow("norm_movement", norm_movement)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # ヒートマップから高移動量の領域を二値化
#     _, thresh = cv2.threshold(norm_movement, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     cv2.imshow('Thresh0', thresh)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # モルフォロジー変換でノイズを除去し、堰の輪郭を明確にする
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     # クロージング処理により、白い部分をつなげ、堰の輪郭を明確にする
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     # オープニング処理により、ノイズを除去する
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#     cv2.imshow('Thresh', thresh)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Otsu's thresholding after Gaussian filtering
#     blur = cv2.GaussianBlur(norm_movement,(5,5),0)
#     ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     cv2.imshow('Thresh3', th3)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#     image = np.zeros_like(thresh)

#     for contour in contours:
#         if cv2.contourArea(contour) > 100:
#             cv2.drawContours(image, [contour], -1, 255, -1)

#     cv2.imshow('Contours', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



#     import matplotlib.pyplot as plt
#     image = thresh

#         # 白い部分（堰の高移動量部分）を検出
#     _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

#     # 下端の検出
#     height, width = binary.shape
#     bottom_points = []

#     # 各列（x方向）について、白い部分の最下端（y座標）を取得
#     for x in range(width):
#         column = binary[:, x]
#         white_pixels = np.where(column == 255)[0]  # 白いピクセルのインデックス
#         if len(white_pixels) > 0:
#             bottom_y = white_pixels[-1]  # 最下端のy座標
#             bottom_points.append((x, bottom_y))

#     # 下端を結ぶ曲線を生成
#     bottom_points = np.array(bottom_points)
#     line_mask = np.zeros_like(binary)

#     # 曲線（堰の位置）を描画
#     for i in range(1, len(bottom_points)):
#         x1, y1 = bottom_points[i - 1]
#         x2, y2 = bottom_points[i]
#         cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)

#     cv2.imshow('Line Mask', line_mask)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#         # # 上流部と下流部を二分するマスクを作成
#         # upper_mask = np.zeros_like(binary)
#         # for i in range(len(bottom_points) - 1):
#         #     x1, y1 = bottom_points[i]
#         #     x2, y2 = bottom_points[i + 1]
#         #     cv2.line(upper_mask, (x1, y1), (x2, y2), 255, 1)

#         # # 堰の下側を黒で塗りつぶす
#         # upper_mask = cv2.fillPoly(upper_mask, [bottom_points], 255)
#         # lower_mask = cv2.bitwise_not(upper_mask)

#         # # 上流部と下流部の画像を作成
#         # upper_region = cv2.bitwise_and(image, image, mask=upper_mask)
#         # lower_region = cv2.bitwise_and(image, image, mask=lower_mask)
        
#         # # 結果を表示
#         # plt.figure(figsize=(10, 5))
#         # plt.subplot(1, 3, 1)
#         # plt.title("Original Image")
#         # plt.imshow(binary, cmap='gray')
#         # plt.subplot(1, 3, 2)
#         # plt.title("Upper Region (Upstream)")
#         # plt.imshow(upper_region, cmap='gray')
#         # plt.subplot(1, 3, 3)
#         # plt.title("Lower Region (Downstream)")
#         # plt.imshow(lower_region, cmap='gray')
#         # plt.tight_layout()
#         # plt.show()
#         # plt.close()

#     # # 輪郭の検出
#     # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # # 面積が最大の輪郭を堰とみなす
#     # max_contour = max(contours, key=cv2.contourArea)

#     # # 堰のマスクを作成
#     # weir_mask = np.zeros((height, width), dtype=np.uint8)
#     # cv2.drawContours(weir_mask, [max_contour], -1, 255, -1)

#     # # 上流部と下流部を分離するマスクを作成
#     # # マスクをコピーして、堰の上流側を白、下流側を黒にする
#     # mask = np.zeros((height, width), dtype=np.uint8)
#     # # 堰の輪郭の上側を白（255）に塗りつぶす
#     # cv2.fillPoly(mask, pts=[np.array([[0, 0], [width, 0], [width, height], [0, height]])], color=255)
#     # cv2.drawContours(mask, [max_contour], -1, 0, -1)

#     # # マスクを適用して上流部を抽出
#     # cap = cv2.VideoCapture(video_path)

#     # while True:
#     #     ret, frame = cap.read()
#     #     if not ret:
#     #         break

#     #     # 上流部を抽出
#     #     upstream = cv2.bitwise_and(frame, frame, mask=mask)

#     #     # 上流部内の不要な動きを検出し、マスクで除去
#     #     # 必要に応じて追加の処理を実装

#     #     # 結果を表示
#     #     cv2.imshow('Upstream', upstream)
#     #     if cv2.waitKey(1) & 0xFF == ord('q'):
#     #         break

#     # cap.release()
#     # cv2.destroyAllWindows()




# def mask_with_mog2(video_path):
#     # 動画の読み込み
#     cap = cv2.VideoCapture(video_path)

#     # MOG2 の初期化
#     fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

#     # 初期設定
#     ret, frame = cap.read()
#     if not ret:
#         print("動画を読み込めませんでした。")
#         cap.release()
#         exit()

#     height, width, _ = frame.shape
#     accumulated_movement = np.zeros((height, width), dtype=np.float32)

#     # ヒートマップの生成（最初の5秒間）
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     num_frames = int(fps * 5)

#     for i in range(num_frames):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # 前景の抽出
#         fgmask = fgbg.apply(frame)
#         # 前景マスクを移動量に加算
#         accumulated_movement += fgmask.astype(np.float32)

#     # ヒートマップを正規化
#     norm_movement = cv2.normalize(accumulated_movement, None, 0, 255, cv2.NORM_MINMAX)
#     norm_movement = norm_movement.astype(np.uint8)

#     # ヒートマップを表示（必要に応じて）
#     # cv2.imshow('Heatmap', norm_movement)
#     # cv2.waitKey(0)

#     # 高移動量の領域を二値化
#     _, thresh = cv2.threshold(norm_movement, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # マスクの作成
#     mask = np.ones((height, width), dtype=np.uint8) * 255  # 初期値は白（255）
#     mask[thresh == 255] = 0  # 高移動量の領域を黒（0）に設定

#     # マスクを適用して下流部を除外した動画を解析
#     cap.release()
#     cap = cv2.VideoCapture(video_path)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # マスクを適用
#         masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
#         # ここでオブジェクトトラッキングの処理を実行
#         # 例: cv2.Cannyを使ってエッジ検出
#         # edges = cv2.Canny(cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY), 50, 150)
#         # 処理結果を表示
#         cv2.imshow('Masked Frame', masked_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# def mask_with_gray_diff(video_path):
#     # 動画の読み込み
#     cap = cv2.VideoCapture(video_path)

#     # フレームレートの取得
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # 最初の5秒間のフレーム数を計算
#     num_frames = int(fps * 5)

#     # 初期設定
#     ret, frame1 = cap.read()
#     if not ret:
#         print("動画を読み込めませんでした。")
#         cap.release()
#         exit()

#     prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#     height, width = prvs.shape
#     accumulated_movement = np.zeros((height, width), dtype=np.float32)

#     # ヒートマップの生成
#     for i in range(num_frames):
#         ret, frame2 = cap.read()
#         if not ret:
#             break
#         next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#         # フレーム間の差分を計算
#         flow = cv2.absdiff(prvs, next)
#         accumulated_movement += flow
#         prvs = next.copy()

#     # ヒートマップを正規化
#     norm_movement = cv2.normalize(accumulated_movement, None, 0, 255, cv2.NORM_MINMAX)
#     norm_movement = norm_movement.astype(np.uint8)

#     # ヒートマップを表示（必要に応じて）
#     # cv2.imshow('Heatmap', norm_movement)
#     # cv2.waitKey(0)

#     # 高移動量の領域を二値化
#     _, thresh = cv2.threshold(norm_movement, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # マスクの作成
#     mask = np.ones((height, width), dtype=np.uint8) * 255  # 初期値は白（255）
#     mask[thresh == 255] = 0  # 高移動量の領域を黒（0）に設定

#     # マスクを適用して下流部を除外した動画を解析
#     cap.release()
#     cap = cv2.VideoCapture(video_path)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # マスクを適用
#         masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
#         # ここでオブジェクトトラッキングの処理を実行
#         # 例: cv2.Cannyを使ってエッジ検出
#         # edges = cv2.Canny(cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY), 50, 150)
#         # 処理結果を表示
#         cv2.imshow('Masked Frame', masked_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()