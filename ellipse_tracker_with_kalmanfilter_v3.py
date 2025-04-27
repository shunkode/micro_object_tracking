"""
変更点
・割当問題(linear_sum_assignment)の箇所がIDスイッチングの原因となっていたため、修正
・新たに検出したオブジェクトと、既にトラッカーが存在するオブジェクトとの距離が一定以下の場合、新しいトラッカーを生成しないように修正
"""

import cv2
import numpy as np
from tqdm import tqdm
from bresenham import bresenham
import pandas as pd
import csv
import sys
import tkinter as tk
from tkinter import simpledialog
import os
from scipy.optimize import linear_sum_assignment

from decorator import error_handler

from mask_video import mask_analysis_region
from set_flow_direction import FlowDirectionDrawer, determine_direction_thres
from calc_angles import calc_angles_n_classify
from set_roi_polygon import set_roi_polygon
from add_ellipses_to_video import add_ellipses_to_video
from confirm_filename import check_file_exists
from calc_iou import classify_based_on_iou


class Tracker:
    count = 0  # トラッカーのユニークな ID を生成するためのクラス変数

    def __init__(self, initial_state, fps):#, initial_frame_n):
        self.id = Tracker.count
        Tracker.count += 1
        self.kf = cv2.KalmanFilter(6, 2)
        dt = 1/fps  # フレーム間隔（フレームレートが一定の場合）
        # 状態遷移行列 A、観測行列 Hの設定
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]], np.float32)
        # プロセスノイズ共分散 Q の設定
        q = 1
        self.kf.processNoiseCov = q * np.array([
    [(dt**4)/4, 0, (dt**3)/2, 0, (dt**2)/2, 0],
    [0, (dt**4)/4, 0, (dt**3)/2, 0, (dt**2)/2],
    [(dt**3)/2, 0, dt**2, 0, dt, 0],
    [0, (dt**3)/2, 0, dt**2, 0, dt],
    [(dt**2)/2, 0, dt, 0, 1, 0],
    [0, (dt**2)/2, 0, dt, 0, 1]], dtype=np.float32)
        # self.kf.processNoiseCov[:5, :5] *= 0.01
        # 観測ノイズ共分散 R の設定
        r = 25  # 観測ノイズの分散（標準偏差が5ピクセルの場合）
        self.kf.measurementNoiseCov = np.array([
        [r, 0],
        [0, r]], dtype=np.float32)
        # self.kf.measurementNoiseCov[2:, 2:] *= 10.0
        # 初期誤差共分散 P の設定
        P = np.array([
    [1,    0,    0,    0,    0,    0],
    [0,    1,    0,    0,    0,    0],
    [0,    0, 1000,    0,    0,    0],
    [0,    0,    0, 1000,    0,    0],
    [0,    0,    0,    0, 1000,    0],
    [0,    0,    0,    0,    0, 1000]], dtype=np.float32)
        # P[5:, 5:] *= 1000.0
        P[4, 4] *= 1000.0
        # P[4:, 4:]にすると、魚の方向転換に対応できなくなる
        # （魚は一度その場で停止してから逆方向に泳ぐことがあるが、P[4:, 4:]にすると、予測がそのまま通り過ぎてしまう）
        # P[5:, 5:] *= 1000.0をしないと、魚のトラッキングが追いつかない
        # P *= 10.0
        self.kf.errorCovPost = P
        # 初期状態の設定
        self.kf.statePost = initial_state
        # self.time_since_update = 0  # 観測されていないフレーム数
        self.hit_streak = 0  # 連続して観測された回数
        self.age = 0  # トラッカーの年齢（フレーム数）

    def predict(self):
        self.kf.predict()
        self.age += 1
        # self.time_since_update += 1
        # 予測された状態を返す
        return self.kf.statePre  

    def update(self, measurement):
        # self.time_since_update = 0
        self.age = 0
        self.hit_streak += 1
        self.kf.correct(measurement)

    def get_state(self):
        # 更新された状態を返す
        return self.kf.statePost  
    

    
def associate_detections_to_trackers(detections, trackers, dist_threshold=50):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0), dtype=int)

    # 距離行列の計算
    dist_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    # trk_id = []
    for d, det in enumerate(detections):
        # trk_id.append([])
        for t, trk in enumerate(trackers):
            # trk_id[d].append(trk.id)

            predicted_state = trk.kf.statePre
            predicted_x, predicted_y = predicted_state[0, 0], predicted_state[1, 0]

            dist = np.linalg.norm(det - np.array([predicted_x, predicted_y]))
            dist_matrix[d, t] = dist

    # 距離に基づくマッチング
    dist_matrix[dist_matrix > dist_threshold] = 10000000  # 閾値以上の距離は非常に大きな値にする（不適な割当を防ぐため）
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    matches = []
    unmatched_detections = []
    unmatched_trackers = []

    for d in range(len(detections)):
        if d not in row_ind:
            unmatched_detections.append(d)

    for t in range(len(trackers)):
        if t not in col_ind:
            unmatched_trackers.append(t)

    # 閾値以下の距離のみをマッチとする
    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] > dist_threshold:
            unmatched_detections.append(r)
            unmatched_trackers.append(c)
        else:
            matches.append([r, c])

    matches = np.array(matches)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



def output_to_csv(output_csv_path, save_trackers_l):
    with open(output_csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)  # すべてのデータを""で囲む
        # データを書く
        for sublist in save_trackers_l:
            writer.writerows(sublist)

@error_handler
def ellipse_tracker_with_kalmanfilter(input_video_path, 
output_video_path, 
output_csv_path, 
contour_thres=20, 
dist_threshold=50, 
max_age=10, 
min_hits=3, 
min_dist_to_matched=10, 
# HEATMAP=True, 
# SHOW_MOG2=False, 
mog2_path=None, # MOG2の動画を保存する場合は、保存先のパスを指定。指定しなければ保存しない。
ROI=None, 
specific_class:list =None,
class_colors:dict ={"fish": (255, 0, 0), "debris": (0, 255, 0), "noise": (0, 0, 255)}, 
OUTPUT_PREDICTION=False):
    if not os.path.exists(input_video_path):
        sys.exit(f"File not found: {input_video_path}")
    output_video_path = check_file_exists(output_video_path)
    output_csv_path = check_file_exists(output_csv_path)

    # 既存のコード...
    save_trackers_l = []
    result_tracking = []
    trackers_list = []  # トラッカーを保持するリスト
    pred_l = []
    
    # max_age = 10  # 観測されなくなってからトラッカーを削除するまでのフレーム数
    # min_hits = 3  # トラッカーを確定するための最低観測回数

    # ROI
    if ROI=="AUTO":
        min_y, mask = mask_analysis_region(input_video_path)
    elif ROI=="RECT":
        roi_x = None
    elif ROI=="POLY":
        points, offset, frame_height, frame_width = set_roi_polygon(input_video_path)
        adjusted_points = [np.array(points, np.int32) - np.array(offset)]
    elif ROI is None:
        pass
    else:
        sys.exit("Please set ROI to AUTO, RECT, or POLY.")

    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if mog2_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_mog2 = cv2.VideoWriter(mog2_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    current_frame_n = 0

    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)  # すべてのデータを""で囲む
        # ヘッダーを書く
        writer.writerow(["id", "x", "y", "MA", "ma", "ellipse_angle", "frame_n", "mean_color", "median_color"])#, "vector_angle", "x_coords", "y_coords", "xy_coords", "distance", "class"])

    if OUTPUT_PREDICTION:
        output_pred_csv_path = output_csv_path.replace(".csv", "_pred.csv")
        with open(output_pred_csv_path, mode="w", newline="") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)  # すべてのデータを""で囲む
            # ヘッダーを書く
            writer.writerow(["id", "x", "y", "frame_n"])#, "class"])

    with tqdm(total=total_frames, desc="Analyzing frames", unit="frame") as pbar:
        while cap.isOpened():
            result_current_frame = []
            save_pred_l = []
            
            # フレームの読み込みと前処理
            ret, frame = cap.read()
            if not ret:
                break
            if ROI=="AUTO":
                frame = frame[min_y:, :]
                frame = cv2.bitwise_and(frame, mask)
            elif ROI=="RECT":
                if roi_x is None:
                    roi_x, roi_y, roi_w, roi_h = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=False)
                frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            elif ROI=="POLY":
                # マスクを作成
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                
                cv2.fillPoly(mask, adjusted_points, 255)
                frame = cv2.bitwise_and(frame, frame, mask=mask)
            else:
                pass

            fg_mask = backSub.apply(frame)

            if current_frame_n < 10:
                current_frame_n += 1
                pbar.update(1)
                continue

            if mog2_path is not None:
                # 白い背景の画像を作成
                contours_img = np.zeros_like(fg_mask, dtype=np.uint8)

            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # オブジェクトの検出（現在の観測値のリストを作成）
            detections = []
            for contour in contours:
                # 楕円フィッティングと中心座標の取得
                if len(contour) >= 5 and cv2.contourArea(contour) > contour_thres:
                    if mog2_path is not None:
                        cv2.drawContours(contours_img, [contour], -1, (255), 2)

                    ellipse = cv2.fitEllipse(contour)
                    (x, y), (MA, ma), angle = ellipse

                    # x や y が NaN でないことを確認
                    if np.isnan(x) or np.isnan(y) or np.isnan(MA) or np.isnan(ma) or np.isnan(angle) or MA >50 or ma > 50 or MA < 1 or ma < 1:#MA == 0 or ma == 0:
                        continue  # NaN の場合はこの楕円をスキップ
                    
                    x = float(round(x, 2))
                    y = float(round(y, 2))

                    ## 楕円内の色の平均値と中央値を計算
                    # 楕円内のマスクを作成
                    mask = np.zeros_like(frame[:, :, 0])  # 同じサイズの空のマスク
                    cv2.ellipse(mask, ellipse, 255, -1)  # 楕円内を白で塗りつぶす
                    # 楕円内の元画像のピクセル値を取得
                    color_info = frame[mask == 255]
                    # 色の平均値を計算
                    mean_color = np.mean(color_info, axis=0)
                    mean_color = np.floor(mean_color).astype(int)
                    # 色の中央値を計算
                    median_color = np.median(color_info, axis=0)

                    # ...
                    detections.append(np.array([x, y], dtype=np.float32))

                    result_current_frame.append((MA, ma, angle, current_frame_n, mean_color, median_color))
            
            if mog2_path is not None:
                out_mog2.write(contours_img)

            # トラッカーの予測ステップ
            for i, trk in enumerate(trackers_list):
                trk.predict()
                if OUTPUT_PREDICTION:
                    # 予測値を保存
                    pred_l[i].append((trk.id, trk.get_state()[0, 0], trk.get_state()[1, 0], current_frame_n))

            # 観測値とトラッカーの関連付け
            matches, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(detections, trackers_list, dist_threshold)
            
            # マッチしたトラッカーの更新
            for m in matches:
                trk = trackers_list[m[1]]
                measurement = detections[m[0]].reshape(2, 1)

                trk.update(measurement)

                id = trk.id
                (x, y) = trk.get_state()[:2].ravel()
                save_tuple = (id, x, y) + result_current_frame[m[0]]
                result_tracking[m[1]].append(save_tuple)

            
            # マッチしなかった観測値から新しいトラッカーを生成
            for idx in unmatched_detections:
                for trk in trackers_list:
                    # if trk.age == 0:  # 観測値で更新されているトラッカーに対してのみ距離を計算
                    #if trk.time_since_update == 0:  # 観測値で更新されているトラッカーに対してのみ距離を計算
                        dist_to_matched = np.linalg.norm(
                            detections[idx] - np.array([trk.get_state()[0, 0], trk.get_state()[1, 0]])
                        )
                        if dist_to_matched < min_dist_to_matched:
                            break
                # for m in matches:
                #     dist_to_matched = np.linalg.norm(detections[idx] - np.array([trk.get_state()[0, 0], trk.get_state()[1, 0]]))# - detections[m[0]])
                #     if dist_to_matched < min_dist_to_matched:
                #         # 10ピクセル以内にマッチした観測値があるので、新しいトラッカーを作らない
                #         break

                else:
                    measurement = detections[idx].reshape(2, 1)
                    initial_state = np.array([
                        [measurement[0, 0]],
                        [measurement[1, 0]],
                        [0],
                        [0],
                        [0],
                        [0]], dtype=np.float32)
                    new_trk = Tracker(initial_state, fps)
                    trackers_list.append(new_trk)
                    save_tuple = (new_trk.id, new_trk.get_state()[0, 0], new_trk.get_state()[1, 0]) + result_current_frame[idx]
                    result_current_frame[idx]
                    result_tracking.append([save_tuple])
                    if OUTPUT_PREDICTION:
                        pred_l.append([(new_trk.id, new_trk.get_state()[0, 0], new_trk.get_state()[1, 0], current_frame_n)])

            # マッチしなかったトラッカーの処理
            for idx in sorted(unmatched_trackers, reverse=True):  # インデックスエラーを防ぐため逆順に処理
                trk = trackers_list[idx]
                # if trk.time_since_update > max_age:
                if trk.age > max_age:
                    if trk.hit_streak < min_hits:
                        trackers_list.pop(idx)
                        result_tracking.pop(idx)
                        if OUTPUT_PREDICTION:
                            pred_l.pop(idx)
                    else:
                        trackers_list.pop(idx)
                        save_trackers_l.append(result_tracking.pop(idx))
                        if OUTPUT_PREDICTION:
                            save_pred_l.append(pred_l.pop(idx))

                # else:
                    #trk.time_since_update += 1
                    # trk.age += 1


            # 角度計算とクラス分け
            if not OUTPUT_PREDICTION:
                # save_trackers_l = calc_angles_n_classify(save_trackers_l, direction_thres, coords_thres, speed_thres, dist_sum_thres, frame_n_thres)
                output_to_csv(output_csv_path, save_trackers_l)
            else:
                # save_trackers_l, save_pred_l = calc_angles_n_classify(save_trackers_l, direction_thres, coords_thres, speed_thres, dist_sum_thres, frame_n_thres, save_pred_l)
                output_to_csv(output_csv_path, save_trackers_l)

                output_to_csv(output_pred_csv_path, save_pred_l)

            save_trackers_l = []

            # 結果の描画や DataFrame への保存
            # ...
            current_frame_n += 1
            pbar.update(1)

    cap.release()
    if mog2_path is not None:
        out_mog2.release()

    # classify_based_on_iou(output_csv_path, output_pred_csv_path, iou_thres=0.5, )

    if ROI=="AUTO":
        roi_args = [min_y, mask]
    elif ROI=="RECT":
        roi_args = [roi_x, roi_y, roi_w, roi_h]
    elif ROI=="POLY":
        roi_args = [points, offset, adjusted_points, frame_height, frame_width]
    else:
        roi_args = None
    # トラッキング結果の描画
    add_ellipses_to_video(input_video_path, 
                          input_csv_path=output_csv_path,
                          output_video_path=output_video_path,
                          ROI=ROI, 
                          roi_args=roi_args, 
                          specific_class=specific_class, 
                          class_colors=class_colors, 
                          pred_csv_path=output_pred_csv_path)
    
    from output_csv import extract_rgb_components
    extract_rgb_components(output_csv_path, output_csv_path)

if __name__ == '__main__':
    import time
    contour_thres = 3
    coord_thres = 15
    dist_threshold = 50
    max_age = 10
    min_hits = 3
    min_dist_to_matched =50
    OUTPUT_PREDICTION=True
    if OUTPUT_PREDICTION:
        pred = "withpred"
    else:
        pred = None
    start = time.time()
    ellipse_tracker_with_kalmanfilter(
        input_video_path="../1_data_raw/ayu.mp4", 
        output_video_path=f"../2_data/kalmanfilter_v3/ayu/manual_roi/ayu/config2/c{contour_thres}_d{dist_threshold}_a{max_age}_h{min_hits}_md{min_dist_to_matched}_largeletter/c{contour_thres}_d{dist_threshold}_a{max_age}_h{min_hits}_md{min_dist_to_matched}.mp4", 
        # output_video_path=f"../2_data/kalmanfilter_v3/2_reduce_saturation/config3/manual_roi/ayu_all_manualroi_c{contour_thres}_d{dist_threshold}_co{coord_thres}_a{max_age}_h{min_hits_history500}_s{speed_thres}_ds{dist_sum_thres}_f{frame_n_thres}_dir{d}_{pred}.mp4", 
        output_csv_path=f"../2_data/kalmanfilter_v3/ayu/manual_roi/ayu/config2/c{contour_thres}_d{dist_threshold}_a{max_age}_h{min_hits}_md{min_dist_to_matched}_largeletter/c{contour_thres}_d{dist_threshold}_a{max_age}_h{min_hits}_md{min_dist_to_matched}.csv", 
        # output_csv_path=f"../2_data/kalmanfilter_v3/ayu/manual_roi/ayu_all_manualroi_c{contour_thres}_d{dist_threshold}_co{coord_thres}_a{max_age}_h{min_hits}_s{speed_thres}_ds{dist_sum_thres}_f{frame_n_thres}_dir{d}_{pred}.csv", 
        contour_thres = contour_thres,
        dist_threshold = dist_threshold,
        max_age = max_age,
        min_hits = min_hits,
        min_dist_to_matched = min_dist_to_matched, 
        ROI="POLY",  # 自動で ROI を設定する場合は、"AUTO", 手動で設定する場合は "RECT" または "POLY"
        mog2_path="C:/Users/koder/OneDrive - Tokyo University of Agriculture and Technology/0_research/202407_back_diff/2_data/kalmanfilter_v3/ayu/manual_roi/ayu/config2/c3_d50_a10_h3_md50_largeletter/c3_mog2.avi",
        OUTPUT_PREDICTION=True
        )
    finish = time.time()
    print(f"Finished in {finish-start:.2f} seconds.")