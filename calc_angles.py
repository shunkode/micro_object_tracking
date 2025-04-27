import numpy as np


def calc_angles_n_classify(save_trackers_l, direction_thres, coords_thres, speed_thres, dist_sum_thres, frame_n_thres, pred_l=None):
    save_l = []
    save_pred_l = []
    for n, s in enumerate(save_trackers_l):
        results_l = []
        # 各IDの初期位置を格納
        results_l.append(list(s[0] + (np.nan, np.nan, np.nan, np.nan, np.nan)))
        id = s[0][0]  # IDを取得
        # x, y座標のリストを抽出
        x_coords = np.std([t[1] for t in s])
        y_coords = np.std([t[2] for t in s])
        xy_coods = np.sqrt(x_coords**2 + y_coords**2)
        # 
        frame_n_min = s[0][6]
        frame_n_max = s[-1][6]
        angle_list = []
        distance_sum = 0
        for i in range(1, len(s)):
            # ベクトルのx, y成分を計算
            x1, y1 = s[i - 1][1], s[i - 1][2]
            x2, y2 = s[i][1], s[i][2]
            dx = x2 - x1
            dy = y2 - y1

            distance = np.sqrt(dx**2 + dy**2)
            
            # ベクトルの角度を計算
            angle = np.arctan2(dy, dx)  # ラジアンで角度を返す
            angle_list.append(angle)

            results_l.append([id, s[i][1], s[i][2], s[i][3], s[i][4], s[i][5], s[i][6], angle, x_coords, y_coords, xy_coods, distance])

            distance_sum += distance

        # 角度の平均を計算
        average_angle = np.mean(angle_list)

        # クラス分け
        # 移動量の標準偏差によるクラス分け
        if xy_coods < coords_thres:
            for r in results_l:
                r.append("noise")
            # 予測結果がある場合
            if pred_l is not None:
                pred_classified_l = [p + ("noise",) for p in pred_l[n]]

        # # １フレームあたりの移動量によるクラス分け
        # elif distance_sum / (frame_n_max - frame_n_min) < speed_thres:
        #     for r in results_l:
        #         r.append("noise")
        #     # 予測結果がある場合
        #     if save_pred_l is not None:
        #         pred_classified_l = [p + ("noise",) for p in save_pred_l[n]]

        # 移動距離の合計によるクラス分け
        elif distance_sum < dist_sum_thres:
            for r in results_l:
                r.append("noise")
            # 予測結果がある場合
            if pred_l is not None:
                pred_classified_l = [p + ("noise",) for p in pred_l[n]]

        elif (frame_n_max - frame_n_min) <  frame_n_thres:
            for r in results_l:
                r.append("noise")
            # 予測結果がある場合
            if pred_l is not None:
                pred_classified_l = [p + ("noise",) for p in pred_l[n]]
                
        # 角度によるクラス分け
        else:
            for class_name, thres_value in direction_thres.items():
                for thres in thres_value:
                    if thres[0] <= average_angle <= thres[1]:
                        for r in results_l:
                            r.append(class_name)
                        # 予測結果がある場合
                        if pred_l is not None:
                            pred_classified_l = [p + (class_name,) for p in pred_l[n]]
                            
        save_l.append(results_l)
        if pred_l is not None:
            save_pred_l.append(pred_classified_l)
        

        # 平均速度を計算
        # average_speed = np.mean([s[2] for s in speed_list]) if speed_list else 0
    

    if pred_l is None:
        return save_l
    else:
        return save_l, save_pred_l




if __name__ == '__main__':
    from set_flow_direction import FlowDirectionDrawer, determine_direction_thres
    import sys
    input_video_path = "../1_data_raw/ayu_clipped.mp4"
    drawer = FlowDirectionDrawer()
    flow_direction_rad = drawer.set_flow_direction_manually(input_video_path)
    direction_thres={"fish": [(-180, -90), (90, 180)], "debris": [(-90, 90)]}
    if flow_direction_rad is not None:
        direction_thres = determine_direction_thres(direction_thres, flow_direction_rad)
    else:
        sys.exit("Flow direction is not set. Please try again.")
    # サンプルデータ
    save_trackers_l = [
        [(1, 1, 1, 1, 9, 9, 9), (1, 2, 3, 2, 9, 9, 9), (1, 3, 5, 3, 9, 9, 9)],  # ID1のデータ
        [(2, 4, 4, 1, 9, 9, 9), (2, 5, 5, 2, 9, 9, 9), (2, 6, 6, 4, 9, 9, 9)]   # ID2のデータ
    ]
    # 実行
    result = calc_angles_n_classify(save_trackers_l, direction_thres, coords_thres=1)
    print(result)