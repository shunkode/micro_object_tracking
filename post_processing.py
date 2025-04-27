import pandas as pd
import numpy as np
from calc_iou import classify_based_on_iou
from calc_iou import calc_iou_from_csv

def calc_parameters(input_csv_path, 
                    output_csv_path=None, 
                    unique_id_output_csv_path=None, 
                    pred_csv_path=None, 
                    radius=3, 
                    fps=60):
    """
    IDごとにベクトルの計算を行い、signed_magnitudeを算出する。
    """

    def vector_with_shift(group):
        """
        各IDグループ内でベクトル計算を実施。
        """
        # 次の行を用いてベクトルを計算（shiftを適用）
        group["x2"] = group["x"].shift(1)
        group["y2"] = group["y"].shift(1)

        # ベクトルの計算
        dx = group["x"] - group["x2"]
        dy = group["y"] - group["y2"]
        # 1つ前のフレームとのフレーム番号の差分を計算
        # group["frame_diff"] = np.where(group["frame_n"].shift(1).isna(), 
        #                                np.nan, 
        #                                group["frame_n"] - group["frame_n"].shift(1))
        magnitude = np.sqrt(dx**2 + dy**2)
        signed_magnitude = np.where(dy >= 0, magnitude, -magnitude)  # y軸方向で正負を決定
        # 結果をグループに追加
        group["distance"] = magnitude
        group["signed_magnitude"] = signed_magnitude

        # 角度の計算
        angle = np.arctan2(dy, dx)
        group["vector_angle"] = angle

        # ベクトル角度差の計算
        group["prev_vector_angle"] = group["vector_angle"].shift(1)
        group["vector_angle_difference"] = np.where(group["prev_vector_angle"].isna(), 
                                             np.nan, 
                                             np.abs(group["vector_angle"] - group["prev_vector_angle"]))

        # 角度差が180度を超える場合の補正
        group["vector_angle_difference"] = np.where(group["vector_angle_difference"] > np.pi, 
                                             2 * np.pi - group["vector_angle_difference"], 
                                             group["vector_angle_difference"])

        return group
    
    # 検出開始時間を計算する関数
    def calc_per_id(df, output_csv_path=None, fps=60):
        df_unique_id = df.drop_duplicates(subset="id").copy()
        # df_unique_id.drop(["MA", "ma", "ellipse_angle", "vector_angle", "distance"], axis=1, inplace=True)
        df_unique_id["detected_time(s)"] = df_unique_id["frame_n"]/fps
        df_unique_id.rename(columns={"frame_n": "detected_frame_n"}, inplace=True)

        # フレーム長を計算し、対応するdf_unique_idにマージ
        frame_range_per_id = df.groupby("id")["frame_n"].agg(["min", "max"]).reset_index()
        frame_range_per_id["frame_range"] = frame_range_per_id["max"] - frame_range_per_id["min"] + 1
        df_unique_id = pd.merge(df_unique_id, frame_range_per_id[["id", "frame_range"]], on="id", how="left")

        # # df_unique_id["average_magnitude"] = df.groupby("id")["signed_magnitude"].mean()
        # # 各IDごとの平均magnitudeを計算し、対応するdf_unique_idにマージ
        # average_magnitude_per_id = df.groupby("id")[["distance", "signed_magnitude", "vector_angle"]].mean().reset_index()
        # average_magnitude_per_id.rename(columns={"distance": "average_distance(speed)", "signed_magnitude": "average_magnitude", "vector_angle": "average_vector_angle"}, inplace=True)
        # average_magnitude_per_idをマージ
        # df_unique_id = pd.merge(df_unique_id, average_magnitude_per_id, on="id", how="left")

        # xとyの標準偏差を計算し、対応するdf_unique_idにマージ
        std_xy_per_id = df.groupby("id")[["x", "y"]].std().reset_index()
        std_xy_per_id.rename(columns={"x": "x_coords", "y": "y_coords"}, inplace=True)
        df_unique_id = pd.merge(df_unique_id, std_xy_per_id, on="id", how="left")
        df_unique_id["xy_coords"] = np.sqrt(df_unique_id["x_coords"]**2 + df_unique_id["y_coords"]**2)

        # 移動量合計（distance_sum)を計算し、対応するdf_unique_idにマージ
        distance_sum_per_id = df.groupby("id")[["distance", "signed_magnitude"]].sum().reset_index()
        distance_sum_per_id.rename(columns={"distance": "distance_sum", "signed_magnitude": "signed_magnitude_sum"}, inplace=True)
        df_unique_id = pd.merge(df_unique_id, distance_sum_per_id, on="id", how="left")

        # 1フレームごとの、移動量と符号付きベクトルの移動量を計算する。
        df_unique_id["average_distance(speed)"] = df_unique_id["distance_sum"] / (df_unique_id["frame_range"] - 1)
        df_unique_id["average_signed_magnitude"] = df_unique_id["signed_magnitude_sum"] / (df_unique_id["frame_range"] - 1)

        # df_unique_id["average_magnitude"] = df.groupby("id")["signed_magnitude"].mean()
        # 各IDごとの平均magnitudeを計算し、対応するdf_unique_idにマージ
        average_vector_per_id = df.groupby("id")[["vector_angle", "vector_angle_difference"]].mean().reset_index()
        average_vector_per_id.rename(columns={"vector_angle": "average_vector_angle", "vector_angle_difference": "average_vector_angle_difference"}, inplace=True)
        # average_magnitude_per_idをマージ
        df_unique_id = pd.merge(df_unique_id, average_vector_per_id, on="id", how="left")

        # 各IDごとに、mean_b, mean_g, mean_r, median_b, median_g, median_r　の標準偏差を計算し、対応するdf_unique_idにマージ
        # Rename the columns for clarity
        # Group by 'id' and calculate the standard deviation for each ID
        id_mean_color_std_per_id = df.groupby('id')[['mean_b', 'mean_g', 'mean_r']].std()
        id_median_color_std_per_id = df.groupby('id')[['median_b', 'median_g', 'median_r']].std()
        id_mean_color_std_per_id.columns = ['mean_b_std', 'mean_g_std', 'mean_r_std']
        id_median_color_std_per_id.columns = ['median_b_std', 'median_g_std', 'median_r_std']
        # mean_b_std, mean_g_std, mean_r_stdの平均二乗平方根を算出
        id_mean_color_std_per_id["mean_color_std_rms"] = np.sqrt(
            id_mean_color_std_per_id["mean_b_std"]**2 + 
            id_mean_color_std_per_id["mean_g_std"]**2 + 
            id_mean_color_std_per_id["mean_r_std"]**2
        )
        # mean_b_std, mean_g_std, mean_r_stdの平均二乗平方根を算出
        id_median_color_std_per_id["median_color_std_rms"] = np.sqrt(
            id_median_color_std_per_id["median_b_std"]**2 + 
            id_median_color_std_per_id["median_g_std"]**2 + 
            id_median_color_std_per_id["median_r_std"]**2
        )
        # id_mean_color_std_per_id, id_median_color_std_per_idをマージ
        df_unique_id = pd.merge(df_unique_id, id_mean_color_std_per_id, on="id", how="left")
        df_unique_id = pd.merge(df_unique_id, id_median_color_std_per_id, on="id", how="left")

        

        
        # frame_range_per_id = df.groupby("id")["frame_n"].agg(["min", "max"]).reset_index()
        # frame_range_per_id.rename(columns={"min": "min_frame_n", "max": "max_frame_n"}, inplace=True)
        # df_unique_id = pd.merge(df_unique_id, frame_range_per_id, on="id", how="left")

        df_unique_id.drop(["x2", "y2", "MA", "ma", "ellipse_angle", "vector_angle", "distance", "signed_magnitude", "mean_b", "mean_g", "mean_r", "median_b", "median_g", "median_r", "prev_vector_angle", "vector_angle_difference"], axis=1, inplace=True)

        # df_unique_id.to_csv(output_csv_path, index=False)s

        return df_unique_id
    
    def calc_iou_per_id(df, df_unique_id):
        """
        IDごとにIoUを計算し、対応するdf_unique_idにマージ
        """
        iou_per_id = df.groupby("id")["iou"].mean().reset_index()
        iou_per_id.rename(columns={"iou": "average_iou"}, inplace=True)
        df_unique_id = pd.merge(df_unique_id, iou_per_id, on="id", how="left")
        df_unique_id.drop(["iou"], axis=1, inplace=True)
        return df_unique_id
    
    # データの読み込み
    if isinstance(input_csv_path, pd.DataFrame):
        df = input_csv_path
    else:
        df = pd.read_csv(input_csv_path)
    if pred_csv_path is not None:
        df = calc_iou_from_csv(df, pred_csv_path, radius)
        df = df.drop(["x_pred", "y_pred"], axis=1)
    # IDごとにベクトルを計算
    df = df.groupby("id", group_keys=False).apply(vector_with_shift).reset_index(drop=True)
    df_unique_id = calc_per_id(df, unique_id_output_csv_path, fps)

    if pred_csv_path is not None:
        df_unique_id = calc_iou_per_id(df, df_unique_id)
    

    # 不要な列を削除
    df = df.drop(columns=["x2", "y2", "prev_vector_angle", "vector_angle_difference"])   
    if "is_predicted" in df.columns:
        df = df.drop(columns=["is_predicted"])
        df_unique_id = df_unique_id.drop(columns=["is_predicted"])

    if output_csv_path is not None:
        df.to_csv(output_csv_path, index=False)
    if unique_id_output_csv_path is not None:
        df_unique_id.to_csv(unique_id_output_csv_path, index=False)

    return df, df_unique_id

if __name__ == "__main__":
    input_csv_path = "../2_data/kalmanfilter/ayu/manual_roi/ayu/c3_d50_a10_h5/with_color/c3_d50_a10_h5_rgb.csv"
    output_csv_path = "../2_data/kalmanfilter/ayu/manual_roi/ayu/c3_d50_a10_h5/with_color/c3_d50_a10_h5_rgb_actual.csv"
    unique_id_output_csv_path = "../2_data/kalmanfilter/ayu/manual_roi/ayu/c3_d50_a10_h5/with_color/c3_d50_a10_h5_rgb_unique.csv"
    calc_parameters(input_csv_path, output_csv_path, unique_id_output_csv_path)