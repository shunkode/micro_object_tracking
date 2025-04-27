import pandas as pd
import csv

from analyze_tracking_result import is_fish_based_on_iou, is_fish, calc_accuracy
from calc_vector import calc_signed_magnitude_per_id
from add_ellipses_to_video import add_ellipses_to_video_is_fish

def extract_specific_ids(input_csv_path, output_csv_path, target_ids):
    """
    指定したIDのデータを抜き出す。
    抽出したいIDをリスト形式で指定
    """
    # CSVファイルを読み込む
    if isinstance(input_csv_path, pd.DataFrame):
        df = input_csv_path
    else:
        df = pd.read_csv(input_csv_path)

    # 指定した複数のIDのデータを抜き出す
    filtered_df = df[df["id"].isin(target_ids)]

    # 抽出結果を新しいCSVファイルに保存
    if output_csv_path is not None:
        filtered_df.to_csv(output_csv_path, index=False)
        print(f"指定したIDのデータを {output_csv_path} に保存しました。")
    return filtered_df

def exclude_specific_ids(input_csv_path, output_csv_path, target_ids):
    """
    指定したIDのデータを除外する。
    除外したいIDをリスト形式で指定
    """
    # CSVファイルを読み込む
    if isinstance(input_csv_path, pd.DataFrame):
        df = input_csv_path
    else:
        df = pd.read_csv(input_csv_path)

    # 指定した複数のIDのデータを抜き出す
    filtered_df = df[~df["id"].isin(target_ids)]

    # 抽出結果を新しいCSVファイルに保存
    if output_csv_path is not None:
        filtered_df.to_csv(output_csv_path, index=False)
        print(f"指定したIDのデータを {output_csv_path} に保存しました。")
    return filtered_df

def extract_ids_n_merge_pred(input_csv_path, 
                             pred_csv_path, 
                             output_csv_path, 
                             target_ids):
    """
    指定したIDのデータを抜き出し、予測結果をマージする。
    """
    # 実測データのうち、指定したIDのデータを抜き出す
    df = extract_specific_ids(input_csv_path, 
                              f"{'/'.join(output_csv_path.split('/')[:-1])}/observed_fish.csv", target_ids)
    # 予測データのうち、指定したIDのデータを抜き出す
    df_pred = extract_specific_ids(pred_csv_path, 
                                   f"{'/'.join(output_csv_path.split('/')[:-1])}/pred_fish.csv", 
                                   target_ids)
    # # 実測データと予測データをマージする
    # df_merged = pd.merge(df, df_pred, on="id", how="left", suffixes=["", "_pred"], how="left")
    df_concat = pd.concat([df, df_pred], axis=0)
    df_concat = df_concat.sort_values(by=["id", "frame_n"])
    df_concat = df_concat.drop_duplicates(subset=["id", "frame_n"], keep="first")
    # Create a new column 'is_predicted' based on whether 'MA', 'ma', and 'ellipse_angle' are NaN
    df_concat['is_predicted'] = df_concat[['MA', 'ma', 'ellipse_angle']].isna().any(axis=1)

    # マージ結果を保存
    df_concat.to_csv(f"{'/'.join(output_csv_path.split('/')[:-1])}/concat_fish_pred.csv", index=False)

    # 実測データのうち、指定したID以外のデータを抜き出す
    df_not_fish = exclude_specific_ids(input_csv_path, 
                                       f"{'/'.join(output_csv_path.split('/')[:-1])}/observed_notfish.csv", target_ids)
    # 予測データのうち、指定したID以外のデータを抜き出す
    df_pred_not_fish = exclude_specific_ids(pred_csv_path, f"{'/'.join(output_csv_path.split('/')[:-1])}/pred_notfish.csv", target_ids)
    # 実測データと予測データをマージする
    # df_merged_not_fish = pd.merge(df_not_fish, df_pred_not_fish, on="id", how="left", suffixes=["", "_pred"], how="left")
    df_concat_not_fish = pd.concat([df_not_fish, df_pred_not_fish], axis=0)
    df_concat_not_fish = df_concat_not_fish.sort_values(by=["id", "frame_n"])
    df_concat_not_fish = df_concat_not_fish.drop_duplicates(subset=["id", "frame_n"], keep="first")
    # Create a new column 'is_predicted' based on whether 'MA', 'ma', and 'ellipse_angle' are NaN
    df_concat_not_fish['is_predicted'] = df_concat_not_fish[['MA', 'ma', 'ellipse_angle']].isna().any(axis=1)
    # マージ結果を保存
    df_concat_not_fish.to_csv(f"{'/'.join(output_csv_path.split('/')[:-1])}/concat_notfish_pred.csv", index=False)  

    return df_concat, df_concat_not_fish

def extract_randam_ids(input_csv_path, output_csv_path=None, num_ids=10):
    """
    num_ids個のIDデータをランダムに抜き出す
    """
    import random
    # 入力されたデータが、dataframeならそのまま読み込む
    if isinstance(input_csv_path, pd.DataFrame):
        df = input_csv_path
    else:
        # CSVファイルを読み込む
        df = pd.read_csv(input_csv_path)

    # ユニークなIDを取得してnum_ids個のデータをランダムに選択
    unique_ids = df['id'].unique()
    # ユニークなIDの個数を取得
    num_unique_ids = len(unique_ids)
    print(f"ユニークなIDの個数: {num_unique_ids}")
    random.seed(0)
    sampled_ids = random.sample(list(unique_ids), num_ids)

    # 選択されたIDに基づいてデータをフィルタリング
    filtered_df = df[df['id'].isin(sampled_ids)]

    # 結果を保存
    if output_csv_path is not None:
        filtered_df.to_csv(output_csv_path, index=False)

    # print(f"{num_ids} 個のIDデータがランダムに選択され、{output_csv_path} に保存されました。")

    return filtered_df

def concat_fish_n_random_noise(input_csv_path, 
                               output_csv_path, 
                               target_ids:list, 
                               WITH_PRED:bool, 
                               pred_csv_path=None, 
                               num_ids=10):
    """
    指定したIDのデータと、
    指定したID以外のデータから、num_ids個のIDをランダムに抜き出したデータを、
    統合するプログラム
    WITH_PRED=Trueならば、カルマンフィルタの予測値を、実測値に統合して出力する
    """
    if WITH_PRED:
        df_fish, df_noise = extract_ids_n_merge_pred(
            input_csv_path, 
            pred_csv_path, 
            output_csv_path, 
            target_ids)
        df_random_noise = extract_randam_ids(df_noise, 
                                             output_csv_path=None, 
                                             num_ids=num_ids)
        df_fish_n_random_noise = pd.concat([df_fish, df_random_noise], axis=0)
        df_fish_n_random_noise = df_fish_n_random_noise.sort_values(by=["id", "frame_n"])
        if output_csv_path is not None:
            df_fish_n_random_noise.to_csv(output_csv_path, index=False)
        return df_fish_n_random_noise

    else:
        # 実測データのうち、指定したIDのデータを抜き出す
        df_fish = extract_specific_ids(input_csv_path, f"{output_csv_path.split(".csv")[0]}_fish.csv", target_ids)
        # 実測データのうち、指定したID以外のデータを抜き出す
        df_noise = exclude_specific_ids(input_csv_path, f"{output_csv_path.split('.csv')[0]}_not_fish.csv", target_ids)
        df_random_noise = extract_randam_ids(df_noise, 
                                             output_csv_path=None, 
                                             num_ids=num_ids)
        df_fish_n_random_noise = pd.concat([df_fish, df_random_noise], axis=0)
        df_fish_n_random_noise = df_fish_n_random_noise.sort_values(by=["id", "frame_n"])
        if output_csv_path is not None:
            df_fish_n_random_noise.to_csv(output_csv_path, index=False)
        return df_fish_n_random_noise
        
def define_fish_notfish_by_id(input_csv_path:pd.DataFrame|str, output_csv_path:str, target_ids:list):
    """
    指定したIDのデータを魚と非魚に分類する
    入力csvファイルの、"class"列に、魚は1、非魚は0を入力する
    """
    if isinstance(input_csv_path, pd.DataFrame):
        df = input_csv_path
    else:
        # CSVファイルを読み込む
        df = pd.read_csv(input_csv_path)
    # "class"列に、魚は1、非魚は0を入力する
    df['class'] = df['id'].apply(lambda x: 1 if x in target_ids else 0)

    # 結果を保存
    if output_csv_path is not None:
        df.to_csv(output_csv_path, index=False)
    
    return df


def extract_rgb_components(input_csv_path, output_csv_path):
    """
    mean_color, median_color列を、それぞれ
    mean_b, mean_g, mean_rと、median_b, median_g, median_rに分割して出力する関数
    """
    # Function to extract RGB components from the color strings
    def split_color_column(color_column):
        return color_column.str.strip('[]').str.split(expand=True).astype(float)

    df = pd.read_csv(input_csv_path)
    # Split the mean_color and median_color columns into separate RGB components
    df[['mean_b', 'mean_g', 'mean_r']] = split_color_column(df['mean_color'])
    df[['median_b', 'median_g', 'median_r']] = split_color_column(df['median_color'])

    # Drop the original mean_color and median_color columns
    df = df.drop(columns=['mean_color', 'median_color'])
    df.to_csv(output_csv_path, index=False)




def extract_specific_class(input_csv_path, output_csv_path, class_name):
    data = pd.read_csv(input_csv_path)
    filtered_data = data[data['class'] == class_name]
    print(filtered_data)
    filtered_data.to_csv(output_csv_path, index=False)


# 検出開始時間を計算する関数
def calc_detected_time(input_csv_path, output_csv_path, fps=30):
    df = pd.read_csv(input_csv_path)
    df_unique_id = df.drop_duplicates(subset="id").copy()
    df_unique_id.drop(["MA", "ma", "ellipse_angle", "vector_angle", "x_coords", "y_coords", "xy_coords", "distance"], axis=1, inplace=True)#(["MA", "ma", "ellipse_angle", "vector_angle", "x_coords", "y_coords", "xy_coords", "distance"], axis=1, inplace=True)
    df_unique_id["detected_time(s)"] = df_unique_id["frame_n"]/fps
    
    df = calc_signed_magnitude_per_id(df, "test4.csv")#, output_csv_path="../2_data/kalmanfilter/ayu/manual_roi/test.csv")

    # df_unique_id["average_magnitude"] = df.groupby("id")["signed_magnitude"].mean()
    # 各IDごとの平均magnitudeを計算し、対応するdf_unique_idにマージ
    average_magnitude_per_id = df.groupby("id")["signed_magnitude"].mean().reset_index()
    average_magnitude_per_id.rename(columns={"signed_magnitude": "average_magnitude"}, inplace=True)
    # df_unique_idと平均magnitudeをマージ
    df_unique_id = pd.merge(df_unique_id, average_magnitude_per_id, on="id", how="left")

    df_unique_id.to_csv(output_csv_path, index=False)




def calc_precision_recall(input_params_csv_path, 
                          actual_csv_path,
                          pred_csv_path,
                          unique_id_csv_path, 
                          ground_truth_csv_path,
                          output_csv_path):
    with open(output_csv_path, mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(["precision", "recall", "f1", "TP", "FP", "FN", "min_iou_thres", "max_iou_thres", "overlap_thes", "radius", "min_x_coords_thres", "max_x_coords_thres", "min_y_coords_thres", "max_y_coords_thres", "min_xy_coords_thres", "max_xy_coords_thres", "min_average_distance_thres", "max_average_distance_thres", "min_average_signed_magnitude_thres", "max_average_signed_magnitude_thres", "min_average_vector_angle_thres", "max_average_vector_angle_thres", "min_distance_sum_thres", "max_distance_sum_thres", "min_signed_magnitude_sum_thres", "max_signed_magnitude_sum_thres", "min_frame_range_thres", "max_frame_range_thres"])

    df_analyzed = pd.read_csv(input_params_csv_path)

    for number, min_iou_thres, max_iou_thres, overlap_thes, radius, min_x_coords_thres, max_x_coords_thres, min_y_coords_thres, max_y_coords_thres, min_xy_coords_thres, max_xy_coords_thres, min_average_distance_thres, max_average_distance_thres, min_average_signed_magnitude_thres, max_average_signed_magnitude_thres, min_average_vector_angle_thres, max_average_vector_angle_thres, min_distance_sum_thres, max_distance_sum_thres, min_signed_magnitude_sum_thres, max_signed_magnitude_sum_thres, min_frame_range_thres, max_frame_range_thres in zip(df_analyzed["number"], df_analyzed["params_min_iou_thres"], df_analyzed["params_max_iou_thres"], df_analyzed["params_overlap_thres"], df_analyzed["params_radius"], df_analyzed["params_min_x_coords_thres"], df_analyzed["params_max_x_coords_thres"], df_analyzed["params_min_y_coords_thres"], df_analyzed["params_max_y_coords_thres"], df_analyzed["params_min_xy_coords_thres"], df_analyzed["params_max_xy_coords_thres"], df_analyzed["params_min_distance_thres"], df_analyzed["params_max_distance_thres"], df_analyzed["params_min_signed_magnitude_thres"], df_analyzed["params_max_signed_magnitude_thres"], df_analyzed["params_min_vector_angle_thres"], df_analyzed["params_max_vector_angle_thres"], df_analyzed["params_min_distance_sum_thres"], df_analyzed["params_max_distance_sum_thres"], df_analyzed["params_min_signed_magnitude_sum_thres"], df_analyzed["params_max_signed_magnitude_sum_thres"], df_analyzed["params_min_frame_range_thres"], df_analyzed["params_max_frame_range_thres"]):

        df_with_iou, unique_df = is_fish_based_on_iou(
            actual_csv_path, 
            pred_csv_path, 
            unique_id_csv_path, 
            iou_thres=[min_iou_thres, max_iou_thres], 
            overlap_thres=0, #overlap_thes, 
            radius=radius, 
            output_csv_path=None)

        unique_df = is_fish(unique_df, min_x_coords_thres, max_x_coords_thres, "x_coords")
        unique_df = is_fish(unique_df, min_y_coords_thres, max_y_coords_thres, "y_coords")
        unique_df = is_fish(unique_df, min_xy_coords_thres, max_xy_coords_thres, "xy_coords")
        unique_df = is_fish(unique_df, min_average_distance_thres, max_average_distance_thres, "average_distance(speed)")
        unique_df = is_fish(unique_df, min_average_signed_magnitude_thres, max_average_signed_magnitude_thres, "average_signed_magnitude")
        unique_df = is_fish(unique_df, min_average_vector_angle_thres, max_average_vector_angle_thres, "average_vector_angle")
        unique_df = is_fish(unique_df, min_distance_sum_thres, max_distance_sum_thres, "distance_sum")
        unique_df = is_fish(unique_df, min_signed_magnitude_sum_thres, max_signed_magnitude_sum_thres, "signed_magnitude_sum")
        unique_df = is_fish(unique_df, min_frame_range_thres, max_frame_range_thres, "frame_range")
        unique_df = is_fish(unique_df, min_iou_thres, max_iou_thres, "average_iou")

        precision, recall, f1, TP, FP, FN = calc_accuracy(unique_df, 
                                            ground_truth_csv_path, 
                                            frame_tolerance=10)
        
        df_with_iou.to_csv(f"{output_csv_path.split('.csv')[0]}_num{number}_iou.csv", index=False)
        unique_df.to_csv(f"{output_csv_path.split('.csv')[0]}_num{number}_unique.csv", index=False)

        df_actual = pd.read_csv(actual_csv_path)
        unique_ids = unique_df[unique_df["class"] == 1]["id"]
        df_actual_filtered = df_actual[df_actual["id"].isin(unique_ids)]
        print(unique_ids)
        print(df_actual_filtered)
        # unique_df_is_fish = unique_df[unique_df['id'] == 1]
        add_ellipses_to_video_is_fish(
            input_video_path="../1_data_raw/ayu.mp4",
            df_with_iou=df_actual_filtered, 
            output_video_path=f"{output_csv_path.split('.csv')[0]}_num{number}_video.mp4", 
            ROI="POLY", 
            roi_args=None, 
            specific_class=1
        )


        
        with open(output_csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([precision, recall, f1, TP, FP, FN,  min_iou_thres, max_iou_thres, overlap_thes, radius, min_x_coords_thres, max_x_coords_thres, min_y_coords_thres, max_y_coords_thres, min_xy_coords_thres, max_xy_coords_thres, min_average_distance_thres, max_average_distance_thres, min_average_signed_magnitude_thres, max_average_signed_magnitude_thres, min_average_vector_angle_thres, max_average_vector_angle_thres, min_distance_sum_thres, max_distance_sum_thres, min_signed_magnitude_sum_thres, max_signed_magnitude_sum_thres, min_frame_range_thres, max_frame_range_thres])


if __name__ == '__main__':
    # input_csv_path = "../2_data/kalmanfilter/ayu/manual_roi/ayu_all_manualroi_c3_d50_co10_a10_h8_s0_ds70_f8_dir130_withpred_i20_o4_r3.csv"
    # output_csv_path = "../2_data/kalmanfilter/ayu/manual_roi/test3.csv"
    # # class_name = "fish"
    # # extract_specific_class(input_csv_path, output_csv_path, class_name)
    # calc_detected_time(input_csv_path, output_csv_path, fps=60)



    # Precision, Recall, F1スコアを計算する
    # input_params_csv_path = "../2_data/kalmanfilter/ayu/manual_roi/observed/c3_d50_a10_h3_optimization_results_best_score.csv"
    # actual_csv_path = "../2_data/kalmanfilter/ayu/manual_roi/observed/c3_d50_a10_h3_actual.csv"
    # pred_csv_path = "../2_data/kalmanfilter/ayu/manual_roi/observed/c3_d50_a10_h3_pred.csv"
    # unique_id_csv_path = "../2_data/kalmanfilter/ayu/manual_roi/observed/c3_d50_a10_h3_unique.csv"
    # ground_truth_csv_path = "../1_data_raw/ground_truth.csv"
    # output_csv_path = "../2_data/kalmanfilter/ayu/manual_roi/observed/c3_d50_a10_h3_with_precision_recall_.csv"

    # calc_precision_recall(input_params_csv_path, 
    #                       actual_csv_path,
    #                       pred_csv_path,
    #                       unique_id_csv_path, 
    #                       ground_truth_csv_path,
    #                       output_csv_path)



    # # 指定したIDのデータを抜き出す
    # input_csv_path = "../2_data/kalmanfilter/ayu/manual_roi/observed/c3_d50_a10_h3/analyzed/ayu_all_manualroi_c3_d50_a10_h5.csv"
    # pred_csv_path = "../2_data/kalmanfilter/ayu/manual_roi/observed/c3_d50_a10_h3/analyzed/ayu_all_manualroi_c3_d50_a10_h5_pred.csv"
    # output_csv_path = "../2_data/kalmanfilter/ayu/manual_roi/observed/c3_d50_a10_h3/analyzed/test.csv"
    # target_ids = [11, 60, 19, 76]
    # extract_ids_n_merge_pred(input_csv_path, 
    #                          pred_csv_path, 
    #                          output_csv_path, 
    #                          target_ids)

    # input_csv_path = "../2_data/kalmanfilter/ayu/manual_roi/ayu/optuna/c3_d50_a10_h5/test_randam_extract_ids.csv"
    # output_csv_path = "../2_data/kalmanfilter/ayu/manual_roi/ayu/optuna/c3_d50_a10_h5/test.csv"
    # target_ids = [11, 60, 19, 76, 20913]
    # # WITH_PRED = True
    # # pred_csv_path = "../2_data/kalmanfilter/ayu/manual_roi/observed/c3_d50_a10_h3/analyzed/ayu_all_manualroi_c3_d50_a10_h5_pred.csv"
    # # num_ids = 5
    # # # random_ids_df = extract_randam_ids(input_csv_path, output_csv_path, num_ids)

    # # df_fish_n_random_noise = concat_fish_n_random_noise(input_csv_path, 
    # #                            output_csv_path, 
    # #                            target_ids, 
    # #                            WITH_PRED, 
    # #                            pred_csv_path, 
    # #                            num_ids
    # #                            )
    # # print(df_fish_n_random_noise)
    # classify_fish_notfish_by_id(input_csv_path, output_csv_path, target_ids)


    input_csv_path = "C:/Users/koder/OneDrive - Tokyo University of Agriculture and Technology/0_research/202407_back_diff/2_data/kalmanfilter_v3/ayu/manual_roi/ayu4_color_thres/config3/c4_d30_a10_h3_md30/c4_d30_a10_h3_md30.csv"
    output_csv_path = "C:/Users/koder/OneDrive - Tokyo University of Agriculture and Technology/0_research/202407_back_diff/2_data/kalmanfilter_v3/ayu/manual_roi/ayu4_color_thres/config3/c4_d30_a10_h3_md30/c4_d30_a10_h3_md30_rgb.csv"
    extract_rgb_components(input_csv_path, output_csv_path)





    # extract_specific_ids(input_csv_path, 
    #                      output_csv_path, 
    #                      target_ids=[11, 60, 19, 76])