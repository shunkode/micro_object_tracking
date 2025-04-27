import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.affinity import scale, rotate
import matplotlib.pyplot as plt
import shapely

def calculate_iou_circle_ellipse(circle_center, circle_radius, ellipse_center, ellipse_axes, ellipse_angle=0.0, is_predicted=None):
    """
    楕円と円のIOUを計算します。

    Parameters:
        circle_center (tuple): (cx, cy) 円の中心
        circle_radius (float): 円の半径
        ellipse_center (tuple): (ex, ey) 楕円の中心
        ellipse_axes (tuple): (a, b) 楕円の長軸、短軸
        ellipse_angle (float): 楕円の回転角度（度単位）
    Returns:
        float: IOUの値
    """
    if is_predicted:
        return 0.0
    # 円の形状を作成（resolution: 1/4円（90度）あたりの頂点数）
    circle = Point(circle_center).buffer(circle_radius, resolution=64)

    # 楕円の基本形状（半径1の円から拡大縮小）
    ellipse = Point(ellipse_center).buffer(1, resolution=64)
    ellipse = scale(ellipse, ellipse_axes[0], ellipse_axes[1], origin=ellipse_center)
    ellipse = rotate(ellipse, ellipse_angle, origin=ellipse_center)

    # 交差領域と和集合の領域を計算
    try:
        intersection_area = circle.intersection(ellipse).area
    except shapely.errors.GEOSException as e:
        if ellipse_axes[0] < 0.1 or ellipse_axes[1] < 0.1:
            print("Ellipse axes: ", ellipse_axes)
            return 0.0
        print(e)

        # 描画
        # 円と楕円の座標を取得
        print("circle area: ", circle.area)
        print("ellipse area: ", ellipse.area)
        print("intersection/circle: ", intersection_area/circle.area)
        circle_coords = np.array(circle.exterior.coords)
        ellipse_coords = np.array(ellipse.exterior.coords)
        plt.figure(figsize=(8, 8))
        plt.plot(circle_coords[:, 0], circle_coords[:, 1], label="Circle", color="blue")
        plt.plot(ellipse_coords[:, 0], ellipse_coords[:, 1], label="Ellipse", color="orange")
        plt.scatter(*circle_center, color="blue", label="Circle Center")
        plt.scatter(*ellipse_center, color="orange", label="Ellipse Center")
        plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        plt.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.title("Circle and Ellipse")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()
        # print("Intersection Area:", intersection_area)
        # print("Union Area:", union_area)
        

    union_area = circle.area + ellipse.area - intersection_area
    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def calculate_iou_ellipses(pred_ellipse_center, ellipse_center, ellipse_axes, ellipse_angle=0.0, is_predicted=None):
    """
    楕円と円のIOUを計算します。

    Parameters:
        circle_center (tuple): (cx, cy) 円の中心
        circle_radius (float): 円の半径
        ellipse_center (tuple): (ex, ey) 楕円の中心
        ellipse_axes (tuple): (a, b) 楕円の長軸、短軸
        ellipse_angle (float): 楕円の回転角度（度単位）
    Returns:
        float: IOUの値
    """
    
    if is_predicted:
        return 0.0
    # 楕円の基本形状（半径1の円から拡大縮小）
    ellipse = Point(ellipse_center).buffer(1, resolution=64)
    ellipse = scale(ellipse, ellipse_axes[0], ellipse_axes[1], origin=ellipse_center)
    ellipse = rotate(ellipse, ellipse_angle, origin=ellipse_center)

    # 楕円の形状を作成（resolution: 1/4円（90度）あたりの頂点数）
    pred_ellipse = Point(pred_ellipse_center).buffer(1, resolution=64)
    pred_ellipse = scale(pred_ellipse, ellipse_axes[0], ellipse_axes[1], origin=pred_ellipse_center)
    pred_ellipse = rotate(pred_ellipse, ellipse_angle, origin=pred_ellipse_center)


    # 交差領域と和集合の領域を計算
    try:
        intersection_area = pred_ellipse.intersection(ellipse).area
    except shapely.errors.GEOSException as e:
        if ellipse_axes[0] < 0.1 or ellipse_axes[1] < 0.1:
            print("Ellipse axes: ", ellipse_axes)
            return 0.0
        print(e)
        

    union_area = pred_ellipse.area + ellipse.area - intersection_area

    if union_area == 0:
        # print("Union area is 0")
        # print("pred ellipse area: ", pred_ellipse.area)
        # print("ellipse area: ", ellipse.area)
        return 0.0

    if intersection_area / union_area>1:
        # 描画
        # 円と楕円の座標を取得
        print("circle area: ", pred_ellipse.area)
        print("ellipse area: ", ellipse.area)
        print("intersection/pred: ", intersection_area/pred_ellipse.area)
        circle_coords = np.array(pred_ellipse.exterior.coords)
        ellipse_coords = np.array(ellipse.exterior.coords)
        plt.figure(figsize=(8, 8))
        plt.plot(circle_coords[:, 0], circle_coords[:, 1], label="Circle", color="blue")
        plt.plot(ellipse_coords[:, 0], ellipse_coords[:, 1], label="Ellipse", color="orange")
        plt.scatter(*pred_ellipse_center, color="blue", label="Circle Center")
        plt.scatter(*ellipse_center, color="orange", label="Ellipse Center")
        plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        plt.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.title("Circle and Ellipse")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()
        # print("Intersection Area:", intersection_area)
        # print("Union Area:", union_area)
            

    union_area = pred_ellipse.area + ellipse.area - intersection_area
    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def calculate_iou_df(actual_pred_df, radius):
    """
    actual_pred_dfから、円（予測）と楕円（実測）のIOUを計算し、
    新たな列"iou"として追加したDataFrameを返します。

    Parameters:
        actual_pred_df (DataFrame): id, x_actual, y_actual, MA, ma, frame_n, x_pred, y_predなどを含むDataFrame
        radius (float): 円の半径

    Returns:
        DataFrame: iou列が追加されたDataFrame
    """
    # 楕円の中心
    # 長軸(MA), 短軸(ma)は実測データ側
    # 回転角度ellipse_angleがデータに無い場合は0度と仮定
    if 'ellipse_angle' not in actual_pred_df.columns:
        actual_pred_df['ellipse_angle'] = 0.0

    # 円の半径が0の場合、楕円の情報を用いてIOUを計算
    if radius != 0:
        # IOUの計算をapplyで一括処理
        actual_pred_df['iou'] = actual_pred_df.apply(
            lambda row: calculate_iou_circle_ellipse(
            circle_center=(row["x_pred"], row["y_pred"]),
            circle_radius=radius,
            ellipse_center=(row["x_actual"], row["y_actual"]),
            ellipse_axes=(row["MA"]/2.0, row["ma"]/2.0),  # 楕円軸はMA, maが全長なので半分を半径相当に
            ellipse_angle=row["ellipse_angle"],
            is_predicted=row.get("is_predicted", None)
            ), axis=1
        )
    # 円の半径が0でない場合、半径情報を用いて円を描き、円と楕円のIOUを計算
    else:
        # IOUの計算をapplyで一括処理
        actual_pred_df['iou'] = actual_pred_df.apply(
            lambda row: calculate_iou_ellipses(
                pred_ellipse_center=(row["x_pred"], row["y_pred"]),
                ellipse_center=(row["x_actual"], row["y_actual"]),
                ellipse_axes=(row["MA"]/2.0, row["ma"]/2.0),  # 楕円軸はMA, maが全長なので半分を半径相当に
                ellipse_angle=row["ellipse_angle"],
                is_predicted=row.get("is_predicted", None)
            ), axis=1
        )

    # actual_iou_df = actual_pred_df[["id", "x_actual", "y_actual", "MA", "ma", "frame_n", "ellipse_angle", "class", "iou"]]
    actual_iou_df = actual_pred_df.rename(columns={"x_actual": "x", "y_actual": "y"})#, "class_actual": "class"})

    return actual_iou_df

def calc_iou_from_csv(actual_csv_path, pred_csv_path, radius):
    if isinstance(actual_csv_path, pd.DataFrame):
        actual_df = actual_csv_path
    else:
        actual_df = pd.read_csv(actual_csv_path)
    pred_df = pd.read_csv(pred_csv_path)
    # idとframe_nが一致するものを抜き出す
    actual_pred_df = pd.merge(actual_df, pred_df, on=["id", "frame_n"], how="inner", suffixes=("_actual", "_pred"))
    actual_iou_df = calculate_iou_df(actual_pred_df, radius=radius)
    return actual_iou_df

def classify_based_on_iou(actual_csv_path, pred_csv_path, iou_thres=0.5, overlap_thres=1, radius=5, output_csv_path=None):
    """
    IOUの値に基づいて、クラス分けを行います。

    Parameters:
        iou_df (DataFrame): id, x, y, MA, ma, frame_n, ellipse_angle, iouを含むDataFrame
        iou_thres (float or list): IOUの閾値. floatの場合は閾値より上か下かで判定、listの場合は閾値の範囲内かで判定
        overlap_thres (int): 閾値の基準を超えたもしくは範囲内にあった回数

    Returns:
        DataFrame: クラス分けされたDataFrame
    """
    iou_df = calc_iou_from_csv(actual_csv_path, pred_csv_path, radius)
    if isinstance(iou_thres, float):
        # 閾値を超えた回数を元のデータフレームに追加
        iou_df['overlap_count'] = iou_df.groupby('id')['iou'].transform(lambda x: (x > iou_thres).sum())
    elif isinstance(iou_thres, list):
        # 閾値範囲内にあった回数を元のデータフレームに追加
        iou_df['overlap_count'] = iou_df.groupby('id')['iou'].transform(lambda x: ((x > iou_thres[0]) & (x <= iou_thres[1])).sum())
    iou_df.loc[iou_df["overlap_count"] < overlap_thres, "class"] = "noise"
    print(iou_df)
    iou_csv_path = actual_csv_path.replace(".csv", "_ie20range30_o4.csv")
    if output_csv_path is not None:
        iou_df.to_csv(iou_csv_path, index=False)
    return iou_df
    

# 使用例
if __name__ == "__main__":
    classify_based_on_iou(
        actual_csv_path="../2_data/kalmanfilter/ayu/manual_roi/ayu_all_manualroi_c3_d50_co10_a10_h8_s0_ds70_f8_dir130_withpred.csv", 
        pred_csv_path= "../2_data/kalmanfilter/ayu/manual_roi/ayu_all_manualroi_c3_d50_co10_a10_h8_s0_ds70_f8_dir130_withpred_pred.csv", 
        iou_thres=[0.2, 0.3], 
        overlap_thres=4, 
        radius=1, 
        output_csv_path="../2_data/kalmanfilter/ayu/manual_roi/ayu_all_manualroi_c3_d50_co10_a10_h8_s0_ds70_f8_dir130_withpred_iou.csv")
    # circle_center = (41.450027, 490.6793)
    # circle_radius = 20
    # ellipse_center = (40.61993, 489.02457)
    # ellipse_axes = (17.076122283935547, 21.227489471435547)
    # ellipse_angle = 10.662163734436035
    # print(calculate_iou_circle_ellipse(circle_center, circle_radius, ellipse_center, ellipse_axes, ellipse_angle))


    # circle_center = (10, 20)
    # ellipse_center = (10, 10)
    # ellipse_axes = (17.076122283935547, 21.227489471435547)
    # ellipse_angle = 10.662163734436035
    # print(calculate_iou_ellipses(circle_center, ellipse_center, ellipse_axes, ellipse_angle))


# import numpy as np
# from shapely.geometry import Point
# from shapely.affinity import scale, rotate
# import matplotlib.pyplot as plt
# import pandas as pd

# def calculate_iou_circle_ellipse(circle_center, circle_radius, ellipse_center, ellipse_axes, ellipse_angle):
#     """
#     楕円と円のIOUを計算します。

#     Parameters:
#         circle_center (tuple): (cx, cy) 円の中心
#         circle_radius (float): 円の半径
#         ellipse_center (tuple): (ex, ey) 楕円の中心
#         ellipse_axes (tuple): (a, b) 楕円の長軸、短軸
#         ellipse_angle (float): 楕円の回転角度（度単位）

#     Returns:
#         float: IOUの値
#     """
#     # 円の形状を作成
#     circle = Point(circle_center).buffer(circle_radius)

#     # 楕円の形状を作成
#     ellipse = Point(ellipse_center).buffer(1)  # 楕円の基本形状を作成
#     ellipse = scale(ellipse, ellipse_axes[0], ellipse_axes[1])  # 軸をスケーリング
#     ellipse = rotate(ellipse, ellipse_angle, origin=ellipse_center)  # 楕円を回転

#     # 円と楕円の座標を取得
#     circle_coords = np.array(circle.exterior.coords)
#     ellipse_coords = np.array(ellipse.exterior.coords)

#     # 描画
#     # plt.figure(figsize=(8, 8))
#     # plt.plot(circle_coords[:, 0], circle_coords[:, 1], label="Circle", color="blue")
#     # plt.plot(ellipse_coords[:, 0], ellipse_coords[:, 1], label="Ellipse", color="orange")
#     # plt.scatter(*circle_center, color="blue", label="Circle Center")
#     # plt.scatter(*ellipse_center, color="orange", label="Ellipse Center")
#     # plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
#     # plt.axvline(0, color="gray", linewidth=0.5, linestyle="--")
#     # plt.gca().set_aspect('equal', adjustable='box')
#     # plt.legend()
#     # plt.title("Circle and Ellipse")
#     # plt.xlabel("X")
#     # plt.ylabel("Y")
#     # plt.grid(True)
#     # plt.show()

#     # 交差領域と和集合の領域を計算
#     intersection_area = circle.intersection(ellipse).area
#     union_area = circle.area + ellipse.area - intersection_area

#     # IOUを計算
#     if union_area == 0:
#         return 0.0  # 和集合が0の場合はIOUを0にする

#     iou = intersection_area / union_area
#     return iou


# # IOU計算のため、実測と予測の情報を取得（IDとframe_nが一致するものを抜き出す）
# def get_actual_pred_info(actual_csv_path, pred_csv_path):
#     actual_df = pd.read_csv(actual_csv_path)
#     pred_df = pd.read_csv(pred_csv_path)
#     # actual_dfとpred_dfのIDとframe_nが一致するものを抜き出す
#     actual_pred_df = pd.merge(actual_df, pred_df, on=["id", "frame_n"], how="inner", suffixes=("_actual", "_pred"))
#     return actual_pred_df
    

# def calculate_iou_circle_ellipse_df():
#     # ellipse_center = (row["x_actual"], row["y_actual"])
#     # ellipse_axes = (row["MA"], row["ma"])
#     # ellipse_angle = row["ellipse_angle"]
#     # circle_center = (row["x_pred"], row["y_pred"])
#     # circle_radius = row["radius"]
#     """
#     楕円と円のIOUを計算します。

#     Parameters:
#         circle_center (tuple): (cx, cy) 円の中心
#         circle_radius (float): 円の半径
#         ellipse_center (tuple): (ex, ey) 楕円の中心
#         ellipse_axes (tuple): (a, b) 楕円の長軸、短軸
#         ellipse_angle (float): 楕円の回転角度（度単位）

#     Returns:
#         float: IOUの値
#     """
#     # 円の形状を作成
#     circle = Point(circle_center).buffer(circle_radius)

#     # 楕円の形状を作成
#     ellipse = Point(ellipse_center).buffer(1)  # 楕円の基本形状を作成
#     ellipse = scale(ellipse, ellipse_axes[0], ellipse_axes[1])  # 軸をスケーリング
#     ellipse = rotate(ellipse, ellipse_angle, origin=ellipse_center)  # 楕円を回転

#     # 円と楕円の座標を取得
#     circle_coords = np.array(circle.exterior.coords)
#     ellipse_coords = np.array(ellipse.exterior.coords)

#     # 描画
#     # plt.figure(figsize=(8, 8))
#     # plt.plot(circle_coords[:, 0], circle_coords[:, 1], label="Circle", color="blue")
#     # plt.plot(ellipse_coords[:, 0], ellipse_coords[:, 1], label="Ellipse", color="orange")
#     # plt.scatter(*circle_center, color="blue", label="Circle Center")
#     # plt.scatter(*ellipse_center, color="orange", label="Ellipse Center")
#     # plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
#     # plt.axvline(0, color="gray", linewidth=0.5, linestyle="--")
#     # plt.gca().set_aspect('equal', adjustable='box')
#     # plt.legend()
#     # plt.title("Circle and Ellipse")
#     # plt.xlabel("X")
#     # plt.ylabel("Y")
#     # plt.grid(True)
#     # plt.show()

#     # 交差領域と和集合の領域を計算
#     intersection_area = circle.intersection(ellipse).area
#     union_area = circle.area + ellipse.area - intersection_area

#     # IOUを計算
#     if union_area == 0:
#         return 0.0  # 和集合が0の場合はIOUを0にする

#     iou = intersection_area / union_area
#     return iou


# if __name__ == '__main__':
#     # テスト
#     circle_center = (0, 0)
#     circle_radius = 5
#     ellipse_center = (0, 2.5)
#     ellipse_axes = (10, 2.5)
#     ellipse_angle = 0
#     iou = calculate_iou_circle_ellipse(circle_center, circle_radius, ellipse_center, ellipse_axes, ellipse_angle)
#     print(iou)  # 0.5708
