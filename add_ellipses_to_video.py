import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd

from mask_video import mask_analysis_region
from set_roi_polygon import set_roi_polygon
from confirm_filename import confirm_filename, check_file_exists

def add_ellipses_to_video(
        input_video_path, 
        input_csv_path, 
        output_video_path, 
        ROI, 
        roi_args:list , 
        specific_class:list=None, 
        class_colors:dict=None, 
        pred_csv_path=None):
    current_frame_n = 0
    cap = cv2.VideoCapture(input_video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    df = pd.read_csv(input_csv_path)
    if pred_csv_path is not None:
        df_pred = pd.read_csv(pred_csv_path)

    if ROI=="AUTO":
        if roi_args is None:
            min_y, mask = mask_analysis_region(input_video_path)
        else:
            min_y, mask = roi_args
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height-min_y))
    elif ROI=="RECT":
        if roi_args is None:
            pass
        else:
            roi_x, roi_y, roi_w, roi_h = roi_args
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (roi_w, roi_h))
    elif ROI=="POLY":
        if roi_args is None:
            points, offset, frame_height, frame_width = set_roi_polygon(input_video_path)
            adjusted_points = [np.array(points, np.int32) - np.array(offset)]
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        else:
            points, offset, adjusted_points, frame_height, frame_width = roi_args
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    else:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    with tqdm(total=total_frames, desc="Adding ellipses to frames", unit="frame") as pbar:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("フレームの読み込みに失敗しました。処理を終了します。")
                break

            if ROI=="AUTO":
                    frame = frame[min_y:, :]
                    frame = cv2.bitwise_and(frame, mask)
            elif ROI=="RECT":
                if roi_x is None:
                    roi_x, roi_y, roi_w, roi_h = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=False)
                frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            elif ROI=="POLY":
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                cv2.fillPoly(mask, adjusted_points, 255)
                frame = cv2.bitwise_and(frame, frame, mask=mask)
            
            current_objs = df[(df["frame_n"] == current_frame_n)]

            for id, x, y, MA, ma, angle in zip(
                current_objs["id"], 
                current_objs["x"], 
                current_objs["y"], 
                current_objs["MA"], 
                current_objs["ma"], 
                current_objs["ellipse_angle"]):
                
                # 楕円とテキストを描画
                cv2.ellipse(frame, ((x, y), (MA, ma), angle), (255, 0, 0), 3)
                # cv2.putText(frame, f"ID: {id}", (int(x)+10, int(y)-10), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            if pred_csv_path is not None:
                current_objs_pred = df_pred[(df_pred["frame_n"] == current_frame_n)]
                for id, x, y in zip(
                    current_objs_pred["id"], 
                    current_objs_pred["x"], 
                    current_objs_pred["y"]):
                    
                    # クラスに対応する色を取得（デフォルトは黄色）
                    color = (0, 255, 0)#class_colors.get(classes, (0, 255, 255))

                    # 楕円とテキストを描画
                    cv2.circle(frame, (int(x), int(y)), radius=3, color=color, thickness=3)
                    cv2.putText(frame, f"PRED_ID: {id}", (int(x)+10, int(y)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            current_frame_n += 1
            cv2.imshow('Tracking', frame)
            cv2.waitKey(1)
            out.write(frame)
            pbar.update(1)
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def add_ellipses_to_video_with_class(
        input_video_path, 
        input_csv_path, 
        output_video_path, 
        ROI, 
        roi_args:list , 
        specific_class:list, 
        class_colors:dict, 
        pred_csv_path=None):
    current_frame_n = 0
    cap = cv2.VideoCapture(input_video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    df = pd.read_csv(input_csv_path)
    if pred_csv_path is not None:
        df_pred = pd.read_csv(pred_csv_path)

    if ROI=="AUTO":
        if roi_args is None:
            min_y, mask = mask_analysis_region(input_video_path)
        else:
            min_y, mask = roi_args
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height-min_y))
    elif ROI=="RECT":
        if roi_args is None:
            pass
        else:
            roi_x, roi_y, roi_w, roi_h = roi_args
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (roi_w, roi_h))
    elif ROI=="POLY":
        if roi_args is None:
            points, offset, frame_height, frame_width = set_roi_polygon(input_video_path)
            adjusted_points = [np.array(points, np.int32) - np.array(offset)]
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        else:
            points, offset, adjusted_points, frame_height, frame_width = roi_args
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    else:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    with tqdm(total=total_frames, desc="Adding ellipses to frames", unit="frame") as pbar:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("フレームの読み込みに失敗しました。処理を終了します。")
                break

            if ROI=="AUTO":
                    frame = frame[min_y:, :]
                    frame = cv2.bitwise_and(frame, mask)
            elif ROI=="RECT":
                if roi_x is None:
                    roi_x, roi_y, roi_w, roi_h = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=False)
                frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            elif ROI=="POLY":
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                cv2.fillPoly(mask, adjusted_points, 255)
                frame = cv2.bitwise_and(frame, frame, mask=mask)
            
            current_objs = df[(df["frame_n"] == current_frame_n)]
            
                
            # print("current_objs", current_objs)
            # print(current_objs.describe())

            for id, x, y, MA, ma, angle, classes in zip(
                current_objs["id"], 
                current_objs["x"], 
                current_objs["y"], 
                current_objs["MA"], 
                current_objs["ma"], 
                current_objs["ellipse_angle"], 
                current_objs["class"]):
                
                # クラスに対応する色を取得（デフォルトは黄色）
                color = class_colors.get(classes, (0, 255, 255))
                
                # 特定クラスのフィルタリングがある場合
                if specific_class is not None and classes not in specific_class:
                    continue

                # 楕円とテキストを描画
                cv2.ellipse(frame, ((x, y), (MA, ma), angle), color, 2)
                cv2.putText(frame, f"ID: {id}, class{classes}", (int(x)+10, int(y)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            
            if pred_csv_path is not None:
                current_objs_pred = df_pred[(df_pred["frame_n"] == current_frame_n)]
                for id, x, y, classes in zip(
                    current_objs_pred["id"], 
                    current_objs_pred["x"], 
                    current_objs_pred["y"], 
                    current_objs_pred["class"]):
                    
                    # クラスに対応する色を取得（デフォルトは黄色）
                    color = (255, 0, 255)#class_colors.get(classes, (0, 255, 255))
                    
                    # 特定クラスのフィルタリングがある場合
                    if classes != "fish":
                        continue

                    # 楕円とテキストを描画
                    cv2.circle(frame, (int(x), int(y)), radius=3, color=color, thickness=2)
                    cv2.putText(frame, f"PRED_ID: {id}, class{classes}", (int(x)+10, int(y)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # if specific_class is None:
            #     for id, x, y, MA, ma, angle, classes in zip(current_objs["id"], current_objs["x"], current_objs["y"], current_objs["MA"], current_objs["ma"], current_objs["angle"]), current_objs["class"]:
            #         cv2.ellipse(frame, ((x, y), (MA, ma), angle), (0, 255, 255), 2)
            #         cv2.putText(frame, f"ID: {id}", (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            # else:
            #     for id, x, y, MA, ma, angle, classes in zip(current_objs["id"], current_objs["x"], current_objs["y"], current_objs["MA"], current_objs["ma"], current_objs["angle"]), current_objs["class"]:
            #         if classes in specific_class:
            #             cv2.ellipse(frame, ((x, y), (MA, ma), angle), (0, 255, 255), 2)
            #             cv2.putText(frame, f"ID: {id}", (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            current_frame_n += 1
            cv2.imshow('Tracking', frame)
            cv2.waitKey(1)
            out.write(frame)
            pbar.update(1)
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def add_ellipses_to_video_with_iou(
        input_video_path, 
        input_csv_path, 
        output_video_path, 
        ROI, 
        roi_args:list , 
        specific_class:list, 
        class_colors:dict, 
        pred_csv_path=None):
    current_frame_n = 0
    cap = cv2.VideoCapture(input_video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    df = pd.read_csv(input_csv_path)
    if pred_csv_path is not None:
        df_pred = pd.read_csv(pred_csv_path)

    if ROI=="AUTO":
        if roi_args is None:
            min_y, mask = mask_analysis_region(input_video_path)
        else:
            min_y, mask = roi_args
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height-min_y))
    elif ROI=="RECT":
        if roi_args is None:
            pass
        else:
            roi_x, roi_y, roi_w, roi_h = roi_args
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (roi_w, roi_h))
    elif ROI=="POLY":
        if roi_args is None:
            points, offset, frame_height, frame_width = set_roi_polygon(input_video_path)
            adjusted_points = [np.array(points, np.int32) - np.array(offset)]
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        else:
            points, offset, adjusted_points, frame_height, frame_width = roi_args
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    else:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    with tqdm(total=total_frames, desc="Adding ellipses to frames", unit="frame") as pbar:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("フレームの読み込みに失敗しました。処理を終了します。")
                break

            if ROI=="AUTO":
                    frame = frame[min_y:, :]
                    frame = cv2.bitwise_and(frame, mask)
            elif ROI=="RECT":
                if roi_x is None:
                    roi_x, roi_y, roi_w, roi_h = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=False)
                frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            elif ROI=="POLY":
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                cv2.fillPoly(mask, adjusted_points, 255)
                frame = cv2.bitwise_and(frame, frame, mask=mask)
            
            current_objs = df[(df["frame_n"] == current_frame_n)]
            
                
            # print("current_objs", current_objs)
            # print(current_objs.describe())

            for id, x, y, MA, ma, angle, classes, x_pred, y_pred, iou in zip(
                current_objs["id"], 
                current_objs["x"], 
                current_objs["y"], 
                current_objs["MA"], 
                current_objs["ma"], 
                current_objs["ellipse_angle"], 
                current_objs["class"], 
                current_objs["x_pred"], 
                current_objs["y_pred"], 
                current_objs["iou"]):
                
                # クラスに対応する色を取得（デフォルトは黄色）
                color = class_colors.get(classes, (0, 255, 255))
                
                # 特定クラスのフィルタリングがある場合
                if specific_class is not None and classes not in specific_class:
                    continue

                # 楕円とテキストを描画
                cv2.ellipse(frame, ((x, y), (MA, ma), angle), color, 2)
                cv2.putText(frame, f"ID: {id}, class{classes}", (int(x)+10, int(y)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                # 予測位置とIoUを描画
                cv2.ellipse(frame, ((x_pred, y_pred), (MA, ma), angle), (255, 0, 255), 2)
                cv2.putText(frame, f"PRED_ID: {id}, IoU: {iou:.2f}", (int(x_pred)+10, int(y_pred)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
            
            # if pred_csv_path is not None:
            #     current_objs_pred = df_pred[(df_pred["frame_n"] == current_frame_n)]
            #     for id, x, y, classes in zip(
            #         current_objs_pred["id"], 
            #         current_objs_pred["x"], 
            #         current_objs_pred["y"], 
            #         current_objs_pred["class"]):
                    
            #         # クラスに対応する色を取得（デフォルトは黄色）
            #         color = (255, 0, 255)#class_colors.get(classes, (0, 255, 255))
                    
            #         # 特定クラスのフィルタリングがある場合
            #         if classes != "fish":
            #             continue

            #         # 楕円とテキストを描画
            #         cv2.circle(frame, (int(x), int(y)), radius=3, color=color, thickness=2)
            #         cv2.putText(frame, f"PRED_ID: {id}, class{classes}", (int(x)+10, int(y)-10), 
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            current_frame_n += 1
            cv2.imshow('Tracking', frame)
            cv2.waitKey(1)
            out.write(frame)
            pbar.update(1)
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

# 魚のデータフレーム（df_with_iou）を引数として受け取り、各フレームごとに魚の位置を描画した動画を作成する関数
def add_ellipses_to_video_is_fish(
        input_video_path, 
        # input_csv_path, 
        df_with_iou,# df_with_iouのデータフレームから、魚であるもののみのデータを代入
        output_video_path, 
        ROI, 
        roi_args:list , 
        specific_class:list, 
        class_colors:dict=None, 
        pred_csv_path=None):
    current_frame_n = 0
    cap = cv2.VideoCapture(input_video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # df = pd.read_csv(input_csv_path)
    # if pred_csv_path is not None:
    #     df_pred = pd.read_csv(pred_csv_path)

    if ROI=="AUTO":
        if roi_args is None:
            min_y, mask = mask_analysis_region(input_video_path)
        else:
            min_y, mask = roi_args
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height-min_y))
    elif ROI=="RECT":
        if roi_args is None:
            pass
        else:
            roi_x, roi_y, roi_w, roi_h = roi_args
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (roi_w, roi_h))
    elif ROI=="POLY":
        if roi_args is None:
            points, offset, frame_height, frame_width = set_roi_polygon(input_video_path)
            adjusted_points = [np.array(points, np.int32) - np.array(offset)]
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        else:
            points, offset, adjusted_points, frame_height, frame_width = roi_args
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    else:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    with tqdm(total=total_frames, desc="Adding ellipses to frames", unit="frame") as pbar:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("フレームの読み込みに失敗しました。処理を終了します。")
                break

            if ROI=="AUTO":
                    frame = frame[min_y:, :]
                    frame = cv2.bitwise_and(frame, mask)
            elif ROI=="RECT":
                if roi_x is None:
                    roi_x, roi_y, roi_w, roi_h = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=False)
                frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            elif ROI=="POLY":
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                cv2.fillPoly(mask, adjusted_points, 255)
                frame = cv2.bitwise_and(frame, frame, mask=mask)
            
            current_objs = df_with_iou[(df_with_iou["frame_n"] == current_frame_n)]
            
                
            # print("current_objs", current_objs)
            # print(current_objs.describe())

            for id, x, y, MA, ma, angle in zip(
                current_objs["id"], 
                current_objs["x"], 
                current_objs["y"], 
                current_objs["MA"], 
                current_objs["ma"], 
                current_objs["ellipse_angle"], 
                # current_objs["x_pred"], 
                # current_objs["y_pred"], 
                ):
                
                # クラスに対応する色を取得（デフォルトは黄色）
                # color = class_colors.get(classes, (0, 255, 255))
                
                # 特定クラスのフィルタリングがある場合
                # if specific_class is not None and classes not in specific_class:
                #     continue

                # 楕円とテキストを描画
                cv2.ellipse(frame, ((x, y), (MA, ma), angle), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {id}, class{"fish"}", (int(x)+10, int(y)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                # # 予測位置とIoUを描画
                # cv2.ellipse(frame, ((x_pred, y_pred), (MA, ma), angle), (255, 0, 255), 2)
                # cv2.putText(frame, f"PRED_ID: {id}, IoU: {iou:.2f}", (int(x_pred)+10, int(y_pred)-10), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
            
            # if pred_csv_path is not None:
            #     current_objs_pred = df_pred[(df_pred["frame_n"] == current_frame_n)]
            #     for id, x, y, classes in zip(
            #         current_objs_pred["id"], 
            #         current_objs_pred["x"], 
            #         current_objs_pred["y"], 
            #         current_objs_pred["class"]):
                    
            #         # クラスに対応する色を取得（デフォルトは黄色）
            #         color = (255, 0, 255)#class_colors.get(classes, (0, 255, 255))
                    
            #         # 特定クラスのフィルタリングがある場合
            #         if classes != "fish":
            #             continue

            #         # 楕円とテキストを描画
            #         cv2.circle(frame, (int(x), int(y)), radius=3, color=color, thickness=2)
            #         cv2.putText(frame, f"PRED_ID: {id}, class{classes}", (int(x)+10, int(y)-10), 
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            current_frame_n += 1
            cv2.imshow('Tracking', frame)
            cv2.waitKey(1)
            out.write(frame)
            pbar.update(1)
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = "../1_data_raw/ayu.mp4"
    input_csv_path = "C:/Users/koder/OneDrive - Tokyo University of Agriculture and Technology/0_research/202407_back_diff/2_data/kalmanfilter_v3/ayu/manual_roi/ayu/config2/c4_d50_a10_h3_md50/c4_d50_a10_h3_md50.csv"
    output_video_path = "C:/Users/koder/OneDrive - Tokyo University of Agriculture and Technology/0_research/202407_back_diff/2_data/kalmanfilter_v3/ayu/manual_roi/ayu/config2/c4_d50_a10_h3_md50/c4_d50_a10_h3_md50_mogresult.avi"
    pred_csv_path = None#"C:/Users/koder/OneDrive - Tokyo University of Agriculture and Technology/0_research/202407_back_diff/2_data/kalmanfilter_v3/ayu/manual_roi/ayu/config2/c4_d50_a10_h3_md50/c4_d50_a10_h3_md50_pred.csv"
    ROI="POLY"
    roi_args = None
    specific_class = None
    class_colors = {"fish": (255, 0, 0), "debris": (0, 255, 0), "noise": (0, 0, 255)}

    output_video_path = check_file_exists(output_video_path)
    
    add_ellipses_to_video(input_video_path, input_csv_path, output_video_path, ROI, roi_args, specific_class, class_colors, pred_csv_path)