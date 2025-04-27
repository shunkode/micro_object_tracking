"""
データ数が少ないときのため、交差検証を導入
"""

import pandas as pd
import os
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
#pydotplus・graghvizで可視化
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report, make_scorer, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib
matplotlib.use('Agg') # 非GUIモードへ切り替え→邪魔なウィンドウの表示を防止する
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.base import clone
import copy
from tqdm import tqdm

from output_csv import concat_fish_n_random_noise
from output_csv import define_fish_notfish_by_id
from post_processing import calc_parameters
from output_csv import extract_ids_n_merge_pred

def output_trees(forest_model, 
                 output_dir:str, 
                 feature_names_l:list,
                 target_names_l:list):
    # 各決定木をループで処理
    for i, estimator in enumerate(forest_model.estimators_):
        dot_data = export_graphviz(estimator,
                                out_file=None,
                                filled=True,
                                rounded=True,
                                class_names=target_names_l,
                                feature_names=feature_names_l,
                                special_characters=True)
        # DOTファイルを保存
        os.makedirs(os.path.join(output_dir, "tree"), exist_ok=True)
        dot_file_path = os.path.join(output_dir,
                                    "tree", 
                                    f"tree_{i}.dot")
        with open(dot_file_path, "w") as f:
            f.write(dot_data)
        # PNG画像として保存
        graph = graph_from_dot_data(dot_data)
        graph.write_png(dot_file_path.replace(".dot", ".png"))

def plot_partial_dependence(models, x_train, feature_names, output_dir, label, model_info):
    """
    指定したモデル群に対して Partial Dependence Plot (PDP) を作成し保存する
    """
    
    pdp_dir = os.path.join(output_dir, "PDP", f"pdp_{label}")
    os.makedirs(pdp_dir, exist_ok=True)

    model_info_path = os.path.join(pdp_dir, f"model_info_{label}.txt")
    with open(model_info_path, "w") as f:
        f.write("rf_seed,kf_seed,cv_fold,f1\n")
        # for info in model_info:
        f.write(",".join(map(str, model_info)) + "\n")
    
    for i, model in enumerate(models):
        for j, feature_name in enumerate(feature_names):
            fig, ax = plt.subplots(figsize=(6, 4))
            disp = PartialDependenceDisplay.from_estimator(
                model,
                X=x_train,
                features=[j],
                kind="average",  # PDP(average) を作成
                #target=1,
                ax=ax, 
                n_jobs=-1
            )
            disp.figure_.suptitle(f"PDP of '{feature_name}' (Model {i})")
            plt.tight_layout()
            plt.savefig(os.path.join(pdp_dir, f"pdp_model{i}_{feature_name}.png"))
            plt.clf()
            plt.close(fig)

def generate_pdp_for_top_bottom_models(output_dir):
    """
    f1スコアの上位5つと下位5つのモデルに対して PDP を作成する
    """
    scores_df = pd.read_csv(os.path.join(output_dir, "scores_all_data.csv"))
    print(scores_df)
    
    # f1スコアが上位5つのモデルを取得
    top_models = scores_df.nlargest(5, 'f1')[['rf_seed', "kf_seed", 'cv_fold', 'f1']]
    print("top_models", top_models)
    # f1スコアが下位5つのモデルを取得
    bottom_models = scores_df.nsmallest(5, 'f1')[['rf_seed', "kf_seed", 'cv_fold', 'f1']]
    print("bottom_models", bottom_models)
    
    top_model_instances = []
    bottom_model_instances = []
    
    # モデルのロード
    for _, row in top_models.iterrows():
        rf_seed = str(int(row['rf_seed']))
        kf_seed = str(int(row['kf_seed']))
        cv_fold = str(int(row['cv_fold']))
        model_path = os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{kf_seed}", f"fold_{cv_fold}", "model.joblib")
        x_train_path = os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{kf_seed}", f"fold_{cv_fold}", "data", "x_train.csv")
        if os.path.exists(model_path):
            top_model_instances.append((joblib.load(model_path), pd.read_csv(x_train_path), [rf_seed, kf_seed, cv_fold, row["f1"]]))
    
    for _, row in bottom_models.iterrows():
        rf_seed = str(int(row['rf_seed']))
        kf_seed = str(int(row['kf_seed']))
        cv_fold = str(int(row['cv_fold']))
        model_path = os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{kf_seed}", f"fold_{cv_fold}", "model.joblib")
        x_train_path = os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{kf_seed}", f"fold_{cv_fold}", "data", "x_train.csv")
        if os.path.exists(model_path):
            bottom_model_instances.append((joblib.load(model_path), pd.read_csv(x_train_path), [rf_seed, kf_seed, cv_fold, row["f1"]]))
    # PDP の作成
    for i, (model, x_train, model_info) in enumerate(top_model_instances):
        plot_partial_dependence([model], x_train.iloc[:, 2:], x_train.columns[2:], output_dir, f"top5_model{i}", model_info)
    
    for i, (model, x_train, model_info) in enumerate(bottom_model_instances):
        plot_partial_dependence([model], x_train.iloc[:, 2:], x_train.columns[2:], output_dir, f"bottom5_model{i}", model_info)


def random_forest(input_feature_df:pd.DataFrame, 
                  input_target_df:pd.DataFrame, 
                  input_feature_id_df:pd.DataFrame, 
                  output_dir:str, 
                  n_split:int=5, 
                  rf_seeds:int=5,# ランダムフォレストのシード数
                  kf_seeds:int=5,  # 交差検証のシード数
                  input_feature_all=None, 
                  input_target_all=None, 
                  input_feature_id_all=None
                  ):
    x = input_feature_df#.values
    y = input_target_df#.values

    scores_all_data_path = os.path.join(output_dir, "scores_all_data.csv")
    with open(scores_all_data_path, "w") as f:
        f.write("rf_seed,kf_seed,cv_fold,accuracy,balanced_accuracy,precision,precision_weighted,recall,recall_weighted,f1,f1_weighted,TP,FP,FN,TN\n")
    with open(os.path.join(output_dir, "scores_all_test.csv"), "w") as f:
        f.write("rf_seed,kf_seed,cv_fold,accuracy,balanced_accuracy,precision,precision_weighted,recall,recall_weighted,f1,f1_weighted\n")
    with open(os.path.join(output_dir, "scores_all_train.csv"), "w") as f:
        f.write("rf_seed,kf_seed,cv_fold,accuracy,balanced_accuracy,precision,precision_weighted,recall,recall_weighted,f1,f1_weighted\n")
    feature_importance_all_path = os.path.join(output_dir, "feature_importance_all.csv")
    with open(os.path.join(feature_importance_all_path), "w") as f:
        f.write("rf_seed,kf_seed,cv_fold," + ",".join(input_feature_df.columns) + "\n")

    for rf_seed in tqdm(range(rf_seeds), desc="Random Forest Seed"):
        # RandomForestClassifier のインスタンスを先に作成
        clf_cv = RandomForestClassifier(n_estimators=200, random_state=rf_seed)
        if os.path.exists(os.path.join(output_dir, "parameters_rf_cv.csv")):
            # パラメータを保存
            parameters = clf_cv.get_params()
            params_file_path = os.path.join(output_dir, "parameters_rf_cv.csv")
            with open(params_file_path, "w") as f:
                f.write("Parameter,Value\n")
                for param, value in parameters.items():
                    f.write(f"{param},{value}\n")

        best_dict = {"best_random_state": [0], "best_fold": [0], "best_f1_mean": [0]}
        for random_state in range(kf_seeds):
            f1_dict = {}
            os.makedirs(os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{random_state}"), exist_ok=True)
            # k分割交差検証 (例: 5分割) でスコアを確認
            # kf = KFold(n_splits=n_split, shuffle=True, random_state=random_state)
            # k分割交差検証 (例: 5分割) でスコアを確認
            kf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=random_state)
            # 取得したいスコアを設定
            # scoring = {"accuracy": "accuracy", 
            #         "balanced_accuracy": "balanced_accuracy", 
            #         "precision": "precision", 
            #         "precision_weighted": "precision_weighted",
            #         "recall": "recall", 
            #         "recall_weighted": "recall_weighted", 
            #         "f1": "f1", 
            #         "f1_weighted": "f1_weighted", 
            #         } 
            scoring = {
                "accuracy": make_scorer(accuracy_score),
                "balanced_accuracy": make_scorer(balanced_accuracy_score),
                "precision": make_scorer(precision_score, zero_division=0),
                "precision_weighted": make_scorer(
                    precision_score, 
                    average="weighted", 
                    zero_division=0
                ),
                "recall": make_scorer(recall_score, zero_division=0),
                "recall_weighted": make_scorer(
                    recall_score, 
                    average="weighted", 
                    zero_division=0
                ),
                "f1": make_scorer(f1_score, zero_division=0),
                "f1_weighted": make_scorer(
                    f1_score, 
                    average="weighted", 
                    zero_division=0
                ),
            }
            
            # 交差検証でスコアを確認
            # cv_scores = cross_val_score(clf_cv, x_train, y_train.ravel(), cv=kf, scoring='accuracy')
            cv_scores = cross_validate(clf_cv, 
                                    x, 
                                    y, 
                                    cv=kf, 
                                    scoring=scoring, 
                                    return_train_score=True, 
                                    return_estimator=True, 
                                    return_indices=True, 
                                    n_jobs=5)
                
            for i in range(n_split):
                os.makedirs(os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{random_state}", f"fold_{i}", "data"), exist_ok=True)
                # train, test のデータを保存
                train_index, test_index = cv_scores["indices"]["train"][i], cv_scores["indices"]["test"][i]
                x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                # x_train, x_test = x[train_index], x[test_index]
                # y_train, y_test = y[train_index], y[test_index]
                # 訓練データのID, テストデータのIDをそれぞれ取得
                # x_train_id, x_test_id = input_feature_id_df.values[train_index], input_feature_id_df.values[test_index]
                x_train_id_df, x_test_id_df = input_feature_id_df.iloc[train_index], input_feature_id_df.iloc[test_index]

                # 訓練データとテストデータをそれぞれ取得
                # x_train_df = pd.DataFrame(x_train, columns=input_feature_df.columns)
                # x_test_df = pd.DataFrame(x_test, columns=input_feature_df.columns)
                # y_train_df = pd.DataFrame(y_train, columns=[input_target_df.name])
                # y_test_df = pd.DataFrame(y_test, columns=[input_target_df.name])
                # x_train_id_df = pd.DataFrame(x_train_id, columns=["id", "frame_n"])
                # x_test_id_df = pd.DataFrame(x_test_id, columns=["id", "frame_n"])
                # 訓練データのID, テストデータのIDをそれぞれ加える
                # x_train_df = pd.concat([x_train_id_df, x_train_df], axis=1)
                # x_test_df = pd.concat([x_test_id_df, x_test_df], axis=1)
                # y_train_df = pd.concat([x_train_id_df, y_train_df], axis=1)
                # y_test_df = pd.concat([x_test_id_df, y_test_df], axis=1)
                # 訓練データのID, テストデータのIDをそれぞれ加える
                x_train_df = pd.concat([x_train_id_df, x_train], axis=1)
                x_test_df = pd.concat([x_test_id_df, x_test], axis=1)
                y_train_df = pd.concat([x_train_id_df, y_train], axis=1)
                y_test_df = pd.concat([x_test_id_df, y_test], axis=1)
                # データを保存
                x_train_df.to_csv(os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{random_state}", f"fold_{i}", "data", "x_train.csv"), index=False)
                x_test_df.to_csv(os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{random_state}", f"fold_{i}", "data", "x_test.csv"), index=False)
                y_train_df.to_csv(os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{random_state}", f"fold_{i}", "data", "y_train.csv"), index=False)
                y_test_df.to_csv(os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{random_state}", f"fold_{i}", "data", "y_test.csv"), index=False)
                os.makedirs(os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{random_state}", f"fold_{i}", "scores"), exist_ok=True)
                # スコアを保存
                cv_scores_file_path = os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{random_state}", f"fold_{i}", "scores", "scores_test_cv.csv")
                with open(cv_scores_file_path, "w") as f:
                    f.write("score,score_value\n")
                    f.write(f"accuracy,{cv_scores['test_accuracy'][i]}\n")
                    f.write(f"balanced_accuracy,{cv_scores['test_balanced_accuracy'][i]}\n")
                    f.write(f"precision,{cv_scores['test_precision'][i]}\n")
                    f.write(f"precision_weighted,{cv_scores['test_precision_weighted'][i]}\n")
                    f.write(f"recall,{cv_scores['test_recall'][i]}\n")
                    f.write(f"recall_weighted,{cv_scores['test_recall_weighted'][i]}\n")
                    f.write(f"f1,{cv_scores['test_f1'][i]}\n")
                    f.write(f"f1_weighted,{cv_scores['test_f1_weighted'][i]}\n")
                
                f1_dict[f"{random_state}_{i}"] = cv_scores['test_f1'][i]
                
                cv_scores_file_path = os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{random_state}", f"fold_{i}", "scores", "scores_train_cv.csv")
                with open(cv_scores_file_path, "w") as f:
                    f.write("score,score_value\n")
                    f.write(f"accuracy,{cv_scores['train_accuracy'][i]}\n")
                    f.write(f"balanced_accuracy,{cv_scores['train_balanced_accuracy'][i]}\n")
                    f.write(f"precision,{cv_scores['train_precision'][i]}\n")
                    f.write(f"precision_weighted,{cv_scores['train_precision_weighted'][i]}\n")
                    f.write(f"recall,{cv_scores['train_recall'][i]}\n")
                    f.write(f"recall_weighted,{cv_scores['train_recall_weighted'][i]}\n")
                    f.write(f"f1,{cv_scores['train_f1'][i]}\n")
                    f.write(f"f1_weighted,{cv_scores['train_f1_weighted'][i]}\n")

                with open(os.path.join(output_dir, "scores_all_test.csv"), "a") as f:
                    f.write(f"{rf_seed},{random_state},{i},{cv_scores['test_accuracy'][i]},{cv_scores['test_balanced_accuracy'][i]},{cv_scores['test_precision'][i]},{cv_scores['test_precision_weighted'][i]},{cv_scores['test_recall'][i]},{cv_scores['test_recall_weighted'][i]},{cv_scores['test_f1'][i]},{cv_scores['test_f1_weighted'][i]}\n")

                with open(os.path.join(output_dir, "scores_all_train.csv"), "a") as f:
                    f.write(f"{rf_seed},{random_state},{i},{cv_scores['train_accuracy'][i]},{cv_scores['train_balanced_accuracy'][i]},{cv_scores['train_precision'][i]},{cv_scores['train_precision_weighted'][i]},{cv_scores['train_recall'][i]},{cv_scores['train_recall_weighted'][i]},{cv_scores['train_f1'][i]},{cv_scores['train_f1_weighted'][i]}\n")

                # モデルの保存
                model_file_path = os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{random_state}", f"fold_{i}", "model.joblib")
                dump(cv_scores["estimator"][i], model_file_path)

                # 特徴量の重要度を出力
                importances = cv_scores["estimator"][i].feature_importances_
                labels = input_feature_df.columns
                plt.figure(figsize=(10, 6))
                plt.barh(y=range(len(importances)), width=importances)
                plt.yticks(ticks=range(len(labels)), labels=labels)
                plt.xlabel("Feature Importance")
                plt.title(f"Feature Importance (Fold {i})")
                # plt.pause(1)
                plt.savefig(os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{random_state}", f"fold_{i}", "scores", "feature_importance.png"))
                plt.clf()
                plt.close()

                # 特徴量の重要度をCSVに保存
                with open(feature_importance_all_path, "a") as f:
                    f.write(f"{rf_seed},{random_state},{i}," + ",".join(map(str, importances)) + "\n")

                # # PDP用の出力先フォルダ(出力に時間が必要なため、必要に応じてコメントアウト)
                # pdp_dir = os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{random_state}", f"fold_{i}", "pdp")
                # os.makedirs(pdp_dir, exist_ok=True)
                # # トレーニングデータを DataFrame に戻す（ID列を取り除く）
                # x_train_features_only = x_train_df[labels]  # 特徴量の列だけに絞る
                # trained_model = cv_scores["estimator"][i]
                # 各特徴量ごとにPDPを保存 (二値分類を想定し, target=1 のPDP)
                # for j, feature_name in enumerate(labels):
                #     fig, ax = plt.subplots(figsize=(6, 4))
                #     disp = PartialDependenceDisplay.from_estimator(
                #         trained_model,
                #         X=x_train_features_only,
                #         features=[j],
                #         kind="average",  # PDP(average) or ICE(individual)を指定
                #         target=1,
                #         ax=ax, 
                #         n_jobs=-1
                #     )
                #     disp.figure_.suptitle(f"PDP of '{feature_name}' (fold_{i})")
                #     plt.tight_layout()
                #     plt.savefig(os.path.join(pdp_dir, f"pdp_fold{i}_{feature_name}.png"))
                #     plt.clf()
                #     plt.close(fig)
                    # pdp_result = partial_dependence(cv_scores["estimator"][i], 
                    #                                 x_train_features_only, 
                    #                                 features=[j], 
                    #                                 kind="average")
                    # print(pdp_result)


                # 分類レポートの生成
                report = classification_report(y_test, cv_scores["estimator"][i].predict(x_test), output_dict=True, labels=[0, 1], zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                report_file_path = os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{random_state}", f"fold_{i}", "scores","classification_report.csv")
                report_df.to_csv(report_file_path, index=True)

                # 予測結果の算出・保存
                y_test_pred = cv_scores["estimator"][i].predict(x_test)
                test_results_df = pd.DataFrame({
                    "id": x_test_id_df["id"].values,
                    "frame_n": x_test_id_df["detected_frame_n"].values,
                    "actual": y_test.values,#.ravel(),
                    "predicted": y_test_pred#.ravel()
                })
                test_results_path = os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{random_state}", f"fold_{i}", "predictions.csv")
                test_results_df.to_csv(test_results_path, index=False)

                # 全データに対する予測結果の算出・保存
                y_pred_all = cv_scores["estimator"][i].predict(input_feature_all)
                accuracy_all = accuracy_score(input_target_all, y_pred_all)
                balanced_accuracy_all = balanced_accuracy_score(input_target_all, y_pred_all)
                precision_all = precision_score(input_target_all, y_pred_all, zero_division=0)
                precision_weighted_all = precision_score(
                    input_target_all, y_pred_all, average="weighted", zero_division=0
                )
                recall_all = recall_score(input_target_all, y_pred_all, zero_division=0)
                recall_weighted_all = recall_score(
                    input_target_all, y_pred_all, average="weighted", zero_division=0
                )
                f1_all = f1_score(input_target_all, y_pred_all, zero_division=0)
                f1_weighted_all = f1_score(
                    input_target_all, y_pred_all, average="weighted", zero_division=0
                )
                cm_all = confusion_matrix(input_target_all, y_pred_all, labels=[0, 1])
                tn_all, fp_all, fn_all, tp_all = cm_all.ravel()
                # 全データのスコアをまとめてCSVに追記 (上で用意した scores_all_data.csv へ)
                with open(scores_all_data_path, "a") as f:
                    f.write(
                        f"{rf_seed},{random_state},{i},"
                        f"{accuracy_all},{balanced_accuracy_all},"
                        f"{precision_all},{precision_weighted_all},"
                        f"{recall_all},{recall_weighted_all},"
                        f"{f1_all},{f1_weighted_all},"
                        f"{tp_all},{fp_all},{fn_all},{tn_all}\n"
                    )

            # スコアの平均値・標準偏差を保存
            cv_scores_file_path = os.path.join(output_dir, f"rf_seed{rf_seed}", f"cv_{random_state}", "scores_cv_all.csv")
            with open(cv_scores_file_path, "w") as f:
                f.write("score,score_value\n")

                best_f1_key = max(f1_dict, key=f1_dict.get)
                best_random_state, best_fold = best_f1_key.split('_')
                best_f1_score = f1_dict[best_f1_key]
                f.write(f"best_random_state,{best_random_state}\n")
                f.write(f"best_fold,{best_fold}\n")
                f.write(f"best_f1_score,{best_f1_score}\n")
                
                f.write(f"accuracy_mean,{cv_scores['test_accuracy'].mean()}\n")
                f.write(f"accuracy_std,{cv_scores['test_accuracy'].std()}\n")
                f.write(f"balanced_accuracy_mean,{cv_scores['test_balanced_accuracy'].mean()}\n")
                f.write(f"balanced_accuracy_std,{cv_scores['test_balanced_accuracy'].std()}\n")
                f.write(f"precision_mean,{cv_scores['test_precision'].mean()}\n")
                f.write(f"precision_std,{cv_scores['test_precision'].std()}\n")
                f.write(f"precision_weighted_mean,{cv_scores['test_precision_weighted'].mean()}\n")
                f.write(f"precision_weighted_std,{cv_scores['test_precision_weighted'].std()}\n")
                f.write(f"recall_mean,{cv_scores['test_recall'].mean()}\n")
                f.write(f"recall_std,{cv_scores['test_recall'].std()}\n")
                f.write(f"recall_weighted_mean,{cv_scores['test_recall_weighted'].mean()}\n")
                f.write(f"recall_weighted_std,{cv_scores['test_recall_weighted'].std()}\n")
                f1_mean = cv_scores['test_f1'].mean()
                f.write(f"f1_mean,{cv_scores['test_f1'].mean()}\n")
                f.write(f"f1_std,{cv_scores['test_f1'].std()}\n")
                f.write(f"f1_weighted_mean,{cv_scores['test_f1_weighted'].mean()}\n")
                f.write(f"f1_weighted_std,{cv_scores['test_f1_weighted'].std()}\n")

            # 最もスコアが良いものを保存
            if f1_mean > best_dict["best_f1_mean"][0]:
                best_dict["best_random_state"] = [best_random_state]
                best_dict["best_fold"] = [best_fold]
                best_dict["best_f1_mean"] = [f1_mean]
            elif f1_mean == best_dict["best_f1_mean"][0]:
                best_dict["best_random_state"].append(best_random_state)
                best_dict["best_fold"].append(best_fold)
                best_dict["best_f1_mean"].append(f1_mean)

            del cv_scores
                
            
    # 最もf1スコアが良いものをCSVファイルで出力
    best_scores_path = os.path.join(output_dir, "best_model_info.csv")
    # 最もf1スコアが高いものを出力
    scores_df = pd.read_csv(os.path.join(output_dir, "scores_all_test.csv"))
    best_f1_score = scores_df['f1'].max()
    best_f1_rows = scores_df[scores_df['f1'] == best_f1_score]
    best_f1_rows.to_csv(best_scores_path, header=True, index=False)
    # 全スコアの平均値・標準偏差を保存
    summary_scores_path = os.path.join(output_dir, "scores_summary_test.csv")
    summary_scores = {
        "score_name": ["accuracy", "balanced_accuracy", "precision", "precision_weighted", "recall", "recall_weighted", "f1", "f1_weighted"],
        "mean": [
            scores_df["accuracy"].mean(),
            scores_df["balanced_accuracy"].mean(),
            scores_df["precision"].mean(),
            scores_df["precision_weighted"].mean(),
            scores_df["recall"].mean(),
            scores_df["recall_weighted"].mean(),
            scores_df["f1"].mean(),
            scores_df["f1_weighted"].mean()
        ],
        "std": [
            scores_df["accuracy"].std(),
            scores_df["balanced_accuracy"].std(),
            scores_df["precision"].std(),
            scores_df["precision_weighted"].std(),
            scores_df["recall"].std(),
            scores_df["recall_weighted"].std(),
            scores_df["f1"].std(),
            scores_df["f1_weighted"].std()
        ]
    }
    summary_df = pd.DataFrame(summary_scores)
    summary_df.to_csv(summary_scores_path, index=False)


    # 最もf1スコアが良いものをCSVファイルで出力
    best_scores_path = os.path.join(output_dir, "best_model_info_all_data.csv")
    # 最もf1スコアが高いものを出力
    scores_df = pd.read_csv(os.path.join(output_dir, "scores_all_data.csv"))
    best_f1_score = scores_df['f1'].max()
    best_f1_rows = scores_df[scores_df['f1'] == best_f1_score]
    best_f1_rows.to_csv(best_scores_path, header=True, index=False)
    # 全スコアの平均値・標準偏差を保存
    summary_scores_path = os.path.join(output_dir, "scores_summary_all_data.csv")
    summary_scores = {
        "score_name": ["accuracy", "balanced_accuracy", "precision", "precision_weighted", "recall", "recall_weighted", "f1", "f1_weighted"],
        "mean": [
            scores_df["accuracy"].mean(),
            scores_df["balanced_accuracy"].mean(),
            scores_df["precision"].mean(),
            scores_df["precision_weighted"].mean(),
            scores_df["recall"].mean(),
            scores_df["recall_weighted"].mean(),
            scores_df["f1"].mean(),
            scores_df["f1_weighted"].mean()
        ],
        "std": [
            scores_df["accuracy"].std(),
            scores_df["balanced_accuracy"].std(),
            scores_df["precision"].std(),
            scores_df["precision_weighted"].std(),
            scores_df["recall"].std(),
            scores_df["recall_weighted"].std(),
            scores_df["f1"].std(),
            scores_df["f1_weighted"].std()
        ]
    }
    summary_df = pd.DataFrame(summary_scores)
    summary_df.to_csv(summary_scores_path, index=False)

    # 特徴量の重要度の平均値・標準偏差を算出
    feature_importance_df = pd.read_csv(feature_importance_all_path)
    # 先頭3列(rf_seed, kf_seed, cv_fold)以降が実際の特徴量の列
    feature_columns = feature_importance_df.columns[3:]
    mean_importances = feature_importance_df[feature_columns].mean()
    std_importances = feature_importance_df[feature_columns].std()
    summary_df = pd.DataFrame({
        "feature": feature_columns,
        "mean_importance": mean_importances,
        "std_importance": std_importances
    })
    summary_df.to_csv(os.path.join(output_dir, "feature_importance_summary.csv"), index=False)

    # 特徴量の重要度の平均値をプロット
    plt.figure(figsize=(10, 6))
    plt.barh(y=range(len(mean_importances)), width=mean_importances, xerr=std_importances, capsize=5)
    plt.yticks(ticks=range(len(feature_columns)), labels=feature_columns)
    plt.xlabel("Mean Feature Importance")
    plt.title("Feature Importance with Standard Deviation")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance_summary.png"))
    plt.clf()
    plt.close()

    # PDPの作成
    generate_pdp_for_top_bottom_models(output_dir)

    

    return# y_train_score, y_test_score

def main(input_csv_path:str, 
         output_dir:str, 
         fish_ids:list, 
         WITH_PRED:bool,
         pred_csv_path:str, 
         num_ids:int|str, 
         radius:int=3, 
         n_split:int=5, 
         rf_seeds:int=5, 
         kf_seeds:int=5):
    # output_dirが重複しないように設定
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    if WITH_PRED:
        output_dir = os.path.join(output_dir, f"rf_v2_numnoise{num_ids}_withpred_{timestamp}")
    else:
        output_dir = os.path.join(output_dir, f"rf_v2_numnoise{num_ids}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 今回設定したパラメータを保存
    with open(os.path.join(output_dir, "parameters_input.csv"), "w") as f:
        f.write("Parameter,Value\n")
        f.write(f"input_csv_path,{input_csv_path}\n")
        f.write(f"fish_ids,{fish_ids}\n")
        f.write(f"WITH_PRED,{WITH_PRED}\n")
        f.write(f"pred_csv_path,{pred_csv_path}\n")
        f.write(f"num_ids,{num_ids}\n")
        f.write(f"radius,{radius}\n")
        f.write(f"n_split,{n_split}\n")
        f.write(f"rf_seeds,{rf_seeds}\n")
        f.write(f"kf_seeds,{kf_seeds}\n")

    # データの読み込み・加工
    if num_ids == "all":
        df_actual = pd.read_csv(input_csv_path)
        if WITH_PRED:
            # 予測結果を取得し、実測結果に結合する
            df_fish, df_noise = extract_ids_n_merge_pred(
                input_csv_path=df_actual, 
                pred_csv_path=pred_csv_path, 
                output_csv_path=f"{output_dir}/merge_observed_pred.csv", 
                target_ids=fish_ids
                )
            df_actual = pd.concat([df_fish, df_noise], axis=0)
            df_actual = df_actual.sort_values(by=["id", "frame_n"])
    else:
        # fish_idsと一致するIDの行と、
        # それ以外の行から、ランダムにnum_ids個の行を抽出
        # WITH_PREDがTrueの場合、pred_csv_pathから予測結果を取得し、実測結果に結合する
        df_actual = concat_fish_n_random_noise(
            input_csv_path, 
            output_csv_path=f"{output_dir}/fish_n_random_noise.csv", 
            target_ids=fish_ids, 
            WITH_PRED=WITH_PRED, 
            pred_csv_path=pred_csv_path, 
            num_ids=num_ids
        )
    # fish_idsに含まれるIDを持つ行に対して、class列に1を代入. それ以外は0を代入
    df_actual = define_fish_notfish_by_id(df_actual, None, fish_ids)

    # 特徴量計算
    output_actual_csv_path = os.path.join(output_dir, "actual.csv")
    output_unique_csv_path = os.path.join(output_dir, "unique.csv")
    _, df_unique = calc_parameters(df_actual, 
                                   output_actual_csv_path, 
                                   output_unique_csv_path, 
                                   pred_csv_path, 
                                   radius, 
                                   fps=float(59.94))

    input_target_df = df_unique["class"].copy()
    input_feature_id_df = df_unique[["id", "detected_frame_n"]].copy()
    # input_feature_df = df_unique.drop(["id", "x", "y", "detected_frame_n", "detected_time(s)", "class"], axis=1).copy()
    input_feature_df = df_unique.drop(["id", "x", "y", "detected_frame_n", "detected_time(s)", "class", "average_iou", "median_r_std", "median_g_std", "median_b_std", "mean_color_std_rms", "mean_r_std", "mean_g_std", "mean_b_std", "average_vector_angle", "average_distance(speed)", "average_signed_magnitude", "distance_sum", "xy_coords"], axis=1).copy()
    # 各パラメータの相関係数を取得（多重共線性の確認）
    corr = input_feature_df.corr()
    corr.to_csv(os.path.join(output_dir, "parameter_correlation.csv"))

    if num_ids != "all":
        df_actual = pd.read_csv(input_csv_path)
        if WITH_PRED:
            # 予測結果を取得し、実測結果に結合する
            df_fish, df_noise = extract_ids_n_merge_pred(
                input_csv_path=df_actual, 
                pred_csv_path=pred_csv_path, 
                output_csv_path=f"{output_dir}/merge_observed_pred.csv", 
                target_ids=fish_ids
                )
            df_actual = pd.concat([df_fish, df_noise], axis=0)
            df_actual = df_actual.sort_values(by=["id", "frame_n"])
        # fish_idsに含まれるIDを持つ行に対して、class列に1を代入. それ以外は0を代入
        df_actual = define_fish_notfish_by_id(df_actual, None, fish_ids)

        # 特徴量計算
        output_actual_csv_path = None
        output_unique_csv_path = None
        _, df_unique = calc_parameters(df_actual, 
                                    output_actual_csv_path, 
                                    output_unique_csv_path, 
                                    pred_csv_path, 
                                    radius, 
                                    fps=float(59.94))
        input_target_all = df_unique["class"].copy()#.values
        input_feature_id_all = df_unique[["id", "detected_frame_n"]].copy()#.values
        # input_feature_all = df_unique.drop(["id", "x", "y", "detected_frame_n", "detected_time(s)", "class"], axis=1).copy()#.values
        input_feature_all = df_unique.drop(["id", "x", "y", "detected_frame_n", "detected_time(s)", "class", "average_iou", "median_color_std_rms", "median_g_std", "median_b_std", "mean_color_std_rms", "mean_g_std", "mean_b_std", "average_vector_angle", "average_distance(speed)", "distance_sum", "xy_coords", "y_coords", "x_coords", "frame_range"], axis=1).copy()
    else:
        input_target_all = input_target_df#.values
        input_feature_id_all = input_feature_id_df#.values
        input_feature_all = input_feature_df#.values


    # メモリ効率化のため、不要になったデータフレームを削除
    del df_actual
    del df_unique
    del _
    del corr
    if WITH_PRED:
        del df_fish
        del df_noise
    
    random_forest(input_feature_df, 
                  input_target_df, 
                  input_feature_id_df, 
                  output_dir, 
                  n_split=n_split, 
                  rf_seeds=rf_seeds, 
                  kf_seeds=kf_seeds, 
                  input_feature_all=input_feature_all,
                  input_target_all=input_target_all,
                  input_feature_id_all=input_feature_id_all)

if __name__ == "__main__":
    # num_ids_l = ["all"]#[12, 30, 60, 120, 240, 480, 960, "all"]
    # WITH_PRED_l = [False]#[True, False]
    # for num_ids in num_ids_l:
    #     for WITH_PRED in WITH_PRED_l:
    #         input_csv_path = "../2_data/kalmanfilter/ayu/manual_roi/ayu/c3_d50_a10_h5/with_color/c3_d50_a10_h5.csv"
    #         output_dir = "../2_data/kalmanfilter/ayu/manual_roi/ayu/c3_d50_a10_h5/with_color/random_forest/training"
    #         fish_ids = [778, 789, 13408, 13421, 13434, 29944, 30368, 31653, 41508, 42868, 47345, 47372]
    #         # WITH_PRED = True
    #         pred_csv_path =  "../2_data/kalmanfilter/ayu/manual_roi/ayu/c3_d50_a10_h5/with_color/c3_d50_a10_h5_pred.csv"
    #         # num_ids = 150
    #         radius = 3
    #         n_split = 5
    #         random_states = 250
    #         main(input_csv_path, output_dir, fish_ids, WITH_PRED, pred_csv_path, num_ids, radius, n_split, random_states)

    num_ids_l = ["all"]#"all"]#[12, 30, 60, 120, 240, 480, 960, "all"]
    WITH_PRED_l = [False]#[True, False]
    for num_ids in num_ids_l:
        for WITH_PRED in WITH_PRED_l:
            input_csv_path = "../2_data/kalmanfilter_v3/ayu/manual_roi/ayu/config2/c4_d50_a10_h3_md50/c4_d50_a10_h3_md50_rgb.csv"
            output_dir = "../2_data/kalmanfilter_v3/ayu/manual_roi/ayu/config2/c4_d50_a10_h3_md50/random_forest/training"
            fish_ids = [296, 5687, 5699, 12273, 12429, 12971, 16805, 17340, 19116]#[296, 5687, 5699, 12273, 12429, 12971, 16805, 19116]#[296, 5687, 5699, 12273, 12429, 12971, 16805, 17340, 19116]
            # WITH_PRED = True
            pred_csv_path =  "../2_data/kalmanfilter_v3/ayu/manual_roi/ayu/config2/c4_d50_a10_h3_md50/c4_d50_a10_h3_md50_pred.csv"
            # num_ids = 150
            radius = 3
            n_split = 5
            rf_seeds = 50
            kf_seeds = 50
            main(input_csv_path, output_dir, fish_ids, WITH_PRED, pred_csv_path, num_ids, radius, n_split, rf_seeds, kf_seeds)
