import datetime
from functools import wraps
import os
import traceback
import sys


"""
複数のファイルパス引数について、上書き防止チェックを行うデコレータ。
file_args に指定した引数名のファイルが存在する場合、プログラムを終了する。
"""
def check_file_exists(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            print("arg", arg)
            print("args", args)
            print("kwargs", kwargs)
            if os.path.exists(arg):
                print(f"エラー: ファイル '{arg}' は既に存在します。")
                sys.exit(1)  # エラーコード1でプログラム終了
        for arg in kwargs.values():
            if os.path.exists(arg):
                print(f"エラー: ファイル '{arg}' は既に存在します。")
                sys.exit(1)
        return func(*args, **kwargs)
    return wrapper
    # return decorator

def error_handler(func):
    if not os.path.exists("log"):
        os.makedirs("log")
    # log.txtが存在しなければ作成
    if not os.path.isfile("./log/log.txt"):
        # エラーログを整理して出力するファイル
        with open ("./log/log.txt", "w") as log:
            log.write("")
    # log_original.txtが存在しなければ作成
    if not os.path.isfile("./log/log_original.txt"):
        #  エラーログをそのまま出力するファイル
        with open ("./log/log_original.txt", "w") as log:
            log.write("")
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            now = datetime.datetime.now()
            now = now.strftime("%Y_%m_%d_%H_%M_%S")
            # 呼び出し元の関数（エラーが発生した関数）の情報を取得
            tb = traceback.format_exc()
            with open ("./log/log_original.txt", "a") as log:
                log.write(f"The time error occurred: {now}/n")
                log.write(f"{tb}/n/n/n")
            print("An error occurred: ", tb)
            tb = traceback.extract_tb(e.__traceback__)
            # 必要に応じてさらにエラーメッセージを出力
            print(f"Error message: {e}")

            with open ("./log/log.txt", "a") as log:
                for err in tb:
                    log.write(f"The time error occurred: {now}, File name: {err.filename}, Function name: {err.name}, Line number: {err.lineno}, Error name: ({type(e).__name__}): {e}, line_content: {err.line}/n/n")
            print(f"An error occurred: {e} ({type(e).__name__})")
    return wrapper


if __name__ == "__main__":
    @check_file_exists
    def test_prevent_overwrite(input_csv_path, output_csv_path, output_video_path, a=2, b=3, c=4):
        print("input_csv_path:", input_csv_path)
        print("output_csv_path:", output_csv_path)
        print("output_video_path:", output_video_path)
        print("a:", a)
        print("b:", b)
        print("c:", c)
        return "Success"
    
    input_csv_path = "C:/Users/koder/OneDrive - Tokyo University of Agriculture and Technology/0_research/202407_back_diff/3_codes/Actual_Test_Data.csv"
    output_csv_path = "C:/Users/koder/OneDrive - Tokyo University of Agriculture and Technology/0_research/202407_back_diff/3_codes/Actual_Test_Data_iou.csv"
    output_video_path = "C:/Users/koder/OneDrive - Tokyo University of Agriculture and Technology/0_research/202407_back_diff/3_codes/ishigaki_river.mp4"
    # a = 1
    # b = 2
    # c = 3
    result = test_prevent_overwrite(input_csv_path, output_csv_path, output_video_path, a=1, b=2, c=3)
    print(result)
    

"""
# 1回目の実行時にはFalseを、
# 2回目の実行時にはTrueを返すデコレータ
def once(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not wrapper.called:
            wrapper.called = True
            #print("1回目の動作を実行します。")
            wrapper.bool = False
            func(*args, **kwargs)
            return wrapper.bool  # デコレートされた関数の戻り値を返す
        else:
            #print("2回目以降は特定の動作をスキップします.")
            wrapper.bool = True
            func(*args, **kwargs)
            return wrapper.bool
    
    wrapper.called = False
    return wrapper
"""
"""
## Usage of once function
@once
def my_function():
    print(my_function.bool)
    print("関数が実行されました.")

# テスト
result_1 = my_function()  # 1回目の呼び出し
result_2 = my_function()  # 2回目の呼び出し
my_function()

print("1回目の結果:", result_1)  # 1回目の呼び出しでは関数の戻り値が返される
print("2回目の結果:", result_2)  # 2回目の呼び出しではデコレータからの False が返される
"""


"""
# Example usage:

@error_handler
def divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError as e:
        print("ERROR OCCURRED")
        print(e)
        # Optionally handle the exception here if needed
        raise  # Re-raise the exception to be caught by the decorator
    

@error_handler
def greet(name:int):
    print(f"Hello, {name}!")

# Testing the decorated functions

divide_result = divide(10, 2)
print("Result of divide function:", divide_result)


divide_error = divide(10, 0)  # This will trigger an error


greet("Alice")
greet(123)  
"""
