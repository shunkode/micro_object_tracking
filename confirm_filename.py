from tkinter import simpledialog, filedialog
import sys

def confirm_filename(input_filepath):
    user_input = simpledialog.askstring("ファイルパスおよびファイル名の確認", f"上書きされないように、ファイルパスを確認してください。\n正しい場合はEnterを、異なる場合は正しいファイルパスを入力するか、fキーを押してプログラムを終了してください。: \n{input_filepath}")
    
    if user_input == "":
        return input_filepath
    elif user_input == "f":
        sys.exit("中断します。ファイル名を再確認してください。")
    else:
        re_input = simpledialog.askstring("ファイルパスおよびファイル名のs最終確認", f"入力されたファイルパス: {user_input} でよろしいですか？: y/n")
        if re_input == "y":
            return user_input
        elif re_input == "n":
            confirm_filename(user_input)
        else:
            sys.exit("中断します。ファイル名を再確認してください。")

import os
import sys

def check_file_exists(filepath):
    """
    指定したパスにファイルが存在する場合、プログラムを終了する。
    """
    if os.path.exists(filepath):
        y_n = input(f"エラー: ファイル '{filepath}' は既に存在します。\n上書きする場合は\"y\"を入力してください。上書きしない場合は\"n\"を入力してください。")
        if y_n == "y":
            return filepath
        elif y_n == "n":
            sys.exit(1)
        sys.exit(1)  # エラーコード1でプログラム終了

    # ディレクトリが存在しない場合、作成する
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"指定したディレクトリが存在しないため、ディレクトリ '{directory}' を作成しました。")

    return filepath
    

if __name__ == "__main__":
    input_filepath = filedialog.askopenfilename()
    output = confirm_filename(input_filepath)
    print(output)