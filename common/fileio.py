#coding:utf-8
import sys, os

def get_ext(file_path):
    # ファイル名から拡張子を取得
    period_idx = file_path.rfind(".")
    if period_idx >= 0:
        return file_path[period_idx:]
    else:
        # READMEなど
        return ""

def get_file_list(tgt_dir, tgt_ext = None, recursive = True):
    # 着目しているディレクトリ以下のファイルを全取得する
    ret = []
    for item_name in os.listdir(tgt_dir):
        item_path = os.path.join(tgt_dir, item_name)
        if os.path.isfile(item_path):
            if tgt_ext is not None:
                if get_ext(item_path) != tgt_ext:
                    continue
            ret.append(item_path)
        elif os.path.isdir(item_path):
            if recursive:
                ret = ret + get_file_list(tgt_dir = item_path,
                                          tgt_ext = tgt_ext)
    return ret

if __name__ == "__main__":
    pass
