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

def get_file_name(file_path):
    return file_path[file_path.rfind(os.sep) + 1:]

def get_file_list(tgt_dir,
                  name_txt = None,
                  include_path_txt = None,
                  exclude_path_txt = None,
                  tgt_ext = None, recursive = True):
    # 着目しているディレクトリ以下のファイルを全取得する
    ret = []
    for item_name in os.listdir(tgt_dir):
        item_path = os.path.join(tgt_dir, item_name)
        if os.path.isfile(item_path):
            if tgt_ext is not None:
                if get_ext(item_path) != tgt_ext:
                    continue
            if not name_txt is None:
                file_name = get_file_name(item_path)
                if file_name.find(name_txt) < 0:
                    continue
            if not include_path_txt is None:
                if item_path.find(include_path_txt) < 0:
                    continue
            if not exclude_path_txt is None:
                if item_path.find(exclude_path_txt) >= 0:
                    continue
            ret.append(item_path)
        elif os.path.isdir(item_path):
            if recursive:
                ret = ret + get_file_list(tgt_dir = item_path,
                                          name_txt = name_txt,
                                          include_path_txt = include_path_txt,
                                          exclude_path_txt = exclude_path_txt,
                                          tgt_ext = tgt_ext,
                                          recursive = recursive)
    return ret

if __name__ == "__main__":
    pass
