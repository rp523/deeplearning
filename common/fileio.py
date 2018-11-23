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
                  tgt_ext = None,
                  only_name = None,
                  max_num = None,
                  recursive = True):
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
            add = item_path
            if only_name is True:
                add = get_file_name(add)
            ret.append(add)
            if max_num is not None:
                if len(ret) >= max_num:
                    return ret
        elif os.path.isdir(item_path):
            if recursive:
                ret = get_file_list(tgt_dir = item_path,
                                          name_txt = name_txt,
                                          include_path_txt = include_path_txt,
                                          exclude_path_txt = exclude_path_txt,
                                          tgt_ext = tgt_ext,
                                          only_name = only_name,
                                          max_num = max_num,
                                          recursive_base = ret,
                                          recursive = recursive)
    if not max_num is None:
        if len(ret) > max_num:
            ret = ret[:max_num]
    return ret

if __name__ == "__main__":
    pass
