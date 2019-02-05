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

# 着目しているディレクトリ以下のファイルを全取得する
def get_file_list_core(tgt_dir,
                       yield_cnt_init,
                       name_txt = None,
                       include_path_txt = None,
                       exclude_path_txt = None,
                       tgt_ext = None,
                       only_name = None,
                       max_num = None,
                       recursive = True):
    yield_cnt = yield_cnt_init
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
            if not max_num is None:
                if yield_cnt >= max_num:
                    break
            # yield実行決定
            yield_cnt += 1
            if only_name is True:
                yield item_name
            else:
                yield item_path
        elif os.path.isdir(item_path):
            if recursive:
                for rec_item in get_file_list_core(tgt_dir = item_path,
                                                   name_txt = name_txt,
                                                   include_path_txt = include_path_txt,
                                                   exclude_path_txt = exclude_path_txt,
                                                   tgt_ext = tgt_ext,
                                                   only_name = only_name,
                                                   yield_cnt_init = yield_cnt,
                                                   max_num = max_num,
                                                   recursive = recursive):
                    yield rec_item

def get_file_list(tgt_dir,
                  name_txt = None,
                  include_path_txt = None,
                  exclude_path_txt = None,
                  tgt_ext = None,
                  only_name = None,
                  max_num = None,
                  recursive = True):
    # 再帰で利用するときに必要な変数yield_cnt_initを隠蔽するために、実処理を別関数化
    for item in get_file_list_core(tgt_dir = tgt_dir,
                                   name_txt = name_txt,
                                   include_path_txt = include_path_txt,
                                   exclude_path_txt = exclude_path_txt,
                                   tgt_ext = tgt_ext,
                                   only_name = only_name,
                                   yield_cnt_init = 0,
                                   max_num = max_num,
                                   recursive = recursive):
        yield item
if __name__ == "__main__":
    for i, item in enumerate(get_file_list("/media/isgsktyktt/EC-PHU3/apolloscape/road01_ins/ColorImage", tgt_ext = ".jpg")):
        print(i, item)
    pass
