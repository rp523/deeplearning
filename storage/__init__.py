#coding:utf-8
import sys, os
import subprocess

class Storage:
    
    def __init__(self):
        # 自身の実行環境を設定する
        pcname = subprocess.getoutput("uname -n")
        if pcname == "isgsktyktt-VJS111":
            self.__location = "notebook"
        elif pcname == "floydhub":
            self.__location = "floydhub"
        else:
            assert(0)
    
    def dataset_path(self, dataset_name):
        # 実行環境ごとに異なるデータセットのパスを取得する
        if dataset_name == "cityscapes":
            if self.__location == "notebook":
                ret_path = "/media/isgsktyktt/EC-PHU3/cityscapes"
            elif self.__location == "floydhub":
                ret_path = "/floyd/input/cityscapes"
        else:
            assert(0)
        assert(os.path.exists(ret_path))
        return ret_path
    
if __name__ == "__main__":
    s = Storage()
    print(s.dataset_path("cityscapes"))
    