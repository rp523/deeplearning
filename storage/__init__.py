#coding:utf-8
import sys, os
import subprocess

class Storage:
    
    def __init__(self):
        # 自身の実行環境を設定する
        pcname = subprocess.getoutput("uname -n")
        if pcname == "isgsktyktt-VJS111":
            self.__location = "notebook"
        elif (pcname == "floydhub") or (pcname == "job-instance"):
            self.__location = "floydhub"
        elif pcname == "Yusuke-PC":
            self.__location = "desktop"
        elif pcname == "auto3tes123":
            self.__location = "auto3tes123"
        else:
            with open("uname_err.txt", "w") as f:
                f.write(pcname)
            print(pcname)
            assert(0)
    
    def dataset_path(self, dataset_name):
        # 実行環境ごとに異なるデータセットのパスを取得する
        if self.__location == "notebook":
            base_path = "/media/isgsktyktt/EC-PHU3"
        elif self.__location == "floydhub":
            base_path = "/floyd/input"
        elif self.__location == "desktop":
            base_path = r"G:\dataset"
        elif self.__location == "auto3tes123":
            base_path = r"/gpfs/auto3tes123/dataset"
        else:
            assert(0)
        dataset_path = os.path.join(base_path, dataset_name)
        assert(os.path.exists(dataset_path))
        return dataset_path
    
if __name__ == "__main__":
    s = Storage()
    print(s.dataset_path("cityscapes"))
    
