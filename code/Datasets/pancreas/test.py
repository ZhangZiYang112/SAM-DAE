import os
from tcia_utils import nbia

os.environ["PYTHONUTF8"] = "1"

collection_name = "Pancreas-CT"
# 注意：官方正确的函数名是带下划线的 make_manifest
try:
    print("正在生成清单...")
    nbia.make_manifest(collection = collection_name, path = "D:/")
    print("生成成功！请去 D 盘双击 manifest-Pancreas-CT.tcia 文件。")
except Exception as e:
    print(f"依然失败，错误信息: {e}")