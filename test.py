import os
import glob
import numpy as np
import nibabel as nib
import pyvista as pv

input_root = r"C:\Users\lenovo\Desktop\LA"
output_dir = os.path.join(input_root, "outputgt_images")
os.makedirs(output_dir, exist_ok=True)

folders = [os.path.join(input_root, f) for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]
for folder in folders:
    pred_path = os.path.join(folder, "VNet_predictions", "04_gt.nii.gz")
    if not os.path.exists(pred_path):
        print(f"未找到: {pred_path}")
        continue
    print(f"正在处理: {pred_path}")
    try:
        img = nib.load(pred_path)
        data = img.get_fdata()
        data = (data > 0).astype(np.uint8)  # 二值化
        # 适配新版 pyvista
        grid = pv.ImageData(dimensions=data.shape, spacing=img.header.get_zooms())
        grid.point_data["values"] = data.flatten(order="F")
        # 提取等值面
        surf = grid.contour(isosurfaces=[0.5], scalars="values")
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(surf, color="red")
        plotter.set_background("white")
        plotter.camera_position = "iso"
        out_name = os.path.basename(os.path.dirname(folder)) + "_" + os.path.basename(folder) + ".png"
        out_path = os.path.join(output_dir, out_name)
        plotter.show(screenshot=out_path, window_size=[512, 512])
        plotter.close()
    except Exception as e:
        print(f"处理文件 {pred_path} 时出错: {e}")

print(f"处理完成！图片保存在: {output_dir}")