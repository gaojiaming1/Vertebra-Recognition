# ----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import time
import os
import cv2
import numpy as np
import argparse
from PIL import Image
from datetime import datetime
from unet import Unet


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='UNet语义分割预测')

    # 必需参数
    parser.add_argument('--image_path', type=str, required=True,
                        help='输入图片路径')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出图片保存路径')
    parser.add_argument('--model_path', type=str, default='',
                        help='模型权重路径（可选）')

    # 可选参数
    parser.add_argument('--count', action='store_true',
                        help='是否进行像素计数')
    parser.add_argument('--show', action='store_true',
                        help='是否显示结果')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='设备类型')

    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    # -------------------------------------------------------------------------#
    unet = Unet()

    # 如果提供了模型路径，加载权重
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model weights from: {args.model_path}")
        # 这里需要根据你的UNet实现来加载权重
        # 如果unet有load_weights方法
        if hasattr(unet, 'load_weights'):
            unet.load_weights(args.model_path)
        else:
            # 否则使用默认的加载方式
            import torch
            checkpoint = torch.load(args.model_path,
                                    map_location=torch.device(args.device))
            if 'model' in checkpoint:
                unet.model.load_state_dict(checkpoint['model'])
            else:
                unet.model.load_state_dict(checkpoint)

    # -------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    # -------------------------------------------------------------------------#
    count = args.count
    name_classes = ["_background_", "chine", "coccyx", "cocc"]

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    try:
        # 读取图片
        print(f"Opening image: {args.image_path}")
        image = Image.open(args.image_path)

        # 执行预测
        print("Running UNet segmentation...")
        r_image = unet.detect_image(image, count=count, name_classes=name_classes)

        # 保存结果
        r_image.save(args.output_path)
        print(f"✅ Successfully saved result to: {args.output_path}")

        # 如果设置了显示标志，显示结果
        if args.show:
            r_image.show()

        # 同时保存一个叠加版本（原图+分割结果）
        try:
            # 创建叠加图像
            original = cv2.imread(args.image_path)
            segmented = cv2.cvtColor(np.array(r_image), cv2.COLOR_RGB2BGR)

            # 如果尺寸不同，调整分割结果尺寸
            if original.shape[:2] != segmented.shape[:2]:
                segmented = cv2.resize(segmented, (original.shape[1], original.shape[0]))

            # 创建50%透明的叠加
            alpha = 0.5
            overlay = cv2.addWeighted(original, 1 - alpha, segmented, alpha, 0)

            # 生成叠加文件名
            base, ext = os.path.splitext(args.output_path)
            overlay_path = f"{base}_overlay{ext}"

            # 保存叠加图像
            cv2.imwrite(overlay_path, overlay)
            print(f"✅ Overlay image saved to: {overlay_path}")

        except Exception as e:
            print(f"⚠️ Could not create overlay image: {e}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()