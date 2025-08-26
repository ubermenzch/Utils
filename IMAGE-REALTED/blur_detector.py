import os
import argparse
import shutil
import cv2
import numpy as np
import concurrent.futures
import time
from tqdm import tqdm

def ensure_output_directory(root_output_dir):
    """确保输出目录存在并清空blur_filter_result子目录"""
    output_dir = os.path.join(root_output_dir, "blur_filter_result")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def detect_rotation_blur(img_path):
    """
    检测单张图片的旋转模糊
    返回(图片路径, 模糊分数, 错误信息)
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return (img_path, 0.0, f"无法读取图片: {os.path.basename(img_path)}")
        
        # 模糊检测算法
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        directional_diff = np.abs(sobelx) - np.abs(sobely)
        rotation_feature = np.abs(directional_diff) * 0.3 + np.abs(laplacian) * 0.7
        
        blur_score = np.var(rotation_feature)
        return (img_path, blur_score, None)
    
    except Exception as e:
        return (img_path, 0.0, f"处理错误: {str(e)}")

def process_images(input_dir, root_output_dir, threshold, threads=os.cpu_count()):
    """使用多线程处理图片模糊检测"""
    start_time = time.time()
    
    # 创建并清理输出目录
    output_dir = ensure_output_directory(root_output_dir)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"模糊阈值: {threshold}")
    print(f"使用线程数: {threads}\n")
    
    # 获取所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_paths = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(input_dir, filename)
            image_paths.append(file_path)
    
    if not image_paths:
        print("⚠️ 警告: 输入目录中没有找到任何图片文件")
        return
    
    total_images = len(image_paths)
    print(f"找到 {total_images} 张图片，开始处理...\n")
    
    # 使用线程池处理图片
    clear_count = 0
    error_count = 0
    
    # 进度条设置
    pbar = tqdm(total=total_images, desc="处理进度", unit="img")
    
    # 使用带线程数的ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        # 提交所有任务
        future_to_img = {executor.submit(detect_rotation_blur, img_path): img_path for img_path in image_paths}
        
        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_img):
            img_path = future_to_img[future]
            try:
                img_path, blur_score, error = future.result()
                
                if error:
                    tqdm.write(f"错误: {error}")
                    error_count += 1
                else:
                    filename = os.path.basename(img_path)
                    
                    # 保存清晰图片
                    if blur_score > threshold:
                        output_path = os.path.join(output_dir, filename)
                        # 实际保存图片（在主线程执行以避免并发写入问题）
                        img = cv2.imread(img_path)
                        cv2.imwrite(output_path, img)
                        clear_count += 1
                        pbar.write(f"{blur_score:>8.1f} | ✅ {filename} - 清晰")
                    else:
                        pbar.write(f"{blur_score:>8.1f} | ⛔ {filename} - 模糊")
            
            except Exception as e:
                tqdm.write(f"处理结果时出错: {str(e)}")
                error_count += 1
            
            # 更新进度
            pbar.update(1)
    
    pbar.close()
    
    # 计算处理时间
    elapsed_time = time.time() - start_time
    img_per_sec = total_images / elapsed_time if elapsed_time > 0 else float('inf')
    
    print("\n" + "=" * 60)
    print(f"处理摘要:")
    print(f"- 总图片数: {total_images}")
    print(f"- 清晰图片数: {clear_count}")
    print(f"- 模糊图片数: {total_images - clear_count - error_count}")
    print(f"- 错误图片数: {error_count}")
    print(f"- 保存目录: {output_dir}")
    print(f"- 处理时间: {elapsed_time:.1f} 秒 ({img_per_sec:.1f} 张/秒)")
    print("=" * 60)
    print(f"所有清晰图片已保存至 'blur_filter_result' 目录")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='多线程旋转模糊检测器')
    parser.add_argument('--input', required=True, help='包含源图片的目录路径')
    parser.add_argument('--output', required=True, help='输出根目录路径（将在此目录下创建blur_filter_result子目录）')
    parser.add_argument('--threshold', type=float, default=150.0, 
                        help='模糊检测阈值（越高要求越严格，默认150）')
    parser.add_argument('--threads', type=int, default=os.cpu_count(), 
                        help=f'并行线程数（默认使用全部CPU核心：{os.cpu_count()}）')
    
    args = parser.parse_args()
    
    print(f"\n=== 多线程旋转模糊检测器 ===")
    process_images(args.input, args.output, args.threshold, args.threads)
