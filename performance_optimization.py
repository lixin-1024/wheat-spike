import time
import cv2
import numpy as np
import psutil
import os
import tracemalloc
from ultralytics import YOLO
from wheat_analysis.pipeline import WheatAnalysisPipeline


def measure_memory_usage():
    """测量内存使用情况"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # 转换为MB


def test_model_inference_speed(model_path, test_image, imgsz=1440, conf=0.5, iterations=10):
    """测试模型推理速度"""
    print(f"测试模型推理速度: {model_path}")
    print(f"测试图片: {test_image}")
    print(f"输入尺寸: {imgsz}x{imgsz}")
    print(f"测试次数: {iterations}")

    # 加载模型
    model = YOLO(model_path)

    # 预热
    print("预热模型...")
    for _ in range(3):
        model.predict(test_image, imgsz=imgsz, conf=conf, verbose=False)

    # 测量推理时间
    tracemalloc.start()
    start_mem = measure_memory_usage()
    start_time = time.time()

    inference_times = []
    for i in range(iterations):
        start_iter = time.time()
        results = model.predict(test_image, imgsz=imgsz, conf=conf, verbose=False)
        end_iter = time.time()
        inference_times.append(end_iter - start_iter)
        print(f"迭代 {i+1}/{iterations}: {end_iter - start_iter:.4f}秒")

    end_time = time.time()
    end_mem = measure_memory_usage()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 计算统计信息
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    mem_increase = end_mem - start_mem
    peak_mem = peak / (1024 * 1024)  # 转换为MB

    print("\n=== 推理速度测试结果 ===")
    print(f"平均推理时间: {avg_time:.4f}秒")
    print(f"标准差: {std_time:.4f}秒")
    print(f"最小时间: {min_time:.4f}秒")
    print(f"最大时间: {max_time:.4f}秒")
    print(f"内存增加: {mem_increase:.2f}MB")
    print(f"峰值内存: {peak_mem:.2f}MB")
    print(f"总测试时间: {end_time - start_time:.2f}秒")

    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'mem_increase': mem_increase,
        'peak_mem': peak_mem
    }


def test_pipeline_performance(test_image, iterations=5):
    """测试完整管线性能"""
    print(f"\n测试完整管线性能: {test_image}")
    print(f"测试次数: {iterations}")

    # 初始化管线
    pipeline = WheatAnalysisPipeline(
        model_path='runs/obb/yolo11_1440_4/weights/best.pt',
        imgsz=1440,
        conf=0.5
    )

    # 预热
    print("预热管线...")
    for _ in range(2):
        pipeline.analyze_single(test_image)

    # 测量性能
    tracemalloc.start()
    start_mem = measure_memory_usage()
    start_time = time.time()

    pipeline_times = []
    for i in range(iterations):
        start_iter = time.time()
        result = pipeline.analyze_single(test_image)
        end_iter = time.time()
        pipeline_times.append(end_iter - start_iter)
        print(f"管线迭代 {i+1}/{iterations}: {end_iter - start_iter:.4f}秒")

    end_time = time.time()
    end_mem = measure_memory_usage()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 计算统计信息
    avg_time = np.mean(pipeline_times)
    std_time = np.std(pipeline_times)
    min_time = np.min(pipeline_times)
    max_time = np.max(pipeline_times)
    mem_increase = end_mem - start_mem
    peak_mem = peak / (1024 * 1024)  # 转换为MB

    print("\n=== 管线性能测试结果 ===")
    print(f"平均处理时间: {avg_time:.4f}秒")
    print(f"标准差: {std_time:.4f}秒")
    print(f"最小时间: {min_time:.4f}秒")
    print(f"最大时间: {max_time:.4f}秒")
    print(f"内存增加: {mem_increase:.2f}MB")
    print(f"峰值内存: {peak_mem:.2f}MB")
    print(f"总测试时间: {end_time - start_time:.2f}秒")

    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'mem_increase': mem_increase,
        'peak_mem': peak_mem
    }


def compare_model_variants():
    """比较不同模型变体的性能"""
    test_image = 'data/train/images/001.jpg'
    models = [
        ('yolo11n-obb.pt', 'YOLO11n (小型)'),
        ('yolo11s-obb.pt', 'YOLO11s (中型)'),
        ('yolo11m-obb.pt', 'YOLO11m (大型)'),
    ]

    results = {}

    for model_path, model_name in models:
        print(f"\n{'='*50}")
        print(f"测试模型: {model_name}")
        print(f"{'='*50}")
        results[model_name] = test_model_inference_speed(model_path, test_image)

    # 比较结果
    print("\n=== 模型性能比较 ===")
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  平均推理时间: {result['avg_time']:.4f}秒")
        print(f"  峰值内存: {result['peak_mem']:.2f}MB")

    return results


def optimize_model_settings():
    """测试不同的模型设置"""
    test_image = 'data/train/images/001.jpg'
    settings = [
        {'imgsz': 640, 'conf': 0.5, 'name': '640x640, conf=0.5'},
        {'imgsz': 960, 'conf': 0.5, 'name': '960x960, conf=0.5'},
        {'imgsz': 1440, 'conf': 0.5, 'name': '1440x1440, conf=0.5'},
        {'imgsz': 1440, 'conf': 0.7, 'name': '1440x1440, conf=0.7'},
    ]

    results = {}

    for setting in settings:
        print(f"\n测试设置: {setting['name']}")
        results[setting['name']] = test_model_inference_speed(
            'runs/obb/yolo11_1440_4/weights/best.pt',
            test_image,
            imgsz=setting['imgsz'],
            conf=setting['conf']
        )

    # 比较结果
    print("\n=== 设置优化比较 ===")
    for setting_name, result in results.items():
        print(f"{setting_name}:")
        print(f"  平均推理时间: {result['avg_time']:.4f}秒")
        print(f"  峰值内存: {result['peak_mem']:.2f}MB")

    return results


if __name__ == '__main__':
    test_image = 'data/train/images/001.jpg'

    print("=== 小麦麦穗表型分析系统 - 性能优化测试 ===")

    # 测试模型推理速度
    model_results = compare_model_variants()

    # 测试管线性能
    pipeline_results = test_pipeline_performance(test_image)

    # 测试不同设置
    settings_results = optimize_model_settings()

    print("\n=== 优化建议 ===")
    print("1. 对于实时应用，建议使用较小的输入尺寸（如640x640）")
    print("2. 根据精度需求调整置信度阈值")
    print("3. 考虑使用轻量级模型（如YOLO11n）以获得更快的推理速度")
    print("4. 批量处理可以进一步提高效率")