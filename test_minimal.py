#!/usr/bin/env python3
"""
最小化测试：逐步导入模块找出问题
"""

import sys
import os

# 添加路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "spatialex"))

print("=== 测试1: 检查Python路径 ===")
print(f"当前工作目录: {os.getcwd()}")
print(f"项目根目录: {project_root}")
print(f"Python路径前3个:")
for i, path in enumerate(sys.path[:3]):
    print(f"  {i}: {path}")

print("\n=== 测试2: 检查文件存在性 ===")
files_to_check = [
    "spatialex/__init__.py",
    "spatialex/SpatialEx/__init__.py",
    "spatialex/SpatialEx/SpatialEx_pyG.py",
    "spatialex/SpatialEx/model.py",
    "spatialex/SpatialEx/utils.py",
    "spatialex/SpatialEx/preprocess.py"
]

for file_path in files_to_check:
    full_path = os.path.join(project_root, file_path)
    exists = os.path.exists(full_path)
    print(f"  {file_path}: {'✓' if exists else '❌'}")

print("\n=== 测试3: 尝试导入spatialex包 ===")
try:
    import spatialex
    print("✓ spatialex包导入成功")
except Exception as e:
    print(f"❌ spatialex包导入失败: {e}")
    print(f"错误类型: {type(e).__name__}")

print("\n=== 测试4: 尝试导入SpatialEx子模块 ===")
try:
    from spatialex import SpatialEx
    print("✓ SpatialEx子模块导入成功")
except Exception as e:
    print(f"❌ SpatialEx子模块导入失败: {e}")
    print(f"错误类型: {type(e).__name__}")

print("\n=== 测试5: 尝试导入具体类 ===")
try:
    from spatialex.SpatialEx import Train_SpatialEx
    print("✓ Train_SpatialEx类导入成功")
except Exception as e:
    print(f"❌ Train_SpatialEx类导入失败: {e}")
    print(f"错误类型: {type(e).__name__}")
    
    # 如果是ImportError，尝试查看详细信息
    if isinstance(e, ImportError):
        print(f"导入错误详情: {e}")
        print(f"导入错误名称: {e.name}")
        print(f"导入错误路径: {e.path}")
