#!/usr/bin/env python3
"""
修复后的测试脚本
测试模块导入是否正常工作
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "spatialex"))

print("=== 环境信息 ===")
print(f"Python版本: {sys.version}")
print(f"当前工作目录: {os.getcwd()}")
print(f"项目根目录: {project_root}")

print("\n=== Python路径 ===")
for i, path in enumerate(sys.path[:5]):
    print(f"  {i}: {path}")

print("\n=== 测试模块导入 ===")

try:
    # 测试1: 导入spatialex包
    print("1. 测试导入spatialex包...")
    import spatialex
    print("   ✓ 成功导入 spatialex 包")
    print(f"   版本: {spatialex.__version__}")
    
    # 测试2: 导入SpatialEx子模块
    print("2. 测试导入SpatialEx子模块...")
    from spatialex import SpatialEx
    print("   ✓ 成功导入 SpatialEx 子模块")
    
    # 测试3: 导入具体类
    print("3. 测试导入具体类...")
    from spatialex.SpatialEx import Train_SpatialEx, Train_SpatialExP
    print("   ✓ 成功导入 Train_SpatialEx 和 Train_SpatialExP 类")
    
    # 测试4: 导入其他模块
    print("4. 测试导入其他模块...")
    from spatialex.SpatialEx import Model, Regression
    print("   ✓ 成功导入 Model 和 Regression 类")
    
    from spatialex.SpatialEx import utils
    print("   ✓ 成功导入 utils 模块")
    
    from spatialex.SpatialEx import preprocess as pp
    print("   ✓ 成功导入 preprocess 模块")
    
    print("\n🎉 所有测试通过！模块导入成功！")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print(f"错误类型: {type(e).__name__}")
    
except Exception as e:
    print(f"❌ 其他错误: {e}")
    print(f"错误类型: {type(e).__name__}")
    
    # 尝试诊断问题
    try:
        import traceback
        print("\n=== 详细错误信息 ===")
        traceback.print_exc()
    except:
        pass

print("\n=== 测试完成 ===")
