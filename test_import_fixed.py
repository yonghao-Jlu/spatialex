#!/usr/bin/env python3
"""
测试脚本：验证spatialex模块是否可以正确导入
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "spatialex"))

print("Python路径:")
for path in sys.path[:5]:  # 只显示前5个路径
    print(f"  {path}")

print("\n尝试导入模块...")

try:
    # 测试导入spatialex包
    import spatialex
    print("✓ 成功导入 spatialex 包")
    print(f"  版本: {spatialex.__version__}")
    
    # 测试导入SpatialEx子模块
    from spatialex import SpatialEx
    print("✓ 成功导入 SpatialEx 子模块")
    
    # 测试导入具体类
    from spatialex.SpatialEx import Train_SpatialEx, Train_SpatialExP
    print("✓ 成功导入 Train_SpatialEx 和 Train_SpatialExP 类")
    
    # 测试导入其他模块
    from spatialex.SpatialEx import Model, Regression
    print("✓ 成功导入 Model 和 Regression 类")
    
    from spatialex.SpatialEx import utils
    print("✓ 成功导入 utils 模块")
    
    from spatialex.SpatialEx import preprocess as pp
    print("✓ 成功导入 preprocess 模块")
    
    print("\n🎉 所有模块导入成功！")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print(f"错误类型: {type(e).__name__}")
    
    # 尝试诊断问题
    try:
        import spatialex
    except Exception as e2:
        print(f"直接导入spatialex失败: {e2}")
    
    try:
        import sys
        print(f"当前工作目录: {os.getcwd()}")
        print(f"脚本位置: {__file__}")
        print(f"项目根目录: {project_root}")
    except Exception as e3:
        print(f"获取路径信息失败: {e3}")

except Exception as e:
    print(f"❌ 其他错误: {e}")
    print(f"错误类型: {type(e).__name__}")
