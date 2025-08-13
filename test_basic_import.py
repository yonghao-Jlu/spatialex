#!/usr/bin/env python3
"""
简化测试脚本：只测试基本的模块结构
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "spatialex"))

print("Python路径:")
for path in sys.path[:5]:
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
    
    print("\n🎉 基本模块导入成功！")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print(f"错误类型: {type(e).__name__}")
    
except Exception as e:
    print(f"❌ 其他错误: {e}")
    print(f"错误类型: {type(e).__name__}")
    
    # 尝试诊断问题
    try:
        print(f"当前工作目录: {os.getcwd()}")
        print(f"脚本位置: {__file__}")
        print(f"项目根目录: {project_root}")
        
        # 检查文件是否存在
        spatialex_init = os.path.join(project_root, "spatialex", "__init__.py")
        print(f"spatialex/__init__.py 存在: {os.path.exists(spatialex_init)}")
        
        spatialex_dir = os.path.join(project_root, "spatialex", "SpatialEx", "__init__.py")
        print(f"spatialex/SpatialEx/__init__.py 存在: {os.path.exists(spatialex_dir)}")
        
    except Exception as e2:
        print(f"诊断信息获取失败: {e2}")
