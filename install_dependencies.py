#!/usr/bin/env python3
"""
依赖安装脚本
解决numpy版本兼容性问题
"""

import subprocess
import sys
import os

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n=== {description} ===")
    print(f"执行命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✓ 成功")
        if result.stdout:
            print("输出:", result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 失败: {e}")
        if e.stderr:
            print("错误:", e.stderr.strip())
        return False

def main():
    print("🚀 开始安装兼容的依赖版本...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 升级pip
    run_command("python -m pip install --upgrade pip", "升级pip")
    
    # 卸载可能有问题的包
    print("\n=== 卸载可能有问题的包 ===")
    packages_to_remove = ["numpy", "pandas", "scipy"]
    for package in packages_to_remove:
        run_command(f"pip uninstall {package} -y", f"卸载 {package}")
    
    # 安装兼容版本的包
    print("\n=== 安装兼容版本的包 ===")
    
    # 首先安装numpy
    run_command("pip install 'numpy>=1.21.0,<1.25.0'", "安装兼容的numpy版本")
    
    # 然后安装其他包
    run_command("pip install 'pandas>=1.3.0,<2.0.0'", "安装兼容的pandas版本")
    run_command("pip install 'scipy>=1.7.0,<1.10.0'", "安装兼容的scipy版本")
    
    # 安装PyTorch相关
    run_command("pip install torch>=1.8.0", "安装PyTorch")
    run_command("pip install torchvision>=0.9.0", "安装torchvision")
    
    # 安装其他依赖
    run_command("pip install 'scanpy>=1.8.0,<2.0.0'", "安装scanpy")
    run_command("pip install 'anndata>=0.8.0,<0.9.0'", "安装anndata")
    run_command("pip install 'scikit-learn>=1.0.0,<1.3.0'", "安装scikit-learn")
    run_command("pip install scikit-misc>=0.2.0", "安装scikit-misc")
    run_command("pip install tqdm>=4.60.0", "安装tqdm")
    run_command("pip install 'matplotlib>=3.3.0,<3.6.0'", "安装matplotlib")
    
    # 安装文档生成工具
    run_command("pip install nbsphinx>=0.8.0", "安装nbsphinx")
    run_command("pip install sphinx_rtd_theme>=1.0.0", "安装sphinx_rtd_theme")
    run_command("pip install sphinx_autodoc_typehints>=1.12.0", "安装sphinx_autodoc_typehints")
    
    print("\n🎉 依赖安装完成！")
    print("\n现在可以尝试运行测试脚本了。")

if __name__ == "__main__":
    main()
