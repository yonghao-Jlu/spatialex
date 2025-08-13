#!/usr/bin/env python3
"""
Test script to verify SpatialEx package imports correctly
"""

import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("Testing SpatialEx package import...")
    
    # Test basic package import
    from spatialex.SpatialEx import SpatialEx
    print("✓ Successfully imported spatialex.SpatialEx")
    
    # Test specific class imports
    from spatialex.SpatialEx import Train_SpatialEx, Train_SpatialExP
    print("✓ Successfully imported Train_SpatialEx and Train_SpatialExP")
    
    # Test model imports
    from spatialex.SpatialEx import Model, Regression
    print("✓ Successfully imported Model and Regression")
    
    # Test utility imports
    from spatialex.SpatialEx import create_optimizer, Compute_metrics
    print("✓ Successfully imported utility functions")
    
    # Test preprocess import
    from spatialex.SpatialEx import pp
    print("✓ Successfully imported preprocess module")
    
    print("\n🎉 All imports successful! SpatialEx package is working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
