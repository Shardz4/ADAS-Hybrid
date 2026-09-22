"""
ADAS HYBRID Benchmark suite
"""

import argparse
import csv
import os
import sys
import time
import numpy as np

try:
    import adas_hybrid
except ImportError:
    sys.exit("Error: 'adas_hybrid' not found")

def get_vram_mb():
    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=3
        )
        return float(res.stdout.strip().split("\n")[0])
    except Exception:
        return 0.0

class MetricTracker:
    def __init__(self, name:str):
        self.name = name
        self.samples = []
    def add(self, ms: float):
        self.samples.append(ms)
    
    @property
    def mean(self):
        return float(np.mean(self.samples)) if self.samples else 0.0
    
    @property
    def std(self):
        return float(np.std(Self.samples)) if self.samples else 0.0
    
    @property
    def p95(self):
        return float(np.percentile(self.samples, 95)) if self.samples else 0.0
    
    @property
    def min_val(self):
        return float(np.min(self.samples)) if self.samples else 0.0
    
    @property
    def max_val(self):
        return float(np.max(self.samples)) if self.samples else 0.0

