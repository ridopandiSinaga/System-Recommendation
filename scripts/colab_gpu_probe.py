"""Run a tiny GPU calculation in a child process, releasing CUDA afterward.

Sent with colab exec -f; this notebook kernel never imports TensorFlow itself.
"""
import json
import subprocess
import sys


PROBE = r'''
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
devices = tf.config.list_physical_devices('GPU')
if not devices:
    raise RuntimeError('TensorFlow cannot detect a GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)
with tf.device('/GPU:0'):
    product = tf.matmul(tf.ones((256, 256)), tf.ones((256, 256)))
    value = float(product[0, 0].numpy())
if value != 256.0 or 'GPU' not in product.device:
    raise RuntimeError('GPU calculation or device placement is incorrect')
hardware = subprocess.run(
    ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
    capture_output=True, text=True, check=True,
).stdout.strip()
print(json.dumps({'status': 'passed', 'tensorflow': tf.__version__,
                  'physical_gpus': [d.name for d in devices],
                  'tensor_device': product.device, 'matrix_value': value,
                  'hardware': hardware}), flush=True)
'''
# Import in the child as well so the probe string is standalone.
PROBE = PROBE.replace("import json\n", "import json\nimport subprocess\n", 1)
result = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=90)
print(result.stdout, flush=True)
if result.returncode:
    print(result.stderr, flush=True)
    raise RuntimeError(f"GPU probe failed with exit code {result.returncode}")
json.loads(result.stdout.strip().splitlines()[-1])
