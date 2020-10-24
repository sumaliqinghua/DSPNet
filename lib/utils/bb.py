#放在要debug的位置前
import os
debug_file = './tmp/debug'#怎么写到config里

if os.path.exists(debug_file):
    import ipdb
    ipdb.set_trace()
#分割线#