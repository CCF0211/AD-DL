import time
import math

def timeSince(since):
    if since is not None:
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    else:
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())