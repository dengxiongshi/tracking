import os
import sys

test = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tracker')


class Opts:
    pass


opts = Opts()

setattr(opts, 'img_size', [640, 640])
setattr(opts, 'conf_thresh', 0.25)
print(opts.img_size)

# opts.img_size = [640,640]
# opts.conf_thresh = 0.25
