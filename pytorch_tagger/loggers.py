from datetime import datetime
from collections import Iterable, OrderedDict
import numpy as np
from poutyne.framework import CSVLogger


class TimedCSVLogger(CSVLogger):

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.keys = sorted(logs.keys())

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        row_dict = OrderedDict({'epoch': epoch, 'time': str(datetime.now())})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csvfile.flush()