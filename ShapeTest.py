#!/usr/bin/env python

import numpy as np

if __name__ == '__main__':

    base = "/tmp/ShapeTestWrite/1"
    node = np.load("%s/node.npy"% base )
    print(node)
