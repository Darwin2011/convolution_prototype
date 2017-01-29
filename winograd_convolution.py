#!/usr/bin/env python

import numpy as np
import scipy.signal as signal
import unittest

def winograd_convolution_2d_8x8(kernel, data):
  AT = np.array(
    [[1, 1, 1, 1, 1, 1, 1, 0],
     [0, 1, -1, 2, -2, 0.5, -0.5, 0],
     [0, 1, 1, 4, 4, 0.25, 0.25, 0],
     [0, 1, -1, 8, -8, 0.125, -0.125, 0],
     [0, 1, 1, 16, 16, 0.0625, 0.0625, 0],
     [0, 1, -1, 32, -32, 0.03125, -0.03125, 1]], dtype=np.float)
  G = np.array(
    [[1, 0, 0],
     [-2.0/9.0, -2.0/9.0, -2.0/9.0],
     [-2.0/9.0, 2.0/9.0, -2.0/9.0],
     [1.0/90.0, 1.0/45.0, 2.0/45.0],
     [1.0/90.0, -1.0/45.0, 2.0/45.0],
     [32.0/45.0, 16.0/45.0, 8.0/45.0],
     [32.0/45.0, -16.0/45.0, 8.0/45.0],
     [0,0,1]], dtype=np.float)
  BT = np.array(
    [[1, 0, -21.0/4.0, 0, 21.0/4.0, 0, -1, 0],
     [0, 1, 1, -17.0/4.0, -17.0/4.0, 1, 1, 0],
     [0, -1, 1, 17.0/4.0, -17.0/4.0, -1, 1, 0],
     [0, 0.5, 0.25, -2.5, -1.25, 2, 1, 0],
     [0, -0.5, 0.25, 2.5, -1.25, -2, 1, 0],
     [0, 2, 4, -2.5, -5, 0.5, 1, 0],
     [0, -2, 4, 2.5, -5, -0.5, 1, 0],
     [0, -1, 0, 21.0/4.0, 0, -21.0/4.0, 0, 1]], dtype=np.float)
  U = np.dot(G, np.dot(kernel, G.T))
  V = np.dot(BT, np.dot(data, BT.T))
  UV = np.multiply(U, V)
  return np.dot(AT, np.dot(UV, AT.T))


def correlation_2d(kernel, data):
  return signal.correlate2d(data, kernel, 'valid')


class TestConvolution(unittest.TestCase):
    def test_winograd(self):
      kernel = np.random.rand(3, 3)
      data = np.random.rand(8, 8)
      self.assertTrue(np.allclose(winograd_convolution_2d_8x8(kernel, data), correlation_2d(kernel, data)))


if __name__ == '__main__':
    unittest.main()
