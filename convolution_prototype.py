#!/usr/bin/env python

import numpy as np
import scipy.signal as signal
import unittest

def winograd_convolution_2d_8x8_ref(kernel, data):
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
  (c_out, c_in, kernel_h, kernel_w) = kernel.shape
  (b, c_in, data_h, data_w) = data.shape
  out = np.zeros((b, c_out, data_h - kernel_h + 1, data_w - kernel_w + 1))
  for img in range(b):
    for c_o in range(c_out):
      for c_i in range(c_in):
        U = np.dot(G, np.dot(kernel[c_o][c_i], G.T))
        V = np.dot(BT, np.dot(data[img][c_i], BT.T))
        UV = np.multiply(U, V)
        out[img][c_o] += np.dot(AT, np.dot(UV, AT.T))
  return out

def winograd_convolution_2d_4x4_ref(kernel, data):
  AT = np.array(
    [[1, 1, 1, 0],
     [0, 1, -1, -1]], dtype=np.float)
  G = np.array(
    [[1, 0, 0],
     [0.5, 0.5, 0.5],
     [0.5, -0.5, 0.5],
     [0, 0, 1]], dtype=np.float)
  BT = np.array(
    [[1, 0, -1, 0],
     [0, 1, 1, 0],
     [0, -1, 1, 0],
     [0, 1, 0, -1]], dtype=np.float)
  (c_out, c_in, kernel_h, kernel_w) = kernel.shape
  (b, c_in, data_h, data_w) = data.shape
  out = np.zeros((b, c_out, data_h - kernel_h + 1, data_w - kernel_w + 1))
  for img in range(b):
    for c_o in range(c_out):
      for c_i in range(c_in):
        U = np.dot(G, np.dot(kernel[c_o][c_i], G.T))
        V = np.dot(BT, np.dot(data[img][c_i], BT.T))
        UV = np.multiply(U, V)
        out[img][c_o] += np.dot(AT, np.dot(UV, AT.T))
  return out

def direct_convolution_2d_8x8_ref_v1(kernel, data):
  (c_out, c_in, kernel_h, kernel_w) = kernel.shape
  (b, c_in, data_h, data_w) = data.shape
  out = np.zeros((b, c_out, data_h - kernel_h + 1, data_w - kernel_w + 1))
  nchw_kernel = kernel
  nhwc_kernel = np.moveaxis(kernel, 1, -1)
  nchw_data = data
  nhwc_data = np.moveaxis(data, 1, -1)
  for img in range(b):
    for o_h in range(data_h - kernel_h + 1):
      for o_w in range(data_w - kernel_w + 1):
        for k_h in range(kernel_h):
          for k_w in range(kernel_w):
            y = o_h + k_h
            x = o_w + k_w
            for c_i in range(c_in):
              d = nhwc_data[img][y][x][c_i]
              for c_o in range(c_out):
                k = nhwc_kernel[c_o][k_h][k_w][c_i]
                out[img][c_o][o_h][o_w] += d * k
  return out 

def direct_convolution_2d_8x8_ref_v2(kernel, data):
  (c_out, c_in, kernel_h, kernel_w) = kernel.shape
  (b, c_in, data_h, data_w) = data.shape
  out = np.zeros((b, c_out, data_h - kernel_h + 1, data_w - kernel_w + 1))
  nchw_kernel = kernel
  nhwc_kernel = np.moveaxis(kernel, 1, -1)
  nchw_data = data
  nhwc_data = np.moveaxis(data, 1, -1)
  for o_h in range(data_h - kernel_h + 1):
    for o_w in range(data_w - kernel_w + 1):
      for k_h in range(kernel_h):
        for k_w in range(kernel_w):
          y = o_h + k_h
          x = o_w + k_w
          for img in range(b):
            for c_i in range(c_in):
              d = nhwc_data[img][y][x][c_i]
              for c_o in range(c_out):
                k = nhwc_kernel[c_o][k_h][k_w][c_i]
                out[img][c_o][o_h][o_w] += d * k
  return out 

def conv_ref(kernel, data):
  (c_out, c_in, kernel_h, kernel_w) = kernel.shape
  (b, c_in, data_h, data_w) = data.shape
  out = np.zeros((b, c_out, data_h - kernel_h + 1, data_w - kernel_w + 1))
  for img in range(b):
    for c_o in range(c_out):
      for c_i in range(c_in):
        out[img][c_o] += signal.correlate2d(data[img][c_i], kernel[c_o][c_i], 'valid')
  return out
  
 
class TestConvolution(unittest.TestCase):

    def test_winograd8x8(self):
      kernel = np.random.rand(4, 4, 3, 3)
      data = np.random.rand(4, 4, 8, 8)
      winograd_out = winograd_convolution_2d_8x8_ref(kernel, data)
      refconv_out = conv_ref(kernel, data)
      self.assertTrue(np.allclose(winograd_out, refconv_out))

    def test_winograd4x4(self):
      kernel = np.random.rand(4, 4, 3, 3)
      data = np.random.rand(4, 4, 4, 4)
      winograd_out = winograd_convolution_2d_4x4_ref(kernel, data)
      refconv_out = conv_ref(kernel, data)
      self.assertTrue(np.allclose(winograd_out, refconv_out))

    def test_directconv(self):
      kernel = np.random.rand(4, 4, 3, 3)
      data = np.random.rand(4, 4, 8, 8)
      direct_conv_out = direct_convolution_2d_8x8_ref_v1(kernel, data) 
      refconv_out = conv_ref(kernel, data)
      self.assertTrue(np.allclose(direct_conv_out, refconv_out))
      
if __name__ == '__main__':
    unittest.main()
