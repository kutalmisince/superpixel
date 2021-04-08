# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:53:26 2021

@author: ada
"""

from Superpixel import Superpixel
import numpy as np
import cv2


class COCOSuperpixel(Superpixel):
    
    def _init_(self):
        super(COCOSuperpixel, self)._init_()
        
    def sp_grid_fill(self, G):
                
        L = self.img_label
            
        spOutputLabels = np.zeros((self.num_sps,1))
        
        gridID = self.label_grid;
        [r, c] = np.shape(gridID)
        
        gridInput = np.zeros((r,c,3))
        gridOutput = np.zeros((r,c))
    
        
        gridInput[:,:,0] = np.reshape(self.mean[:-1,0],(r,c))
        gridInput[:,:,1] = np.reshape(self.mean[:-1,1],(r,c))
        gridInput[:,:,2] = np.reshape(self.mean[:-1,2],(r,c))
        
        for i in range(c):
            for j in range(r):
                
                
                n = int(gridID[j,i])
                            
                b = self.bbox[n, :]
                
                mask = L[b[1]:b[3], b[0]:b[2]] == n

                labels = G[b[1]:b[3], b[0]:b[2],1]
                
                h = cv2.calcHist([labels[mask]],[0],None,[256],[0,256])
                
                gtLabel = np.where(h == h.max())[0][0]
                
                gridOutput[j, i] = gtLabel
                
                spOutputLabels[n] = gtLabel
                
                
        return gridInput, gridOutput, spOutputLabels
    
    def sp_image_fill(self,inp):
        
        L = self.img_label
        
        [height, width] = np.shape(L)
        
        _,channels = np.shape(inp)
        
        I = np.zeros((height, width, channels))
        
        for n in range(self.num_sps):
            
            b = self.bbox[n,:]
            
            mask = L[b[1]:b[3], b[0]:b[2]] == n
            
            for c in range(channels):
                
                I[b[1]:b[3], b[0]:b[2],c] = I[b[1]:b[3], b[0]:b[2],c] + mask*inp[n,c]
                
            
        return I