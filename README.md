# CBAM-brain-tumor-segmentation

The aim of this project is to add the convolutional block attention module (CBAM) to an already existing U-Net model, developed by the German Cancer Research Center.

This model is designed to produce a 3D segmentation of brain tumors, using 3D scans. 

Note that my contribution here is adding the CBAM module, this contribution can be found under the file "network_architecture" in files with the word cbam in their name.
I have also added residual blocks to the original code.

Test results  (with the added CBAM and/or residual layers) show notable improvement of segmentation results (Hausdorff metric and DICE index).

