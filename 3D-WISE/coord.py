import SimpleITK as sitk
import numpy as np
import torch

# HR = sitk.ReadImage("/home/shijie/Fetal_Reconstruction/T1_SR/DATA/data_56_t2/train2/HR/tra_2.nii.gz")
# LR = sitk.ReadImage("/home/shijie/Fetal_Reconstruction/T1_SR/DATA/data_56_t2/train2/LR/subject_1_t2_2.nii.gz")

def make_coord(HR,LR):
    HR_shape = HR.GetSize()
    HR_spacing = HR.GetSpacing()
    HR_origin = HR.GetOrigin()
    HR_Direction = np.array(HR.GetDirection()).reshape(3,3)
    S=np.zeros_like(HR_Direction)
    S[0,0]=HR_spacing[0]
    S[1,1]=HR_spacing[1]
    S[2,2]=HR_spacing[2]
    O=np.array(HR_origin).reshape(3,1)


    HR_x = torch.linspace(0, HR_shape[0] - 1, HR_shape[0])
    HR_y = torch.linspace(0, HR_shape[1] - 1, HR_shape[1])
    HR_z = torch.linspace(0, HR_shape[2] - 1, HR_shape[2])
    HR_X, HR_Y, HR_Z = torch.meshgrid(HR_x, HR_y, HR_z)
    HR_coord = torch.cat((HR_X.unsqueeze(0), HR_Y.unsqueeze(0), HR_Z.unsqueeze(0)), dim=0)
    W = np.dot(np.dot(HR_Direction, S), HR_coord.reshape(3,-1)) + O


    LR_shape = LR.GetSize()
    LR_spacing = LR.GetSpacing()
    LR_origin = LR.GetOrigin()
    LR_Direction = np.array(LR.GetDirection()).reshape(3,3)
    S=np.zeros_like(LR_Direction)
    S[0,0]=LR_spacing[0]
    S[1,1]=LR_spacing[1]
    S[2,2]=LR_spacing[2]
    O=np.array(LR_origin).reshape(3,1)

    C=np.dot(np.linalg.inv(np.dot(LR_Direction, S)), (W - O))
    C[0,:]=C[0,:]/LR_shape[0]
    C[1,:]=C[1,:]/LR_shape[1]
    C[2,:]=C[2,:]/LR_shape[2]
    C=2*C-1
    # C_interval=np.array([2/HR_shape[0],2/HR_shape[1],2/HR_shape[2]])
    return C.reshape(3,HR_shape[0],HR_shape[1],HR_shape[2])








