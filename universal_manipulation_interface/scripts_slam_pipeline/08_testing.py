import roboticstoolbox as rtb
import numpy as np
from spatialmath import SE3

ur10 = rtb.models.DH.UR10()
initial_guess_deg = [-37.26, -119.02,-122.68, 57.14, 88.44, -0.46]
initial_guess_rad = np.deg2rad(initial_guess_deg)


# pose = ur10.fkine(initial_guess_rad) 
# print(pose)

pose = [ 0.03058861, -0.29987455,  0.23660799, -2.07251733, -0.03173392, 0.02153551]
# pose = [0.03034556 ,-0.05935223  ,0.05730286 ,-2.14903608 ,-0.29645124 , 0.26392691]
t = pose[:3]  # translational components (x, y, z)
rpy = pose[3:]

R = SE3.Rx(rpy[0]) * SE3.Ry(rpy[1]) * SE3.Rz(rpy[2])
T = SE3(t) * R

print(T)
joint_solutions = ur10.ikine_LM(T, q0=initial_guess_rad)
if joint_solutions.success == True:
    print("Joints config:", joint_solutions.q)
else:
    print("Inverse kinematics failed to converge. No solution found.")