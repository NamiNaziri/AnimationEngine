# AnimationEngine

This project contains animation features built upon one of my Computer Graphics assignments, with additional features added to the base code.

## Featrues



https://user-images.githubusercontent.com/49837425/233210116-610a1317-8071-490c-b00a-7e2b9ff4d4cf.mp4



Here, you can see the features and a blog post that explains each feature.

- GPU-Accelerated SSD
- Jacobian Inverse Kinematics [blog](https://naminaziri.github.io/inverse-kinematics)
- GPU-Based Dual Quaternion Skinning [blog](https://naminaziri.github.io/dual-quaternion-skinning)
- Keyframe Animation


## How to use keyframe animation and IK

You can choose any bone by right clicking on it. Then, you can change its rotation using the arrow keys and the PageUp and PageDown keys. Additionally, you can change the rotation by solving IK. When you select a bone, you can reset the IK target to that bone's location by pressing the "R" key on your keyboard. You can move the target by holding down the "X", "Y", or "Z" keys on your keyboard and moving your mouse in the corresponding plane. When you're satisfied with the target location, press the "Solve IK" button. Once you're happy with the pose, you can take a snapshot by clicking on the "Take Snapshot" button. Then, you can specify the time at which this snapshot should be played by typing in the desired second.

## Dual Quaternion Skinning On GPU



https://user-images.githubusercontent.com/49837425/233210545-61115954-2c34-4282-8f20-838d04994729.mp4


## Reference

### Matrix calculation is done using Eigen library

https://eigen.tuxfamily.org/index.php?title=Main_Page

### IK
http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/handouts/IK.pdf

https://cseweb.ucsd.edu/classes/sp16/cse169-a/slides/CSE169_09.pdf

https://math.stackexchange.com/questions/728666/calculate-jacobian-matrix-without-closed-form-or-analytical-form

https://robotics.stackexchange.com/a/22634

### Dual Quaternion Skinning

https://users.cs.utah.edu/~ladislav/kavan08geometric/kavan08geometric.html
