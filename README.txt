# CS-C3100 Computer Graphics, Fall 2021
# Lehtinen / Aho, Kaskela, Kynkäänniemi
#
# Assignment 3: Hierarchical transforms

Student name:
Student number:
Hours spent on requirements (approx.): 3 hours
Hours spent on extra credit (approx.):

# First, a 10-second poll about this assignment period:

Did you go to exercise sessions?

Did you work on the assignment using Aalto computers, your own computers, or both?

# Which parts of the assignment did you complete? Mark them 'done'.
# You can also mark non-completed parts as 'attempted' if you spent a fair amount of
# effort on them. If you do, explain the work you did in the problems/bugs section
# and leave your 'attempt' code in place (commented out if necessary) so we can see it.

           R1 Calculating joint positions (1 p): done
                   R2 Rotating the joints (2 p): done
  R3 Visualizing joint coordinate systems (2 p): done
       R4 Skeletal subspace deformation (4 pts): done
                R5 Skinning for normals (1 pts): done

# Did you do any extra credit work?

Easy: SSD on the GPU (2p) : 35min
Medium: Add a \wrist" joint to both hands (5p) : 5 hours
Medium/hard: IK (8p) : 16 hours
Animation (3 pts)    : 6 hours
Dual quaternion skinning (4 p) : 8 hours

(Describe what you did and, if there was a substantial amount of work involved, how you did it. Also describe how to use/activate your extra features, if they are interactive.)

# Are there any known problems/bugs remaining in your code?

(Please provide a list of the problems. If possible, describe what you think the cause is, how you have attempted to diagnose or fix the problem, and how you would attempt to diagnose or fix it if you had more time or motivation. This is important: we are more likely to assign partial credit if you help us understand what's going on.)


Altough I have some idea of how to use dual numbers I decided to not use dual numbers for calculating the derivatives for the Jacobian matrix. 
I just used the derivative definition. Eigen C++ template library is used for matrix calculations.

refrences:

http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/handouts/IK.pdf

https://cseweb.ucsd.edu/classes/sp16/cse169-a/slides/CSE169_09.pdf

https://math.stackexchange.com/questions/728666/calculate-jacobian-matrix-without-closed-form-or-analytical-form

https://robotics.stackexchange.com/a/22634


The dual quaternion has some artifact in some orientations. I'm not sure this artifact arises because
of the wrong implementation or it's just how it is :) since it happens in unusual orientations.
"The artifact is significantly less noticeable for the mocapguy so I assume it depends on the character and the rigging.

https://www.youtube.com/watch?v=BbtSTWsiasE&feature=youtu.be



# Did you collaborate with anyone in the class?

(Did you help others? Did others help you? Let us know who you talked to, and what sort of help you gave or received.)

# Any other comments you'd like to share about the assignment or the course so far?

(Was the assignment too long? Too hard? Fun or boring? Did you learn something, or was it a total waste of time? Can we do something differently to help you learn? Please be brutally honest; we won't take it personally.)

