# test if the iteration method converges in discrete situation

import numpy as np
import matplotlib.pyplot as plt


# define probabilities on 11 states
p = np.array([2/11,0.5/11,0.5/11,2/11,0.5/11,0.5/11,1/11,1/11,1/11,1/11,1/11]
             )
a = np.zeros(11)
# define marginal probabilities 
# x
p_x = np.array([4/6,1/3])
# y
p_y = np.array([1/8,1/4,1/8,1/2])

N = 100
diff = np.zeros(N)
diff_x = np.zeros(N)
diff_y = np.zeros(N)    
# start iteration
for i in range(N):
    if i%2 ==1:
        # first contour
        a[0] = p_x[0] * p[0]/( p[0] + p[1]+p[3]+p[6]+p[10]+p[5] + p[9]) 
        a[1] = p_x[0] * p[1]/( p[0] + p[1]+p[3]+p[6]+p[10]+p[5] + p[9]) 
        a[3] = p_x[0] * p[3]/( p[0] + p[1]+p[3]+p[6]+p[10]+p[5] + p[9])
        a[6] = p_x[0] * p[6]/ ( p[0] + p[1]+p[3]+p[6]+p[10]+p[5] + p[9])
        a[10] = p_x[0] * p[10]/( p[0] + p[1]+p[3]+p[6]+p[10]+p[5] + p[9])
        a[5] = p_x[0] * p[5]/( p[5] + p[9]+p[0] + p[1]+p[3]+p[6]+p[10])
        a[9] = p_x[0] * p[9]/(p[5] + p[9]+p[0] + p[1]+p[3]+p[6]+p[10])
        
        # second contour
        a[2] = p_x[1] * p[2]/(p[2] + p[4] + p[7] +p[8])
        a[4] = p_x[1] *p[4]/(p[2] + p[4] + p[7] +p[8])
        a[7] = p_x[1] * p[7]/(p[2] + p[4] + p[7] +p[8])
        a[8] = p_x[1] * p[8]/(p[2] + p[4] + p[7] +p[8])

    else:
        # first row
        a[0] = p_y[0] * p[0]/(p[0] +p[1])
        a[1] = p_y[0] * p[1]/(p[0] +p[1])
        # second row
        a[2] = p_y[1] * p[2]/(p[2] +p[3])
        a[3] = p_y[1] * p[3]/(p[2] +p[3])
        # third row
        a[4] = p_y[2] * p[4]/(p[4] + p[5] +p[6])
        a[5] = p_y[2] * p[5]/(p[4] + p[5] +p[6])
        a[6] = p_y[2] * p[6]/(p[4] + p[5] +p[6])
        # fourth row
        a[7] = p_y[3] * p[7]/(p[7] + p[8] + p[9] + p[10])
        a[8] = p_y[3] * p[8]/(p[7] + p[8] + p[9] + p[10])
        a[9] = p_y[3] * p[9]/(p[7] + p[8] + p[9] + p[10])
        a[10] = p_y[3] * p[10]/(p[7] + p[8] + p[9] + p[10])
    
    # compute the difference between p and a
    diff[i] = np.linalg.norm(p-a)
    # update p
    p = a.copy()
    print(p[1:9])
    # compute the marginal difference
    diff_x[i] = np.linalg.norm(p_x - np.array([ p[0]+p[1]+p[3]+p[6]+p[10]+p[5] + p[9], p[2] + p[4] + p[7] +p[8]]))
    diff_y[i] = np.linalg.norm(p_y - np.array([p[0]+p[1], p[2]+p[3], p[4]+p[5]+p[6], p[7]+p[8]+p[9]+p[10]]))

plt.plot(diff)
plt.plot(diff_x)
plt.plot(diff_y)
plt.xlabel('iteration')
plt.ylabel('difference')
plt.title('Difference between p and a over iterations')
plt.show()

