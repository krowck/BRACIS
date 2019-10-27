## Generate a contour plot
# Import some other libraries that we'll need
# matplotlib and numpy packages must also be installed
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# define objective function
def f(x):
    x1 = x[0]
    x2 = x[1]
    obj = 0.2 + x1**2 + x2**2 - 0.1*math.cos(6.0*3.1415*x1) - 0.1*math.cos(6.0*3.1415*x2)
    return obj

def sa(x_start, f):
# Start location

    ##################################################
    # Simulated Annealing
    ##################################################
    # Number of cycles
    n = 50
    # Number of trials per cycle
    m = 50
    # Number of accepted solutions
    na = 0.0
    # Probability of accepting worse solution at the start
    p1 = 0.7
    # Probability of accepting worse solution at the end
    p50 = 0.001
    # Initial temperature
    t1 = -1.0/math.log(p1)
    # Final temperature
    t50 = -1.0/math.log(p50)
    # Fractional reduction every cycle
    frac = (t50/t1)**(1.0/(n-1.0))
    # Initialize x
    x = np.zeros((n+1,2))
    x[0] = x_start
    xi = np.zeros(2)
    xi = x_start
    na = na + 1.0
    # Current best results so far
    xc = np.zeros(2)
    xc = x[0]
    fc = f.evaluate(xi)
    fs = np.zeros(n+1)
    fs[0] = fc
    # Current temperature
    t = t1
    # DeltaE Average
    DeltaE_avg = 0.0
    for i in range(n):
        print('Cycle: ' + str(i) + ' with Temperature: ' + str(t))
        for j in range(m):
            # Generate new trial points
            xi[0] = xc[0] + random.random() - 0.5
            xi[1] = xc[1] + random.random() - 0.5
            # Clip to upper and lower bounds
            xi[0] = max(min(xi[0],1.0),-1.0)
            xi[1] = max(min(xi[1],1.0),-1.0)
            DeltaE = abs(f.evaluate(xi)-fc)
            if (f.evaluate(xi)>fc):
                # Initialize DeltaE_avg if a worse solution was found
                #   on the first iteration
                if (i==0 and j==0): DeltaE_avg = DeltaE
                # objective function is worse
                # generate probability of acceptance
                p = math.exp(-DeltaE/(DeltaE_avg * t))
                # determine whether to accept worse point
                if (random.random()<p):
                    # accept the worse solution
                    accept = True
                else:
                    # don't accept the worse solution
                    accept = False
            else:
                # objective function is lower, automatically accept
                accept = True
            if (accept==True):
                # update currently accepted solution
                xc[0] = xi[0]
                xc[1] = xi[1]
                fc = f.evaluate(xc)
                # increment number of accepted solutions
                na = na + 1.0
                # update DeltaE_avg
                DeltaE_avg = (DeltaE_avg * (na-1.0) +  DeltaE) / na
        # Record the best x values at the end of every cycle
        x[i+1][0] = xc[0]
        x[i+1][1] = xc[1]
        fs[i+1] = fc
        # Lower the temperature for next cycle
        t = frac * t

    # print solution
    print('Best solution: ' + str(xc))
    print('Best objective: ' + str(fc))
