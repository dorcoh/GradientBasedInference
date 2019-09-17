import os
from timeit import default_timer as timer

learning_rates = [1e-3, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
iterations = [15, 30, 45, 60]
regularization = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]
data = 'failed'

if not os.path.exists('logs_exp_one'):
    os.makedirs('logs_exp_one')


for i in iterations:
    for l in learning_rates:
        for r in regularization:
            start = timer()
            params = (data, l, r, i)
            print("Exp ", params)
            command = "python -u main.py --load %s -l %f -a %f -i %d " % params
            command += "> logs_exp_one/out_lr_%f_reg_%f_it_%d " % (l, r, i)
            os.system(command)
            end = timer()
            print("Time: ", end-start)
