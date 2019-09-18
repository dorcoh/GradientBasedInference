import os
from timeit import default_timer as timer

learning_rates = [1e-3, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
iterations = [15, 30, 45]
regularization = [1e-3, 1e-7, 1e-9, 1e-11, 0]
data = 'fixed+failed'

if not os.path.exists('logs_exp_one'):
    os.makedirs('logs_exp_one')


for i in iterations:
    for l in learning_rates:
        for r in regularization:
            # skip done exps
            if l == 1e-3 and r == 1e-3 and i == 15:
                continue
            start = timer()
            params = (data, l, r, i)
            print("Exp ", params)
            command = "python -u main.py --load %s -l %.20f -a %.20f -i %d -c " % params
            command += "> logs_exp_one/out_lr_%.20f_reg_%.20f_it_%d " % (l, r, i)
            print(command)
            os.system(command)
            end = timer()
            print("Time: ", end-start)
