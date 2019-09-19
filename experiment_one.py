import os
from timeit import default_timer as timer

learning_rates = [1e0, 1e1, 1e2]
iterations = [15, 30]
regularization = [0, 1e-11, 1e-5]
pickle_paths = ['expdata/baseline_dev', 'expdata/baseline_test']
data = 'fixed+failed'

if not os.path.exists('logs_exp_one'):
    os.makedirs('logs_exp_one')


for i in iterations:
    for l in learning_rates:
        for r in regularization:
            for path in pickle_paths:
                # skip baselines
                if l == 1e01 and r == 0 and i == 15:
                    continue
                start = timer()
                params = (data, l, r, i, path)
                print("Exp ", params)
                command = "python -u main.py --load %s -l %.20f -a %.20f -i %d -p %s" % params
                command += "> logs_exp_one/out_lr_%.20f_reg_%.20f_it_%d " % (l, r, i)
                print(command)
                os.system(command)
                end = timer()
                print("Time: ", end-start)
