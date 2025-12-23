import os
import matplotlib.pyplot as plt
from iohinspector import DataManager

manager = DataManager()

result_dir = "data"
folders = [os.path.join(result_dir, f) for f in os.listdir(result_dir)]
manager.add_folders(folders)

selection = manager.select(
    function_ids=[18],
    dimensions=[50],
    algorithms=["genetic_algorithm"]
)

df = selection.load(
    monotonic=True,
    include_meta_data=True
)

final_eval = []
final_y = []

eval_temp = [0]
y_temp = [0]

for evaluation, y in zip(df["evaluations"], df["raw_y"]):
    if evaluation < eval_temp[-1]:
        final_eval.append(eval_temp)
        final_y.append(y_temp)
        eval_temp = [0]
        y_temp = [0]

    eval_temp.append(evaluation)
    y_temp.append(y)

fig, ax = plt.subplots(1, 2, figsize=(15, 8))
for evaluations, y in zip(final_eval, final_y):
    ax[0].plot(evaluations, y)
ax[0].set_ylabel('Fitness score')
ax[0].set_xlabel('Evaluations')
ax[0].grid(alpha=0.6)
ax[0].set_xscale('log')
ax[0].set_title('LABS')


manager = DataManager()

result_dir = "data"
folders = [os.path.join(result_dir, f) for f in os.listdir(result_dir)]
manager.add_folders(folders)

selection = manager.select(
    function_ids=[23],
    dimensions=[49],
    algorithms=["genetic_algorithm"]
)

df = selection.load(
    monotonic=True,
    include_meta_data=True
)

final_eval = []
final_y = []

eval_temp = [0]
y_temp = [0]

for evaluation, y in zip(df["evaluations"], df["raw_y"]):
    if evaluation < eval_temp[-1]:
        final_eval.append(eval_temp)
        final_y.append(y_temp)
        eval_temp = [0]
        y_temp = [0]

    eval_temp.append(evaluation)
    y_temp.append(y)

for evaluations, y in zip(final_eval, final_y):
    ax[1].plot(evaluations, y)
ax[1].set_xlabel('Evaluations')
ax[1].grid(alpha=0.6)
ax[1].set_xscale('log')
ax[1].set_title('N-Queens')


plt.savefig('results.png')

plt.close()
