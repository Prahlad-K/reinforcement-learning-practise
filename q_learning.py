import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from monte_carlo_es import max_dict
from TD0_prediction import random_action

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


if __name__ == "__main__":
	grid = negative_grid(step_cost=-0.1)

	# print rewards
	print("rewards:")
	print_values(grid.rewards, grid)

	Q = {}
	states= grid.all_states()
	for s in states:
		Q[s] = {}
		for a in ALL_POSSIBLE_ACTIONS:
			Q[s][a] = 0

	update_counts = {}
	update_counts_sa = {}

	for s in states:
		update_counts[s] = 1.0
		update_counts_sa[s] = {}
		for a in ALL_POSSIBLE_ACTIONS:
			update_counts_sa[s][a] = 1.0

	t = 1.0
	deltas = []
	for it in range(10000):
		if it%100 == 0:
			t += 1e-2
		if it%2000 == 0:
			print("it:", it)

		s = (2, 0)
		grid.set_state(s)

		a= max_dict(Q[s])[0]
		

		biggest_change = 0
		while not grid.game_over():
			a = random_action(a, eps = 0.5/t)
			r = grid.move(a)
			s2 = grid.current_state()


			alpha = ALPHA / update_counts_sa[s][a]
			update_counts_sa[s][a] += 0.005
			old_qsa = Q[s][a]
			a2, max_q_s2a2 = max_dict(Q[s2])

			Q[s][a] = Q[s][a] + alpha*(r + GAMMA*max_q_s2a2 - Q[s][a])
			biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))

			update_counts[s] = update_counts.get(s, a) + 1
			s = s2
			a = a2

		deltas.append(biggest_change)
	plt.plot(deltas)
	plt.show()

	policy = {}
	V = {}

	for s in grid.actions.keys():
		a, max_q = max_dict(Q[s])
		policy[s] = a
		V[s] = max_q

	print("update counts:")
	total = np.sum(list(update_counts.values()))
	for k, v in update_counts.items():
		update_counts[k] = float(v) / total
		print_values(update_counts, grid)

	print("values:")
	print_values(V, grid)
	print("policy:")
	print_policy(policy, grid)