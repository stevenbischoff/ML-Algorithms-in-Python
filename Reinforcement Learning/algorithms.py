"""
This module contains a learner class that can implement value iteration,
Q-learning, and SARSA and run trials.

Author: Steve Bischoff
"""
import numpy as np
import random
import itertools
import copy

seed = 1682779292
random.seed(seed)
np.random.seed(seed)

class learner():
    """
    Parameters:
        car:
        track:
        gamma: float in [0, 1], future discount rate
        lr: float, learning rate
        T: int, training runs
    """
    def __init__(self, car, track, gamma, lr, T=100000):

        self.t = 0 # training rounds
        self.q_updates = 0 # number of times self.Q is updated
        self.q_updates_list = []
        #self.test_results = []

        self.car = car
        self.track = track
        self.epsilon = max(1/(self.t+4), 0.001)
        self.gamma = gamma
        self.lr = lr
        self.T = T

        self.S = list(itertools.product(track.positions, car.velocities))
        self.A = car.accelerations

        self.Q = self._initial_Q()
        self.values_to_policy()


    def _initial_Q(self):
        """
        Randomly initializes Q values around 0.
        Returns:
            Q: dict of value estimates
        """
        Q = {}
        for s in self.S:
            Q[s] = {}
            for a in self.A:
                Q[s][a] = np.random.normal(loc=0.0, scale=0.01)
        return Q


    def values_to_policy(self):
        """
        Converts the agent's Q function to a policy by determining the current
        best action for each state.
        Returns:
            self.policy: A dictionary giving a function from states to actions
        """
        self.policy = {}
        for s in self.Q.keys():
            best_q = -10000
            Q_s = self.Q[s]
            for a in self.A:
                q = Q_s[a]
                if q > best_q:
                    best_q = q
                    best_a = a
            self.policy[s] = best_a
        return self.policy


    def value_iteration_sweep(self, deterministic):

        V_prime = copy.copy(self.V)
        new_positions = {}
        for s in self.S:
            pos = s[0]
            vel = s[1]
            Q_s = self.Q[s]

            for a in self.A:
                if deterministic:
                    s_prime, r, crash = self.track.get_s_prime(
                        pos, vel, a, self.car)
                    Q_s[a] = r + self.gamma*self.V[s_prime]
                else: # weighted sum of both possibilities
                    s_prime_s, r_s, crash = self.track.get_s_prime(
                        pos, vel, a, self.car)
                    s_prime_f, r_f, crash = self.track.get_s_prime(
                        pos, vel, tuple([0,0]), self.car)
                    Q_s[a] = 0.8*r_s + 0.2*r_f + self.gamma*(
                        0.8*self.V[s_prime_s] + 0.2*self.V[s_prime_f])
                
            V_prime[s] = max(Q_s.values())

        return V_prime


    def value_iteration(self, deterministic=False):
        """
        Parameters:
            deterministic: Boolean
        Returns:
            self.V: An estimate of the cumulative value of each possible state.
        """
        # initialize V(s)
        self.V = {}
        for s in self.S:
            self.V[s] = np.random.normal(loc=0.0, scale=0.01)

        converged = False     
        while not converged:
            self.t += 1

            V_prime = self.value_iteration_sweep(deterministic)
            
            converged = True
            for s in self.S:
                if abs(V_prime[s] - self.V[s]) > 0.0001:
                    converged = False
                    break
            self.V = V_prime

        return self.V
    

    def q_learning(self, T_round):
        """
        Parameters:
            T_round: number of training rounds to perform
        Returns:
            self.Q: An estimate of the cumulative value of each possible state
                / action pair
        """       
        for i in range(T_round):

            self.t += 1

            # update learning rate
            if self.t == self.T*15//16:
                self.lr = self.lr/2
            elif self.t == self.T*7//8:
                self.lr = self.lr/2
            elif self.t == self.T*3//4:
                self.lr = self.lr/2
            elif self.t == self.T//2:
                self.lr = self.lr/2
            elif self.t == self.T//4:
                self.lr = self.lr/2
            
            # update /eps/
            self.epsilon = max(1/(self.t+4), 0.001) # +4 else too long at first

            # initialize /s/
            s = tuple([self.car.pos, self.car.vel])

            score = 0
            while True:
                pos = s[0]
                vel = s[1]

                # choose /a/ using policy derived from Q, e.g. /eps/-greedy
                if np.random.uniform() < self.epsilon:
                    best_a = random.choice(self.A)
                else:
                    best_q = -10000
                    Q_s = self.Q[s]
                    for a in self.A:
                        q = Q_s[a]
                        if q > best_q:
                            best_q = q
                            best_a = a
                # take action /a/, observe /r/ and /s_prime/
                if np.random.uniform() < 0.2:
                    s_prime, r, crash = self.track.get_s_prime(pos, vel, tuple([0, 0]), self.car)
                else:
                    s_prime, r, crash = self.track.get_s_prime(pos, vel, best_a, self.car)            

                # Update /Q(s,a)/
                # get max Q(s_prime, a_prime)
                best_q_prime = -1000
                for a_prime in self.A:
                    q_prime = self.Q[s_prime][a_prime]
                    if q_prime > best_q_prime:
                        best_q_prime = q_prime
                        best_a_prime = a_prime

                self.Q[s][best_a] = self.Q[s][best_a] + self.lr*(r + self.gamma*best_q_prime - self.Q[s][best_a])
                self.q_updates += 1

                s = s_prime

                score += 1

                if s[0] in self.track.finish_idx:
                    break
                #if score >= 300:
                 #   break

            self.q_updates_list.append(score)

        return self.Q


    def SARSA(self, T_round):
        """
        Parameters:
            T_round:
        Returns:
            self.Q: An estimate of the cumulative value of each possible state
                / action pair
        """
        for i in range(T_round):

            self.t += 1

            # update learning rate
            if self.t == self.T*15//16:
                self.lr = self.lr/2
            elif self.t == self.T*7//8:
                self.lr = self.lr/2
            elif self.t == self.T*3//4:
                self.lr = self.lr/2
            elif self.t == self.T//2:
                self.lr = self.lr/2
            elif self.t == self.T//4:
                self.lr = self.lr/2

            # update /eps/
            self.epsilon = max(1/(self.t+4), 0.001) # +4 else too long at first

            # initialize /s/
            s = tuple([self.car.pos, self.car.vel])

            # choose action /a/ using (policy derived from) /Q/, e.g. /eps/-greedy
            if np.random.uniform() < self.epsilon:
                best_a = random.choice(self.A)
            else:
                best_q = -10000
                Q_s = self.Q[s]
                for a in self.A:
                    q = Q_s[a]
                    if q > best_q:
                        best_q = q
                        best_a = a

            score = 0           
            while True:
                pos = s[0]
                vel = s[1]

                # take action /a/, observe /r/ and /s_prime/
                if np.random.uniform() < 0.2:
                    s_prime, r, crash = self.track.get_s_prime(
                        pos, vel, tuple([0, 0]), self.car)
                else:
                    s_prime, r, crash = self.track.get_s_prime(
                        pos, vel, best_a, self.car)

                # Choose /a_prime/ using policy derived from /Q/, e.g. /eps/-greedy
                if np.random.uniform() < self.epsilon:
                    best_a_prime = random.choice(self.A)
                    best_q_prime = self.Q[s_prime][best_a_prime]
                else:
                    best_q_prime = -10000
                    Q_s_prime = self.Q[s_prime]
                    for a in self.A:
                        q = Q_s_prime[a]
                        if q > best_q_prime:
                            best_q_prime = q
                            best_a_prime = a

                # Update /Q(s,a)/
                self.Q[s][best_a] = self.Q[s][best_a] + self.lr*(
                    r + self.gamma*best_q_prime - self.Q[s][best_a])
                self.q_updates += 1

                s = s_prime
                best_a = best_a_prime

                score += 1

                if s[0] in self.track.finish_idx:
                    break
                #if score >= 300:
                 #   break
            self.q_updates_list.append(score)
                
        return self.Q


    def trial(self, deterministic=False, max_score=300, verbose=False):
        """
        Performs a single "time trial" using the current policy.
        Parameters:
            deterministic:
            max_score: the number of moves until the trial terminates
            verbose:
        Returns:
            score: the number of moves required to reach the finish.
            path: an ordered list of each state reached during the trial
        """
        s = tuple([self.car.pos, self.car.vel])
        S = list(itertools.product(self.track.positions, self.car.velocities))

        if verbose:
            self.track.track[s[0]] = 'X'

        score = 0
        path = [s]
        actions = []
        while True:

            if deterministic:
                a = self.policy[s]
            else:
                if np.random.uniform() < 0.2:
                    a = tuple([0, 0])
                else:
                    a = self.policy[s]

            actions.append(a)

            pos = s[0]
            vel = s[1]

            s, r, crash = self.track.get_s_prime(pos, vel, a, self.car)
            if crash:
                path.append('Crash!')
            path.append(s)

            if verbose:
                self.track.track[s[0]] = 'X'

            score += 1

            if s[0] in self.track.finish_idx:
                break
            if score >= max_score:
                break

        if verbose:       
            self.track.print_track()
            self.track.reset_track()

        return score, path


    def n_trials(self, test_trials, deterministic=False, max_score=300,
                 verbose=False):
        """
        Performs a given number of trials with the current policy.
        Parameters:
            test_trials: the number of trials to perform
            deterministic:
            max_score: the number of moves until the trial terminates
            verbose:
        Returns:
            trial_results: a list of trial scores
        """
        trial_results = []
        for i in range(test_trials):
            trial_results.append(
                self.trial(deterministic, max_score, verbose)[0])
        return trial_results


    def train_with_learning_curves(self, algorithm, test_intvl=10,
            test_trials=10, deterministic=False, max_score=300, verbose=False):
        """
        Trains the agent using an input algorithm, periodically performing test
        trial runs.
        Parameters:
            algorithm: self.q_learning or self.SARSA
            test_intvl: the number of training intervals between test trial runs
            test_trials: the number of trials to perform
            deterministic:
            max_score: the number of moves until the trial terminates
            verbose:
        """
        self.test_results = []
        self.cum_q_scores = []

        trial_results = self.n_trials(
            test_trials, deterministic, max_score, verbose)
        self.test_results.append(trial_results)
        self.cum_q_scores.append(self.q_updates)

        while self.t < self.T:
            self.Q = algorithm(test_intvl)
            self.values_to_policy()

            trial_results = self.n_trials(
                test_trials, deterministic, max_score, verbose)
            self.test_results.append(trial_results)
            self.cum_q_scores.append(self.q_updates)

        self.test_means = [np.mean(i) for i in self.test_results]
