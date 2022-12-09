from lnn import Model, Predicate, Variable, And, Or, Predicates, Fact, Loss, Propositions, Implies, World, Variables
import torch.nn as nn
import gym
import numpy as np
import torch
import collections

Observation = collections.namedtuple("Observation", ("Position", "Velocity", "Angle", "AngVelocity"))

class LNNCartpole():
    def __init__(n_pos, n_vel, n_ang, n_angvel, num_nodes):
        predicate_list = []
        def create_predicates(n_nodes, name):
            for i in range(n_nodes):
                predicate_list.append(Predicate(name + str(i+1)))
                predicate_list.append(Predicate(name + str(-(i+1))))
        return predicate_list

        def create_n_ary_and(num_nodes, predicate_list):
            and_list = []
            for i in range(num_nodes):
                and_list.append(And(*predicate_list))
            return and_list

        def create_n_ary_or(and_list):
            return Or(*and_list)

        self.model = Model()
        self.preds = {
            "Position": create_predicates(n_pos, "pos"),
            "Velocity": create_predicates(n_vel, "vel"),
            "Angle": create_predicates(n_ang, "ang"),
            "AngVelocity": create_predicates(n_angvel, "angvel")
        }
        self.and_nodes = create_n_ary_and(num_nodes, preds)
        self.or_node = create_n_ary_or(and_nodes)

        self.model.add_knowledge(*and_nodes, or_node)
    def generate_initial_state_dictionary(self, raw_bin_dict):
        d = []
        for key in self.preds:
            arr = [{0: Fact.FALSE}]*(len(self.preds[key])*2)
            
            positive, value = raw_bin_dict[key]
            
            index = 2*(val-1) if positive else 2*val-1
            arr[index][0] = Fact.TRUE
            d.append(dict(zip(self.preds[key], arr)))
        res = {**d[0], **d[1], **d[2], **d[3]}
        return res
    
class FOLCartpoleAgent():
    MAXLEN = 10_000
    MIN_REPLAY_SIZE = 1_000
    BATCH_SIZE = 64
    GAMMA = 0.9
    LR = 0.01
    
    def __init__(self, n_bin_args, n_nodes, limits):
        self.left_lnn = create_lnn(*n_bin_args, n_nodes)
        self.right_lnn = create_lnn(*n_bin_args, n_nodes)
        self.limits = limits
        self.bin_args = n_bin_args
        self.bin_sizes = {}
        for key in self.bin_args:
            self.bin_sizes[key] = self.limits[key][1]/self.bin_args
            
        self.replay_memory = collections.deque([], maxlen = self.MAXLEN)
        
    def env2fol(self, obs):
        assert obs.shape == (4,)
        obs = Observation(*obs)
        ret = {}
        for key in self.limits:
            val = getattr(obs, key)
            positive = (val >= 0)
            if positive:
                val_bin = math.ceil(val/self.bin_sizes[key])
                if val/self.bin_sizes[key] - int(val/self.bin_sizes[key]) == 0:
                    val_bin += 1
                val_bin = min(val_bin, self.bin_args[key])
            else:
                val_bin = math.floor(val/self.bin_sizes[key])
                val_bin = max(val_bin, -self.bin_args[key])
            
            ret[key] = (positive, abs(val_bin))
        return ret
    
    def remember(obs):
        '''
            obs: namedtuple given by (state, action, reward, next_state, done)
        '''
        self.replay_memory.append(obs)
    
    def optimize():
        if len(self.replay_memory) < self.MIN_REPLAY_SIZE:
            return
        
        