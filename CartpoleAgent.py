from lnn import Model, Predicate, Variable, And, Or, Predicates, Fact, Loss, Propositions, Implies, World, Variables
import numpy as np	
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import math

#MATTHEW BRANCH HERE

Observation = collections.namedtuple("Observation", ("Position", "Velocity", "Angle", "AngVelocity"))
Transition = collections.namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))

class LNNCartpole():
  def __init__(self, num_nodes, n_pos, n_vel, n_ang, n_angvel):#, left):

    def create_predicates(n_nodes, name, var):
      predicate_list = []
      for i in range(n_nodes):
        predicate_list.append(Predicate(name + str(i+1))(var))
        predicate_list.append(Predicate('not' + name + str(i+1))(var))
        predicate_list.append(Predicate(name + str(-(i+1)))(var))
        predicate_list.append(Predicate('not' + name + str(-(i+1)))(var))     
      return predicate_list

    def create_n_ary_and(num_nodes, preds):
      and_list = []
      for _ in range(num_nodes):
        and_list.append(And(*preds["Position"], *preds["Velocity"], *preds["Angle"], *preds["AngVelocity"]))
      return and_list

    def create_n_ary_or(and_list):
      return Or(*and_list)

    self.model = Model()
    x = Variable('x')
    self.preds = {
      "Position": create_predicates(n_pos, "pos", x),
      "Velocity": create_predicates(n_vel, "vel", x),
      "Angle": create_predicates(n_ang, "ang", x),
      "AngVelocity": create_predicates(n_angvel, "angvel", x)
    }
    self.and_nodes = create_n_ary_and(num_nodes, self.preds)
    self.or_node = create_n_ary_or(self.and_nodes)

    self.model.add_knowledge(*self.and_nodes, self.or_node)
    
    #self.left = left

  def generate_state_dictionary(self, processed_fol_arr):
    '''
      params: processed_fol_arr - array of dictionaries that determine which bins should be evaluated to true on initial downward pass
      returns: dictionary that can be directly inputted into lnn for training/inference
    '''

    ###FIX HERE TO INDEXING ISSUES

    ### false false faslef aslefalflflasfelafla flasefla  flasef lfae al ture  flasleflasflasflasfelas f  // ture turue tutue tuet eut etuet uet uet eutetfalse false true
    d = []
    for key in self.preds:
      value_array = []
      for i, fol in enumerate(processed_fol_arr):
        positive, value = fol[key]
        # if self.left:
        #   positive = not(positive)
        index = 2*(value-1) if positive else 2*value-1
        index *= 2
        for j in range(len(self.preds[key])//2):
          if len(value_array) <= 2*j:
            value_array.append({})
            value_array.append({})
          
          if 2*j == index:
            value_array[j][str(i)] = Fact.TRUE
            value_array[j+1][str(i)] = Fact.FALSE
          else:
            value_array[j][str(i)] = Fact.FALSE
            value_array[j+1][str(i)] = Fact.TRUE
        
        predicate_array = np.array(self.preds[key], dtype = object)[:, 0]
        
        d.append(dict(zip(predicate_array, value_array)))
    res = {**d[0], **d[1], **d[2], **d[3]}
    return res

  def generate_label_dictionary(self, qval_arr, err=0.05):
    '''
      params:
        qval_arr: array of qvals for training
        err: float error radius on truth bounds

      returns:
        label_dict: dictionary {str(i): (qval-err, qval+err)} for each qval in qval_arr
    '''
    label_dict = {self.or_node: {str(i): (max(qval-err, 0.), min(qval+err, 1.)) for i, qval in enumerate(qval_arr)}}
    return label_dict

  def forward(self, processed_fol_arr):
    '''
      params:
        processed_fol_arr: array of fol observations used to generate state dict

      returns:
        output: bsz x 2 tensor of lower/upper bounds for each batch example
    '''
    self.model.flush()
    state_dict = self.generate_state_dictionary(processed_fol_arr)
    self.model.add_data(state_dict)
    self.model.infer()
    return self.or_node.get_data()

  def train_step(self, obs, labels, lr, steps=1):
    '''
      params:
        obs: array of dictionaries corresponding to first order logic of input nodes
        labels: array of floats corresponding to the labels of observations

      returns:
        loss: loss over training	
    '''
    assert len(obs) == labels.shape[0]

    self.model.flush()

    state_dict = self.generate_state_dictionary(obs)
    self.model.add_data(state_dict)
    label_dict = self.generate_label_dictionary(labels)
    self.model.add_labels(label_dict)
    
    epochs, loss = self.model.train(losses=[Loss.SUPERVISED], learning_rate = lr, epochs=steps)
    return loss

class FOLCartpoleAgent():
  MAXLEN = 10_000
  MIN_REPLAY_SIZE = 1_000
  BATCH_SIZE = 4
  GAMMA = 0.9
  LR = 0.001

  def __init__(self, n_bin_args, n_nodes, limits):
    # self.left_lnn = LNNCartpole(n_nodes, **n_bin_args, left = True)
    # self.right_lnn = LNNCartpole(n_nodes, **n_bin_args, left = False)
    self.lnn = LNNCartpole(n_nodes, **n_bin_args)
    self.limits = limits
    self.bin_args = n_bin_args
    self.bin_sizes = {}
    for (key1, key2) in zip(self.limits, self.bin_args):
      self.bin_sizes[key1] = self.limits[key1][1]/self.bin_args[key2]

    self.replay_memory = collections.deque([], maxlen = self.MAXLEN)

  def envs2fol(self, obs_arr):
    ret = []
    for obs in obs_arr:
      ret.append(self.env2fol(obs))
    return ret

  def env2fol(self, obs):
    assert obs.shape == (4,)
    obs = Observation(*obs)
    ret = {}
    for (key1, key2) in zip(self.limits, self.bin_args):
      val = getattr(obs, key1)
      positive = (val >= 0)
      if positive:
        val_bin = math.ceil(val/self.bin_sizes[key1])
        if val/self.bin_sizes[key1] - int(val/self.bin_sizes[key1]) == 0:
          val_bin += 1
        val_bin = min(val_bin, self.bin_args[key2])
      else:
        val_bin = math.floor(val/self.bin_sizes[key1])
        val_bin = max(val_bin, -self.bin_args[key2])

      ret[key1] = (positive, abs(val_bin))
    return ret

  def remember(self, obs):
    '''
      obs: namedtuple given by (state, action, reward, next_state, done)
    '''
    self.replay_memory.append(obs)

  def optimize(self):
    '''
      returns: total loss over the two lnns over one training step
    '''
    if len(self.replay_memory) < self.MIN_REPLAY_SIZE:
      return

    ##NEEDS TO ONLY USE ONE LNN


    #Randomly samples from memory buffer and transforms into namedtuple
    transitions = [self.replay_memory[idx] for idx in np.random.permutation(len(self.replay_memory))[:self.BATCH_SIZE]]
    batch = Transition(*zip(*transitions))

    #Mask to separate inputs when calculating state_values (direction masks) and next_state_values (final mask)
    final_mask = torch.tensor([val == False for val in batch.done])
    
    
    # left_mask = torch.tensor([val == 0 for val in batch.action], device = self.device) #True is left, False is Right
    # right_mask = left_mask == False		

    #perform next_state_calculations and fill expected values tensor

    if final_mask.sum().item() > 0:
      next_state_batch = self.envs2fol(np.array(batch.next_state)[final_mask])

      next_values = self.lnn.forward(next_state_batch).mean(dim=1)

      next_state_values = torch.zeros(self.BATCH_SIZE)
      next_state_values[final_mask] = next_values[:final_mask.sum().item()]

      # right_next_values = self.right_lnn.forward(next_state_batch).mean(dim=1)
      # left_next_values = self.left_lnn.forward(next_state_batch).mean(dim=1)
      ##next_state_values[final_mask] = torch.stack((right_next_values, left_next_values), dim=1).max(dim=1).values

      reward_batch = torch.tensor(batch.reward)
      expected_next_state_values = next_state_values * self.GAMMA + reward_batch
    
    else:
      expected_next_state_values = reward_batch

    #perform state_calculations to derive loss from current state -> expected values
    state_batch = self.envs2fol(batch.state)
    loss = self.lnn.train_step(state_batch, expected_next_state_values, lr = self.LR, steps = 1)

    return loss

    # state_batch_right = self.envs2fol(np.array(batch.state)[right_mask])
    # state_batch_left = self.envs2fol(np.array(batch.state)[left_mask])

    # loss_left = self.left_lnn.train_step(state_batch_left, expected_next_state_values[left_mask])
    # loss_right = self.right_lnn.train_step(state_batch_right, expected_next_state_values[right_mask])
    # return loss_left + loss_right

  def sample_random_action(self):
    '''
      0: left
      1: right
    '''
    return np.random.randint(2)

  def get_action(self, state):
    state_fol = [self.env2fol(state)]
    val = self.lnn.forward(state_fol).mean(dim=1)[0]
    if val > 0.5:
      return 1
    else:
      return 0
    # left_q = self.left_lnn.forward(state_fol).mean(dim=1)
    # right_q = self.right_lnn.forward(state_fol).mean(dim=1)
    # return torch.argmax(torch.cat((left_q, right_q), dim = 0))



