import json
import numpy as np
from utils.misc import display_table

basic_config = {
  "resources": {0: 1},
  "acuities": 1, 
  "prob_acuities": [1], 
  "weighted_wait": {0: 1},
  "order": [0],
  "spawn": 5, 
  "treatment_times": {0: {0: 1}}, 
  "max_time": 60, 
  "set_seed": 0
}

class Config:
  def __init__(self, filepath, patient_distribution="poisson", patient_params=5, patient_priority_func=lambda p: p["waiting_time"], warm_up_time=0, max_time=60, seed=0):
    with open(filepath, mode='r') as f:
      self.data = json.load(f)
    self.patient_distribution = patient_distribution
    self.patient_params = patient_params
    self.patient_priority_func = patient_priority_func
    self.warm_up_time = warm_up_time
    self.max_time = max_time
    self.seed = seed

  def gymify(self):
    resource_to_id_mapping = {
      resource['label']: i for i, resource in enumerate(self.data['resources'])
    }
    for treatment in self.data['treatments']:
      treatment['resource_id'] = resource_to_id_mapping[treatment['resource']]
    treatment_to_id_mapping = {
      treatment['label']: i for i, treatment in enumerate(self.data['treatments'])
    }
    for pattern in self.data['patterns']:
      pattern['order_ids'] = [treatment_to_id_mapping[treatment] for treatment in pattern['order']]
    return {
      "acuities": self.data['acuities'],
      "resources": self.data['resources'], 
      "treatments": self.data['treatments'],
      "patterns": self.data['patterns'], 
      "patient_distribution_func": self.patient_distribution,
      "patient_params": self.patient_params,
      "patient_priority_func": self.patient_priority_func,
      "warm_up_time": self.warm_up_time,
      "max_time": self.max_time,
      "seed": self.seed,
    }

  def display(self):
      self.display_acuities()
      self.display_resources()
      self.display_treatments()
      self.display_patterns()
      
  def display_acuities(self):
      print("\n=== acuities information table ===\n")
      headers = ["acuity_level", "weighted_waiting_time", "weighted_acuity", "weighted_freq"]
      data = []
      for dict_info in self.data['acuities']:
          row = [
              dict_info['label'],
              dict_info['weighted_waiting_time'],
              dict_info['weighted_acuity'],
              dict_info['weighted_occurrence'],
          ]
          data.append(row)
      display_table(data, headers)

  def display_resources(self):
      print("\n=== resouces information table ===\n")
      headers = ["resource", "quantity"]
      data = []
      for dict_info in self.data['resources']:
          row = [
              dict_info['label'],
              dict_info['quantity'],
          ]
          data.append(row)
      display_table(data, headers)

  def display_treatments(self):
      print("\n=== treatments information table ===\n")
      headers = ["treatment", "description", "distribution", "resource"]
      data = []
      for dict_info in self.data['treatments']:
          row = [
              dict_info["label"],
              dict_info['description'],
              f"{dict_info['distribution']}({','.join(map(str,dict_info['params']))})",
              dict_info['resource'],
          ]
          data.append(row)
      display_table(data, headers)

  def display_patterns(self):
      print("\n=== patterns information table ===\n")
      headers = ["pattern", "order", "weighted_freq"]
      data = []
      for dict_info in self.data['patterns']:
          row = [
              dict_info['label'],
              " > ".join(dict_info["order"]),
              dict_info["weight"]
          ]
          data.append(row)
      display_table(data, headers)
