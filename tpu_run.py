import functools
import os
import subprocess
import time
import argparse

import glob
import requests
from fabric import Connection
from dataclasses import dataclass
import multiprocessing.pool
import regex as re
import pandas as pd
import random
multiprocessing.set_start_method("spawn")

from func_timeout import func_set_timeout
import time
import pdb
import json

@functools.lru_cache()
def get_bearer():
  return subprocess.check_output("gcloud auth print-access-token", shell=True).decode("utf-8").strip()


@functools.lru_cache()
def get_project():
  return subprocess.check_output("gcloud config list --format 'value(core.project)'", shell=True).decode(
    "utf-8").strip()

@dataclass
class TPUCreator:
  """
  Utility for creating TPUs and stuff
  """
  name: str
  tpu_size: int
  zone: str = 'us-east1-d'
  preemptible: bool = False
  network: str='rowan'
  subnetwork: str='rowan'
  version: str='v2-alpha'
  accelerator_type: str='v3'

  @property
  def base_url(self):
    # https://cloud.google.com/tpu/docs/reference/rest/v2alpha1/projects.locations.nodes/create
    return f'https://tpu.googleapis.com/v2alpha1/projects/{get_project()}/locations/{self.zone}/nodes'

  def check_tpu(self):
    response = requests.get(f'{self.base_url}/{self.name}',
                headers={'Authorization': f'Bearer {get_bearer()}'})
    return response.json()

  def create_tpu(self):
    """
    Tries to create a TPU,
    :return: returns True if successful and False otherwise
    """
    if not os.path.expanduser('~/.ssh/google_compute_engine'):
      raise ValueError("Must create SSH keys in legacy mode with something like"
               "ssh-keygen -m PEM -t rsa -b 4096 -C \"$(whoami)@$(hostname)\" -f ~/.ssh/google_compute_engine")

    try:
      status = self.check_tpu()

      if status["state"] not in ["CREATING", "READY"]:
        print("deleting TPU")
        self.delete_tpu()

        while True:
          try:
            print("deleting check: {}".format(self.check_tpu()["state"]), flush=True)
            time.sleep(1)
          except:
            break
    except:
      pass

    data = {
      "accelerator_type": f'{self.accelerator_type}-{self.tpu_size}',
      "runtime_version": f'{self.version}',
      "network_config": {"enable_external_ips": True, "network": self.network, "subnetwork": self.subnetwork},
      "tags": "unified_io",
    }

    if self.preemptible:
      data["schedulingConfig"] = {"preemptible": True}

    response = requests.post(self.base_url,
                 headers={'Authorization': f'Bearer {get_bearer()}',
                      'Content-Type': 'application/json', },
                 params=(('node_id', self.name),), json=data)
    print(response.json())
    return response.status_code == 200

  def delete_tpu(self):
    response = requests.delete(f'{self.base_url}/{self.name}', headers={'Authorization': f'Bearer {get_bearer()}'})
    return response.json()

  def wait_until_tpu_ready(self):
    desired_state = {'state': 'READY', 'health': 'HEALTHY'}
    # desired_state = {'state': 'READY'}
    while True:
      ret = self.check_tpu()

      print(f"wait_until_tpu_ready check: {ret}", flush=True)

      if ("error" in ret) or (ret["state"] == "TERMINATED"):
        return False

      matches = True
      for k, expected_v in desired_state.items():
        if k not in ret:
          matches = False
          continue
        if ret[k] != expected_v:
          matches = False

      if matches:
        return True
      time.sleep(30)


  def get_connections(self):
    host = self.name
    zone = self.zone
    key_path = os.path.expanduser('~/.ssh/google_compute_engine')

    out = subprocess.getoutput(f"gcloud alpha compute tpus tpu-vm describe --zone {zone} {host} --format json")
    out = json.loads(out)
    ips = [x["accessConfig"]["externalIp"] for x in out["networkEndpoints"]]
    print(f"Identified {ips} ips for host {host}")

    # This will (sometimes?) take care of some know-host issue that would otherwise prevent us
    # from ssh-ing in normally
    # Might be some ssh things we could do to fix this in a better way
    print(f"Testing connection with gcloud ssh....")
    exit_code = os.system('gcloud alpha compute tpus tpu-vm ssh {} --zone {} --command="echo gcloud connected"'.format(host, zone))
    if exit_code != 0:
      raise ValueError(f"gcloud connection failed, host {host} might be not be reachable")

    conns = [Connection(h, connect_kwargs={"key_filename": key_path}) for h in ips]
    return conns

def install_dependencies(conn):
  """
  Upload all the code
  :param conn:
  :param address:
  :return:
  """
  try:
      conn.run('pkill -9 train.py')
  except Exception as e:
      print(e)
  # try:
      # conn.run('pkill -9 train_eval.py')
  # except Exception as e:
      # print(e)
  try:
      conn.run('killall -9 screen')
  except Exception as e:
      print(e)

  print(f"Starting on {conn}", flush=True)
  conn.run('rm -rf *.py')
  conn.run('rm -rf *.json')
  conn.run('rm -rf screenlog.0')
  conn.run('rm -rf *.sh')

  # copy credential for some error
  conn.run(f"mkdir /home/jiasenl/.config/gcloud -p")
  conn.put('/home/jiasenl/.config/gcloud/application_default_credentials.json', f'/home/jiasenl/.config/gcloud')
  
  # conn.sudo('rm -rf *')
  local_code_path = os.path.expanduser('~/LL3M/')
  # Copy files
  for i in glob.glob(os.path.join(local_code_path, '*.py')):
    conn.put(i, f'')
  
  for i in glob.glob(os.path.join(local_code_path, '*.md')):
    conn.put(i, f'')

  for ok_folder in ['module', 'data', 'evaluations']:
    conn.sudo(f'rm -rf {ok_folder}')
    conn.run(f"mkdir {ok_folder} -p")
    for i in glob.glob(os.path.join(local_code_path,  ok_folder, '*.*')):
      conn.put(i, f'{ok_folder}/')

  for ok_folder in ['models']:
    conn.sudo(f'rm -rf {ok_folder}')
    all_paths = glob.glob(os.path.join(local_code_path, ok_folder, '**', '*.*'), recursive=True)
    # get unique path for folder creation.
    folder_paths = set(['/'.join(p.split('/')[:-1]) for p in all_paths])
    for p in folder_paths: 
      new_path = p.replace(local_code_path, '')
      conn.run(f"mkdir {new_path} -p")
    
    for p in all_paths:
      new_path = '/'.join(p.replace(local_code_path, '').split('/')[:-1])      
      conn.put(p, new_path)

  conn.put(os.path.join(local_code_path, 'tpu_startup_script.sh'), "tpu_startup_script.sh")
  conn.sudo('chmod +x tpu_startup_script.sh', hide=True)
  conn.run('~/tpu_startup_script.sh', hide=True)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--tpu', type=str, 
                    help='tpu to launch')
  parser.add_argument('--script', type=str, 
                    help='script to run')
  
  args = parser.parse_args()

  if not (args.tpu and args.script):
    parser.error('No action requested, add --tpu or --script')
  
  tpu_creator = TPUCreator(name=args.tpu, tpu_size=32)
  conns = tpu_creator.get_connections()
  
  with multiprocessing.pool.ThreadPool(processes=len(conns)) as p:
    p.map(install_dependencies, conns)
  time.sleep(30)

  def _run_pretrain(conn):
    with conn.cd(''):
      local_code_path = os.path.expanduser('~/LL3M/')
      conn.put(os.path.join(local_code_path, args.script), "run.sh")
      conn.sudo('chmod +x run.sh', hide=True)
      conn.run(f'screen -d -m -L bash -c ./run.sh', pty=False)
      print('done')
  
  with multiprocessing.pool.ThreadPool(processes=len(conns)) as p:
    p.map(_run_pretrain, conns)
    
    
    