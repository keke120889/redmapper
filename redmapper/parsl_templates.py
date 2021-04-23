PARSL_LOCAL_CONFIG_TEMPLATE = """
from parsl.executors import WorkQueueExecutor, ThreadPoolExecutor
from parsl.providers import LocalProvider
from parsl.addresses import address_by_hostname
from parsl.utils import get_all_checkpoints


provider = LocalProvider(init_blocks=0, min_blocks=0, max_blocks=1)
executors = [WorkQueueExecutor(label="work_queue", port=9000, shared_fs=True,
                               provider=provider, autolabel=False,
                               address=address_by_hostname()),
             ThreadPoolExecutor(max_threads=1, label="submit-node")]

config = parsl.config.Config(strategy="simple",
                             garbage_collect=False,
                             app_cache=True,
                             executors=executors,
                             retries=1)

DFK = parsl.load(config)
"""

PARSL_SLURM_CONFIG_TEMPLATE = """
from parsl.executors import WorkQueueExecutor, ThreadPoolExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher


PROVIDER_OPTIONS = dict(nodes_per_block={nodes},
                        exclusive=True,
                        init_blocks=0,
                        min_blocks=0,
                        max_blocks=1,
                        parallelism=0,
                        launcher=SrunLauncher(
                            overrides='-K0 -k --slurmd-debug=verbose'),
                        cmd_timeout=300)

SCHEDULER_OPTIONS = ("#SBATCH --constraint={constraint}\\n"
                     "#SBATCH --qos={qos}\\n")

provider = SlurmProvider('None', walltime='00:{walltime}:00',
                         scheduler_options=SCHEDULER_OPTIONS,
                         **PROVIDER_OPTIONS)

executors = [WorkQueueExecutor(label='work_queue', port=9000, shared_fs=True,
                               provider=provider, autolabel=False),
             ThreadPoolExecutor(max_threads=1, label='submit-node')]
config = parsl.config.Config(strategy='simple',
                             garbage_collect=False,
                             app_cache=True,
                             executors=executors,
                             retries=1)

DFK = parsl.load(config)
"""


PARSL_RUN_TEMPLATE = """
#!/bin/env python

import os
import parsl
{parsl_config}

@parsl.bash_app(executors=['work_queue'], cache=True,
                ignore_for_cache=['stdout', 'stderr'])
def run_command(command, inputs=(), stdout=None, stderr=None, parsl_resource_specification=None):
    return command

command = '{parsl_command}'
resource_spec = {{'memory': {memory}, 'cores': 1, 'disk': 0}}

futures = []
for pixel in {hpix_list_str}:
    comm = command.format(pixel=pixel)
    logfile = f'{jobname}-{{pixel}}.log'
    futures.append(run_command(comm, stdout=logfile, stderr=logfile,
                               parsl_resource_specification=resource_spec))

[_.result() for _ in futures]
"""
