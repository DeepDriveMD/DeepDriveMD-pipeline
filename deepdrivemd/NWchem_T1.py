#!/usr/bin/env python3

# - initial ML force field exists
# - while iteration < X (configurable):
#   - start DTF ( Ab-initio MD simulation ) (with all reasources) CPU only
#   - start force field training task (FFTrain) (with all resources) CPU only
#   - if DFT partially satisfy the uncertainty
#     - Kill Half of the Ab-initio Tasks
#     - Start DDMD with %50 CPU and %100 GPU
#   - If DFT fully satisfy:
#       - run 2nd DDMD loop (divide available resources between bot loop)
#   - If DDMD1 finish run DDMD 2 with full resoureces

# lower / upper bound on active num of simulations
# ddmd.get_last_n_sims ...

# ------------------------------------------------------------------------------
#


import os
import math, sys, argparse
import json
import time
import random
import signal
import threading as mt

from collections import defaultdict

import radical.pilot as rp
import radical.utils as ru


# ------------------------------------------------------------------------------
#
class DDMD(object):

    # define task types (used as prefix on task-uid)
    TASK_TRAIN_MODEL = 'task_train_model' # DDMD
    TASK_TRAIN_FF    = 'task_train_ff'    # AB-initio
    TASK_MD_DDMD     = 'task_md_ddmd'     # DDMD
    TASK_MD_AI       = 'task_md_ai'       # AB-initio MD
    TASK_DFT         = 'task_dft'         # Ab-inito
    TASK_SELECT      = 'task_select'      # DDMD
    TASK_AGENT       = 'task_agent'       # DDMD

    TASK_TYPES       = [TASK_TRAIN_MODEL,
                        TASK_TRAIN_FF,
                        TASK_MD_DDMD,
                        TASK_MD_AI,
                        TASK_DFT,
                        TASK_SELECT,
                        TASK_AGENT]

    # these alues fall from heaven....
    ITER_AB_INITIO = 6
    ITER_DDMD   = 6
    ITER_DDMD_1 = int(math.floor(ITER_AB_INITIO / 2))
    ITER_DDMD_2 = ITER_AB_INITIO

    # keep track of core usage
    cores_used     = 0
    gpus_used      = 0
    avail_cores    = 0
    avail_gpus     = 0

    # keep track the stage
    stage = 0 # 0 no tasks started
              # 1 only ab-initio
              # 2 ab-initio + DDM1
              # 3 DDMD1 + DDMD2
              # 4 only DDMD2
              # 5 all done

    # --------------------------------------------------------------------------
    #
    def __init__(self):

        # control flow table
        self._protocol = {self.TASK_TRAIN_MODEL: self._control_train_model,
                          self.TASK_TRAIN_FF   : self._control_train_ff   ,
                          self.TASK_MD_DDMD    : self._control_md_ddmd    ,
                          self.TASK_MD_AI      : self._control_md_ai    ,
                          self.TASK_DFT        : self._control_dft        ,
                          self.TASK_SELECT     : self._control_select     ,
                          self.TASK_AGENT      : self._control_agent      }

        self._glyphs   = {self.TASK_TRAIN_MODEL: 'T',
                          self.TASK_TRAIN_FF   : 't',
                          self.TASK_MD_DDMD    : 'S',
                          self.TASK_MD_DDMD    : 's',
                          self.TASK_DFT        : 'd',
                          self.TASK_SELECT     : 'L',
                          self.TASK_AGENT      : 'A',}

        # bookkeeping
        self._iter           =  0
        self._iterDDMD1      =  0
        self._iterDDMD2      =  0
        self._threshold      =  1
        self._cores          = 16  # available cpu resources
        self._gpus           =  4  # available gpu resources
        self._avail_cores    = self._cores
        self._avail_gpus     = self._gpus
        self._cores_used     =  0
        self._gpus_used      =  0
        self._ddmd_tasks     =  0

        self._lock   = mt.RLock()
        self._series = [1, 2]
        self._uids   = {s:list() for s in self._series}

        self._tasks  = {s: {ttype: dict() for ttype in self.TASK_TYPES}
                            for s in self._series}

        self._final_tasks = list()

        # silence RP reporter, use own
        os.environ['RADICAL_REPORT'] = 'false'
        self._rep = ru.Reporter('ddmd')
        self._rep.title('DDMD')

        # RP setup
        self._session = rp.Session()
        self._pmgr    = rp.PilotManager(session=self._session)
        self._tmgr    = rp.TaskManager(session=self._session)

        pdesc = rp.PilotDescription({'resource': 'local.localhost',
                                     'runtime' : 30,
#                                     'runtime' : 4,
                                     'cores'   : self._cores})
#                                     'cores'   : 1})
        self._pilot = self._pmgr.submit_pilots(pdesc)

        self._tmgr.add_pilots(self._pilot)
        self._tmgr.register_callback(self._state_cb)
        # Parser
        self.env_work_dir = os.getenv("MINI_APP_DeepDriveMD_DIR")
        if self.env_work_dir is None:
            print("Warning: Did not set up work_dir using env var, need to set it up in parser manually!")
        self.set_argparse()
        self.get_json()

    # --------------------------------------------------------------------------
    #

    def set_resource(self, res_desc):
        self.resource_desc = res_desc

    # --------------------------------------------------------------------------
    #
    def set_argparse(self):
        parser = argparse.ArgumentParser(description="DeepDriveMD_miniapp_EnTK_serial")

        parser.add_argument('--num_phases', type=int, default=3,
                        help='number of phases in the workflow')
        parser.add_argument('--mat_size', type=int, default=5000,
                        help='the matrix with have size of mat_size * mat_size')
        parser.add_argument('--data_root_dir', default='./',
                        help='the root dir of gsas output data')
        parser.add_argument('--num_step', type=int, default=1000,
                        help='number of step in MD simulation')
        parser.add_argument('--num_epochs_train', type=int, default=150,
                        help='number of epochs in training task')
        parser.add_argument('--model_dir', default='./',
                        help='the directory where save and load model')
        parser.add_argument('--conda_env', default=None,
                        help='the conda env where numpy/cupy installed, if not specified, no env will be loaded')
        parser.add_argument('--num_sample', type=int, default=500,
                        help='num of samples in matrix mult (training and agent)')
        parser.add_argument('--num_mult_train', type=int, default=4000,
                        help='number of matrix mult to perform in training task')
        parser.add_argument('--dense_dim_in', type=int, default=12544,
                        help='dim for most heavy dense layer, input')
        parser.add_argument('--dense_dim_out', type=int, default=128,
                        help='dim for most heavy dense layer, output')
        parser.add_argument('--preprocess_time_train', type=float, default=20.0,
                        help='time for doing preprocess in training')
        parser.add_argument('--preprocess_time_agent', type=float, default=10.0,
                        help='time for doing preprocess in agent')
        parser.add_argument('--num_epochs_agent', type=int, default=10,
                        help='number of epochs in agent task')
        parser.add_argument('--num_mult_agent', type=int, default=4000,
                        help='number of matrix mult to perform in agent task, inference')
        parser.add_argument('--num_mult_outlier', type=int, default=10,
                        help='number of matrix mult to perform in agent task, outlier')
        parser.add_argument('--enable_darshan', action='store_true',
                        help='enable darshan analyze')
        parser.add_argument('--project_id', required=True,
                        help='the project ID we used to launch the job')
        parser.add_argument('--queue', required=True,
                        help='the queue we used to submit the job')
        parser.add_argument('--work_dir', default=self.env_work_dir,
                        help='working dir, which is the dir of this repo')
        parser.add_argument('--num_sim', type=int, default=12,
                        help='number of tasks used for simulation')
        parser.add_argument('--num_nodes', type=int, default=3,
                        help='number of nodes used for simulation')
        parser.add_argument('--io_json_file', default="io_size.json",
                        help='the filename of json file for io size')

        args = parser.parse_args()
        self.args = args

    def get_json(self):
        json_file = "{}/launch-scripts/{}".format(self.args.work_dir, self.args.io_json_file)
        with open(json_file) as f:
            self.io_dict = json.load(f)

    def get_arguments(self, ttype, argument_val=""):

        #FIXME  OK: find correct Phase and i= task id
        phase_idx = 0
        i = 0

        args = []

        if ttype == self.TASK_MD_DDMD:
            args = ['{}/sim/lammps/main_ase_lammps.py'.format(self.args.work_dir),
                           '{}'.format(argument_val.split("|")[0]), # get pbd file path here #FIXME
                           '{}'.format(argument_val.split("|")[1])] # get test dir  path here #FIXME

        elif ttype == self.TASK_DFT1:
            args = ['{}/sim/nwchem/main1_nwchem.py'.format(self.args.work_dir)]
        elif ttype == self.TASK_DFT2:
            args = ['{}/sim/nwchem/main2_nwchem.py'.format(self.args.work_dir),
                           '{}'.format(argument_val)] # this will need to get the instance
        elif ttype == self.TASK_DFT3:
            args = ['{}/sim/nwchem/main3_nwchem.py'.format(self.args.work_dir)]

        elif ttype == self.TASK_TRAIN_FF:
            args = ['{}/model/deepm/main_deepmd.py'.format(self.args.work_dir),
                       '{}'.format(argument_val)] #training folder name

        elif ttype == self.TASK_TRAIN_MODEL:
            args = ['{}/Executables/training.py'.format(self.args.work_dir),
                       '--num_epochs={}'.format(self.args.num_epochs_train),
                       '--device=gpu',
                       '--phase={}'.format(phase_idx),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--model_dir={}'.format(self.args.model_dir),
                       '--num_sample={}'.format(self.args.num_sample * (1 if phase_idx == 0 else 2)),
                       '--num_mult={}'.format(self.args.num_mult_train),
                       '--dense_dim_in={}'.format(self.args.dense_dim_in),
                       '--dense_dim_out={}'.format(self.args.dense_dim_out),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--preprocess_time={}'.format(self.args.preprocess_time_train),
                       '--write_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["train"]["write"]),
                       '--read_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["train"]["read"])]


        elif ttype == self.TASK_AGENT:
            args = ['{}/Executables/agent.py'.format(self.args.work_dir),
                       '--num_epochs={}'.format(self.args.num_epochs_agent),
                       '--device=gpu',
                       '--phase={}'.format(phase_idx),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--model_dir={}'.format(self.args.model_dir),
                       '--num_sample={}'.format(self.args.num_sample),
                       '--num_mult={}'.format(self.args.num_mult_agent),
                       '--num_mult_outlier={}'.format(self.args.num_mult_outlier),
                       '--dense_dim_in={}'.format(self.args.dense_dim_in),
                       '--dense_dim_out={}'.format(self.args.dense_dim_out),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--preprocess_time={}'.format(self.args.preprocess_time_agent),
                       '--write_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["agent"]["write"]),
                       '--read_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["agent"]["read"])]


        elif ttype == self.TASK_SELECT:
            args = ['{}/Executables/selection.py'.format(self.args.work_dir),
                       '--phase={}'.format(phase_idx),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--write_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["selection"]["write"]),
                       '--read_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["selection"]["read"])]


        return args



    # --------------------------------------------------------------------------
    #
    def __del__(self):

        self.close()


    # --------------------------------------------------------------------------
    #
    def close(self):

        if self._session is not None:
            self._session.close(download=True)
            self._session = None


    # --------------------------------------------------------------------------
    #
    def dump(self, task=None, msg=''):
        '''
        dump a representation of current task set to stdout
        '''

        # this assumes one core per task

        self._rep.plain('<<|')

        idle = self._cores

        for ttype in self.TASK_TYPES:

            n = 0
            for series in self._series:
                n   += len(self._tasks[series][ttype])
                idle -= n

            self._rep.ok('%s' % self._glyphs[ttype] * n)

        self._rep.plain('%s' % '-' * idle +
                        '| %4d [%4d]' % (self._cores_used, self._cores))

        if task and msg:
            self._rep.plain(' %-15s: %s\n' % (task.uid, msg))
        else:
            if task:
                msg = task
            self._rep.plain(' %-15s: %s\n' % (' ', msg))


    # --------------------------------------------------------------------------
    #
    def start(self):
        '''
        submit initial set of Ab-initio MD similation tasks DFT
        '''

        self.dump('submit MD simulations')

        # start ab-initio loop
        self.stage = 1
        self._ab_initio(self.TASK_DFT, series=1)

    # def control_exec(self, rules = None):
    #     if rules["stage"] < 0:
    #         self.dump("Stage cannot be negatif there is an error")
    #         self.stop()

    #     if self._avail_cores >= rules["n_cpus"] and self._avail_gpus >= rules["n_gpus"]:
    #         self._submit_task(rules)
    #     else:
    #         #FIXME OK this is hard I need to think  about this a bit more for know there will be multiple controler





    # --------------------------------------------------------------------------
    #
    def _ab_initio(self, ttype, series):


        # Ab-Inito tasks only uses CPU

        self.dump('next ab-initio iter: requested')

        # FIXME OK: currently we assume only 1 CPU per task
        # To fix this we need to ask user to define how may cpus 1 task may use.
        n_tasks = self._cores

        if ttype == self.TASK_TRAIN_FF:
            if self.stage == 2:
                n_tasks = int(math.floor((self._cores - self._ddmd_tasks)/2))
            self._submit_task(self.TASK_MD_AI, n=n_tasks, series=1)
            return

        if ttype == self.TASK_DFT:
            if self.stage == 2:
                n_tasks = int(math.floor((self._cores - self._ddmd_tasks)/2))

            for i in range(4):
                self._submit_task(self.TASK_TRAIN_FF, n=1, series=1, argvals="train{}".format(i))#FIXME argvals needs to be 4
            return


        self._iter += 1

        uids = list()
        for series in self._series:
            for ttype in self._tasks[series]:
                if ttype in [self.TASK_MD_DDMD, self.TASK_TRAIN_MODEL,
                             self.TASK_AGENT,   self.TASK_SELECT]:
                    continue
                uids.extend(self._tasks[series][ttype].keys())

        # cancel necessary tasks from ab-initio iteration
        if self._iter == self.ITER_DDMD_1:
            self.dump('ab-initio iter Partially Satisfy: Start DDMD1')


            tts  = list(self._tasks[series][self.TASK_TRAIN_FF].keys())
            dtfs = list(self._tasks[series][self.TASK_DFT].keys())

            # to_cancel  = int(math.floor(len(uids) / 2))
            to_cancel = int(math.floor((self._cores - self._ddmd_tasks)/2))

            self.dump('Number of tasks to CANCEL: %s' % to_cancel)
            self._ddmd_tasks = to_cancel

            if len(dtfs) >= to_cancel:
                random_uids = random.sample(dtfs, to_cancel)
                self._cancel_tasks(random_uids)

            else:
                self._cancel_tasks(dtfs)
                to_cancel = to_cancel - len(dtfs)

                if len(tts) >= to_cancel:
                    random_uids = random.sample(tts, to_cancel)
                    self._cancel_tasks(random_uids)

                else:
                    self._cancel_tasks(tts)

            # we use use 50% of resources for DDMD tasks now
            # (other 50% are reserved for ab-initio)
            self.stage = 2

            # self._submit_task(self.TASK_TRAIN_MODEL, n=self._ddmd_tasks, series=1)
            self.control_DDMD(self.TASK_MD_DDMD, series=1)


        elif self._iter >= self.ITER_DDMD_2:
            self.dump('Ab-initio Is done Start DDMD2')


            self._cancel_tasks(uids)

            # ab-initio completed, we use up to 100% for MD tasks
            self.stage =3

            # self._submit_task(self.TASK_TRAIN_MODEL, n=self._ddmd_tasks, series=2)
            self.control_DDMD(self.TASK_MD_DDMD, series=2)
            return


        # If I reach hear I will start next batch of DFT tasks (assume one core per task)
        # FIXME: task numbers
        n_tasks = self._cores - self._ddmd_tasks
        self._submit_task(self.TASK_DFT, n=n_tasks, series=1)

        self.dump('next ab-initio iter: started %s DFT'
                  % (self._cores - self._cores_used))

    # --------------------------------------------------------------------------
    def control_DDMD (self, ttype, series):
        # This function control how many resources available for and given DDMD task
        self.dump("Starting %s "%ttype)

        # Check which stage we are in and set Core counts accordingly
        # FIXME  OK: for now I will assume we create N task with mutliple resources
        #            Where N  =  #cpus/#gpus
        #            we can always change this later

        cpus = 0
        gpus = 0
        ntask = 0

        if self.stage <=0 or self.stage >=5:
            self.dump("Something went wrong")
            self.stop()
        elif self.stage == 1:
            # In here we only should have ab-initio code running
            self.dump("Error: Only ab-initio should have been running here")
            self.stop()
        elif self.stage == 2:
            # here we have ab-initio and DDMD 1
            self.dump('Ab-initio still runs use  50% CPU and 100% GPU')
            cpus = int (math.floor(self._cores / 2)) # or self._ddmd_tasks
            gpus = self._gpus
        elif self.stage == 3:
            # here we have ab-initio and DDMD 1
            self.dump('DDMD1 and DDMD2 runs use  50% CPU and  GPU')
            cpus = int (math.floor(self._cores / 2))
            gpus = int (math.floor(self._gpus  / 2))
        elif self.stage == 4:
            # here we have ab-initio and DDMD 1
            self.dump('DDMD2 only use 100% CPU and  GPU')
            cpus = self._cores
            gpus = self._gpus

        if ttype == self.TASK_AGENT:
            ntask = 1
            cpus = 1
            gpus = 1

        if ttype == self.TASK_SELECT:
            ntask = 1
#            cpus = self._cores

        if ttype in [self.TASK_TRAIN_MODEL, self.TASK_MD_DDMD]:
            ntask = gpus
            cpus = int (math.floor(cpus / gpus))
            gpus = 1

        if ttype == self.TASK_MD_DDMD:
            if self._iterDDMD1 >= ITER_DDMD and self._iterDDMD2 >= ITER_DDMD:
                self.dump("We are done:")
                self.stop()

            if series == 1:
                if self._iterDDMD1 >= ITER_DDMD:
                    self.stage = 4
                    #FIXME OK We need to decide what to do at this point
                    #For now I will wait for any stage from DDMD2 to finish
                    return
                else:
                    self._iterDDMD1+=1
            elif series == 2:
                if self._iterDDMD2 >= ITER_DDMD:
                    self.dump("ERROR: For some reason there is still DDMD 1 running")
                    self.stop()
                else:
                    self._iterDDMD2+=1

        self._submit_task(ttype, n=ntask, cpu=cpus, gpu=gpus, series=1)


    # --------------------------------------------------------------------------
    #
    def stop(self):

        os.kill(os.getpid(), signal.SIGKILL)
        os.kill(os.getpid(), signal.SIGTERM)


    # --------------------------------------------------------------------------
    #
    def _get_ttype(self, uid):
        '''
        get task type from task uid
        '''

        ttype = uid.split('.')[0]

        assert ttype in self.TASK_TYPES, 'unknown task type: %s' % uid
        return ttype


    # --------------------------------------------------------------------------
    #
    def _submit_task(self, ttype, args=None, n=1, cpu=1, gpu=0, series: int=1, argvals=''):
        '''
        submit 'n' new tasks of specified type

        NOTE: all tasks are uniform for now: they use a single core and sleep
              for a random number (0..3) of seconds.
        '''

        # with self._lock:

        #     tds   = list()
        #     for _ in range(n):

        #         t_sleep = int(random.randint(0,30) / 10) + 3
        #         result  = int(random.randint(0,10) /  1)

        #         uid = ru.generate_id('%s.%03d' % (ttype, self._iter))
        #         tds.append(rp.TaskDescription({
        #                    'uid'          : uid,
        #                    'cpu_processes': cpu,
        #                    'gpus'         : gpu,
        #                    'executable'   : '/bin/sh',
        #                    'arguments'    : ['-c', 'sleep %s; echo %s %s' %
        #                                            (t_sleep, result, args)]
        #                    }))

        #     tasks = self._tmgr.submit_tasks(tds)

        # NOTE Here I will try to add all Mini-app tasks

        cur_args = self.get_arguments(ttype, argument_val=argvals)

        with self._lock:

            tds   = list()
            for _ in range(n):

                tds.append(rp.TaskDescription({
                           'uid'            : ttype,
                           'ranks'          : 1,
                           'cores_per_rank' : cpu,
                           'gpus_per_rank'  : gpu,
                           'executable'     : 'python',
                           'arguments'      : cur_args
                           }))

            tasks = self._tmgr.submit_tasks(tds)

            for task in tasks:
                self._register_task(task, series=series)


    # --------------------------------------------------------------------------
    #
    def _cancel_tasks(self, uids):
        '''
        cancel tasks with the given uids, and unregister them
        '''

        uids = ru.as_list(uids)

        # FIXME AM: does not work
        self._tmgr.cancel_tasks(uids)

        for uid in uids:

            series = self._get_series(uid=uid)
            ttype = self._get_ttype(uid)
            task  = self._tasks[series][ttype][uid]
            self.dump(task, 'cancel [%s]' % task.state)

            self._unregister_task(task)

        self.dump('cancelled')


    # --------------------------------------------------------------------------
    #
    def _register_task(self, task, series: int):
        '''
        add task to bookkeeping
        '''

        with self._lock:

            ttype = self._get_ttype(task.uid)

            self._uids[series].append(task.uid)

            self._tasks[series][ttype][task.uid] = task

            cores = task.description['cpu_processes'] \
                  * task.description['cpu_threads']
            self._cores_used += cores

            gpus = task.description['gpu_processes']
            self._gpus_used += gpus


    # --------------------------------------------------------------------------
    #
    def _unregister_task(self, task):
        '''
        remove completed task from bookkeeping
        '''

        with self._lock:

            series = self._get_series(task)
            ttype = self._get_ttype(task.uid)

            if task.uid not in self._tasks[series][ttype]:
                return

            # remove task from bookkeeping
            self._final_tasks.append(task.uid)
            del self._tasks[series][ttype][task.uid]
            self.dump(task, 'unregister %s' % task.uid)

            cores = task.description['cpu_processes'] \
                  * task.description['cpu_threads']
            self._cores_used -= cores

            gpus = task.description['gpu_processes']
            self._gpus_used -= gpus


    # --------------------------------------------------------------------------
    #
    def _state_cb(self, task, state):
        '''
        act on task state changes according to our protocol
        '''

        try:
            return self._checked_state_cb(task, state)

        except Exception as e:
            self._rep.exception('\n\n---------\nexception caught: %s\n\n' % repr(e))
            ru.print_exception_trace()
            self.stop()


    # --------------------------------------------------------------------------
    #
    def _checked_state_cb(self, task, state):

        # this cb will react on task state changes.  Specifically it will watch
        # out for task completion notification and react on them, depending on
        # the task type.

        if state in [rp.TMGR_SCHEDULING] + rp.FINAL:
            self.dump(task, ' -> %s' % task.state)

        # ignore all non-final state transitions
        if state not in rp.FINAL:
            return

        # ignore tasks which were already completed
        if task.uid in self._final_tasks:
            return

        # lock bookkeeping
        with self._lock:

            # raise alarm on failing tasks (but continue anyway)
            if state == rp.FAILED:
                self._rep.error('task %s failed: %s' % (task.uid, task.stderr))
                self.stop()

            # control flow depends on ttype
            ttype  = self._get_ttype(task.uid)
            action = self._protocol[ttype]
            if not action:
                self._rep.exit('no action found for task %s' % task.uid)
            action(task)

            # remove final task from bookkeeping
            self._unregister_task(task)


    # --------------------------------------------------------------------------
    #
    def _get_series(self, task=None, uid=None):

        if uid:
            # look up by uid
            for series in self._series:
                if uid in self._uids[series]:
                    return series

        else:
            # look up by task type
            for series in self._series:
                if task.uid in self._uids[series]:
                    return series

        raise ValueError('task does not belong to any serious')


    # --------------------------------------------------------------------------
    #
    def _control_train_model(self, task):
        '''
        react on completed MD simulation task
        '''
        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_TRAIN_MODEL]) > 1:
            return

        self.dump(task, 'completed model train')

        # FIXME OK: allways trigger control_DDMD()
        self.control_DDMD(self.TASK_SELECT, series)

    # --------------------------------------------------------------------------
    #
    def _control_md_ai(self, task):
        '''
        react on completed ff training task
        '''

        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_MD_AI]) > 1:
            return

        self.dump(task, 'completed ab-initio md ')
        self._ab_initio(self.TASK_MD_AI, series)



    # --------------------------------------------------------------------------
    #
    def _control_train_ff(self, task):
        '''
        react on completed ff training task
        '''

        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_TRAIN_FF]) > 1:
            return

        self.dump(task, 'completed ff train')
        self._ab_initio(self.TASK_DFT, series)

    # --------------------------------------------------------------------------
    #
    def _control_dft1(self, task):
        '''
        react on completed DFT task
        '''
        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_DFT1]) > 1:
            return

        # TODO READ the inputs.txt
        # submit self.TASK_DFT2 for each line

        self.dump(task, 'completed dft1')
        self._submit_task() #TODO submit self.TASK_DFT2

    # --------------------------------------------------------------------------
    #
    def _control_dft2(self, task):
        '''
        react on completed DFT task
        '''
        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_DFT2]) > 1:
            return

        self.dump(task, 'completed dft')
#        self._ab_initio(self.TASK_TRAIN_FF, series)
    #TODO : submit self.TASK_DFT3


    # --------------------------------------------------------------------------
    #
    def _control_dft3(self, task):
        '''
        react on completed DFT task
        '''
        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_DFT3]) > 1:
            return

        self.dump(task, 'completed dft')
        self._ab_initio(self.TASK_TRAIN_FF, series)


    # --------------------------------------------------------------------------
    #
    def _control_select(self, task):
        '''
        react on completed DDMD selection task
        '''
        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_SELECT]) > 1:
            return

        self.dump(task, 'completed selection')
        self.control_DDMD(self.TASK_AGENT, series)

    # --------------------------------------------------------------------------
    #
    def _control_md_ddmd(self, task):
        '''
        react on completed DDMD selection task
        '''
        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_SELECT]) > 1:
            return

        self.dump(task, 'completed MD')
        self.control_DDMD(self.TASK_TRAIN_MODEL,  series)



    # --------------------------------------------------------------------------
    #
    def _control_agent(self, task):
        '''
        react on completed DDMD agent task
        '''
        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_AGENT]) > 1:
            return

        self.dump(task, 'completed Agent')



        # FIXME OK: allways trigger control_DDMD()
        self.control_DDMD(self.TASK_MD_DDMD, series)


# ------------------------------------------------------------------------------
#
if __name__ == '__main__':
    # Apparently there is no main(?)
    pass

# ------------------------------------------------------------------------------
