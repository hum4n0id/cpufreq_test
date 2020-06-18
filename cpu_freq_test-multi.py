# !/usr/bin/env python3

from os import path
from collections import deque
import itertools
import multiprocessing
import threading
import psutil
import random
import signal
import sys
import math
import time
import datetime
# from pudb import set_trace; set_trace()


class cpuFreqExec(Exception):
    def __init__(self, message):
        self.message = message
        # logging error
        print (message)


class cpuFreqTest:
    def __init__(self):
        self._core_list = []
        self._thread_siblings = []
        self.freq_results = []
        self.pid_list = []
        # time to stay at frequency under load (s)
        # keep above 10s to assist lower freq sampling
        # should be gt observe_interval
        self.scale_duration = 4
        # frequency sampling interval (s)
        # should be lt scale_duration
        self.observe_interval = 1

        # attributes common to all cores
        self.path_root = '/sys/devices/system/cpu'
        path_scaling_driver = path.join(
            'cpu0', 'cpufreq', 'scaling_driver')
        path_scaling_gvrnrs = path.join(
            'cpu0', 'cpufreq', 'scaling_available_governors')
        path_scaling_freqs = path.join(
            'cpu0', 'cpufreq', 'scaling_available_frequencies')

        self.scaling_driver = self._read_cpu(
            path_scaling_driver).rstrip('\n')
        self.scaling_gvrnrs = self._read_cpu(
            path_scaling_gvrnrs).rstrip('\n').split()
        # flip order of freqs, ascending
        self.scaling_freqs_str = self._read_cpu(
            path_scaling_freqs).rstrip('\n').split()
        # cast freqs to int
        self.scaling_freqs = list(
            map(
                int, self.scaling_freqs_str))

    def _read_cpu(self, fname):
        abs_path = path.join(
            self.path_root, fname)
        print('read: ', abs_path)
        # open abs_path in binary mode, read
        try:
            with open(abs_path, 'rb') as f:
                data = f.read().decode('utf-8')
        except Exception:
            raise cpuFreqExec(
                'ERROR: unable to read file: ', abs_path)
        else:
            return data

    def _write_cpu(self, fname, data):
        abs_path = path.join(
            self.path_root, fname)
        print('write: ', abs_path)
        # open abs_path in binary mode, write
        try:
            with open(abs_path, 'wb') as f:
                f.write(data)
        except Exception:
            raise cpuFreqExec(
                'ERROR: unable to write file: ', abs_path)
        else:
            return data

    def _list_core_rng(self, core_rng):
        core_list = []
        if not core_rng:
            return core_list
        # may be unnecessary
        for core in core_rng.split(','):
            first_last = core.split('-')
            if len(first_last) == 2:
                core_list += list(
                    range(
                        int(first_last[0]),
                        int(first_last[1]) + 1))
            else:
                core_list += [int(first_last[0])]
        print ('core_list: ', core_list)
        self._core_list = core_list
        return core_list

    def _get_cores(self, fname):
        abs_path = path.join(
            self.path_root, fname)
        core_rng = self._read_cpu(
            abs_path).strip('\n').strip()
        core_list = self._list_core_rng(core_rng)
        return core_list

    def _process_data(self):
        pass

    # common
    def enable_all_cpu(self):
        to_enable = (
            set(self._get_cores('present'))
            & set(self._get_cores('offline')))
        print ('enabling the following cpus: ', to_enable)
        for cpu in to_enable:
            abs_path = path.join(
                ('cpu' + str(cpu)), 'online')
            print('en_cpu: ', abs_path)
            self._write_cpu(abs_path, b'1')

    # def _get_thread_siblings(self):
    #     thread_siblings = []
    #     online_cpus = self._get_cores('online')
    #     for cpu in online_cpus:
    #         abs_path = path.join(
    #             self.path_root, ('cpu' + str(cpu)),
    #             'topology', 'thread_siblings_list')
    #         thread_siblings.append(self._get_cores(abs_path)[1:])
    #     return thread_siblings

    # common
    def disable_thread_siblings(self):
        """ Disable all threads attached to the same core.
        """
        # prefer lists for iterations
        thread_siblings = []
        online_cpus = self._get_cores('online')
        for cpu in online_cpus:
            abs_path = path.join(
                self.path_root, ('cpu' + str(cpu)),
                'topology', 'thread_siblings_list')
            # second thread is sibling
            thread_siblings += self._get_cores(abs_path)[1:]
        self._thread_siblings = thread_siblings
        # prefer set for binary &
        to_disable = set(thread_siblings) & set(online_cpus)

        for sib in to_disable:
            abs_path = path.join(
                self.path_root, ('cpu' + str(sib)),
                'online')
            self._write_cpu(abs_path, b'0')

    # make callable by argparse '-r' arg
    def reset(self):
        """ Enable all offline cpus,
        and reset max and min frequencies files.
        """
        self.enable_all_cpu()
        self.set_governors('ondemand')
        present_cores = self._get_cores('present')

        for cpu in present_cores:
            # reset max freq
            abs_path = path.join(
                self.path_root, ('cpu' + str(cpu)),
                'cpufreq', 'scaling_max_freq')
            max_freq = str(
                max(
                    self.scaling_freqs)).encode()
            print ('max freq: ', max_freq)
            self._write_cpu(abs_path, max_freq)

            # reset min freq
            abs_path = path.join(
                self.path_root, ('cpu' + str(cpu)),
                'cpufreq', 'scaling_min_freq')
            min_freq = str(
                min(
                    self.scaling_freqs)).encode()
            print ('min freq: ', min_freq)
            self._write_cpu(abs_path, min_freq)

    # common
    def set_governors(self, governor):
        print ('setting governor: ', governor)
        online_cores = self._get_cores('online')
        for cpu in online_cores:
            abs_path = path.join(
                ('cpu' + str(cpu)),
                'cpufreq', 'scaling_governor')
            self._write_cpu(abs_path, governor.encode())

    def run_test(self):
        # disable hyperthread cores
        self.disable_thread_siblings()

        # userspace governor required for scaling_setspeed
        if self.scaling_driver == 'acpi-cpufreq':
            self.set_governors('userspace')
        else:
            self.set_governors('performance')

        # spawn core tests concurrently
        self.spawn_core_test()
        self._process_data()

        # reset state and cleanup
        print('reseting cpus')
        self.reset()

        # observe + tally results
        print('freq_results')
        print(self.freq_results)

    # log PIDs
    # cleanup w/ join
    def spawn_core_test(self):
        proc_list = list()
        result_queue = multiprocessing.Queue()
        n_cpus = len(self._get_cores('online'))

        for cpu in range(n_cpus):
            print ('cpu: ', cpu)
            affinity = [cpu]
            aff_dict = dict(
                affinity=affinity)
            proc = multiprocessing.Process(
                target=self.run_child,
                args=(result_queue,),
                kwargs=aff_dict)
            proc.start()
            proc_list.append(proc)

        for cpu in range(n_cpus):
            child_queue = result_queue.get()
            self.freq_results.append(child_queue)

        for proc in proc_list:
            proc.join()
            print ('process collapsed, joined parent')

    # def execute_core_test(self, core):
    #     cpu_freq_ctest = cpuFreqCoreTest(core)
    #     return cpu_freq_ctest.scale_all_freq()

# fix formatting, print, comments
    # def run_child(self, affinity):
    def run_child(self, output, affinity):
        # get self pid
        proc = psutil.Process()

        print ('PID: ', proc.pid)
        self.pid_list.append(proc.pid)

        proc.cpu_affinity(affinity)
        core_aff = proc.cpu_affinity()
        print ('core affinity : ', core_aff)
        core = int(affinity[0])
        print('* testing core: ', core)

        cpu_freq_ctest = cpuFreqCoreTest(core)
        cpu_freq_ctest.scale_all_freq()
        # thread safe
        observed_freqs = cpu_freq_ctest.__call__()
        # concurrent/simultaneous getter access deadlocking,
        # unmangled attribute...?
        # observed_freqs =e cpu_freq_ctest.observed_freq
        freq_result = {'cpu': core, 'observd_freqs': observed_freqs}
        output.put(freq_result)
        # freq_ntpl = namedtuple('cpu', [self.scaling_freqs_str])
        # core = freq_ntpl._make(result)


class cpuFreqCoreTest(cpuFreqTest):
    """ Subclass to facilitate concurrent frequency scaling.
    Every physical core will self-test freq. scaling capabilities at once.
    """
    def __init__(self, cpu_num):
        super().__init__()
        # mangle instance attributes
        self.__instance_core = int(cpu_num)
        self.__instance_cpu = 'cpu' + str(cpu_num)
        self.__instance_cpu_dict = {'cpu': cpu_num}
        self.__stop_loop = 0
        self.__observed_freqs = []
        self.__observed_freqs_dict = {}

    def __call__(self):
        return self.__observed_freqs_dict

    @property
    def observed_freqs(self):
        return self.__observed_freqs

    @observed_freqs.setter
    def observed_freqs(self, freq):
        freq = str(freq)
        # freq = '###' + str(freq) + '###'
        self.__observed_freqs_dict.update({freq: self.__observed_freqs})
        self.__observed_freqs = []
        # self.__observed_freqs.append(freq)

    class ObserveFreq:
        """ Class for instantiating observation thread;
        note: interval scaling will occur as freq is ramped up.
        """
        def __init__(self, interval, callback, **kwargs):
            self.interval = interval
            self.callback = callback
            self.kwargs = kwargs
            # cleanup thread
            self.daemon = True

        def observe(self):
            self.callback(**self.kwargs)
            thread_timer = threading.Timer(
                self.interval,
                self.observe)
            thread_timer.daemon = self.daemon
            thread_timer.start()

    def observe_freq_cb(self):
        self.__observed_freqs.append(self.get_cur_freq())
        print (self.__observed_freqs)

    def get_cur_freq(self):
        """ Get current frequency.
        """
        online_cores = super()._get_cores('online')
        if self.__instance_core in online_cores:
            abs_path = path.join(
                self.path_root, self.__instance_cpu,
                'cpufreq', 'scaling_cur_freq')
            freqs = super()._read_cpu(
                abs_path).rstrip('\n').split()[0]
            return freqs

    # def sample_freq_cb(self):
    #     next_call = time.time()
    #     while True:
    #         self.__observed_freqs.append(self.get_cur_freq())
    #         print (self.__observed_freqs)
    #         next_call = next_call + self.sample_time
    #         time.sleep(next_call - time.time())

    def scale_all_freq(self):
        """ Primary class method to get running core freqs,
        nested fns for encapsulation.
        """
        # def moving_average(iterable, n=3):
        #     # moving_average([40, 30, 50, 46, 39, 44]) --> 4
        # 0.0 42.0 45.0 43.0
        #     it = iter(iterable)
        #     d = deque(itertools.islice(it, n - 1))
        #     d.appendleft(0)
        #     s = sum(d)
        #     for elem in it:
        #         s += elem - d.popleft()
        #         d.append(elem)
        #         yield s / n

        def execute_workload(x):
            """ Perform maths to load cpu/core.
            """
            print (' generating cpu load')
            print (' x=', x)
            while not self.__stop_loop:
                math.factorial(x)

        def handle_alarm(*args):
            """ Alarm trigger callback.
            """
            # record observed frequency under load
            # self.__last_freq = self.get_cur_freq()
            # unload core
            self.__stop_loop = 1

# revisit / tune, try using observed_freq
        def verify_freq(freq):
            print ('##########################')
            print ('CPU: ', self.__instance_cpu)
            print('set freq: ', freq.decode())
            observed_freq = self.get_cur_freq()
            # self.__observed_freqs.append(int(self.__last_freq))
            print ('observed_freq: ', observed_freq)
            print ('self.__observed_freq: ', self.__observed_freqs)
            print ('##########################')

        def scale_to_freq(freq, idx):
            """ Proxy method to scale core to freq.
            """
            # setup async alarm to kill load gen
            signal.signal(signal.SIGALRM, handle_alarm)
            # time to gen load
            signal.alarm(self.scale_duration)
            # instantiate ObserveFreq for data sampling
            observe_freq = self.ObserveFreq(
                interval=self.observe_interval,
                callback=self.observe_freq_cb)
            # init data sampling
            observe_freq.observe()
            # set remaining frequency markers
            if idx > 0:
                self.observed_freqs = freq.decode()
            # pass random int for load generation
            execute_workload(
                random.randint(34512, 67845))
            # timerThread.start()
            verify_freq(freq)
            # reset signal alarm trigger bit
            self.__stop_loop = 0

        # scaling_freqs_set = set([reversed(self.scaling_freqs)])

        # setup paths
        print('* scaling: ', self.__instance_cpu)
        abs_path_setspd = path.join(
            self.path_root, self.__instance_cpu,
            'cpufreq', 'scaling_setspeed')
        abs_path_maxspd = path.join(
            self.path_root, self.__instance_cpu,
            'cpufreq', 'scaling_max_freq')
        abs_path_minspd = path.join(
            self.path_root, self.__instance_cpu,
            'cpufreq', 'scaling_min_freq')
        print (' freq scaling table: ', self.scaling_freqs)

# add conditional to test order of scaling_freqs, reverse if necessary
        for idx, freq in enumerate(reversed(self.scaling_freqs)):
            # set initial frequency marker
            if idx == 0:
                self.observed_freqs = freq
            print (' scaling to: ', freq)
            # set marker for sorting/processing
            freq = str(freq).encode()
            # userspace governor required to write to ./scaling_setspeed
            if self.scaling_driver == 'acpi-cpufreq':
                try:
                    super()._write_cpu(abs_path_setspd, freq)
                except Exception:
                    raise cpuFreqExec(
                        'ERROR: setting invalid frequency,\
                        scaling_setspeed!')
                else:
                    scale_to_freq(freq, idx)
            else:
                # per cpufreq, set max_freq before min_freq
                try:
                    super()._write_cpu(abs_path_maxspd, freq)
                except Exception:
                    raise cpuFreqExec(
                        'ERROR: setting invalid frequency,\
                        scaling_max_freq!')
                else:
                    try:
                        super()._write_cpu(abs_path_minspd, freq)
                    except Exception:
                        raise cpuFreqExec(
                            'ERROR: setting invalid frequency,\
                            scaling_min_freq!')
                    else:
                        scale_to_freq(freq, idx)


def main():
    cpu_freq_test = cpuFreqTest()
    cpu_freq_test.run_test()
    n_cpus = psutil.cpu_count()
    print ('cpu_count: ', n_cpus)
    print ('PID list: ', cpu_freq_test.pid_list)
    return 0


if __name__ == '__main__':
    sys.exit(main())

# -*- coding: utf-8 -*-
