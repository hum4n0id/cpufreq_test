# !/usr/bin/env python3

"""
todo/verify:
* implement argparse (almost complete)
* change header to Canonical official syntax
* expand exeception handling
* make reset() callable with '-r' arg
* get min/max from min/max files and write to attributes
    - currently using min/max from freq scaling table

todo optional (nice to have):
* check if workload needs to scale with processor power
"""

from os import path
import collections
import itertools
import multiprocessing
import threading
import random
import signal
import sys
import math
import pprint
import psutil
# from pudb import set_trace; set_trace()


class CpuFreqExec(Exception):
    """ Exception handling.
    """

    def __init__(self, message):
        self.message = message
        # logging error
        print(message)


class CpuFreqTest:
    """ Test cpufreq scaling capabilities.
    """

    def __init__(self):
        def append_max_min():
            scaling_freqs = []
            path_max = path.join(
                'cpu0', 'cpufreq', 'scaling_max_freq')
            path_min = path.join(
                'cpu0', 'cpufreq', 'scaling_min_freq')
            scaling_freqs.append(self._read_cpu(
                path_max).rstrip('\n'))
            scaling_freqs.append(self._read_cpu(
                path_min).rstrip('\n'))
            return scaling_freqs

        self.pid_list = []  # pids for core affinity assignment
        self.freq_result_map = {}  # final results
        # chainmap object constructor
        self.freq_chainmap = collections.ChainMap()

        # time to stay at frequency under load (sec)
        # more time = more resolution
        # should be gt observe_interval
        self.scale_duration = 15

        # frequency sampling interval (thread timer)
        # too low time = duplicate samples
        # should be lt scale_duration
        self._observe_interval = .6

        # factorial to calculate during core test, positive int
        self.workload_n = random.randint(34512, 67845)

        # max, min percentage of avg freq allowed to pass,
        # relative to target freq
        # ex: max = 110, min = 90 is 20% passing tolerance
        self._max_freq_pct = 110
        self._min_freq_pct = 90
        self._fail_count = 0  # init fail bit

        # attributes common to all cores
        self.path_root = '/sys/devices/system/cpu'
        # cpufreq driver
        path_scaling_driver = path.join(
            'cpu0', 'cpufreq', 'scaling_driver')
        self.scaling_driver = self._read_cpu(
            path_scaling_driver).rstrip('\n')

        # available governors
        path_scaling_gvrnrs = path.join(
            'cpu0', 'cpufreq', 'scaling_available_governors')
        path_startup_governor = path.join(
            'cpu0', 'cpufreq', 'scaling_governor')
        self.scaling_gvrnrs = self._read_cpu(
            path_scaling_gvrnrs).rstrip('\n').split()
        self.startup_governor = self._read_cpu(
            path_startup_governor).rstrip('\n')

        # ensure the correct freq table is populated
        if self.scaling_driver == 'acpi-cpufreq':
            path_scaling_freqs = path.join(
                'cpu0', 'cpufreq',
                'scaling_available_frequencies')
            scaling_freqs = self._read_cpu(
                path_scaling_freqs).rstrip('\n').split()
        else:
            scaling_freqs = append_max_min()
        # cast freqs to int
        self.scaling_freqs = list(
            map(
                int, scaling_freqs))

    @property
    def observe_interval(self):
        return self._observe_interval

    @observe_interval.setter
    def observe_interval(self, idx):
        """ Pad/throttle observ_freq_cb()
        prevents race condition (need to verify)?
        """
        for idx in range(2, 5):
            for idx in range(6, 9):
                for idx in range(10, 14):
                    for idx in range(15, 20):
                        while idx >= 21:
                            self._observe_interval += .3
                        self._observe_interval += .2
                    self._observe_interval += .2
                self._observe_interval += .1
            self._observe_interval += .1

    def _read_cpu(self, fname):
        """ Read sysfs/cpufreq file.
        """
        abs_path = path.join(
            self.path_root, fname)
        # debug print('read: ', abs_path)
        # open abs_path in binary mode, read
        try:
            with open(abs_path, 'rb') as f:
                data = f.read().decode('utf-8')
        except Exception:
            raise CpuFreqExec(
                'ERROR: unable to read file:', abs_path)
        else:
            return data

    def _write_cpu(self, fname, data):
        """ Write sysfs/cpufreq file.
        """
        abs_path = path.join(
            self.path_root, fname)
        # debug print('write: ', abs_path)
        # open abs_path in binary mode, write
        try:
            with open(abs_path, 'wb') as f:
                f.write(data)
        except Exception:
            raise CpuFreqExec(
                'ERROR: unable to write file:', abs_path)
        else:
            return data

    def _list_core_rng(self, core_rng):
        """ Method to convert core range to list prior
        to iteration.
        """
        core_list = []
        if not core_rng:
            return core_list
        # allow iteration over range: rng
        for core in core_rng.split(','):
            first_last = core.split('-')
            if len(first_last) == 2:
                core_list += list(
                    range(
                        int(first_last[0]),
                        int(first_last[1]) + 1))
            else:
                core_list += [int(first_last[0])]
        return core_list

    def _get_cores(self, fname):
        """ Get various core ranges, convert to list.
        """
        abs_path = path.join(
            self.path_root, fname)
        core_rng = self._read_cpu(
            abs_path).strip('\n').strip()
        core_list = self._list_core_rng(core_rng)
        return core_list

    def _process_results(self):
        """ Process results from CpuFreqCoreTest()
        """
        # transpose and append results from subclass
        def comp_freq_dict(inner_key, inner_val):
            if inner_val:
                # get % avg_freq/target_freq
                result_pct = int((inner_val / inner_key) * 100)
                # append result %
                new_inner_val = [result_pct]
                if self._min_freq_pct <= result_pct <= self._max_freq_pct:
                    # append result P/F
                    new_inner_val.append('Pass')
                else:
                    # ""
                    new_inner_val.append('Fail')
                    # increment fail bit
                    self._fail_count += 1
            # append avg freq
            new_inner_val.append(int(inner_val))
            return new_inner_val

        # create master result table with dict comp.
        self.freq_result_map = {
            outer_key: {
                inner_key: comp_freq_dict(inner_key, inner_val)
                for inner_key, inner_val in outer_val.items()}
            for outer_key, outer_val in self.freq_chainmap.items()}

        return self.freq_result_map

    def enable_all_cpu(self):
        """ Enable all present and offline cores.
        """
        to_enable = (
            set(self._get_cores('present'))
            & set(self._get_cores('offline')))
        print('enabling the following cores:', to_enable)
        for core in to_enable:
            abs_path = path.join(
                ('cpu' + str(core)), 'online')
            self._write_cpu(abs_path, b'1')

    def disable_thread_siblings(self):
        """ Disable all threads attached to the same core,
        aka hyperthreading.
        """
        thread_siblings = []
        online_cpus = self._get_cores('online')
        for core in online_cpus:
            abs_path = path.join(
                self.path_root, ('cpu' + str(core)),
                'topology', 'thread_siblings_list')
            # second core is sibling
            thread_siblings += self._get_cores(abs_path)[1:]
        self._thread_siblings = thread_siblings
        # prefer set for binary &
        to_disable = set(thread_siblings) & set(online_cpus)

        for core in to_disable:
            abs_path = path.join(
                self.path_root, ('cpu' + str(core)),
                'online')
            self._write_cpu(abs_path, b'0')

    def set_governors(self, governor):
        """ Set/change cpu governor, perform on
        all cores.
        """
        print('setting governor:', governor)
        online_cores = self._get_cores('online')
        for core in online_cores:
            abs_path = path.join(
                ('cpu' + str(core)),
                'cpufreq', 'scaling_governor')
            self._write_cpu(abs_path, governor.encode())

    def reset(self):
        """ Enable all offline cpus,
        and reset max and min frequencies files.
        """
        self.enable_all_cpu()
        if self.scaling_driver == 'acpi-cpufreq':
            self.set_governors('ondemand')
        else:
            self.set_governors(self.startup_governor)

        present_cores = self._get_cores('present')

        for core in present_cores:
            # reset max freq
            abs_path = path.join(
                self.path_root, ('cpu' + str(core)),
                'cpufreq', 'scaling_max_freq')
            max_freq = str(
                max(
                    self.scaling_freqs)).encode()
            # debug print('max freq: ', max_freq)
            self._write_cpu(abs_path, max_freq)

            # reset min freq
            abs_path = path.join(
                self.path_root, ('cpu' + str(core)),
                'cpufreq', 'scaling_min_freq')
            min_freq = str(
                min(
                    self.scaling_freqs)).encode()
            # debug print('min freq: ', min_freq)
            self._write_cpu(abs_path, min_freq)

    def run_test(self):
        """ Execute cpufreq test, process results and return
        appropriate exit code.
        """
        # scaling_freqs = list(reversed(self.scaling_freqs))
        # disable hyperthread cores
        self.disable_thread_siblings()

        # userspace governor required for scaling_setspeed
        if self.scaling_driver == 'acpi-cpufreq':
            self.set_governors('userspace')
        else:
            self.set_governors('performance')

        # spawn core tests concurrently
        self.spawn_core_test()
        print('\n##[--------------]##')
        print('##[TEST COMPLETE!]##')
        print('##[--------------]##\n')

        # reset state and cleanup
        print('##[resetting cpus]##')
        self.reset()
        print('active threads:', threading.active_count())
        print('dangling pids:', self.pid_list)
        # process results
        print('\n##[results]##')
        print('-legend:')
        print(' core: target_freq: [sampled_avg_%, P/F, sampled_avg],\n')
        pprint.pprint(self._process_results())

        if self._fail_count:
            print('\nTest Failed')
            print('fail_count =', self._fail_count)
            return 1

        print('\nTest Passed')
        return 0

    def spawn_core_test(self):
        """ Spawn concurrent scale testing on all online cores.
        """
        proc_list = []
        # create queue for piping results
        result_queue = multiprocessing.Queue()
        online_cores = self._get_cores('online')

        for core in online_cores:
            # assign affinity per online_core list
            affinity = [core]
            affinity_dict = dict(
                affinity=affinity)
            proc = multiprocessing.Process(
                target=self.run_child,
                args=(result_queue,),
                kwargs=affinity_dict)
            # invoke core test
            proc.start()
            proc_list.append(proc)

        for core in online_cores:
            # pipe results from core test
            child_queue = result_queue.get()
            # append to chainmap object
            self.freq_chainmap = self.freq_chainmap.new_child(
                child_queue)

        for proc in proc_list:
            # terminate core test process
            proc.join()
            print('process collapsed, joined parent')

    def run_child(self, output, affinity):
        """ Subclass instantiation & constructor for individual
        core.
        """
        # get self pid
        proc = psutil.Process()
        # record pid for tracking
        self.pid_list.append(proc.pid)
        # print('PIDs:', self.pid_list)
        # assign affinity to process
        proc.cpu_affinity(affinity)
        core = int(affinity[0])
        # intantiate core test
        cpu_freq_ctest = CpuFreqCoreTest(core)
        # execute freq scaling
        cpu_freq_ctest.scale_all_freq()
        freq_map = cpu_freq_ctest.__call__()

        # map results to core
        output.put(freq_map)


class CpuFreqCoreTest(CpuFreqTest):
    """ Subclass to facilitate concurrent frequency scaling.
    Every physical core will self-test freq. scaling capabilities at once.
    """

    def __init__(self, core):
        super().__init__()
        # mangle instance attributes
        self.__instance_core = int(core)  # core under test
        self.__instance_cpu = 'cpu' + str(core)  # str cpu ref
        self.__stop_loop = 0  # signal.alarm semaphore
        self.__observed_freqs = []  # recorded freqs
        self.__observed_freqs_dict = {}  # core: recorded freqs
        self.__observed_freqs_rdict = {}  # raw recorded freqs (float)

    def __call__(self):
        """ Have subclass return dict '{core: avg_freq}'
        when called.
        """
        freq_map = (
            {self.__instance_core: self.__observed_freqs_dict})
        return freq_map

    @property
    def observed_freqs(self):
        """ Expose core's sampled freqs.
        """
        return self.__observed_freqs

    @property
    def observed_freqs_rdict(self):
        """ Expose raw freq samples.
        """
        return self.__observed_rfreqs

    class ObserveFreq:
        """ Class for instantiating observation thread (async);
        note: interval scaling will occur with freq scaling.
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
        """ Callback method to sample frequency.
        """
        # sample current frequency
        self.__observed_freqs.append(
            self.get_cur_freq())
        # matrix mode
        print(self.__observed_freqs)

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
            return int(freqs)

    def calc_freq_avg(self, freqs, n=3):
        """ Calculate moving average of observed_freqs.
        """
        freq_itr = iter(freqs)
        freq_deq = collections.deque(
            itertools.islice(freq_itr, n - 1))
        freq_deq.appendleft(0)
        freq_sum = sum(freq_deq)
        for elm in freq_itr:
            freq_sum += elm - freq_deq.popleft()
            freq_deq.append(elm)
            yield freq_sum / n

    def map_observed_freqs(self, idx):
        """ Align freq key/values and split result lists
        for grouping.
        """
        scaling_freqs = list(reversed(self.scaling_freqs))
        target_freq = scaling_freqs[idx]
        freq_avg = self.calc_freq_avg(self.__observed_freqs)

        for freq in freq_avg:
            # append observed_freq list to dict value with target freq as key
            self.__observed_freqs_dict.update(
                {target_freq: freq})
            # pack results for raw data record
            self.__observed_freqs_rdict.update(
                {target_freq: self.__observed_freqs})

    def scale_all_freq(self):
        """ Primary class method to get running core freqs,
        nested fns for encapsulation.
        """
        def execute_workload(n):
            """ Perform maths to load core.
            """
            while not self.__stop_loop:
                math.factorial(n)

        def handle_alarm(*args):
            """ Alarm trigger callback,
            unload core
            """
            self.__stop_loop = 1

        def visualize_freq(freq):
            """ Method to provide feedback for debug/verbose
            logging.
            """
            print('##########################')
            print('testing: ', self.__instance_cpu)
            print('target freq: ', freq.decode())
            print('workload n:', self.workload_n)
            print('##########################')

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
            # logging
            visualize_freq(freq)
            # start data sampling
            observe_freq.observe()
            # pass random int for load generation
            execute_workload(
                self.workload_n)
            # stop workload loop
            self.__stop_loop = 0
            # map freq results to core
            self.map_observed_freqs(idx)
            # reset list for next frequency
            self.__observed_freqs = []

        # setup paths
        abs_path_setspd = path.join(
            self.path_root, self.__instance_cpu,
            'cpufreq', 'scaling_setspeed')
        abs_path_maxspd = path.join(
            self.path_root, self.__instance_cpu,
            'cpufreq', 'scaling_max_freq')
        abs_path_minspd = path.join(
            self.path_root, self.__instance_cpu,
            'cpufreq', 'scaling_min_freq')

        # iterate over core supported freqs
        for idx, freq in enumerate(reversed(self.scaling_freqs)):
            freq = str(freq).encode()
            # userspace governor required to write to ./scaling_setspeed
            if self.scaling_driver == 'acpi-cpufreq':
                try:
                    super()._write_cpu(abs_path_setspd, freq)
                except Exception:
                    raise CpuFreqExec(
                        'ERROR: setting invalid frequency,\
                        scaling_setspeed!')
                else:
                    # begin scaling
                    scale_to_freq(freq, idx)
                    # pad observe_interval callback
                    self.observe_interval = idx
            else:
                # per cpufreq, set max_freq before min_freq
                try:
                    super()._write_cpu(abs_path_maxspd, freq)
                except Exception:
                    raise CpuFreqExec(
                        'ERROR: setting invalid frequency,\
                        scaling_max_freq!')
                else:
                    try:
                        super()._write_cpu(abs_path_minspd, freq)
                    except Exception:
                        raise CpuFreqExec(
                            'ERROR: setting invalid frequency,\
                            scaling_min_freq!')
                    else:
                        # begin scaling
                        scale_to_freq(freq, idx)
                        # pad observe_interval callback
                        self.observe_interval = idx


def main():
    cpu_freq_test = CpuFreqTest()
    return(cpu_freq_test.run_test())


if __name__ == '__main__':
    sys.exit(main())

# -*- coding: utf-8 -*-
