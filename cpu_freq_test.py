# !/usr/bin/env python3

"""
todo/verify:
* implement argparse
* change header to Canonical official syntax
* expand exeception handling
* make reset() callable with '-r' arg
* get min/max from min/max files and write to attributes
    - currently using min/max from freq scaling table

todo optional (nice to have):
* check if workload needs to scale with processor power
    - can add setter or fn to update workload_n
* add capabilities to track spawned threads from
    callback timers
* do rca on startup offset in scale_all_freq()
"""

from os import path
import collections
import itertools
import multiprocessing
import threading
import psutil
import random
import signal
import sys
import math
import pprint
#from pudb import set_trace


class cpuFreqExec(Exception):
    """ Exception handling.
    """
    def __init__(self, message):
        self.message = message
        # logging error
        print (message)


class cpuFreqTest:
    def __init__(self):
        self._core_list = []  # ephermeral core list
        self._thread_siblings = []  # sut hyperthread cores
        self.pid_list = []  # pids for core affinity assignment
        self.freq_result_map = {}  # final results
        # ChainMap object constructor
        self.freq_chainmap = collections.ChainMap()

        # factorial to calculate during core test, positive int
        self._workload_n = random.randint(34512, 67845)

        # time to stay at frequency under load (sec)
        # more time = more resolution
        # should be gt observe_interval
        self.scale_duration = 15

        # frequency sampling interval (thread timer)
        # too low time = duplicate samples
        # should be lt scale_duration
        self._observe_interval = .75

        # max, min percentage of avg freq allow to pass
        # relative to target frequency
        # ex: max = 110, min = 90 is 20% passing tolerance
        # self._max_freq_pct = 110
        self._max_freq_pct = 110
        self._min_freq_pct = 90
        self._fail_count = 0

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
        scaling_freqs = self._read_cpu(
            path_scaling_freqs).rstrip('\n').split()
        self.scaling_freqs = list(
            map(
                int, scaling_freqs))

    @property
    def observe_interval(self):
        return self._observe_interval

    @observe_interval.setter
    def observe_interval(self, idx):
        """ Setter to pad/throttle observ_freq_cb()
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
            raise cpuFreqExec(
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
            raise cpuFreqExec(
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
        self._core_list = core_list
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
        """ Process results from cpuFreqCoreTest()
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
                    # append result P/F
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
        print ('enabling the following cpus:', to_enable)
        for cpu in to_enable:
            abs_path = path.join(
                ('cpu' + str(cpu)), 'online')
            self._write_cpu(abs_path, b'1')

    def disable_thread_siblings(self):
        """ Disable all threads attached to the same core,
        aka hyperthreading.
        """
        thread_siblings = []
        online_cpus = self._get_cores('online')
        for cpu in online_cpus:
            abs_path = path.join(
                self.path_root, ('cpu' + str(cpu)),
                'topology', 'thread_siblings_list')
            # second core is sibling
            thread_siblings += self._get_cores(abs_path)[1:]
        self._thread_siblings = thread_siblings
        # prefer set for binary &
        to_disable = set(thread_siblings) & set(online_cpus)

        for sib in to_disable:
            abs_path = path.join(
                self.path_root, ('cpu' + str(sib)),
                'online')
            self._write_cpu(abs_path, b'0')

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
            # debug print ('max freq: ', max_freq)
            self._write_cpu(abs_path, max_freq)

            # reset min freq
            abs_path = path.join(
                self.path_root, ('cpu' + str(cpu)),
                'cpufreq', 'scaling_min_freq')
            min_freq = str(
                min(
                    self.scaling_freqs)).encode()
            # debug print ('min freq: ', min_freq)
            self._write_cpu(abs_path, min_freq)

    def set_governors(self, governor):
        """ Set/change cpu governor, perform on
        all cores.
        """
        print ('setting governor:', governor)
        online_cores = self._get_cores('online')
        for cpu in online_cores:
            abs_path = path.join(
                ('cpu' + str(cpu)),
                'cpufreq', 'scaling_governor')
            self._write_cpu(abs_path, governor.encode())

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
        print ('active threads:', threading.active_count())

        # process results
        print('\n##[results]##')
        print('-legend:')
        print(' core: target_freq: [sampled_avg_%, P/F, sampled_avg],\n')
        pprint.pprint(self._process_results())

        if self._fail_count:
            print('\nTest Failed')
            print('fail_count =', self._fail_count)
            return 1
        else:
            print('\nTest Passed')
            return 0

    def spawn_core_test(self):
        """ Spawn concurrent scale testing on all online cores.
        """
        proc_list = list()
        result_queue = multiprocessing.Queue()
        online_cores = self._get_cores('online')

        for core in online_cores:
            print ('cpu: ', core)
            affinity = [core]
            aff_dict = dict(
                affinity=affinity)
            proc = multiprocessing.Process(
                target=self.run_child,
                args=(result_queue,),
                kwargs=aff_dict)
            proc.start()
            proc_list.append(proc)

        for core in online_cores:
            child_queue = result_queue.get()
            self.freq_chainmap = self.freq_chainmap.new_child(
                child_queue)

        for proc in proc_list:
            proc.join()
            print ('process collapsed, joined parent')

    def run_child(self, output, affinity):
        """ Subclass instantiation & constructor for individual
        core.
        """
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
        # set_trace()
        # thread safe
        freq_map = cpu_freq_ctest.__call__()

        # map results to core
        output.put(freq_map)


class cpuFreqCoreTest(cpuFreqTest):
    """ Subclass to facilitate concurrent frequency scaling.
    Every physical core will self-test freq. scaling capabilities at once.
    """
    def __init__(self, cpu_num):
        super().__init__()
        # mangle instance attributes
        self.__instance_core = int(cpu_num)  # core under test
        self.__instance_cpu = 'cpu' + str(cpu_num)  # str cpu ref
        self.__stop_loop = 0  # signal.alarm semaphore
        self.__observed_freqs = []  # recorded freqs
        self.__observed_freqs_dict = {}  # core: recorded freqs
        self.__observed_freqs_rdict = {}  # raw recorded freqs (float)
        self.__freq_avg = []  # running freq avg

    def __call__(self):
        """ Have subclass return dict '{core: avg_freq}'
        when called.
        """
        freq_map = (
            {self.__instance_core: self.__observed_freqs_dict})
        return freq_map

    @property
    def observed_freqs(self):
        """ Getter to expose core's sampled freqs.
        """
        return self.__observed_freqs

    @observed_freqs.setter
    def observed_freqs(self, idx, avg_freq=0):
        """ Setter to align freq key/values and split result lists
        for grouping.
        """
        scaling_freqs = list(reversed(self.scaling_freqs))
        # offset keys to correct startup offset
        if idx:
            target_freq = scaling_freqs[idx - 1]
        else:
            target_freq = scaling_freqs[idx]

        self.__freq_avg = self.calc_freq_avg(
            self.__observed_freqs)

        for freq in self.__freq_avg:
            avg_freq = freq
        # append observed_freq list to dict value with target freq as key
        self.__observed_freqs_dict.update(
            {target_freq: avg_freq})
        # pack results for raw data record
        self.__observed_freqs_rdict.update(
            {target_freq: self.__observed_freqs})
        # reset list for next frequency
        self.__observed_freqs = []

    @property
    def observed_freqs_rdict(self):
        """ Getter to expose raw freq samples.
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
            yield (freq_sum / n)

    def scale_all_freq(self):
        """ Primary class method to get running core freqs,
        nested fns for encapsulation.
        """
        def execute_workload(n):
            """ Perform maths to load core.
            """
            print (' generating cpu load')
            print (' n =', n)
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
            # print ('##########################')
            # print ('CPU: ', self.__instance_cpu)
            # print('set freq: ', freq.decode())
            # print ('observed_freq: ', self.get_cur_freq())
            # print ('##########################')
            pass

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
            # call setter in parent class
            self.observed_freqs = idx
            # pass random int for load generation
            execute_workload(
                self._workload_n)
            # timerThread.start()
            visualize_freq(freq)
            # reset signal alarm trigger bit
            self.__stop_loop = 0

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

        # iterate over core supported freqs
        for idx, freq in enumerate(reversed(self.scaling_freqs)):
            print (' scaling to: ', freq)
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
                    # begin scaling
                    scale_to_freq(freq, idx)
                    # pad observe_interval callback
                    self.observe_interval = idx
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
                        # begin scaling
                        scale_to_freq(freq, idx)
                        # pad observe_interval callback
                        self.observe_interval = idx


def main():
    cpu_freq_test = cpuFreqTest()
    return(cpu_freq_test.run_test())


if __name__ == '__main__':
    sys.exit(main())

# -*- coding: utf-8 -*-
