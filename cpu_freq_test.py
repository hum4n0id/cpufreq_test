# !/usr/bin/env python3

"""
Copyright (C) 2020 Canonical Ltd.

Authors
  Adrian Lane <adrian.lane@canonical.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3,
as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

* Test and validate sut cpu scaling capabilities.

todo optional (nice to have):
* check if workload needs to scale with processor power
"""

from os import path
import multiprocessing
import collections
import threading
import argparse
import logging
import random
import signal
import pprint
import math
import time
import sys
import psutil


class CpuFreqExec(Exception):
    """ Exception handling (stderr).
    """
    def __init__(self, message):
        super().__init__(message)
        logging.error(message, exc_info=True)


class CpuFreqTest():
    """ Test cpufreq scaling capabilities.
    """
    # class attributes / statics

    # time to stay at frequency under load (sec)
    # more time = more resolution
    # should be gt observe_interval
    scale_duration = 15
    # frequency sampling interval (thread timer)
    # should be lt scale_duration
    observe_interval = 1
    # max, min percentage of avg freq allowed to pass,
    # relative to target freq
    # ex: max = 110, min = 90 is 20% passing tolerance
    max_freq_pct = 110
    min_freq_pct = 90
    # time to wait for child join to complete
    proc_join_timeout = 5

    def __init__(self):
        """ Instance attributes.
        """
        def append_max_min():
            """ Create scaling table from max_freq,
            min_freq cpufreq files.
            """
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

        self.path_root = '/sys/devices/system/cpu'
        self.__fail_count = 0
        # attributes common to all cores
        self.__proc_list = []  # pids for core affinity assignment

        # chainmap object constructor
        self.freq_chainmap = collections.ChainMap()
        self.freq_result_map = {}  # final results

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
        if 'acpi-cpufreq' in self.scaling_driver:
            path_scaling_freqs = path.join(
                'cpu0', 'cpufreq',
                'scaling_available_frequencies')
            scaling_freqs = self._read_cpu(
                path_scaling_freqs).rstrip('\n').split()
            # cast freqs to int
            self.scaling_freqs = list(
                map(int, scaling_freqs))
            # test freqs in ascending order
            self.scaling_freqs.sort()
        # intel p_state, etc
        else:
            # setup path and status for intel pstate directives
            if 'intel_' in self.scaling_driver:
                # /sys/devices/system/cpu/intel_pstate/status
                self.path_ipst_status = path.join(
                    'intel_pstate', 'status')
                self.startup_ipst_status = self._read_cpu(
                    self.path_ipst_status).rstrip('\n')
            # use max, min freq for scaling table
            self.scaling_freqs = list(
                map(int, append_max_min()))
            self.scaling_freqs.sort()
            self.startup_max_freq = self.scaling_freqs[1]
            self.startup_min_freq = self.scaling_freqs[0]

    def _read_cpu(self, fpath):
        """ Read sysfs/cpufreq file.
        """
        abs_path = path.join(
            self.path_root, fpath)
        # open abs_path in binary mode, read
        try:
            with open(abs_path, 'rb') as _:
                data = _.read().decode('utf-8')
        except OSError:
            raise CpuFreqExec(
                'ERROR: unable to read file: %s' % abs_path)
        else:
            return data

    def _write_cpu(self, fpath, data):
        """ Write sysfs/cpufreq file.
        """
        def return_bytes_utf():
            """ Data type conversion to utf,
            for sysfs writes.
            """
            try:
                # str type
                data_enc = data.encode()
            except (AttributeError, TypeError):
                # int, float type
                data_enc = str(data).encode()
            data_utf = bytes(data_enc)
            return data_utf

        if not isinstance(data, bytes):
            data_utf = return_bytes_utf()
        else:
            # do not convert bytes()
            data_utf = data

        abs_path = path.join(
            self.path_root, fpath)
        # write utf bytes to cpufreq sysfs
        try:
            with open(abs_path, 'wb') as _:
                _.write(data_utf)
        except OSError:
            raise CpuFreqExec(
                'ERROR: unable to write file: %s' % abs_path)
        else:
            return data

    def _get_cpufreq_param(self, parameter):
        """ Get base cpufreq param from online cores.
        Used for method calls via argparse.
        """
        data = {}
        online_cores = self._get_cores('online')
        for core in online_cores:
            fpath = path.join(
                'cpu%i' % core,
                'cpufreq', parameter)
            data[int(core)] = self._read_cpu(
                fpath).rstrip('\n').split()[0]
        return data

    def _get_cores(self, fpath):
        """ Get various core ranges, convert to list.
        """
        def list_core_rng(core_rng):
            """ Method to convert core range to list prior
            to iteration.
            """
            core_list = []
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

        core_rng = self._read_cpu(
            fpath).strip('\n').strip()
        core_list = list_core_rng(core_rng)
        return core_list

    def _process_results(self):
        """ Process results from CpuFreqCoreTest()
        """
        def comp_freq_dict(inner_key, inner_val):
            """ Transpose and append results from subclass.
            """
            if inner_val:
                # get % avg_freq/target_freq
                result_pct = int((inner_val / inner_key) * 100)
                # append result %
                new_inner_val = [result_pct]
                if CpuFreqTest.min_freq_pct <= result_pct <= (
                        CpuFreqTest.max_freq_pct):
                    # append result P/F
                    new_inner_val.append('Pass')
                else:
                    new_inner_val.append('Fail')
                    # increment fail bit
                    self.__fail_count += 1
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

    def disable_thread_siblings(self):
        """ Disable all threads attached to the same core,
        aka hyperthreading.
        """
        def get_thread_siblings():
            """ Get hyperthread cores to offline.
            """
            thread_siblings = []
            online_cores = self._get_cores('online')
            for core in online_cores:
                fpath = path.join(
                    'cpu%i' % core,
                    'topology', 'thread_siblings_list')
                # second core is sibling
                thread_siblings += self._get_cores(fpath)[1:]
            # prefer set for binary &
            to_disable = set(thread_siblings) & set(online_cores)
            return to_disable

        to_disable = get_thread_siblings()
        logging.info('  - disabling cores: %s', to_disable)
        for core in to_disable:
            fpath = path.join(
                'cpu%i' % core,
                'online')
            self._write_cpu(fpath, 0)

    def get_governors(self):
        """ Return active governors on all cores.
        """
        governors = self._get_cpufreq_param(
            'scaling_governor')
        return governors

    def set_governors(self, governor):
        """ Set/change cpu governor, perform on
        all cores.
        """
        logging.info('  - setting governor: %s', governor)
        online_cores = self._get_cores('online')
        for core in online_cores:
            fpath = path.join(
                'cpu%i' % core,
                'cpufreq', 'scaling_governor')
            self._write_cpu(fpath, governor)

    def reset(self):
        """ Enable all offline cpus,
        and reset max and min frequencies files.
        """
        def reset_intel_pstate():
            """ Reset fn for pstate driver.
            """
            self._write_cpu(
                self.path_ipst_status, 'off')
            ipst_status = self._write_cpu(
                self.path_ipst_status, self.startup_ipst_status)
            logging.info('  - setting mode: %s', ipst_status)

        def enable_off_cores():
            """ Enable all present and offline cores.
            """
            present_cores = self._get_cores('present')
            # duck-typed for -r flag invokation
            try:
                offline_cores = self._get_cores('offline')
            except ValueError:
                return
            else:
                to_enable = set(present_cores) & set(offline_cores)

            logging.info('* enabling thread siblings/hyperthreading:')
            logging.info('  - enabling cores: %s', to_enable)
            for core in to_enable:
                fpath = path.join(
                    'cpu%i' % core,
                    'online')
                self._write_cpu(fpath, 1)

        def set_max_min():
            """ Set max_frequency and min_frequency cpufreq files.
            """
            present_cores = self._get_cores('present')
            for core in present_cores:
                path_max = path.join(
                    'cpu%i' % core,
                    'cpufreq', 'scaling_max_freq')
                path_min = path.join(
                    'cpu%i' % core,
                    'cpufreq', 'scaling_min_freq')
                # reset max freq
                self._write_cpu(
                    path_max, self.startup_max_freq)
                # reset min freq
                self._write_cpu(
                    path_min, self.startup_min_freq)

        logging.info('* restoring startup governor:')
        # in case test ends prematurely from prior run
        # and facilitate reset() called from args
        if 'userspace' in self.startup_governor:
            # need to validate these assumptions
            self.startup_governor = 'ondemand'
        elif 'performance' in self.startup_governor:
            self.startup_governor = 'powersave'
        self.set_governors(self.startup_governor)

        # enable offline cores
        enable_off_cores()

        # reset sysfs for non-acpi_cpufreq systems
        if 'acpi-cpufreq' not in self.scaling_driver:
            # intel_pstate, intel_cpufreq
            if 'intel_' in self.scaling_driver:
                logging.info('* resetting intel p_state cpufreq driver')
                # will reset max, min freq files
                reset_intel_pstate()
            else:
                logging.info('* manually restoring max, min freq files')
                set_max_min()

    def execute_test(self):
        """ Execute cpufreq test, process results and return
        appropriate exit code.
        """
        # disable thread siblings/hyperthread cores
        logging.info('* disabling thread siblings (hyperthreading):')
        self.disable_thread_siblings()

        # if intel, reset and start best available driver (passive pref.)
        if 'intel_' in self.scaling_driver:
            # reset intel driver for clean slate
            self._write_cpu(
                self.path_ipst_status, 'off')

            # see if we can use the intel_cpufreq driver (fullest featured)
            try:
                logging.info(
                    '* starting intel_cpufreq driver:')
                self._write_cpu(self.path_ipst_status, 'passive')
            except CpuFreqExec:
                # active mode available for all intel p_state systems
                logging.info(
                    '  - failed: setting p_state mode to active')
                self._write_cpu(self.path_ipst_status, 'active')

            cur_ipst_status = self._read_cpu(
                self.path_ipst_status).rstrip('\n')
            logging.info(
                '  - p_state mode: %s', cur_ipst_status)

        logging.info('* configuring cpu governors:')
        # userspace governor required for scaling_setspeed
        if 'acpi-cpufreq' in self.scaling_driver:
            self.set_governors('userspace')
        else:
            self.set_governors('performance')

        # spawn core_tests concurrently
        self.spawn_core_test()
        print('\n##[--------------]##')
        print('##[TEST COMPLETE!]##')
        print('##[--------------]##\n')

        # reset state and cleanup
        logging.info('##[resetting cpus]##')
        self.reset()

        # facilitate house cleaning
        # terminate dangling processes post .join()
        if self.__proc_list:
            logging.info('* terminating dangling pids')
            for proc in self.__proc_list:
                proc.terminate()
                # logging.info('  - PID %i terminated', proc.pid)
        logging.info('* active threads: %i', threading.active_count())

        # process results
        print('\n##[results]##')
        print('-legend:')
        print(' core: target_freq: [sampled_avg_%, P/F, sampled_avg],\n')
        # pretty result output
        pprint.pprint(self._process_results())

        if self.__fail_count:
            print('\nTest Failed')
            print('fail_count =', self.__fail_count)
            return 1

        print('\nTest Passed')
        return 0

    def spawn_core_test(self):
        """ Spawn concurrent scale testing on all online cores.
        """
        def run_child(output, affinity):
            """ Subclass instantiation & constructor for individual
            core.
            """
            proc = psutil.Process()
            # assign affinity to process
            proc.cpu_affinity(affinity)
            core = int(affinity[0])
            # intantiate core_test
            cpu_freq_ctest = CpuFreqCoreTest(core)
            # execute freq scaling
            cpu_freq_ctest.scale_all_freq()
            freq_map = cpu_freq_ctest.__call__()

            # map results to core
            output.put(freq_map)

        proc_list = []
        pid_list = []
        # create queue for piping results
        result_queue = multiprocessing.Queue()
        online_cores = self._get_cores('online')
        # first spawn children, then start workload
        online_cores.sort(reverse=True)

        # assign affinity and spawn core_test
        for core in online_cores:
            affinity = [core]
            affinity_dict = dict(
                affinity=affinity)
            proc = multiprocessing.Process(
                target=run_child,
                args=(result_queue,),
                kwargs=affinity_dict)
            # invoke core_test
            proc.start()
            proc_list.append(proc)
            pid_list.append(proc.pid)

        logging.debug('* spawned PIDs performing core_testing:')
        logging.debug('  - %r', pid_list)

        # get queued core_test results
        for core in online_cores:
            # pipe results from core_test
            child_queue = result_queue.get()
            # append to chainmap object
            self.freq_chainmap = self.freq_chainmap.new_child(
                child_queue)

        # cleanup spawned core_test pids
        for proc in proc_list:
            # join core_test processes
            proc.join(
                CpuFreqTest.proc_join_timeout)
            logging.debug(
                '* PID %s joined parent', proc.pid)
            time.sleep(.1)

        # update attribute to perfrom 2nd pass terminate
        self.__proc_list = proc_list


class CpuFreqCoreTest(CpuFreqTest):
    """ Subclass to facilitate concurrent frequency scaling.
    Every physical core will self-test freq. scaling capabilities at once.
    """
    class ObserveFreq:
        """ Class for instantiating observation thread.
        Non-blocking and locked to system time to avoid
        exponentional drift as frequency scaling occurs.
        """
        # dev note: encapsulating this functionality in
        # a core_test nested class was cleanest way to
        # sample data while not blocking testing threads.
        def __init__(self, interval, callback):
            """ Execute start_timer on instantiation.
            """
            self.timer_running = False
            self.thread_timer = None
            self.interval = interval
            self.callback = callback
            self.next_call = time.time()
            self.start_timer()

        def start_timer(self):
            """ Facilitate callbacks at specified interval,
            accounts and corrects for drift.
            """
            if not self.timer_running:
                # offset interval
                self.next_call += self.interval
                # create time delta for consistent timing
                time_delta = self.next_call - time.time()
                # call observe() after time_delta passes
                self.thread_timer = threading.Timer(
                    time_delta, self.observe)
                # cleanup timer threads on exit
                self.thread_timer.daemon = True
                self.thread_timer.start()
                self.timer_running = True

        def observe(self):
            """ Trigger callback to sample frequency.
            """
            # reset timer_running
            self.timer_running = False
            # ObserveFreq not subclassed; callback to outer scope
            self.callback()
            # start another tt cycle
            self.start_timer()

        def stop(self):
            """ Called when frequency scaling completed.
            """
            if self.thread_timer:
                # event loop end
                self.thread_timer.cancel()
            self.timer_running = False

    def __init__(self, core):
        super().__init__()
        # mangle instance attributes
        self.__instance_core = int(core)  # core under test
        self.__instance_cpu = 'cpu%i' % core  # str cpu ref
        self.__stop_loop = False  # init signal.alarm semaphore
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
        """ Expose sampled freqs for core.
        """
        return self.__observed_freqs

    @property
    def observed_freqs_rdict(self):
        """ Expose raw freq samples, mapped to core.
        """
        return self.__observed_freqs_rdict

    def _observe_freq_cb(self):
        """ Callback method to sample frequency.
        """
        def get_cur_freq():
            """ Get current frequency.
            """
            fpath = path.join(
                self.__instance_cpu,
                'cpufreq', 'scaling_cur_freq')
            freqs = self._read_cpu(
                fpath).rstrip('\n').split()[0]
            return int(freqs)

        self.__observed_freqs.append(
            get_cur_freq())
        # matrix mode
        logging.debug(self.__observed_freqs)

    def scale_all_freq(self):
        """ Primary method to scale full range of freqs.
        """
        def calc_freq_median(freqs):
            """ Get the median value of observed freqs.
            """
            n_samples = len(freqs)
            # floor division req.
            index = n_samples // 2
            # odd number of samples in observed_freqs
            if n_samples % 2:
                freq_median = sorted(freqs)[index]
            # even number of samples in observed_freqs
            else:
                freq_median = sum(
                    sorted(freqs)[(index - 1):(index + 1)]) / 2
            return freq_median

        def map_observed_freqs(target_freq):
            """ Align freq key/values and split result lists
            for grouping.
            """
            # get median of observed freqs
            freq_med = calc_freq_median(self.__observed_freqs)
            # target_freq = key, freq_median = value
            self.__observed_freqs_dict.update(
                {target_freq: freq_med})
            # raw data record
            self.__observed_freqs_rdict.update(
                {target_freq: self.__observed_freqs})

        def handle_alarm(*args):
            """ Alarm trigger callback,
            unload core
            """
            # args unused; *args present for signal callback
            del args
            # stop workload loop
            self.__stop_loop = True

        def execute_workload(workload_n):
            """ Perform maths to load core.
            """
            while not self.__stop_loop:
                math.factorial(workload_n)

        def log_freq_scaling(freq, workload_n):
            """ Method to provide feedback for debug/verbose
            logging.
            """
            logging.info(
                '* testing: %s || target freq: %i || workload n: %i',
                self.__instance_cpu, freq, workload_n)

        def load_observe_map(freq):
            """ Proxy fn to scale core to freq.
            """
            # gen randint for workload factorial calcs
            # helps salt concurrent workloads
            workload_n = random.randint(34512, 37845)
            # setup async alarm to kill load gen
            signal.signal(signal.SIGALRM, handle_alarm)
            # time to gen load
            signal.alarm(CpuFreqTest.scale_duration)
            # instantiate ObserveFreq and start data sampling
            observe_freq = self.ObserveFreq(
                interval=CpuFreqTest.observe_interval,
                callback=self._observe_freq_cb)
            # provide feedback on test status
            log_freq_scaling(freq, workload_n)
            # start loading core
            execute_workload(
                workload_n)
            # stop sampling
            observe_freq.stop()
            # map freq results to core
            map_observed_freqs(freq)

        # set paths relative to core
        path_set_speed = path.join(
            self.__instance_cpu,
            'cpufreq', 'scaling_setspeed')
        path_max_freq = path.join(
            self.__instance_cpu, 'cpufreq',
            'scaling_max_freq')

        # iterate over supported frequency scaling table
        for idx, freq in enumerate(self.scaling_freqs):
            # re-init some attributes after 1st pass
            if idx:
                # reset freq list
                self.__observed_freqs = []
                # reset workload loop bit
                self.__stop_loop = False

            # acpi supports full freq table scaling
            if 'acpi-cpufreq' in self.scaling_driver:
                # write to sysfs
                self._write_cpu(path_set_speed, freq)
                # facilitate testing
                load_observe_map(freq)
            # others support max, min freq scaling
            else:
                self._write_cpu(path_max_freq, freq)
                load_observe_map(freq)


def parse_args_logging():
    """ Ingest arguments and init logging.
    """
    def init_logging(args):
        """ Parse and set logging levels, start logging.
        """
        # stdout for argparsed logging lvls
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(args.log_level)
        # stderr for exceptions
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.ERROR)

        # setup base/top-lvl logger
        base_logging = logging.getLogger()
        # set base logging level to pipe StreamHandler() thru
        base_logging.setLevel(logging.NOTSET)

        # start logging
        base_logging.addHandler(stdout_handler)
        base_logging.addHandler(stderr_handler)

    parser = argparse.ArgumentParser()
    # levels: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    # lvlnum: 50      , 40   , 30     , 20  , 10   , 0
    # only allow one arg
    parser_mutex_grp = parser.add_mutually_exclusive_group()
    parser_mutex_grp.add_argument(
        '-d', '--debug',
        dest='log_level',
        action='store_const',
        const=logging.DEBUG,
        # default logging level
        default=logging.INFO,
        help='debug/verbose output (stdout)')
    parser_mutex_grp.add_argument(
        '-q', '--quiet',
        dest='log_level',
        action='store_const',
        const=logging.WARNING,
        help='suppress output')
    # 'dev mode' args
    parser_mutex_grp.add_argument(
        '-r', '--reset',
        action='store_true',
        help='reset cpufreq sysfs (governor, max/min, pstate)')
    parser_mutex_grp.add_argument(
        '-g', '--gov',
        action='store_true',
        help='get active governor (global/all cpu)')
    args = parser.parse_args()
    init_logging(args)
    return args


def main():
    args = parse_args_logging()
    # instantiate CpuFreqTest as cpu_freq_test
    cpu_freq_test = CpuFreqTest()
    if args.reset:
        cpu_freq_test.reset()
        print('reset cpufreq sysfs')
    elif args.gov:
        print(cpu_freq_test.get_governors())
    else:
        return cpu_freq_test.execute_test()


if __name__ == '__main__':
    sys.exit(main())
