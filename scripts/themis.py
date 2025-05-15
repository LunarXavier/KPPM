import os
import subprocess
import time
from argparse import ArgumentParser, Namespace
from multiprocessing.pool import ThreadPool


def get_time_in_seconds(testing_time):
    if 'h' in testing_time:
        testing_time_in_secs = int(testing_time[:-1]) * 60 * 60
    elif 'm' in testing_time:
        testing_time_in_secs = int(testing_time[:-1]) * 60
    elif 's' in testing_time:
        testing_time_in_secs = int(testing_time[:-1])
    else:
        print("Warning: the given time is ZERO seconds!!")
        testing_time_in_secs = 0  # error!

    return testing_time_in_secs

def run_fastbot(apk, avd_serial, avd_name, output_dir, testing_time, event_counts):
    print("run fastbot bash ")
    command = 'bash -x run_fastbot.sh %s %s %s %s %s %s' % (apk, avd_serial, avd_name,
                                                              output_dir,
                                                              testing_time,
                                                              event_counts)
    print('execute fastbot: %s' % command)
    os.system(command)

def run_hybirddroid(apk, avd_serial, avd_name, output_dir, testing_time, event_counts):
    print("run hybirddroid bash ")
    command = 'bash -x run_hybirdDroid.sh %s %s %s %s %s %s' % (apk, avd_serial, avd_name,
                                                              output_dir,
                                                              testing_time,
                                                              event_counts)
    print('execute hybirddroid: %s' % command)
    os.system(command)

def run_hybirddroid_with_industry(apk, avd_serial, avd_name, output_dir, testing_time, event_counts , login_script, all_apks):
    print("run hybirddroid bash ")
    command = 'bash -x run_hybirdDroid_with_industry.sh %s %s %s %s %s %s %s %s' % (apk, avd_serial, avd_name,
                                                              output_dir,
                                                              testing_time,
                                                              event_counts,
                                                              login_script,
                                                              all_apks)
    print('execute hybirddroid with industry: %s' % command)
    os.system(command)


def run_hybirddroid_no_llm(apk, avd_serial, avd_name, output_dir, testing_time, event_counts):
    print("run hybirddroid bash ")
    command = 'bash -x run_hybirdDroid_no_llm.sh %s %s %s %s %s %s' % (apk, avd_serial, avd_name,
                                                              output_dir,
                                                              testing_time,
                                                              event_counts)
    print('execute hybirddroid no llm: %s' % command)
    os.system(command)

def run_monkey(apk, avd_serial, avd_name, output_dir, testing_time, event_counts):
    command = 'bash -x run_monkey.sh %s %s %s %s %s %s' % (apk, avd_serial, avd_name,
                                                              output_dir,
                                                              testing_time,
                                                              event_counts)
    print('execute monkey: %s' % command)
    os.system(command)

def run_ape(apk, avd_serial, avd_name, output_dir, testing_time, screen_option, login_script):
    command = 'bash -x run_ape.sh %s %s %s %s %s %s %s' % (apk, avd_serial, avd_name,
                                                        output_dir,
                                                        testing_time,
                                                        screen_option,
                                                        login_script)
    print('execute ape: %s' % command)
    os.system(command)

def run_aurora(apk, avd_serial, avd_name, output_dir, testing_time, screen_option, login_script,test_count):
    command = 'bash -x ../aurora/Code/run_aurora.sh %s %s %s %s %s %s %s %s' % (apk, avd_serial, avd_name,
                                                        output_dir,
                                                        testing_time,
                                                        screen_option,
                                                        login_script,
                                                        test_count)
    print('execute aurora: %s' % command)
    os.system(command)

def get_all_apks(apk_list_file):
    file = open(apk_list_file, 'r')
    apk_paths = []
    apk_login_scripts = []
    for line in file.readlines():
        if line.strip().startswith('#'):
            # skip commented apk files
            continue
        if "," in line:
            content = line.split(",")
            apk_paths.append(content[0].strip())
            apk_login_scripts.append(content[1].strip())
        else:
            # 去除行首尾的空格和引号，然后添加到列表中
            apk_path = line.strip().strip('"')
            apk_paths.append(apk_path)
            apk_login_scripts.append("\"\"")
    print("Total %s apks under test" % len(apk_paths))
    return apk_paths, apk_login_scripts

def run_industry_apk(args: Namespace):
    if not os.path.exists(args.o):
        os.mkdir(args.o)

    # allocate emulators for an apk
    start_avd_serial = 5554 + args.offset * 2
    avd_serial_list = []
    for apk_index in range(args.number_of_devices):
        avd_serial = 'emulator-' + str(start_avd_serial + apk_index * 2)
        avd_serial_list.append(avd_serial)
        print('allocate emulators: %s' % avd_serial)

        # only for run_ape.sh
        screen_option = "-no-window"

    # multiple apks 
    all_apks, all_apks_login_scripts = get_all_apks(args.apk_list)

    number_of_xapks_content = len(all_apks)
    print("the number of content xapks:", number_of_xapks_content)
    print("the apk list: %s" % str(all_apks))

    number_of_apks = len(all_apks)
    apk_index = 0

    while 0 <= apk_index < args.repeat:
        print('test run count:',apk_index)
        p = ThreadPool(args.number_of_devices)
        for avd_serial in avd_serial_list:
            time.sleep(10)
            if apk_index >= number_of_apks:
                break

            print("Now allocate the apk: %s on %s" % (str(all_apks), avd_serial))
            login_script = all_apks_login_scripts[0]
            print("its login script: %s" % login_script)

            # 保存主apk路径
            current_apk = all_apks[0]
            # 将 APK 列表转换为以空格分隔的字符串
            all_apks_str = ' '.join(all_apks)
  
            if args.hybirddroid:
                p.apply_async(run_hybirddroid_with_industry, args=(current_apk, avd_serial, args.avd_name,
                                                args.o, args.time, args.event_count, login_script,  all_apks_str))              
            else:
                pass

            apk_index += 1

        print("wait the allocated devices to finish...")
        p.close()
        p.join()

def main(args: Namespace):
    if not os.path.exists(args.o):
        os.mkdir(args.o)

    # allocate emulators for an apk
    start_avd_serial = 5554 + args.offset * 2
    avd_serial_list = []
    for apk_index in range(args.number_of_devices):
        avd_serial = 'emulator-' + str(start_avd_serial + apk_index * 2)
        avd_serial_list.append(avd_serial)
        print('allocate emulators: %s' % avd_serial)

        # only for run_ape.sh
        screen_option = "-no-window"

    if args.apk is not None:
        # single apk mode
        all_apks = [args.apk]
        if args.login_script is None:
            all_apks_login_scripts = ["\"\""]
        else:
            all_apks_login_scripts = [args.login_script]
    else:
        # multiple apks mode
        all_apks, all_apks_login_scripts = get_all_apks(args.apk_list)

    if args.repeat > 1:
        copy_all_apks = all_apks.copy()
        copy_all_apks_login_scripts = all_apks_login_scripts.copy()
        for i in range(1, args.repeat):
            all_apks = all_apks + copy_all_apks
            all_apks_login_scripts = all_apks_login_scripts + copy_all_apks_login_scripts

    print("the apk list to fuzz: %s" % str(all_apks))

    number_of_apks = len(all_apks)
    apk_index = 0

    while 0 <= apk_index < number_of_apks:
        print('test run count:',apk_index)
        p = ThreadPool(args.number_of_devices)
        for avd_serial in avd_serial_list:
            time.sleep(10)
            if apk_index >= number_of_apks:
                break
            current_apk =  all_apks[apk_index]

            print(os.path.exists(current_apk))

            print("Now allocate the apk: %s on %s" % (current_apk, avd_serial))
            login_script = all_apks_login_scripts[apk_index]
            print("its login script: %s" % login_script)

            if args.monkey:
                p.apply_async(run_monkey, args=(current_apk, avd_serial, args.avd_name,
                                                args.o, args.time, args.event_count))                
            elif args.fastbot:
                p.apply_async(run_fastbot, args=(current_apk, avd_serial, args.avd_name,
                                                args.o, args.time, args.event_count))    
            elif args.ape:
                p.apply_async(run_ape, args=(current_apk, avd_serial, args.avd_name,
                                             args.o, args.time, screen_option,
                                             login_script))   
            elif args.hybirddroid:
                p.apply_async(run_hybirddroid_with_industry, args=(current_apk, avd_serial, args.avd_name,
                                                args.o, args.time, args.event_count))
            elif args.aurora:
                p.apply_async(run_aurora, args=(current_apk, avd_serial, args.avd_name,
                                                args.o, args.time, screen_option, login_script,
                                                apk_index))
            elif args.hybirddroid_no_llm:
                p.apply_async(run_hybirddroid_no_llm, args=(current_apk, avd_serial, args.avd_name,
                                                args.o, args.time, args.event_count))                
            else:
                pass

            apk_index += 1

        print("wait the allocated devices to finish...")
        p.close()
        p.join()


if __name__ == '__main__':
    ap = ArgumentParser()

    # by default, we run each bug/tool for 6h & 5r.
    # Each emulator is configured as 2GB RAM, 1GB internal storage and 1GB SDCard

    ap.add_argument('--avd', type=str, dest='avd_name', help="the device name")
    ap.add_argument('--apk', type=str, dest='apk')
    ap.add_argument('-n', type=int, dest='number_of_devices', default=1,
            help="number of emulators created for testing, default: 1")
    ap.add_argument('--apk-list', type=str, dest='apk_list', help="list of apks under test")
    ap.add_argument('-o', required=True, help="output dir")
    ap.add_argument('--time', type=str, default='6h', help="the fuzzing time in hours (e.g., 6h), minutes (e.g., 6m),"                                                          " or seconds (e.g., 6s), default: 6h")
    ap.add_argument('--repeat', type=int, default=1, help="the repeated number of runs, default: 1")
    ap.add_argument('--max-emu', type=int, default=16, help="the maximum allowed number of emulators")
    ap.add_argument('--no-headless', dest='no_headless', default=False, action='store_true', help="show gui")
    ap.add_argument('--login', type=str, dest='login_script', help="the script for app login")
    ap.add_argument('--wait', type=int, dest='idle_time',
                    help="the idle time to wait before starting the fuzzing")
    ap.add_argument('--is_industry', default=False, action='store_true', help='is industry app')

    # supported fuzzing tools
    ap.add_argument('--monkey', default=False, action='store_true')
    ap.add_argument('--ape', default=False, action='store_true')
    ap.add_argument('--fastbot', default=False, action='store_true')
    ap.add_argument('--hybirddroid', default=False, action='store_true')
    ap.add_argument('--hybirddroid_no_llm', default=False, action='store_true')
    ap.add_argument('--aurora', default=False, action='store_true')
    ap.add_argument('--gptdroid', default=False, action='store_true')

    ap.add_argument('--offset', type=int, default=0, help="device offset number w.r.t emulator-5554")
    ap.add_argument('--event_count', type=str, default=None, help="event counts, defaut=3600")

    args = ap.parse_args()
    
    if args.number_of_devices + args.offset > 16:
        if not args.timemachine:
            # TimeMachine is allowed to run more than 16 instances due to it runs in the docker containers.
            ap.error('n + offset should not be ge 16')

    if args.apk is None and args.apk_list is None:
        ap.error('please specify an apk or an apk list')

    if args.apk_list is not None and not os.path.exists(args.apk_list):
        ap.error('No such file: %s' % args.apk_list)

    if args.idle_time is not None:
        for i in range(1, int(args.idle_time)):
            print("%d minutes remaining to wait ..." % (args.idle_time - i))
            time.sleep(60)
    # 如果路径不存在，则创建路径
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    if args.is_industry:
        run_industry_apk(args)
    else:
        main(args)
