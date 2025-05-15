#! /usr/bin/python

import os, sys, traceback, subprocess

ADB=os.getenv('ADB', 'adb')

APE_ROOT='/data/local/tmp/'
APE_JAR=APE_ROOT + 'ape.jar'
APE_MAIN='com.android.commands.monkey.Monkey'
APP_PROCESS='/system/bin/app_process'

BASE_CMD=['CLASSPATH=' + APE_JAR, APP_PROCESS, APE_ROOT, APE_MAIN]

for k in ['MAX_CT_WAIT_FOR_ACTIVITY', 'CMD_APP_RESTART', 'USE_EXT_UIAUTOMATOR']:
    v = os.getenv(k)
    if v: BASE_CMD = [k + '=' + v] + BASE_CMD

SERIAL=os.getenv('SERIAL')
if SERIAL: BASE_CMD=[ADB, '-s', SERIAL, 'shell'] + BASE_CMD
else: BASE_CMD=[ADB, 'shell'] + BASE_CMD

def run_cmd(*args):
    print('Run cmd: ' + (' '.join(*args)))
    subprocess.check_call(*args)

def run_ape(args):
    run_cmd(BASE_CMD + list(args))

if __name__ == '__main__':
    try:
        run_ape(sys.argv[1:])
    except:
        traceback.print_exc()

