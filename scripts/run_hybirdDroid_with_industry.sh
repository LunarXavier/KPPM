#!/bin/bash

APK_PATH="$1"
DEVICE_SERIAL="$2"
AVD_NAME="$3"
OUTPUT_PATH="$4"
TEST_TIME="$5"
EVENT_COUNT="$6"
LOGIN_SCRIPT=$7 # the script for app login via uiautomator2
APK_LISTS="$8"

PYTHON_SCRIPT="../start.py"
HEADLESS=true
current_date_time=$(date +'%Y-%m-%d-%H-%M-%S')
apk_file_name=$(basename "$APK_PATH")
RESULT_DIR="$OUTPUT_PATH/$apk_file_name.hybirddroid.result.$DEVICE_SERIAL.$AVD_NAME#$current_date_time"
LOG_FILE="$RESULT_DIR/log.txt"

# 提取端口号
if [[ "$DEVICE_SERIAL" =~ emulator-([0-9]+) ]]; then
    AVD_PORT="${BASH_REMATCH[1]}"
    echo "The port is: $AVD_PORT"
else
    echo "No port found in the string."
fi

# 确保输出目录存在
function EnsureDirectoryExists {
    local directory="$1"
    if [ ! -d "$directory" ]; then
        mkdir -p "$directory" && echo "Directory '$directory' created successfully." || echo "Failed to create directory '$directory'."
    else
        echo "Directory '$directory' already exists."
    fi
}
EnsureDirectoryExists "$RESULT_DIR"

# 检查模拟器是否已经启动
function IsEmulatorRunning {
    local port="$1"
    adb devices | grep -q "emulator-$port"
}

# 启动模拟器
function StartEmulator {
    local avdName="$1"
    local port="$2"
    local headless="$3"
    local headlessFlag=""
    if [ "$headless" = true ]; then
        headlessFlag="-no-window"
    fi
    emulator -port "$port" -avd "$avdName" -read-only $headlessFlag &
}

# 等待设备准备好
function WaitForDevice {
    local deviceSerial="$1"

    function GetBootAnimStatus {
        local deviceSerial="$1"
        adb -s "$deviceSerial" shell getprop init.svc.bootanim | tr -d '\r'
    }

    echo "Waiting for device $deviceSerial to be ready..."

    adb -s "$deviceSerial" wait-for-device

    bootAnimStatus=$(GetBootAnimStatus "$deviceSerial")
    i=0

    while [ "$bootAnimStatus" != "stopped" ]; do
        echo "   Waiting for emulator ($deviceSerial) to fully boot (#$i Times) ..."
        sleep 5
        ((i++))
        if [ "$i" -eq 10 ]; then
            echo "Cannot connect to the device: ($deviceSerial) after (#$i Times)..."
            break
        fi
        bootAnimStatus=$(GetBootAnimStatus "$deviceSerial")
    done

    if [ "$bootAnimStatus" = "stopped" ]; then
        echo "Device $deviceSerial is fully booted."
    fi
}

# # 判断设备是否启动，如果已经启动则杀死
# while IsEmulatorRunning "$AVD_PORT"; do
#     echo "Emulator on port $AVD_PORT is already running."
#     echo "Stopping emulator $DEVICE_SERIAL..."
#     adb -s "$DEVICE_SERIAL" emu kill
#     sleep 3
# done

# 启动模拟器
echo "Starting emulator..."
StartEmulator "$AVD_NAME" "$AVD_PORT" "$HEADLESS"
WaitForDevice "emulator-$AVD_PORT"

sleep 3

# 获取应用包名
packageInfo=$(aapt dump badging "$APK_PATH")
PACKAGE_NAME=$(echo "$packageInfo" | grep -oP "package: name='\K[^']+")

echo "** PROCESSING APP (${DEVICE_SERIAL}): $PACKAGE_NAME"

# 授予权限
adb -s "$DEVICE_SERIAL" root
sleep 2

# 检查设备连接状态
echo "** CHECKING DEVICE CONNECTION ($DEVICE_SERIAL)"
device=$(adb devices | grep "$DEVICE_SERIAL")
if [ -z "$device" ]; then
    echo "Device $DEVICE_SERIAL not connected."
    exit 1
fi

# 最大重试次数
MAX_RETRIES=3
attempt=0

# 有时候droidbot会因为无法清除日志而启动失败，加-d可以解决这个问题
while [ "$attempt" -lt "$MAX_RETRIES" ]; do
    adb -s "$DEVICE_SERIAL" logcat -c -d
    if [ $? -eq 0 ]; then
        echo "success"
        break
    else
        ((attempt++))
        echo "failed,retry $attempt Times."
    fi
    sleep 1
done

# 从第8个参数开始，将其余的作为 APK 文件列表
APK_FILES=("${@:8}")
# 安装应用并自动授予权限
# 使用 adb install-multiple 命令安装 APK 文件
adb -s $DEVICE_SERIAL install-multiple -g "${APK_FILES[@]}"&> $RESULT_DIR/install.log

sleep 2
#anroid 11 通过以下命令授予存储权限
adb -s $DEVICE_SERIAL shell appops set --uid $PACKAGE_NAME MANAGE_EXTERNAL_STORAGE allow
sleep 2
echo "install app"

# execute pre if necessary
if [[ $LOGIN_SCRIPT != "" ]]
then
    echo "** APP LOGIN (${DEVICE_SERIAL})"

    # enable if use the login script
    # sleep 5 # wait for a few seconds before installation to avoid such error: "adb: connect error for write: closed"
    python3 $LOGIN_SCRIPT ${DEVICE_SERIAL} hybirddroid 2>&1 | tee $RESULT_DIR/login.log

    # enable if use the snapshot (already login, do not need to install the app)
    echo " *** Login SUCCESS ****" >> $RESULT_DIR/login.log
fi

# 获取注册的广播接收器名称
RECEIVER_NAME=$(adb -s $DEVICE_SERIAL shell pm dump $PACKAGE_NAME | grep "jacocoInstrument.SMSInstrumentedReceiver" | awk '{print $2}')
# 打印获取到的接收器名称
echo "Registered Broadcast Receiver: $RECEIVER_NAME"
sleep 1

# start logcat
echo "** START LOGCAT (${AVD_SERIAL}) "
adb -s $AVD_SERIAL logcat -c
adb -s $AVD_SERIAL logcat AndroidRuntime:E CrashAnrDetector:D System.err:W CustomActivityOnCrash:E ACRA:E WordPress-EDITOR:E *:F *:S > $result_dir/logcat.log &

# 启动覆盖率数据转储
echo "** START COVERAGE ($DEVICE_SERIAL)"
bash -x ./dump_coverage.sh $DEVICE_SERIAL $PACKAGE_NAME $RESULT_DIR $RECEIVER_NAME & 2>&1 | tee -a "$RESULT_DIR/coverage.log"
echo "Coverage dump started in background"

adb -s "$DEVICE_SERIAL" shell date "+%Y-%m-%d-%H:%M:%S" > "$RESULT_DIR/hybirddroid_testing_time_on_emulator.txt"

# 运行Python脚本并传递参数
if [ "$TEST_TIME" != "None" ]; then
    python3 "$PYTHON_SCRIPT" -d "$DEVICE_SERIAL" -a "$APK_PATH" -o "$RESULT_DIR" -timeout "$TEST_TIME" -grant_perm -is_emulator -accessibility_auto 2>&1 | tee -a "$LOG_FILE"
elif [ "$EVENT_COUNT" != "None" ]; then
    python3 "$PYTHON_SCRIPT" -d "$DEVICE_SERIAL" -a "$APK_PATH" -o "$RESULT_DIR" -count "$EVENT_COUNT" -grant_perm -is_emulator 2>&1 | tee -a "$LOG_FILE"
fi

adb -s "$DEVICE_SERIAL" shell date "+%Y-%m-%d-%H:%M:%S" >> "$RESULT_DIR/hybirddroid_testing_time_on_emulator.txt"

# stop coverage dumping
echo "** STOP COVERAGE (${DEVICE_SERIAL})"
kill `ps aux | grep "dump_coverage.sh ${DEVICE_SERIAL}" | grep -v grep |  awk '{print $2}'`

# stop logcat
echo "** STOP LOGCAT (${DEVICE_SERIAL})"
kill `ps aux | grep "${DEVICE_SERIAL} logcat" | grep -v grep | awk '{print $2}'`

echo "Stopping emulator $DEVICE_SERIAL..."
sleep 5
adb -s $DEVICE_SERIAL emu kill
echo "@@@@@@ Finish ($DEVICE_SERIAL): $PACKAGE_NAME @@@@@@@"