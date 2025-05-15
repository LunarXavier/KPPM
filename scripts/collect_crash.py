from argparse import ArgumentParser
import os
import re
from collections import defaultdict

def extract_crash_info(log_content):
    # 正则表达式匹配崩溃信息，包括多行堆栈跟踪，限制堆栈跟踪行以时间戳和 "E AndroidRuntime" 开头
    crash_regex = re.compile(
        r'(\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\s*\d+\s*\d+ E AndroidRuntime: FATAL EXCEPTION:.*?)\n((?:\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\s*\d+\s*\d+ E AndroidRuntime:.*?\n)*)',
        re.DOTALL
    )
    return crash_regex.findall(log_content)

def clean_stack_trace(crash_log):
    """
    从崩溃日志中提取堆栈跟踪，并去除时间戳信息。
    """
    # 按行分割堆栈跟踪
    stack_trace = crash_log.strip().splitlines()
    
    # 清理每一行，去除时间戳
    cleaned_trace = []
    for line in stack_trace:
        # 分割行，去掉时间戳（假设时间戳在第一部分）
        parts = line.split(' ', 5)  # 只分割前5个部分
        if len(parts) > 5:
            cleaned_line = parts[5]  # 获取去掉时间戳后的部分
            cleaned_trace.append(cleaned_line)
    
    return cleaned_trace

def read_logcat_file(file_path):
    crash_info = []  
    unique_crash_info = set()

    with open(file_path, 'r') as file:
        log_content = file.read()        
        # 找到 "--------- beginning of crash" 的行并截取之后的内容
        start_index = log_content.find("--------- beginning of crash")
        if start_index != -1:
            log_content = log_content[start_index + len("--------- beginning of crash\n"):]  # 从下一行开始

        crashes = extract_crash_info(log_content)
        for crash in crashes:
            crash_string = f"{crash[0].strip()}\n{crash[1].strip()}"  # 合并为一个字符串
            crash_info.append(crash_string)  # 添加到列表中
            
            cleaned_trace = []
            # 删除 crash[1].strip() 的第一行内容
            crash_lines = crash[1].strip().split('\n')
            if len(crash_lines) > 2:
                # 过滤掉第一行
                for line in crash_lines:
                    # 提取 E AndroidRuntime: 后的内容
                    error_message = line.split('E AndroidRuntime: ', 1)[1]
                    cleaned_trace.append(f"E AndroidRuntime: {error_message}")
                filtered_crash_string = '\n'.join(cleaned_trace[2:])
            else:
                # 如果只有一行，结果为空
                filtered_crash_string = ''

            # 合并最终结果
            stack_trace_string = f"{filtered_crash_string}".strip()
            unique_crash_info.add(stack_trace_string)

    return crash_info, unique_crash_info

def get_tool_name(testing_result_dir: str):
    base_name = os.path.basename(testing_result_dir)
    base_name = str(base_name.split(".apk.")[1])
    tool_name = str(base_name.split(".result")[0])
    return tool_name

def write_crash_info_to_file(crash_data, output_file):
    all_crash_num = 0 
    crash_list = {}
    crash_from_diff_tool = {'hybirddroid':0, 'fastbot':0, 'monkey':0}
    with open(output_file, 'w') as file:
        for log_file, crashes in crash_data.items():
            file.write(f"File: {log_file}\n") 
            each_crash_num = 0
            for crash in crashes:
                each_crash_num += 1
                file.write(f"find crash {each_crash_num}:\n{crash}\n")
            file.write(f"crash num: {each_crash_num}\n")
            crash_list[os.path.basename(log_file)] = each_crash_num
            all_crash_num += each_crash_num
            
            if get_tool_name(log_file) == 'hybirddroid':
                crash_from_diff_tool['hybirddroid'] += each_crash_num
            elif get_tool_name(log_file) == 'fastbot':
                crash_from_diff_tool['fastbot'] += each_crash_num
            elif get_tool_name(log_file) == 'monkey':
                crash_from_diff_tool['monkey'] += each_crash_num

        file.write(f"-------all crashes num:{all_crash_num}---------\n")
        for key, value in crash_list.items():
            file.write(f"{key}: {value}\n")
        file.write(f'-------crashes from diff tool----------\n')
        for key, value in crash_from_diff_tool.items():
            file.write(f"{key}: {value}\n")

def main(root_dir):
    all_crashes_info = {}
    all_unique_crashed_info = {}
    for subdir in os.listdir(root_dir): 
        subdir = os.path.join(root_dir,subdir)
        if os.path.isdir(subdir):
            print("subdir: ", subdir)
            logcat_directory = os.path.join(subdir,'logcat.log') if os.path.exists(os.path.join(subdir,'logcat.log')) else os.path.join(subdir,'logcat.txt') 
            all_crashes_info[subdir], all_unique_crashed_info[subdir] = read_logcat_file(logcat_directory)
    all_crashes_output_file = os.path.join(root_dir,'all_crashes_info.txt')  
    all_unique_crashes_output_file = os.path.join(root_dir,'all_unqiue_crashes_info.txt')

    write_crash_info_to_file(all_crashes_info, all_crashes_output_file)
    write_crash_info_to_file(all_unique_crashed_info, all_unique_crashes_output_file)
    print(f"All crashes information has been written to {all_crashes_output_file}")
    print(f"All unqiue crashes information has been written to {all_unique_crashes_output_file}")

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-dir', type=str, dest="root_dir", help="test root dir")
    arg = ap.parse_args()
    root_dir = arg.root_dir 
    main(root_dir)