import os
import subprocess
import argparse
from pathlib import Path


def get_properties_by_app():
    """
    Parse the directory structure and organize properties by app name.
    Returns a dictionary with app names as keys and lists of _new.py files as values.
    """
    properties_dir = Path("Properties")
    app_properties = {}

    # Go through each app directory
    for app_dir in properties_dir.iterdir():
        if app_dir.is_dir():
            app_name = app_dir.name
            app_properties[app_name] = []

            # Get all *_new.py files
            for file_path in app_dir.glob("*_new.py"):
                # Skip duplicate files (like markor_1019_new 2.py)
                if " " not in file_path.name:
                    app_properties[app_name].append(file_path)

    return app_properties


def run_kea_test(app_name, property_files, timeout=None):
    """
    Run Kea test for a specific app with its property files.

    Args:
        app_name: Name of the app to test
        property_files: List of property file paths
        timeout: Optional timeout in seconds
    """
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    property_args = [str(file) for file in property_files]

    # Construct the command
    cmd = ["kea", "-f"] + property_args + ["-a", f"apk/{app_name}.apk", "-p", "random", "-o", f"output_random/{app_name}_output"]
    # cmd = ["kea", "-f"] + property_args + ["-a", f"apk/{app_name}.apk", "-p", "llm", "-o", f"output/{app_name}_output"]

    # Add timeout if specified
    if timeout:
        cmd.extend(["-t", str(timeout)])

    # Print command for logging
    print(f"Running: {' '.join(cmd)}")

    # Execute command and show real-time output
    try:
        # Use Popen to capture output in real-time
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   universal_newlines=True)

        # Print output as it comes
        output_lines = []
        for line in process.stdout:
            print(line, end='')  # Print output in real-time
            output_lines.append(line)

        # Wait for process to complete and get return code
        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd, output=''.join(output_lines))

        print(f"Finished testing {app_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error testing {app_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run Kea tests on multiple apps with multiple properties')
    parser.add_argument('--timeout', type=int, help='Timeout in seconds for each test')
    parser.add_argument('--apps', nargs='*', help='Specific apps to test (if not specified, all apps will be tested)')
    # Keep threads parameter for backward compatibility, but it won't be used
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of concurrent tests to run (ignored in sequential mode)')
    args = parser.parse_args()

    # Get properties organized by app
    app_properties = get_properties_by_app()

    # Filter apps if specified
    if args.apps:
        app_properties = {app: props for app, props in app_properties.items() if app in args.apps}

    # Run tests sequentially
    for app_name, property_files in app_properties.items():
        if property_files:  # Only test if there are property files
            run_kea_test(app_name, property_files, args.timeout)

    print("All tests have been completed.")


if __name__ == "__main__":
    main()