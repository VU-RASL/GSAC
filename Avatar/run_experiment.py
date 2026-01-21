import subprocess

def run_command(command):
    """Runs a shell command and waits for it to complete."""
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {command}")
        exit(1)  # Exit if a command fails

# Run commands sequentially

#########################   Original ##############################################
# male3_cmd = 'python main.py --base=./configs/gaussians_docker_male3.yaml '
# male4_cmd = 'python main.py --base=./configs/gaussians_docker_male4.yaml '
# female3_cmd = 'python main.py --base=./configs/gaussians_docker_female3.yaml '
# female4_cmd = 'python main.py --base=./configs/gaussians_docker_female4.yaml '


# run_command(male3_cmd )
# run_command(male4_cmd )
# run_command(female3_cmd)
# run_command(female4_cmd)

# run_command(male3_cmd )
# run_command(male4_cmd )
# run_command(female3_cmd)
# run_command(female4_cmd)

# run_command(male3_cmd )
# run_command(male4_cmd )
# run_command(female3_cmd)
# run_command(female4_cmd)

# run_command(male3_cmd )
# run_command(male4_cmd )
# run_command(female3_cmd)
# run_command(female4_cmd)

# run_command(male3_cmd )
# run_command(male4_cmd )
# run_command(female3_cmd)
# run_command(female4_cmd)




#########################   Custom ##############################################
male3_cmd = 'python main.py --base=./configs/gaussians_docker_custom_male3.yaml '
male4_cmd = 'python main.py --base=./configs/gaussians_docker_custom_male4.yaml '
female3_cmd = 'python main.py --base=./configs/gaussians_docker_custom_female3.yaml '
female4_cmd = 'python main.py --base=./configs/gaussians_docker_custom_female4.yaml '


run_command(male3_cmd )
run_command(male4_cmd )
run_command(female3_cmd)
run_command(female4_cmd)

run_command(male3_cmd )
run_command(male4_cmd )
run_command(female3_cmd)
run_command(female4_cmd)

run_command(male3_cmd )
run_command(male4_cmd )
run_command(female3_cmd)
run_command(female4_cmd)

run_command(male3_cmd )
run_command(male4_cmd )
run_command(female3_cmd)
run_command(female4_cmd)

run_command(male3_cmd )
run_command(male4_cmd )
run_command(female3_cmd)
run_command(female4_cmd)