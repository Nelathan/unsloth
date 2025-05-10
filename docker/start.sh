#!/bin/bash
set -e # Exit the script if any statement returns a non-true return value

# Execute script if exists
execute_script() {
  local script_path=$1
  local script_msg=$2
  if [[ -f ${script_path} ]]; then
    echo "${script_msg}"
    bash ${script_path}
  fi
}

# Setup ssh
setup_ssh() {
  if [[ $PUBLIC_KEY ]]; then
    echo "Setting up SSH..."
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >>~/.ssh/authorized_keys
    chmod 700 -R ~/.ssh

    if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
      ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -q -N ''
      echo "ED25519 key fingerprint:"
      ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
    fi

    service ssh start
  fi
}

# Export env vars
export_env_vars() {
  echo "Exporting environment variables..."
  printenv | grep -E '^RUNPOD_|^PATH=|^_=' | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >>/etc/rp_environment
  echo 'source /etc/rp_environment' >>~/.bashrc
}

echo "Pod Started"

setup_ssh

export_env_vars

echo "Start script(s) finished, pod is ready to use."

# if folder /workspace/unsloth not exists, clone the repository
if [ ! -d "/workspace/unsloth" ]; then
  echo "Cloning unsloth repository..."
  git clone https://github.com/Nelathan/unsloth.git /workspace/unsloth
fi

# echo "Executing train script..."
# python /workspace/unsloth/recipes/Llama-3.1-8B-sugarquill.py

sleep infinity
