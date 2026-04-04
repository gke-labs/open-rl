#!/usr/bin/env python3
import os
import subprocess
import sys


def run_command(cmd, exit_on_error=True):
  try:
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    return result.stdout.strip()
  except subprocess.CalledProcessError as e:
    if exit_on_error:
      print(f"Error running command: {cmd}\n{e.stderr}")
      sys.exit(1)
    return None


def main():
  if len(sys.argv) < 2:
    print("Usage: ./gen_gcp_ssh_config.py [INSTANCE_NAME] [ALIAS_NAME (optional)]")
    sys.exit(1)

  instance = sys.argv[1]
  alias = sys.argv[2] if len(sys.argv) > 2 else instance

  # Get instance details
  print(f"# Fetching details for {instance}...")

  # Try multiple formats to get the external IP
  ip = None
  formats = [
    "value(networkInterfaces[0].accessConfigs[0].externalIp)",
    "value(networkInterfaces[].accessConfigs[].externalIp.list())",
    "value(EXTERNAL_IP)",  # Only works with 'list', but let's try 'describe' first
  ]

  for fmt in formats[:2]:
    cmd = f"gcloud compute instances describe {instance} --format='{fmt}'"
    result = run_command(cmd, exit_on_error=False)
    if result and result != "[None]":
      ip = result
      break

  if not ip or ip == "[None]":
    # Fallback to 'list'
    print("# No external IP found in 'describe'. Trying 'list' fallback...")
    cmd = f"gcloud compute instances list --filter='name={instance}' --format='value(EXTERNAL_IP)'"
    result = run_command(cmd)
    if result and result != "[None]":
      ip = result

  if not ip or ip == "[None]" or ip == "TERMINATED":
    print(f"Error: Could not find a valid external IP for '{instance}'.")
    print(f"# Debug: gcloud returned '{ip}'")
    print("Possible reasons:")
    print(" 1. The instance is stopped/terminated.")
    print(" 2. The instance does not have an external IP (check if you need IAP).")
    print(" 3. You are in the wrong project (check 'gcloud config list').")
    sys.exit(1)

  # In case multiple IPs are returned, take the first one
  ip = ip.split(",")[0].strip()

  # Get project number (needed for HostKeyAlias)
  project_id = run_command("gcloud config get-value project")
  print(f"# Current GCP Project: {project_id}")
  project_num = run_command(f"gcloud projects describe {project_id} --format='value(projectNumber)'")

  user = run_command("whoami")
  home = os.path.expanduser("~")

  config_block = f"""
# --- Added by gen_gcp_ssh_config.py ---
Host {alias}
    HostName {ip}
    User {user}_google_com
    IdentityFile {home}/.ssh/google_compute_engine
    IdentitiesOnly yes
    CheckHostIP no
    HashKnownHosts no
    StrictHostKeyChecking no
    HostKeyAlias compute.{project_num}
    UserKnownHostsFile {home}/.ssh/google_compute_known_hosts
# --------------------------------------
"""
  print("\nCopy the following block into your ~/.ssh/config:\n")
  print(config_block)


if __name__ == "__main__":
  main()
