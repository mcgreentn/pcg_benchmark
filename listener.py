import os
import time
import subprocess

from config import get_output_dir

class Listener:
    def __init__(self, listener_id, check_interval=2, run_dir=None):
        self.listener_id = listener_id
        # Matches driver.run_dir, which lives under the untracked run-output directory.
        self.run_dir = run_dir or os.path.join(get_output_dir(), "run")
        self.running = True
        self.check_interval = check_interval

    def listen(self):
        print(f"Listener {self.listener_id} started. Monitoring {self.run_dir} for run-{self.listener_id}.wts...")
        while self.running:
            weights_file = f"run-{self.listener_id}.wts"
            weights_path = os.path.join(self.run_dir, weights_file)
            if os.path.exists(weights_path):
                print(f"Found weights file: {weights_path}. Running runner.py...")
                subprocess.run([
                    "python", "runner.py", weights_path
                ])
                os.remove(weights_path)
                print(f"Completed run and removed {weights_path}.")
            time.sleep(self.check_interval)

    def stop(self):
        self.running = False

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python listener.py <id>")
        sys.exit(1)
    listener_id = sys.argv[1]
    listener = Listener(listener_id)
    listener.listen()
