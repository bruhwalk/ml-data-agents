"""Start Label Studio, create project, and import tasks.

Usage::

    python start_ls.py "My Project" "World,Sports,Business,Sci/Tech"
    python start_ls.py "My Project" "World,Sports,Business,Sci/Tech" data/annotation/labelstudio_tasks.json

Starts Label Studio on localhost:8080, creates a project with the given labels,
imports tasks, and opens the browser.

Prerequisites:
    pip install label-studio label-studio-sdk
"""

import io
import json
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))


def _find_label_studio_cmd() -> list[str] | None:
    """Find the label-studio executable. Returns command list or None."""
    import shutil

    # 1. Try label-studio on PATH
    if shutil.which("label-studio"):
        return ["label-studio"]

    # 2. Try label-studio.exe in Python Scripts dir (Windows)
    if sys.platform == "win32":
        scripts_dir = Path(sys.executable).parent / "Scripts"
        ls_exe = scripts_dir / "label-studio.exe"
        if ls_exe.exists():
            return [str(ls_exe)]

    # 3. Try python -m label_studio (works on some installs)
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import label_studio; print('ok')"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return [sys.executable, "-m", "label_studio"]
    except (subprocess.TimeoutExpired, OSError):
        pass

    return None


def wait_for_ls(url: str = "http://localhost:8080", timeout: int = 60) -> bool:
    """Wait for Label Studio to become available."""
    import urllib.request
    import urllib.error

    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url + "/api/health", timeout=2)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    return False


def main(
    project_name: str = "Annotation",
    labels_str: str = "",
    tasks_path: str = "data/annotation/labelstudio_tasks.json",
    port: int = 8080,
) -> None:
    labels = [l.strip() for l in labels_str.split(",") if l.strip()]
    tasks_file = ROOT / tasks_path
    config_file = ROOT / "data" / "annotation" / "labelstudio_config.xml"

    if not tasks_file.exists():
        print(f"Tasks file not found: {tasks_file}")
        print("Run export_ls.py first.")
        sys.exit(1)

    url = f"http://localhost:{port}"

    # Check if LS is already running
    already_running = False
    try:
        import urllib.request
        urllib.request.urlopen(url + "/api/health", timeout=2)
        already_running = True
        print(f"Label Studio already running at {url}")
    except (Exception,):
        pass

    if not already_running:
        print(f"Starting Label Studio on port {port}...")

        # Find the label-studio executable
        ls_cmd = _find_label_studio_cmd()
        if ls_cmd is None:
            print("label-studio not found. Install with: pip install label-studio")
            sys.exit(1)

        # Start LS in background
        subprocess.Popen(
            ls_cmd + ["start", "--port", str(port), "--no-browser"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

        if not wait_for_ls(url, timeout=90):
            print("Label Studio failed to start within timeout.")
            print("Try starting manually: label-studio start")
            sys.exit(1)

        print(f"Label Studio started at {url}")

    # Use label-studio-sdk to create project and import
    try:
        from label_studio_sdk import Client

        # Try to get API key from env or default
        import os
        api_key = os.environ.get("LABEL_STUDIO_API_KEY", "")

        if not api_key:
            print("\nLabel Studio is running. To connect via API:")
            print(f"  1. Open {url} in browser")
            print("  2. Sign up / log in")
            print("  3. Go to Account & Settings -> Access Token")
            print("  4. Set LABEL_STUDIO_API_KEY environment variable")
            print(f"\nOr import tasks manually:")
            print(f"  - Tasks JSON: {tasks_file}")
            if config_file.exists():
                print(f"  - Project config: {config_file}")
            webbrowser.open(url)
            return

        ls = Client(url=url, api_key=api_key)

        # Build labeling config
        if config_file.exists():
            config_xml = config_file.read_text(encoding="utf-8")
        else:
            choices = "\n".join(f'    <Choice value="{l}" />' for l in labels)
            config_xml = f"""<View>
  <Text name="text" value="$text" />
  <Choices name="label" toName="text" choice="single" showInLine="true">
{choices}
  </Choices>
</View>"""

        # Create project
        project = ls.start_project(
            title=project_name,
            label_config=config_xml,
        )
        print(f"Created project: {project_name} (id={project.id})")

        # Import tasks
        with open(tasks_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        project.import_tasks(tasks)
        print(f"Imported {len(tasks)} tasks")

        project_url = f"{url}/projects/{project.id}"
        print(f"\nProject URL: {project_url}")
        webbrowser.open(project_url)

    except ImportError:
        print("label-studio-sdk not installed. Install with: pip install label-studio-sdk")
        print(f"\nImport tasks manually at {url}:")
        print(f"  - Tasks JSON: {tasks_file}")
        if config_file.exists():
            print(f"  - Project config: {config_file}")
        webbrowser.open(url)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "Annotation"
    labels = sys.argv[2] if len(sys.argv) > 2 else ""
    tasks = sys.argv[3] if len(sys.argv) > 3 else "data/annotation/labelstudio_tasks.json"
    main(name, labels, tasks)
