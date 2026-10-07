import subprocess
import sys

requirements_file = "requirements.txt"

packages = []
pip_options = []

with open(requirements_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        # Preserve options such as --extra-index-url
        if line.startswith("-"):
            pip_options.append(line)
        else:
            packages.append(line)

failed = []

for package in packages:
    print("\n" + "=" * 80)
    print(f"Installing: {package}")
    print("=" * 80)

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        package,
        # *pip_options,
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nFAILED: {package}")
        failed.append(package)
    else:
        print(f"\nOK: {package}")

print("\n" + "=" * 80)
print("Installation finished.")

if failed:
    print("\nPackages that failed:")
    for package in failed:
        print(f"  {package}")
else:
    print("\nAll packages installed successfully.")