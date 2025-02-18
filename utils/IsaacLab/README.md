# External Tools

This folder is reserved for cloning the IsaacLab repository. To get started, follow these steps:

## Cloning the Repo
```bash
git clone https://github.com/your-org/IsaacLab.git external_tools/IsaacLab
```

## Usage
Ensure that your scripts reference this directory correctly. Example:
```python
import sys
sys.path.append("external_tools/IsaacLab")
import isaaclab
```

## Notes
- Do **not** modify the contents of `IsaacLab` directly; instead, contribute to the upstream repo.
- Ensure you have the necessary dependencies installed.

## Updating the Repo
```bash
cd external_tools/IsaacLab
git pull origin main
```

