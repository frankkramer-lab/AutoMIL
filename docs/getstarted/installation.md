# Installation

This guide provides a quick overview of how to install **AutoMIL** on your system

## Github

**AutoMIL** can be installed directly from its public :material-github: [GitHub](https://github.com/your/project) repository. To download the source code, open a terminal, navigate to any directory and run:

```bash
git clone https://github.com/WaibelJonas/AutoMIL.git
```

This will clone the projects source code inside a new directory called `./automil`. Navigate to this directory and install **AutoMIL** in your current python environment:

```bash
pip install .
```

!!! warning "Use a Virtual Environment"

    We strongly recommend creating a virtual environment inside the project directory.  
    This allows you to install dependencies in isolation without affecting your system-wide Python installation.

    Inside the project directory, create a virtual environment **before** installing AutoMIL:

    ```bash
    python3 -m venv .venv
    ```

    Activate the virtual environment:

    === "Bash (Linux/macOS)"

        ```bash
        source .venv/bin/activate
        ```

    === "PowerShell (Windows)"

        ```powershell
        ./venv/Scripts/Activate.ps1
        ```

    Then install AutoMIL:

    ```bash
    pip install .
    ```

## Verify Installation

To verify correct Installation, try calling the help page via:

```bash
automil -h
```

You can also check the installation path with:

=== "Bash (Linux / macOS)"

    ```bash
    which automil
    ```

=== "PowerShell (Windows)"

    ```powershell
    Get-Command automil
    ```


If the command resolves to a path or a command registry entry, the installation was successful