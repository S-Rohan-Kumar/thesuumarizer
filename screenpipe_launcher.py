"""
screenpipe_launcher.py - A module to open the Screenpipe executable file on Windows in a separate terminal window
"""

import os
import subprocess
import sys
if sys.platform == 'win32':  # 'win32' indicates Windows
    import winreg
    # Use winreg here for Windows-specific functionality
else:
    # Handle the non-Windows case (e.g., skip or use an alternative)
    pass
from pathlib import Path
from typing import Optional, List


def search_directory_for_screenpipe(directory: Path) -> Optional[Path]:
    """
    Recursively search a directory for the Screenpipe executable.
    
    Args:
        directory: The directory to search in
        
    Returns:
        Path to the executable if found, None otherwise
    """
    try:
        if not directory.exists() or not directory.is_dir():
            return None
            
        # Check for Screenpipe.exe directly in this directory
        exe_path = directory / "Screenpipe.exe"
        if exe_path.exists() and exe_path.is_file():
            return exe_path
            
        # Try searching for any file containing "screenpipe" in the name
        for item in directory.glob("*screenpipe*.exe"):
            if item.is_file():
                return item
                
        # Search subdirectories recursively (but not too deep to avoid long searches)
        for item in directory.iterdir():
            if item.is_dir():
                result = search_directory_for_screenpipe(item)
                if result:
                    return result
                    
        return None
    except (PermissionError, OSError):
        # Skip directories we can't access
        return None


def get_installed_programs_from_registry() -> List[str]:
    """
    Get a list of installation directories from the Windows registry.
    
    Returns:
        List of potential installation directories
    """
    installation_dirs = []
    
    # Registry keys to check for installed software
    registry_keys = [
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
        r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
    ]
    
    for base_key in registry_keys:
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base_key) as key:
                for i in range(winreg.QueryInfoKey(key)[0]):
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        with winreg.OpenKey(key, subkey_name) as subkey:
                            try:
                                display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                                
                                # If it's related to Screenpipe
                                if "screenpipe" in display_name.lower():
                                    try:
                                        install_location = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                                        if install_location and len(install_location) > 0:
                                            installation_dirs.append(install_location)
                                    except (WindowsError, IndexError):
                                        pass
                            except (WindowsError, IndexError):
                                continue
                    except (WindowsError, IndexError):
                        continue
        except (WindowsError, IndexError):
            pass
            
    return installation_dirs


def find_screenpipe_executable() -> Optional[Path]:
    """
    Find the Screenpipe executable on Windows with thorough searching.
    
    Returns:
        Path to the Screenpipe executable if found, None otherwise.
    """
    # Check common installation directories
    common_dirs = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Screenpipe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Screenpipe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Screenpipe",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Screenpipe",
        Path(os.environ.get("APPDATA", "")) / "Screenpipe",
        Path(os.environ.get("APPDATA", "")) / "Local" / "Screenpipe",
    ]
    
    # Add direct executable paths
    direct_exes = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Screenpipe" / "Screenpipe.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Screenpipe" / "Screenpipe.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Screenpipe" / "Screenpipe.exe",
    ]
    
    # Check desktop shortcuts
    desktop = Path(os.path.join(os.path.expanduser("~"), "Desktop"))
    
    # Check user's download folder
    downloads = Path(os.path.join(os.path.expanduser("~"), "Downloads"))
    
    # Check Start Menu folders
    start_menu_dirs = [
        Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
        Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    ]
    
    # Add registry-based paths
    registry_dirs = [Path(dir_path) for dir_path in get_installed_programs_from_registry()]
    
    # First check the direct executable paths for efficiency
    for exe_path in direct_exes:
        if exe_path.exists():
            return exe_path
    
    # Then check all the search directories
    search_dirs = common_dirs + registry_dirs + start_menu_dirs + [desktop, downloads]
    for directory in search_dirs:
        result = search_directory_for_screenpipe(directory)
        if result:
            return result
    
    # Last resort: search system drives for portable installations
    # Limited to common portable app directories to avoid excessive searching
    drives = []
    for drive_letter in "CDEFG":  # Limit to primary drives
        drive_path = f"{drive_letter}:\\"
        if os.path.exists(drive_path):
            drives.append(Path(drive_path))
    
    for drive in drives:
        # Common places for portable apps
        portable_dirs = [
            drive / "PortableApps",
            drive / "Portable",
            drive / "Programs",
            drive / "Applications"
        ]
        
        for portable_dir in portable_dirs:
            if portable_dir.exists():
                result = search_directory_for_screenpipe(portable_dir)
                if result:
                    return result
    
    return None


def launch_screenpipe_in_terminal(executable_path: Optional[Path] = None) -> bool:
    """
    Launch the Screenpipe application in a separate terminal window.
    
    Args:
        executable_path: Optional path to the Screenpipe executable.
                         If None, the function will try to find it.
    
    Returns:
        True if Screenpipe was launched successfully, False otherwise.
    """
    if executable_path is None:
        print("Searching for Screenpipe executable...")
        executable_path = find_screenpipe_executable()
    
    if executable_path is None or not executable_path.exists():
        print("Screenpipe executable not found after thorough search.")
        return False
    
    try:
        print(f"Found Screenpipe at: {executable_path}")
        
        # Create a batch script to launch Screenpipe
        batch_path = Path(os.environ.get("TEMP", ".")) / "launch_screenpipe.bat"
        
        with open(batch_path, 'w') as batch_file:
            batch_file.write(f'@echo off\n')
            batch_file.write(f'echo Starting Screenpipe...\n')
            batch_file.write(f'cd /d "{executable_path.parent}"\n')
            batch_file.write(f'start "" "{executable_path}"\n')
            batch_file.write(f'echo Screenpipe has been launched.\n')
            batch_file.write(f'pause\n')
        
        # Run the batch file in a new terminal window
        subprocess.Popen(['start', 'cmd', '/c', str(batch_path)], shell=True)
        
        print("Screenpipe has been launched in a new terminal window.")
        return True
        
    except Exception as e:
        print(f"Error launching Screenpipe: {e}")
        return False


def main():
    """
    Main function to run when the script is executed directly.
    """
    print("Screenpipe Launcher Utility")
    print("===========================")
    
    # Check if a path was provided as a command-line argument
    executable_path = None
    if len(sys.argv) > 1:
        potential_path = Path(sys.argv[1])
        if potential_path.exists() and potential_path.is_file():
            executable_path = potential_path
            print(f"Using provided executable path: {executable_path}")
    
    # Launch Screenpipe
    success = launch_screenpipe_in_terminal(executable_path)
    
    if not success:
        print("\nFailed to launch Screenpipe.")
        print("You can try specifying the path to Screenpipe.exe as a command-line argument:")
        print("  python screenpipe_launcher.py \"C:\\Path\\To\\Screenpipe.exe\"")
        input("Press Enter to exit...")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
