import sys
import os
import importlib.util
import glob

def check_libraries():
    """Checks if required libraries are installed."""
    required = ['numpy', 'pandas', 'pymoo', 'matplotlib', 'seaborn']
    missing = []
    for lib in required:
        if importlib.util.find_spec(lib) is None:
            missing.append(lib)
    if missing:
        print(f"Error: Missing required libraries: {', '.join(missing)}")
        print("Please install them using: pip install " + " ".join(missing))
        sys.exit(1)
    print("Libraries check passed.")

def check_directories():
    """Checks if essential directories exist."""
    required_dirs = ['data', 'config', 'src']
    for d in required_dirs:
        if not os.path.exists(d):
            print(f"Warning: Directory '{d}' not found. Creating it...")
            os.makedirs(d, exist_ok=True)
    print("Directory structure check passed.")

def select_config():
    """Allows the user to select a configuration file from the config directory."""
    config_dir = 'config'
    # Find all json files in config dir (non-recursive)
    files = glob.glob(os.path.join(config_dir, '*.json'))
    
    if not files:
        print(f"Error: No configuration files found in '{config_dir}'.")
        sys.exit(1)
        
    print("\nAvailable Configurations:")
    for i, f in enumerate(files):
        print(f"{i+1}. {os.path.basename(f)}")
        
    choice = input("\nSelect configuration (number) [1]: ").strip()
    
    if not choice:
        idx = 0
    else:
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(files):
                raise ValueError
        except ValueError:
            print("Invalid selection. Using default (1).")
            idx = 0
            
    selected = files[idx]
    print(f"Selected: {selected}\n")
    return selected

if __name__ == "__main__":
    print("=== MOEA/D for Association Rule Mining ===")
    
    check_libraries()
    check_directories()
    
    config_file = select_config()
    
    try:
        from orchestrator import Orchestrator
        orch = Orchestrator(config_file)
        orch.run()
    except KeyboardInterrupt:
        print("\n\nExecution cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred during execution:\n{e}")
        import traceback
        traceback.print_exc()
