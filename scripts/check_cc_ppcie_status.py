#!/usr/bin/env python3
import sys

# If needed, adapt this to your environment/package structure
try:
    from verifier.nvml import NvmlHandler
except ImportError as ex:
    print(f"Error: Could not import NvmlHandler. Make sure you're in the correct virtual environment.\n{ex}")
    sys.exit(1)

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

def colorize_result(is_enabled: bool) -> str:
    """Return a colored True/False string plus emoji for quick scanning."""
    if is_enabled:
        return f"{GREEN}True ✅{RESET}"
    else:
        return f"{RED}False ❌{RESET}"

def main():
    try:
        # Initialize NVML
        NvmlHandler.init_nvml()

        # Check CC (Confidential Compute) and PPCIE (Protected PCIe) status
        cc_enabled = NvmlHandler.is_cc_enabled()
        ppcie_enabled = NvmlHandler.is_ppcie_mode_enabled()

        # Colorize results
        cc_colored_result = colorize_result(cc_enabled)
        ppcie_colored_result = colorize_result(ppcie_enabled)

        print("------------------------------------------------------------")
        print("Statuses:")
        print(f"Confidential Computing Enabled : {cc_colored_result}")
        print(f"Protected PCIe Mode Enabled   : {ppcie_colored_result}")
        print("------------------------------------------------------------")

        # If you'd like to fail if neither feature is enabled, uncomment:
        # if not cc_enabled and not ppcie_enabled:
        #     sys.exit("Neither CC nor PPCIE is enabled. Exiting with error code 1.")

    except Exception as error:
        print(f"Error checking CC/PPCIE status: {error}")
        sys.exit(1)

if __name__ == "__main__":
    main()