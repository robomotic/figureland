#!/usr/bin/env python3
"""
Command line interface entry point for pip installed package.
"""

import sys
import os


def main():
    """Entry point for `figureland` CLI command."""
    # Add package directory to path
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(pkg_dir)

    # Look for config in multiple locations
    config_paths = [
        os.path.join(parent_dir, 'config'),
        os.path.join(os.getcwd(), 'config'),
        '/etc/figureland/config',
    ]

    # Add first existing config path
    for config_path in config_paths:
        if os.path.exists(config_path):
            sys.argv.extend(['--config-dir', config_path])
            break

    # Import and run hydra main
    from figureland.main import main as hydra_main
    hydra_main()


if __name__ == "__main__":
    main()
