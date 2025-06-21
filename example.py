#!/usr/bin/env python3
"""
Example script demonstrating OptionsFlowX usage.

This script shows how to initialize and run the OptionsFlowX
high-frequency trading scanner.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import OptionsFlowX
from src.utils.config import load_config


async def main():
    """Main example function."""
    print("OptionsFlowX - High-Frequency Trading Scanner")
    print("=" * 50)
    
    try:
        # Load configuration
        print("Loading configuration...")
        config = load_config()
        
        # Initialize OptionsFlowX
        print("Initializing OptionsFlowX...")
        scanner = OptionsFlowX(config=config)
        
        # Get initial status
        status = scanner.get_status()
        print(f"Initial status: {status['is_running']}")
        
        # Start scanning
        print("Starting OptionsFlowX scanner...")
        print("Press Ctrl+C to stop")
        
        # Run for a limited time in example mode
        await asyncio.wait_for(scanner.start(), timeout=30)
        
    except asyncio.TimeoutError:
        print("\nExample completed after 30 seconds")
    except KeyboardInterrupt:
        print("\nStopping OptionsFlowX...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'scanner' in locals():
            await scanner.stop()


if __name__ == "__main__":
    asyncio.run(main()) 