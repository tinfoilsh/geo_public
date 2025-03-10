#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import base64
from key_manager import KeyManager

def main():
    """Generate key pairs for each landmark in the registry."""
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    registry_path = os.path.join(current_dir, "landmark_registry.json")
    
    # Load registry
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    # Create keys directory if it doesn't exist
    keys_dir = os.path.join(current_dir, "keys")
    os.makedirs(keys_dir, exist_ok=True)
    
    # Generate keys for each landmark
    for landmark in registry['landmarks']:
        hostname = landmark['hostname']
        landmark_keys_dir = os.path.join(keys_dir, hostname)
        os.makedirs(landmark_keys_dir, exist_ok=True)
        
        # Initialize key manager for this landmark
        key_manager = KeyManager(landmark_keys_dir, registry_path)
        
        try:
            # Generate or load keys
            private_key_path, public_key_path = key_manager.load_or_generate()
            
            # Read public key and encode as base64
            with open(public_key_path, 'rb') as f:
                public_key_bytes = f.read()
                public_key_b64 = base64.b64encode(public_key_bytes).decode('utf-8')
            
            # Update registry with public key
            landmark['public_key'] = public_key_b64
            
            print(f"✅ Generated/loaded keys for {hostname}:")
            print(f"  - Private key: {private_key_path}")
            print(f"  - Public key: {public_key_path}")
            
        except Exception as e:
            print(f"❌ Error generating keys for {hostname}: {e}")
            raise
    
    # Save updated registry with public keys
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    print("\n✨ Updated landmark_registry.json with public keys")

if __name__ == "__main__":
    main()
