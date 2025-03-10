#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import base64
from typing import Tuple, Optional
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
)

class KeyManager:
    def __init__(self, keys_dir: str, registry_path: str):
        """Initialize KeyManager with paths to keys directory and registry."""
        self.keys_dir = keys_dir
        self.registry_path = registry_path
        self.private_key_path = os.path.join(keys_dir, "private_key.pem")
        self.public_key_path = os.path.join(keys_dir, "public_key.pem")
        
    def generate_key_pair(self, force: bool = False) -> Tuple[str, str]:
        """
        Generate a new RSA key pair and save to files.
        
        Args:
            force: If True, overwrite existing keys. If False, raise error if keys exist.
        
        Returns:
            Tuple of (private_key_path, public_key_path)
            
        Raises:
            FileExistsError: If keys already exist and force=False
        """
        if not force and (os.path.exists(self.private_key_path) or os.path.exists(self.public_key_path)):
            raise FileExistsError(
                f"Keys already exist at {self.private_key_path} or {self.public_key_path}. "
                "Use force=True to overwrite."
            )
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Get public key
        public_key = private_key.public_key()
        
        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Serialize public key
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Save keys to files
        os.makedirs(self.keys_dir, exist_ok=True)
        with open(self.private_key_path, 'wb') as f:
            f.write(private_pem)
        with open(self.public_key_path, 'wb') as f:
            f.write(public_pem)
            
        return self.private_key_path, self.public_key_path
        
    def load_or_generate(self) -> Tuple[str, str]:
        """
        Load existing keys if present, otherwise generate new ones.
        
        Returns:
            Tuple of (private_key_path, public_key_path)
        """
        try:
            if os.path.exists(self.private_key_path) and os.path.exists(self.public_key_path):
                return self.private_key_path, self.public_key_path
            return self.generate_key_pair()
        except Exception as e:
            print(f"Error loading/generating keys: {e}")
            raise
    
    def load_private_key(self) -> RSAPrivateKey:
        """Load private key from file."""
        if not os.path.exists(self.private_key_path):
            raise FileNotFoundError(f"Private key not found at {self.private_key_path}")
            
        with open(self.private_key_path, 'rb') as f:
            private_pem = f.read()
            
        return load_pem_private_key(private_pem, password=None)
    
    def load_public_key_from_registry(self, hostname: str) -> Optional[RSAPublicKey]:
        """Load public key for a specific hostname from the registry."""
        if not os.path.exists(self.registry_path):
            raise FileNotFoundError(f"Registry not found at {self.registry_path}")
            
        with open(self.registry_path, 'r') as f:
            registry = json.load(f)
            
        for landmark in registry.get('landmarks', []):
            if landmark['hostname'] == hostname:
                public_key_b64 = landmark['public_key']
                public_key_bytes = base64.b64decode(public_key_b64)
                return load_pem_public_key(public_key_bytes)
                
        return None
    
    def sign_message(self, message: bytes) -> bytes:
        """Sign a message using the private key."""
        private_key = self.load_private_key()
        
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature
    
    def verify_signature(self, hostname: str, message: bytes, signature: bytes) -> bool:
        """Verify a signature using the public key from the registry."""
        public_key = self.load_public_key_from_registry(hostname)
        if not public_key:
            raise ValueError(f"No public key found for hostname: {hostname}")
            
        try:
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
