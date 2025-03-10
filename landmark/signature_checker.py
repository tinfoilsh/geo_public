#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from nv_attestation_sdk import attestation

LOCAL_POLICY_FILE = "policies/policy.json"

def verify_token(token):
    """
    Use an Attestation client to validate the token against a local policy.
    """
    client = attestation.Attestation()
    policy_path = os.path.join(os.path.dirname(__file__), LOCAL_POLICY_FILE)
    try:
        with open(policy_path, "r") as json_file:
            policy_data = json.load(json_file)
        policy_data_str = json.dumps(policy_data)
    except FileNotFoundError:
        print("[signature_checker] Policy file not found; cannot validate token.")
        return False

    valid = client.validate_token(policy_data_str, token)
    if not valid:
        print("[signature_checker] Validation failed for the provided token.")
        return False
    else:
        print("[signature_checker] Token validated successfully.")
        return True 