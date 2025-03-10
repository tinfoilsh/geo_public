#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import random
import string
import requests
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import math
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(message)s',
    datefmt='%M:%S',
    handlers=[
        logging.FileHandler(f'landmark_{os.environ.get("LANDMARK_PORT", "unknown")}.log'),
        logging.StreamHandler()
    ]
)

# Suppress Flask startup logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)
# Add this line to suppress Flask startup messages
cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *args, **kwargs: None

# Ensure Python can import from the current folder
sys.path.append(os.path.dirname(__file__))

from signature_checker import verify_token
from key_manager import KeyManager

@dataclass
class GPUData:
    """Data structure for storing GPU location data from a landmark."""
    timestamp: float
    radius: float
    confidence: float
    
class GPULocationTracker:
    """Tracks GPU location data from multiple landmarks."""
    def __init__(self):
        self.gpu_data: Dict[str, Dict[str, GPUData]] = defaultdict(dict)
        
    def add_data(self, gpu_id: str, hostname: str, data: Dict[str, Any]) -> None:
        """Add new data point for a GPU from a specific landmark."""
        self.gpu_data[gpu_id][hostname] = GPUData(
            timestamp=data["timestamp"],
            radius=data["radius"],
            confidence=data.get("confidence", 1.0)
        )
        
    def get_data(self, gpu_id: str) -> Dict[str, GPUData]:
        """Get all data points for a specific GPU."""
        return self.gpu_data.get(gpu_id, {})
        
    def find_overlap_region(self, gpu_id: str) -> Optional[List[Tuple[float, float]]]:
        """
        Calculate the region of possible GPU locations based on overlapping circles from landmarks.
        Returns a list of (latitude, longitude) points defining the boundary polygon of the overlap region.
        Returns None if insufficient data points exist.
        
        The algorithm:
        1. Each landmark measurement creates a circle of possible locations
        2. For 2 landmarks: calculate the intersection points of their circles
        3. For 3+ landmarks: iteratively find intersections with each additional circle
        4. Return the boundary points of the resulting intersection region
        """
        data_points = self.get_data(gpu_id)
        if len(data_points) < 2:
            return None
            
        # Get landmark positions from registry
        try:
            with open(os.path.join(os.path.dirname(__file__), "landmark_registry.json"), 'r') as f:
                registry = json.load(f)
                positions = {
                    landmark["hostname"]: (
                        float(landmark.get("latitude", 0)),
                        float(landmark.get("longitude", 0))
                    )
                    for landmark in registry.get("landmarks", [])
                    if "latitude" in landmark and "longitude" in landmark
                }
        except Exception as e:
            print(f"Warning: Could not load landmark positions: {e}")
            return None
            
        # Convert to list of circles (center_lat, center_lon, radius)
        circles = []
        for hostname, data in data_points.items():
            if hostname not in positions:
                continue
            lat, lon = positions[hostname]
            circles.append((lat, lon, data.radius))
            
        if len(circles) < 2:
            return None
            
        def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
            """Calculate great-circle distance between two points in kilometers."""
            R = 6371  # Earth's radius in kilometers
            
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            lat1, lat2 = math.radians(lat1), math.radians(lat2)
            
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            return R * c
            
        def circle_intersection_points(c1: Tuple[float, float, float], c2: Tuple[float, float, float]) -> List[Tuple[float, float]]:
            """
            Calculate intersection points of two circles on Earth's surface.
            Returns list of (lat, lon) points.
            """
            lat1, lon1, r1 = c1
            lat2, lon2, r2 = c2
            
            # Convert radii from meters to kilometers
            r1, r2 = r1/1000, r2/1000
            
            # Calculate distance between centers
            d = haversine_distance(lat1, lon1, lat2, lon2)
            
            # Check if circles are too far apart or one contains the other
            if d > r1 + r2:  # Circles are separate
                return []
            if d < abs(r1 - r2):  # One circle contains the other
                return []
            if d == 0 and r1 == r2:  # Circles are identical
                return []
                
            # Calculate intersection points
            # This is an approximation using flat Earth geometry near the intersection
            # Convert to local x,y coordinates (in km) centered at c1
            bearing = math.atan2(
                math.sin(math.radians(lon2-lon1)) * math.cos(math.radians(lat2)),
                math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
                math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                math.cos(math.radians(lon2-lon1))
            )
            
            # Distance from first circle's center to the line connecting the intersection points
            a = (r1*r1 - r2*r2 + d*d) / (2*d)
            
            # Distance from the line connecting the intersection points to either intersection point
            h = math.sqrt(r1*r1 - a*a)
            
            # Calculate intersection points in local coordinates
            x2 = a * math.cos(bearing)
            y2 = a * math.sin(bearing)
            
            xi = x2 + h * math.sin(bearing)
            yi = y2 - h * math.cos(bearing)
            
            xi_prime = x2 - h * math.sin(bearing)
            yi_prime = y2 + h * math.cos(bearing)
            
            # Convert back to lat/lon
            R = 6371  # Earth's radius in kilometers
            lat_scale = 180 / (math.pi * R)
            lon_scale = lat_scale / math.cos(math.radians(lat1))
            
            intersection1 = (
                lat1 + yi * lat_scale,
                lon1 + xi * lon_scale
            )
            intersection2 = (
                lat1 + yi_prime * lat_scale,
                lon1 + xi_prime * lon_scale
            )
            
            return [intersection1, intersection2]
            
        def find_polygon_points(circles: List[Tuple[float, float, float]]) -> List[Tuple[float, float]]:
            """Find all intersection points between circles and return boundary points."""
            points = []
            
            # Find all pairwise intersections
            for i in range(len(circles)):
                for j in range(i + 1, len(circles)):
                    points.extend(circle_intersection_points(circles[i], circles[j]))
                    
            if not points:
                return []
                
            # Filter points to keep only those that lie within or on all circles
            valid_points = []
            for point in points:
                valid = True
                for circle in circles:
                    dist = haversine_distance(point[0], point[1], circle[0], circle[1])
                    if dist > (circle[2]/1000 + 0.1):  # Add small tolerance for floating point
                        valid = False
                        break
                if valid:
                    valid_points.append(point)
                    
            return valid_points
            
        # Calculate intersection region
        boundary_points = find_polygon_points(circles)
        
        # If we found valid intersection points, return them
        if boundary_points:
            return boundary_points
            
        # Fallback: if no intersection found but circles are close,
        # return the midpoint of the closest approach
        if len(circles) == 2:
            lat1, lon1, _ = circles[0]
            lat2, lon2, _ = circles[1]
            return [(
                (lat1 + lat2) / 2,
                (lon1 + lon2) / 2
            )]
            
        return None
        
    def find_overlap_center(self, gpu_id: str) -> Optional[Tuple[float, float]]:
        """
        Calculate an estimated center point from the overlap region.
        This is a convenience method that uses find_overlap_region() and returns the centroid.
        """
        points = self.find_overlap_region(gpu_id)
        if not points:
            return None
            
        # Calculate centroid of boundary points
        lat_sum = sum(p[0] for p in points)
        lon_sum = sum(p[1] for p in points)
        return (lat_sum / len(points), lon_sum / len(points))

@dataclass
class PingData:
    """Data structure for storing ping measurements."""
    time_sent: float
    round_trip_time: float
    target_gpu: str

@dataclass
class RadiusEstimate:
    """Data structure for storing radius estimates for a GPU."""
    target_gpu: str
    estimate: float
    timestamp: float
    confidence: float

class LandmarkTracker:
    """Tracks ping data and radius estimates for each landmark."""
    def __init__(self):
        self.landmarks: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'pings': [],  # List[PingData]
                'radius_estimates': {}  # Dict[str, RadiusEstimate] keyed by GPU ID
            }
        )

    def add_ping(self, landmark: str, ping: PingData) -> None:
        """Add a new ping measurement for a landmark, skipping duplicates."""
        existing_pings = self.landmarks[landmark]['pings']
        # Skip if this ping was already received
        if any(
            p.time_sent == ping.time_sent 
            and p.round_trip_time == ping.round_trip_time 
            and p.target_gpu == ping.target_gpu 
            for p in existing_pings
        ):
            return
        
        existing_pings.append(ping)
        # Keep only last N pings (e.g., 10)
        self.landmarks[landmark]['pings'] = existing_pings[-10:]

    def update_radius_estimate(self, landmark: str, estimate: RadiusEstimate) -> None:
        """Update radius estimate for a specific GPU."""
        self.landmarks[landmark]['radius_estimates'][estimate.target_gpu] = estimate

    def get_radius_estimate(self, landmark: str, gpu_id: str) -> Optional[RadiusEstimate]:
        """Get the current radius estimate for a specific GPU from a landmark."""
        return self.landmarks[landmark]['radius_estimates'].get(gpu_id)

    def get_all_estimates(self, gpu_id: str) -> Dict[str, RadiusEstimate]:
        """Get all landmarks' radius estimates for a specific GPU."""
        return {
            landmark: data['radius_estimates'][gpu_id]
            for landmark, data in self.landmarks.items()
            if gpu_id in data['radius_estimates']
        }

# Initialize Flask app
app = Flask(__name__)

# Initialize key manager and get hostname from environment
current_dir = os.path.dirname(os.path.abspath(__file__))
LANDMARK_HOSTNAME = os.environ.get("LANDMARK_HOSTNAME", "localhost")
key_manager = KeyManager(
    keys_dir=os.path.join(current_dir, "keys", LANDMARK_HOSTNAME),
    registry_path=os.path.join(current_dir, "landmark_registry.json")
)

# Initialize GPU location tracker
gpu_tracker = GPULocationTracker()

# Initialize the tracker
landmark_tracker = LandmarkTracker()

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--simulated', action='store_true', help='Use simulated coordinates')
args = parser.parse_args()

def broadcast_to_landmarks(gpu_id: str, ping_data: Optional[PingData] = None, radius_estimate: Optional[RadiusEstimate] = None) -> None:
    """
    Broadcast GPU detection and measurements to other landmarks.
    
    Args:
        gpu_id: Unique identifier for the GPU being located
        ping_data: Optional PingData containing timing information
        radius_estimate: Optional RadiusEstimate containing radius calculation
    """
    logging.info(f"📢 Broadcasting data for GPU {gpu_id}")
    
    # Get our latest data
    our_data = landmark_tracker.landmarks[LANDMARK_HOSTNAME]
    
    # Prepare the message with our current data
    message = {
        "hostname": LANDMARK_HOSTNAME,
        "timestamp": time.time(),
        "data": {
            "pings": [
                {
                    "time_sent": ping.time_sent,
                    "round_trip_time": ping.round_trip_time,
                    "target_gpu": ping.target_gpu
                }
                for ping in our_data.get('pings', [])
            ],
            "radius_estimates": {
                gpu_id: {
                    "target_gpu": estimate.target_gpu,
                    "estimate": estimate.estimate,
                    "timestamp": estimate.timestamp,
                    "confidence": estimate.confidence
                }
                for gpu_id, estimate in our_data.get('radius_estimates', {}).items()
            }
        }
    }
    
    logging.info(f"📢 Broadcasting message: {json.dumps(message, indent=2)}")
    
    # Convert to bytes and sign
    message_bytes = json.dumps(message, sort_keys=True).encode()
    signature = key_manager.sign_message(message_bytes)
    message["signature"] = signature.hex()
    
    # Load registry and broadcast
    with open(key_manager.registry_path, 'r') as f:
        registry = json.load(f)
        
    # Broadcast to all landmarks except self
    for landmark in registry.get('landmarks', []):
        if landmark['hostname'] != LANDMARK_HOSTNAME:
            try:
                response = requests.post(
                    f"http://{landmark['hostname']}/landmark-estimates",
                    json=message,
                    timeout=5
                )
                response.raise_for_status()
                logging.info(f"Successfully broadcast to {landmark['hostname']}")
            except Exception as e:
                logging.error(f"Failed to broadcast to {landmark['hostname']}: {str(e)}")

# Fallback coordinates for San Francisco (used if geolocation fails)
LANDMARK_LAT = 37.7749
LANDMARK_LON = -122.4194

# Initialize Flask app with CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Store the latest radius calculations
latest_results = {
    'light_radius': None,
    'network_radius': None
}

SPEED_OF_LIGHT = 299792458  # meters per second


def calculate_radius(round_trip_time, verification_time):
    """Calculate the maximum possible distance based on speed of light and network transit time."""
    # Calculate network transit time by subtracting verification time
    network_transit_time = round_trip_time - verification_time
    # Use network transit time divided by 2 since signal needs to travel both ways
    return SPEED_OF_LIGHT * (network_transit_time / 2)

def calculate_network_speed_radius(round_trip_time, verification_time):
    """Calculate a more realistic radius bound based on empirical network speeds."""    
    # Calculate the speed of light radius first
    light_radius = calculate_radius(round_trip_time, verification_time)
    
    # Network radius should be 13.3% of the speed of light radius
    return light_radius * 0.133  # 13.3% of speed of light

@app.route('/')
def serve_map():
    """Serve the React frontend application."""
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'dist')
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/api/radius')
def get_radius():
    """Return the latest calculated radii."""
    return jsonify({
        'radius': latest_results.get('light_radius'),  # Keep old key for backward compatibility
        'lightRadius': latest_results.get('light_radius'),
        'networkRadius': latest_results.get('network_radius'),
        'landmark': {
            'lat': LANDMARK_LAT,
            'lon': LANDMARK_LON
        }
    })

def generate_nonce(length=32):
    """Generate a random hex nonce of specified byte length."""
    # Generate random bytes and convert to hex string
    random_bytes = os.urandom(length)
    return random_bytes.hex()

def verify_nonce_and_calculate_radius():
    """Run the nonce verification process and calculate the radius."""
    global latest_results, landmark_tracker
    
    try:
        nonce = generate_nonce()
        logging.info(f"Generated nonce: {nonce}")

        start_time = time.time()
        
        port = int(os.environ.get("LANDMARK_PORT", "5001"))
        if port != 5001:
            time.sleep(random.uniform(0.4, 0.8))

        # Prepare request data
        request_data = {"nonce": nonce}
        
        # Add simulated coordinates if in simulation mode
        if args.simulated:
            # Load our simulated coordinates from registry
            with open(os.path.join(os.path.dirname(__file__), "landmark_registry.json"), 'r') as f:
                registry = json.load(f)
                our_coords = None
                for landmark in registry["landmarks"]:
                    if landmark["hostname"] == LANDMARK_HOSTNAME:
                        our_coords = landmark.get("simulated_coords")
                        break
                
            if our_coords:
                request_data["simulated_host_coords"] = our_coords
                logging.info(f"Using simulated coordinates: {our_coords}")

        # Send nonce to the host service
        host_url = os.environ.get("HOST_URL", "127.0.0.1:5000")
        logging.info(f"Sending request to host at http://{host_url}/sign")
        response = requests.post(f"http://{host_url}/sign", json=request_data)
        json_resp = response.json()
        
        # If in simulation mode, calculate and apply artificial delay
        if args.simulated and our_coords:
            host_coords = json_resp.get("host_coords", {})
            if host_coords:
                # Calculate distance in kilometers
                from math import radians, sin, cos, sqrt, atan2
                
                def haversine_distance(lat1, lon1, lat2, lon2):
                    R = 6371  # Earth's radius in kilometers
                    
                    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    c = 2 * atan2(sqrt(a), sqrt(1-a))
                    return R * c
                
                distance = haversine_distance(
                    our_coords["latitude"], our_coords["longitude"],
                    host_coords["latitude"], host_coords["longitude"]
                )
                
                # Calculate required network delay (distance * 2 for round trip / speed of light)
                # Use 13.3% of speed of light as typical network speed
                network_speed = SPEED_OF_LIGHT * 0.133  # meters per second
                required_delay = (distance * 1000 * 2) / network_speed  # seconds
                
                logging.info(f"Simulating network delay of {required_delay:.3f}s for {distance:.1f}km distance")
                time.sleep(required_delay)

        # Continue with existing code
        token = json_resp.get("token")
        receipt = json_resp.get("receipt", {})
        timings = json_resp.get("timings", {})
        logging.debug(f"Received response from host: {json_resp}")

        end_time = time.time()
        round_trip_time = end_time - start_time
        
        # Get verification timings from host
        timings = receipt.get("timings", {})
        gpu_id = receipt.get("gpu_id")
        gpu_info = receipt.get("gpu_info", {})

        total_verification_time = timings.get('total_verification_time', 0)
        
        # Calculate both radius bounds
        light_radius = calculate_radius(round_trip_time, total_verification_time)
        network_radius = calculate_network_speed_radius(round_trip_time, total_verification_time)
        
        logging.info(f"Calculated radius bounds - Light: {light_radius/1000:.2f}km, Network: {network_radius/1000:.2f}km")
        
        # Always store data, even with mock GPU IDs in development mode
        if gpu_id:
            # Store ping data
            ping_data = PingData(
                time_sent=start_time,
                round_trip_time=round_trip_time,
                target_gpu=gpu_id
            )
            landmark_tracker.add_ping(LANDMARK_HOSTNAME, ping_data)
            logging.debug(f"Stored ping data for GPU {gpu_id}: {ping_data}")

            # Store radius estimate
            radius_estimate = RadiusEstimate(
                target_gpu=gpu_id,
                estimate=network_radius,
                timestamp=time.time(),
                confidence=1.0
            )
            landmark_tracker.update_radius_estimate(LANDMARK_HOSTNAME, radius_estimate)
            logging.debug(f"Updated radius estimate for GPU {gpu_id}: {radius_estimate}")

            # Broadcast data to other landmarks
            logging.info(f"📢 Broadcasting data for GPU {gpu_id} to other landmarks...")
            broadcast_latest_estimates()

        # Store latest results for API endpoint
        latest_results = {
            'light_radius': light_radius,
            'network_radius': network_radius,
            'round_trip_time': round_trip_time,
            'gpu_id': gpu_id,
            'gpu_info': gpu_info,
            'timings': timings  # Store the timings from the host
        }
                
           # Print verification timings and results
        print(f"[landmark] Received H100-signed token: {token[:15]}...")
        print(f"[landmark] Host verification timings:")
        print(f"  - Evidence gathering: {timings.get('evidence_gathering', 0):.6f} seconds")
        print(f"  - Attestation: {timings.get('attestation', 0):.6f} seconds")
        print(f"  - Token validation: {timings.get('validation', 0):.6f} seconds")
        print(f"  - Total verification time: {total_verification_time:.6f} seconds")
        print(f"[landmark] 🏁 Round-trip time: {round_trip_time:.6f} seconds")
        print(f"[landmark] 📊 Time spent in network transit: {round_trip_time - total_verification_time:.6f} seconds")
        print(f"[landmark] 📍 Maximum possible distances:")
        print(f"  - Speed of light bound: {light_radius/1000:.2f} km")
        print(f"  - Network speed bound: {network_radius/1000:.2f} km")

        # Validate the token (optional)
        result = verify_token(token)
        print(f"[landmark] Token validation result: {result}")

        return gpu_id, network_radius

    except Exception as e:
        logging.error(f"Error during verification: {e}", exc_info=True)
        latest_results = {
            'light_radius': None,
            'network_radius': None,
            'gpu_id': None,
            'timings': {}
        }
        return None, None

def broadcast_latest_estimates():
    """Broadcast our latest radius estimates and ping data to all other landmarks."""
    try:
        # Get our latest data
        our_data = landmark_tracker.landmarks[LANDMARK_HOSTNAME]
        current_time = time.time()
        
        logging.info(f"📢 Broadcast")
        logging.debug(f"Current landmark data: {our_data}")
        
        # Prepare the message with our current data in the required format
        message = {
            "hostname": LANDMARK_HOSTNAME,
            "timestamp": current_time,
            "data": {
                "pings": [
                    {
                        "time_sent": ping.time_sent,
                        "round_trip_time": ping.round_trip_time,
                        "target_gpu": ping.target_gpu
                    }
                    for ping in our_data.get('pings', [])
                ],
                "radius_estimates": {
                    gpu_id: {
                        "target_gpu": estimate.target_gpu,
                        "estimate": estimate.estimate,
                        "timestamp": estimate.timestamp,
                        "confidence": estimate.confidence
                    }
                    for gpu_id, estimate in our_data.get('radius_estimates', {}).items()
                }
            }
        }
        logging.debug(f"Prepared broadcast message: {json.dumps(message, indent=2)}")
        
        # Sign the message
        message_bytes = json.dumps(message, sort_keys=True).encode()
        signature = key_manager.sign_message(message_bytes)
        message["signature"] = signature.hex()
        
        # Load registry and broadcast
        with open(key_manager.registry_path, 'r') as f:
            registry = json.load(f)
        
        # logging.info(f"Broadcasting data: {message}")
            
        for landmark in registry.get('landmarks', []):
            if landmark['hostname'] != LANDMARK_HOSTNAME:
                try:
                    response = requests.post(
                        f"http://{landmark['hostname']}/landmark-estimates",
                        json=message,
                        timeout=5
                    )
                    response.raise_for_status()
                    logging.debug(f"Successfully broadcast to {landmark['hostname']}")
                except Exception as e:
                    logging.error(f"Failed to broadcast to {landmark['hostname']}: {str(e)}")
    except Exception as e:
        logging.error(f"Error broadcasting estimates: {e}")
        raise

@app.route("/landmark-estimates", methods=["POST"])
def receive_landmark_estimates():
    """Receive and process data from another landmark."""
    try:
        data = request.get_json(force=True)
        sender = data.get("hostname", "unknown")
        logging.info(f"📻 Received from {sender}")
        logging.debug(f"Full received data: {json.dumps(data, indent=2)}")
        
        # Validate required fields
        if not all(field in data for field in ["hostname", "timestamp", "data"]):
            logging.error(f"Missing required fields in data from {sender}")
            return jsonify({"error": "Missing required fields"}), 400
            
        # Verify signature if present
        if "signature" in data:
            message_data = {k: v for k, v in data.items() if k != "signature"}
            message_bytes = json.dumps(message_data, sort_keys=True).encode()
            try:
                signature = bytes.fromhex(data["signature"])
                if not key_manager.verify_signature(data["hostname"], message_bytes, signature):
                    logging.error(f"Invalid signature from {sender}")
                    return jsonify({"error": "Invalid signature"}), 401
            except Exception as e:
                logging.error(f"Signature verification failed for {sender}: {str(e)}")
                return jsonify({"error": f"(landmark-estimates)Signature verification failed: {str(e)}"}), 401
            
        # Update our tracker with the received data
        hostname = data["hostname"]
        timestamp = data["timestamp"]
        
        # Process ping data
        for ping_data in data["data"].get("pings", []):
            try:
                # Validate required fields
                required_fields = ["time_sent", "round_trip_time", "target_gpu"]
                if not all(field in ping_data for field in required_fields):
                    missing = [f for f in required_fields if f not in ping_data]
                    logging.warning(f"Missing required fields {missing} in ping data from {hostname}")
                    continue
                    
                ping = PingData(
                    time_sent=ping_data["time_sent"],
                    round_trip_time=ping_data["round_trip_time"],
                    target_gpu=ping_data["target_gpu"]
                )
                landmark_tracker.add_ping(hostname, ping)
                logging.debug(f"Added ping for GPU {ping_data['target_gpu']} from {hostname}: {ping}")
            except Exception as e:
                logging.error(f"Error processing ping data from {hostname}: {e}")
                continue
        
        # Process radius estimates
        for gpu_id, radius_data in data["data"].get("radius_estimates", {}).items():
            try:
                if not isinstance(radius_data, dict):
                    logging.warning(f"Invalid radius data format for GPU {gpu_id} from {hostname}: not a dict")
                    continue
                    
                # Validate required fields
                required_fields = ["target_gpu", "estimate"]
                if not all(field in radius_data for field in required_fields):
                    missing = [f for f in required_fields if f not in radius_data]
                    logging.warning(f"Missing required fields {missing} in radius data for GPU {gpu_id} from {hostname}")
                    continue
                
                estimate = RadiusEstimate(
                    target_gpu=radius_data["target_gpu"],
                    estimate=radius_data["estimate"],
                    timestamp=radius_data.get("timestamp", timestamp),
                    confidence=radius_data.get("confidence", 1.0)
                )
                landmark_tracker.update_radius_estimate(hostname, estimate)
                logging.debug(f"Updated radius estimate for GPU {gpu_id} from {hostname}: {estimate}")
            except Exception as e:
                logging.error(f"Error processing radius data for GPU {gpu_id} from {hostname}: {e}")
                continue
        # logging.info(f"Successfully processed data from {hostname}")
        return jsonify({"status": "ok"}), 200
        
    except Exception as e:
        logging.error(f"Error processing landmark data: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def start_broadcast_thread():
    """Start a thread to periodically broadcast our estimates."""
    def broadcast_loop():
        while True:
            # Add random jitter between 4-6 seconds to prevent servers from always broadcasting at the same time
            jitter = random.uniform(3.0, 30.0)
            time.sleep(jitter)
            broadcast_latest_estimates()
    
    thread = threading.Thread(target=broadcast_loop, daemon=True)
    thread.start()

def start_verification_thread():
    """Start a thread to periodically verify nonces and calculate radii."""
    def verification_loop():
        logging.info("Starting verification loop...")
        while True:
            try:
                logging.debug("Running verification cycle...")
                gpu_id, radius = verify_nonce_and_calculate_radius()
                if gpu_id and radius:
                    logging.info(f"Successfully verified GPU {gpu_id} with radius {radius:.2f}m")
                else:
                    logging.warning("Verification cycle completed but no data generated")
            except Exception as e:
                logging.error(f"Error in verification loop: {e}", exc_info=True)
            time.sleep(1)  # Run verification every second
    
    thread = threading.Thread(target=verification_loop, daemon=True)
    thread.start()
    return thread

def main():
    """Start the Flask server and run verification in a separate thread."""
    logging.info("Starting landmark service...")
    
    try:
        # Start verification thread
        logging.info("Starting verification thread...")
        verification_thread = start_verification_thread()
        logging.info("Verification thread started successfully")
        
        # Start broadcast thread
        logging.info("Starting broadcast thread...")
        start_broadcast_thread()
        logging.info("Broadcast thread started successfully")
        
        # Start the Flask server
        port = int(os.environ.get("LANDMARK_PORT", "5001"))
        logging.info(f"Starting web server on port {port}...")
        app.run(host="0.0.0.0", port=port, debug=False)
    except Exception as e:
        logging.error(f"Error in main: {e}", exc_info=True)
        raise

@app.route("/landmark-data", methods=["POST"])
def receive_landmark_data():
    """Receive data from another landmark, verify its signature, and process the data."""
    try:
        data = request.get_json(force=True)
        
        # Validate required fields
        required_fields = ["hostname", "gpu_id", "signature"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
            
        # Convert the data (excluding signature) to bytes for verification
        message_data = {k: v for k, v in data.items() if k != "signature"}
        message_bytes = json.dumps(message_data, sort_keys=True).encode()
        
        # Verify the signature
        try:
            signature = bytes.fromhex(data["signature"])
            if not key_manager.verify_signature(data["hostname"], message_bytes, signature):
                return jsonify({"error": "Invalid signature"}), 401
        except Exception as e:
            return jsonify({"error": f"(landmark-data) Signature verification failed: {str(e)}"}), 401
            
        # Get GPU ID and trigger our own measurement
        gpu_id = data["gpu_id"]
        
        # Start our own verification for this GPU
        verify_nonce_and_calculate_radius()
        
        return jsonify({"status": "ok"}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ping", methods=["POST"])
def handle_ping():
    """Handle ping requests and trigger nonce verification."""
    try:
        # Run the verification process
        gpu_id, network_radius = verify_nonce_and_calculate_radius()
        
        # Get the latest results and handle None values
        round_trip_time = latest_results.get('round_trip_time')
        light_radius = latest_results.get('light_radius')
        network_radius = latest_results.get('network_radius')
        timings = latest_results.get('timings', {})
        
        # Convert round_trip_time to milliseconds, handling None
        round_trip_ms = (round_trip_time or 0) * 1000
        
        # Log the ping results
        logging.info(f"Ping results - RTT: {round_trip_time}, Light: {light_radius}, Network: {network_radius}")
        
        response_data = {
            "status": "success",
            "roundTripTime": round_trip_ms,
            "lightRadius": light_radius,
            "networkRadius": network_radius,
            "gpuId": gpu_id,
            "timings": timings
        }
        
        # Log the response
        logging.debug(f"Sending response: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"Error handling ping: {str(e)}"
        logging.error(error_msg)
        return jsonify({
            "status": "error",
            "error": error_msg
        }), 500

@app.route("/")
def index():
    """Serve the frontend page."""
    return """
    <html>
    <head>
        <title>Landmark Status</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            pre { background: #f5f5f5; padding: 10px; border-radius: 5px; }
            .error { color: red; }
            .success { color: green; }
        </style>
    </head>
    <body>
        <h1>Landmark Data</h1>
        <div id="status" class="success">Connected</div>
        <h2>All Landmarks Data:</h2>
        <pre id="landmarkData">Loading...</pre>
        <h2>Latest Ping Results:</h2>
        <pre id="pingData">No pings yet</pre>
        
        <script>
            function updateStatus(connected) {
                const status = document.getElementById('status');
                status.className = connected ? 'success' : 'error';
                status.innerText = connected ? 'Connected' : 'Disconnected';
            }
            
            function fetchData() {
                // Fetch all landmarks data
                fetch('/api/all_landmarks_data')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('landmarkData').innerText = 
                            JSON.stringify(data, null, 2);
                        updateStatus(true);
                    })
                    .catch(err => {
                        console.error('Error fetching data:', err);
                        updateStatus(false);
                    });
                    
                // Trigger a ping
                fetch('/api/ping', { method: 'POST' })
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('pingData').innerText = 
                            JSON.stringify(data, null, 2);
                    })
                    .catch(err => {
                        console.error('Error pinging:', err);
                    });
            }
            
            // Update every second
            setInterval(fetchData, 1000);
            fetchData();
        </script>
    </body>
    </html>
    """

@app.route("/api/all_landmarks_data", methods=["GET"])
def get_all_landmarks_data():
    """Return data for all landmarks including their radius estimates."""
    try:
        # Load registry to get landmark positions
        with open(key_manager.registry_path, 'r') as f:
            registry = json.load(f)

        response_data = {"landmarks": []}

        for landmark_info in registry.get("landmarks", []):
            hostname = landmark_info.get("hostname")
            lat = landmark_info.get("latitude", 0.0)
            lon = landmark_info.get("longitude", 0.0)

            # Get all radius estimates and pings for this landmark
            landmark_data = landmark_tracker.landmarks.get(hostname, {})
            
            # Format data according to required structure
            formatted_data = {
                "hostname": hostname,
                "lat": lat,
                "lon": lon,
                "data": {
                    "pings": [
                        {
                            "time_sent": ping.time_sent,
                            "round_trip_time": ping.round_trip_time,
                            "target_gpu": ping.target_gpu
                        }
                        for ping in landmark_data.get('pings', [])
                    ],
                    "radius_estimates": {
                        gpu_id: {
                            "target_gpu": estimate.target_gpu,
                            "estimate": estimate.estimate,
                            "timestamp": estimate.timestamp,
                            "confidence": estimate.confidence
                        }
                        for gpu_id, estimate in landmark_data.get('radius_estimates', {}).items()
                    }
                }
            }
            
            response_data["landmarks"].append(formatted_data)

        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/<path:path>')
def serve_static(path):
    frontend_build = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'dist')
    return send_from_directory(frontend_build, path)

if __name__ == "__main__":
    # Start verification and broadcast threads regardless of mode
    # verification_thread = start_verification_thread()
    broadcast_thread = start_broadcast_thread()
    
    # Start the Flask server
    port = int(os.environ.get("LANDMARK_PORT", "5001"))
    app.run(host="0.0.0.0", port=port)                                                                          
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    broadcast_thread = start_broadcast_thread()
    
    # Start the Flask server
    port = int(os.environ.get("LANDMARK_PORT", "5001"))
    app.run(host="0.0.0.0", port=port)                                                                          
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          