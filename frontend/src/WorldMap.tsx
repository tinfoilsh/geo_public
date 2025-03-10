import React, { useEffect, useState } from 'react';
// import { MapContainer, TileLayer, Marker, Popup, Circle, useMap } from 'react-leaflet';
import { MapContainer, TileLayer, useMap, Marker, Popup, Circle } from 'react-leaflet'
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

// Fix the default icon configuration
const DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

L.Marker.prototype.options.icon = DefaultIcon;


// Define interfaces for our data structures
interface Landmark {
  hostname: string;
  lat: number;
  lon: number;
  data: {
    pings: Array<{
      round_trip_time: number;
      target_gpu: string;
      time_sent: number;
    }>;
    radius_estimates: {
      [gpuId: string]: {
        confidence: number;
        estimate: number;
        target_gpu: string;
        timestamp: number;
      };
    };
  };
}

interface CircleData {
  lat: number;
  lon: number;
  radius: number;
  color: string;
}

interface MarkerData {
  lat: number;
  lon: number;
  hostname: string;
}

// FitBounds component with TypeScript props
interface FitBoundsProps {
  markers: MarkerData[];
  circles: CircleData[];
}

function FitBounds({ markers, circles }: FitBoundsProps) {
  const map = useMap();

  useEffect(() => {
    if ((!markers || markers.length === 0) && (!circles || circles.length === 0)) return;

    // Calculate bounds that include both markers and the full extent of circles
    const bounds: [number, number][] = [];
    
    markers?.forEach((m) => {
      if (m.lat !== undefined && m.lon !== undefined) {
        bounds.push([m.lat, m.lon]);
      }
    });

    circles?.forEach((c) => {
      if (c.lat !== undefined && c.lon !== undefined && c.radius && c.radius > 0) {
        // Add points at the north, south, east, and west extremes of each circle
        const radiusInDegrees = c.radius / 111000; // Approximate degrees per meter at the equator
        bounds.push([c.lat + radiusInDegrees, c.lon]); // North
        bounds.push([c.lat - radiusInDegrees, c.lon]); // South
        bounds.push([c.lat, c.lon + radiusInDegrees]); // East
        bounds.push([c.lat, c.lon - radiusInDegrees]); // West
      }
    });

    if (bounds.length > 0) {
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [markers, circles, map]);

  return null;
}

// Styles interface
interface Styles {
  [key: string]: React.CSSProperties;
}

// New components
interface MapComponentProps {
  landmarkMarkers: MarkerData[];
  singleLandmarkCircles: CircleData[];
  gpuCircles: CircleData[];
}

const LANDMARK_LAT = 37.7749;
const LANDMARK_LON = -122.4194;

// Color constants
const LIGHT_SPEED_COLOR = 'rgba(255, 0, 51, 0.5)';  // #f03
const NETWORK_LATENCY_COLOR = 'rgba(0, 51, 255, 0.5)';  // #03f
const LIGHT_SPEED_TEXT_COLOR = '#f03';
const NETWORK_LATENCY_TEXT_COLOR = '#03f';

function MapComponent({ landmarkMarkers, singleLandmarkCircles, gpuCircles }: MapComponentProps) {
  return (
    <MapContainer 
      center={[LANDMARK_LAT, LANDMARK_LON]} // Use the SF coordinates instead of London
      zoom={10} 
      style={{height: '600px', width: '100%'}} // Increased height
    >
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      
      {landmarkMarkers.map((marker, idx) => (
        <Marker key={`marker-${idx}`} position={[marker.lat, marker.lon]}>
          <Popup>{marker.hostname}</Popup>
        </Marker>
      ))}
      {[...singleLandmarkCircles, ...gpuCircles].map((circle, idx) => (
        <Circle
          key={`circle-${idx}`}
          center={[circle.lat, circle.lon]}
          radius={circle.radius}
          pathOptions={{ color: circle.color }}
        />
      ))}
      <FitBounds 
        markers={landmarkMarkers} 
        circles={[...singleLandmarkCircles, ...gpuCircles]} 
      />
    </MapContainer>
  );
}

interface StatusSectionProps {
  pingLoading: boolean;
  sendPing: () => void;
  lightRadius: number | null;
  networkRadius: number | null;
  pingMessage: string;
  styles: Styles;
  timings?: Timings;
  roundTripTime: number | null;
}

// Status Section Components
interface PingButtonProps {
  pingLoading: boolean;
  sendPing: () => void;
  styles: Styles;
}

function PingButton({ pingLoading, sendPing, styles }: PingButtonProps) {
  return (
    <button
      onClick={sendPing}
      style={styles.actionButton}
      disabled={pingLoading}
    >
      Send Ping
    </button>
  );
}

interface RadiusInfoProps {
  lightRadius: number | null;
  networkRadius: number | null;
  timings?: Timings;
  roundTripTime: number | null;
}

function RadiusInfo({ lightRadius, networkRadius, timings, roundTripTime }: RadiusInfoProps) {
  
  const tv = timings?.total_verification_time ?? 0
  
  let travelTime: string | null = null
  
  if (roundTripTime && tv) {
    travelTime = ((roundTripTime - (1000* tv))).toFixed(2)
  } 

  return (
    <div id="radius-info">
      <p>
        Maximum distance bounds:
        <br />
        • Speed of light:{' '}
        <span style={{ color: LIGHT_SPEED_TEXT_COLOR }}>
          {lightRadius && !isNaN(lightRadius)
            ? `${(lightRadius / 1000).toFixed(2)} km`
            : 'waiting for ping...'}
        </span>
        <br />
        • Network latency:{' '}
        <span style={{ color: NETWORK_LATENCY_TEXT_COLOR }}>
          {networkRadius && !isNaN(networkRadius)
            ? `${(networkRadius / 1000).toFixed(2)} km`
            : 'waiting for ping...'}
        </span>
      </p>
      {timings && (
        <p style={{ fontSize: '0.9em', color: '#666' }}>
          Timing breakdown:
          <br />
          • Evidence gathering: {(timings.evidence_gathering * 1000).toFixed(2)} ms
          <br />
          • Attestation: {(timings.attestation * 1000).toFixed(2)} ms
          <br />
          • Validation: {(timings.validation * 1000).toFixed(2)} ms
          <br />
          • Total verification: {(tv * 1000).toFixed(2)} ms
          <br />
          • Travel time: {travelTime} ms
        </p>
      )}
    </div>
  );
}

interface StatusMessageProps {
  pingMessage: string;
  styles: Styles;
}

function StatusMessage({ pingMessage, styles }: StatusMessageProps) {
  return (
    <>
      <p style={pingMessage.includes('Error') ? styles.statusError : styles.statusSuccess}>
        {pingMessage}
      </p>
      <p style={{ fontSize: '0.9em', color: '#666' }}>
        The network latency bound (blue) is based on empirical measurements showing ~100ms
        round-trip time globally. This accounts for real-world factors like fiber optic cable
        routing, network switching delays, and TCP/IP overhead, resulting in an effective speed
        of about 13.3% of light speed.
      </p>
    </>
  );
}

// Updated StatusSection
function StatusSection({ 
  pingLoading, 
  sendPing, 
  lightRadius, 
  networkRadius, 
  pingMessage, 
  styles,
  timings,
  roundTripTime 
}: StatusSectionProps) {
  return (
    <>
      <PingButton pingLoading={pingLoading} sendPing={sendPing} styles={styles} />
      <div style={styles.info}>
        <p>Landmark Location: San Francisco (37.7749°N, 122.4194°W)</p>
        <RadiusInfo 
          lightRadius={lightRadius} 
          networkRadius={networkRadius} 
          timings={timings}
          roundTripTime={roundTripTime}
        />
        <StatusMessage pingMessage={pingMessage} styles={styles} />
      </div>
    </>
  );
}

interface LandmarksTableProps {
  landmarks: Landmark[];
  styles: Styles;
}

// Landmarks Table Components
interface GPUInfoProps {
  gpuId: string;
  gpuData: {
    estimate: number;
    target_gpu: string;
  };
  pingCount: number;
}

function GPUInfo({ gpuId, gpuData, pingCount }: GPUInfoProps) {
  return (
    <p style={{ margin: '2px 0' }}>
      <strong>GPU {gpuId.slice(0, 8)}...</strong>
      <br />
      Radius: {gpuData.estimate ? (gpuData.estimate / 1000).toFixed(2) : '---'} km
      <br />
      Pings: {pingCount}
    </p>
  );
}

interface TableRowProps {
  landmark: Landmark;
  styles: Styles;
}

function TableRow({ landmark: lm, styles }: TableRowProps) {
  const gpuInfo = Object.entries(lm.data?.radius_estimates || {}).map(([gpuId, gpuData]) => (
    <GPUInfo
      key={gpuId}
      gpuId={gpuId}
      gpuData={gpuData}
      pingCount={lm.data?.pings.filter(p => p.target_gpu === gpuId).length || 0}
    />
  ));

  const lastUpdate =
    lm.data && (lm.data.pings.length > 0 || Object.keys(lm.data.radius_estimates).length > 0)
      ? new Date().toLocaleTimeString()
      : 'No data';

  return (
    <tr>
      <td style={styles.tableCell}>{lm.hostname}</td>
      <td style={styles.tableCell}>
        {gpuInfo.length > 0 ? gpuInfo : 'No GPU data'}
      </td>
      <td style={styles.tableCell}>{lastUpdate}</td>
    </tr>
  );
}

// Updated LandmarksTable
function LandmarksTable({ landmarks, styles }: LandmarksTableProps) {
  return (
    <div style={{ marginTop: 20 }}>
      <h3>All Landmarks Data</h3>
      <table style={{ borderCollapse: 'collapse', width: '100%' }}>
        <thead>
          <tr>
            <th style={styles.tableCell}>Hostname</th>
            <th style={styles.tableCell}>GPU Data</th>
            <th style={styles.tableCell}>Last Update</th>
          </tr>
        </thead>
        <tbody>
          {landmarks.map((landmark, idx) => (
            <TableRow key={`lm-${idx}`} landmark={landmark} styles={styles} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface Timings {
  evidence_gathering: number;
  attestation: number;
  validation: number;
  total_verification_time: number;
}

function WorldMap() {
  // Constants from the original file
  const LANDMARK_LAT = 37.7749;
  const LANDMARK_LON = -122.4194;

  // State for the single "radius" data
  const [lightRadius, setLightRadius] = useState<number | null>(null);
  const [networkRadius, setNetworkRadius] = useState<number | null>(null);
  const [roundTripTime, setRoundTripTime] = useState<number | null>(null);

  // State for the ping UI
  const [pingMessage, setPingMessage] = useState<string>('');
  const [pingLoading, setPingLoading] = useState<boolean>(false);

  // State for the multi-landmark data
  const [landmarks, setLandmarks] = useState<Landmark[]>([]);

  // State for the timings
  const [timings, setTimings] = useState<Timings | null>(null);

  // Fetch /api/radius once to initialize the radius for a single landmark
  const fetchData = async () => {
    try {
      const response = await fetch('/api/radius');
      const data = await response.json();
      setLightRadius(data.lightRadius);
      setNetworkRadius(data.networkRadius);
    } catch (error) {
      console.error('Error fetching data:', error);
      setLightRadius(NaN);
      setNetworkRadius(NaN);
    }
  };

  // Ping logic
  const sendPing = async () => {
    try {
      setPingLoading(true);
      setPingMessage('Sending ping...');
      const response = await fetch('/api/ping', {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log("Ping response:", data);
      setPingMessage(
        `✅ Ping successful! Round-trip time: ${data.roundTripTime.toFixed(3)} ms`
      );
      setRoundTripTime(data.roundTripTime.toFixed(3));
      // Update timings if available
      if (data.timings) {
        setTimings(data.timings);
      }
      console.log("data.roundTripTime", data);
      // Trigger an immediate fetch of new radius data
      await fetchData();
    } catch (error) {
      console.error('Error sending ping:', error);
      setPingMessage(`❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setPingLoading(false);
    }
  };

  // A basic color hashing function to consistently color circles for each hostname
  const getHostnameColor = (hostname: string) => {
    let hash = 0;
    for (let i = 0; i < hostname.length; i++) {
      hash = hostname.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = Math.abs(hash) % 360;
    return `hsl(${hue}, 70%, 50%)`;
  };

  // Polls /api/all_landmarks_data for the array of landmarks
  const fetchAllLandmarksData = async () => {
    try {
      const resp = await fetch('/api/all_landmarks_data');
      const data = await resp.json();
      setLandmarks(data.landmarks || []);
    } catch (err) {
      console.error('Error fetching all landmarks data:', err);
    }
  };

  // Initial data fetch
  useEffect(() => {
    fetchData();
    fetchAllLandmarksData();
  }, []);

  // Poll for new multi-landmark data every 1 second
  useEffect(() => {
    const intervalId = setInterval(fetchAllLandmarksData, 1000);
    return () => clearInterval(intervalId);
  }, []);

  // Prepare the circles for the single "landmark" from /api/radius
  const singleLandmarkCircles = [];
  if (lightRadius && !isNaN(lightRadius)) {
    singleLandmarkCircles.push({
      lat: LANDMARK_LAT,
      lon: LANDMARK_LON,
      radius: lightRadius,
      color: LIGHT_SPEED_COLOR,
    });
  }
  
  if (networkRadius && !isNaN(networkRadius)) {
    singleLandmarkCircles.push({
      lat: LANDMARK_LAT,
      lon: LANDMARK_LON,
      radius: networkRadius,
      color: NETWORK_LATENCY_COLOR,
    });
  }

  // Build sets of circles for each landmark's GPU data
  // We'll also keep track of markers for each landmark
  const gpuCircles: CircleData[] = [];
  const landmarkMarkers: MarkerData[] = [];

  landmarks.forEach((landmark) => {
    landmarkMarkers.push({
      lat: landmark.lat,
      lon: landmark.lon,
      hostname: landmark.hostname,
    });

    // If the landmark has radius estimates, show circles
    if (landmark.data) {
      // Iterate over radius_estimates directly
      Object.entries(landmark.data.radius_estimates).forEach(([gpuId, gpuData]) => {
        if (gpuData.estimate && gpuData.estimate > 0) {
          gpuCircles.push({
            lat: landmark.lat,
            lon: landmark.lon,
            radius: gpuData.estimate,
            color: getHostnameColor(landmark.hostname),
          });
        }
      });
    }
  });
  
  
  const position = [51.505, -0.09]


  return (
    <div style={{ margin: 0, padding: 20 }}>
      <h2>GPU Location Bound Visualization</h2>

      <MapComponent 
        landmarkMarkers={landmarkMarkers}
        singleLandmarkCircles={singleLandmarkCircles}
        gpuCircles={gpuCircles}
      />

      <StatusSection 
        pingLoading={pingLoading}
        sendPing={sendPing}
        lightRadius={lightRadius}
        networkRadius={networkRadius}
        pingMessage={pingMessage}
        styles={styles}
        timings={timings || undefined}
        roundTripTime={roundTripTime}
      />

      <LandmarksTable 
        landmarks={landmarks}
        styles={styles}
      />
    </div>
  );
}

// Basic styling
const styles: Styles = {
  actionButton: {
    backgroundColor: '#4CAF50',
    border: 'none',
    color: 'white',
    padding: '10px 20px',
    textAlign: 'center',
    textDecoration: 'none',
    display: 'inline-block',
    fontSize: 16,
    margin: '10px 0',
    cursor: 'pointer',
    borderRadius: 4,
  },
  info: {
    marginTop: 10,
    fontFamily: 'sans-serif',
  },
  statusSuccess: {
    marginTop: 10,
    fontFamily: 'monospace',
    color: '#4CAF50',
  },
  statusError: {
    marginTop: 10,
    fontFamily: 'monospace',
    color: '#f44336',
  },
  tableCell: {
    border: '1px solid #ccc',
    textAlign: 'left',
    padding: 5,
  },
};

export default WorldMap; 