import React, { useState, useEffect } from 'react';
import { Skeleton, Card, ErrorBanner } from './ui';
import Plot from 'react-plotly.js';

export default function DashboardMap({ state, apiBase }) {
  const [drilldownData, setDrilldownData] = useState(null);
  const [geoData, setGeoData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    async function fetchGeo() {
      try {
        const res = await fetch(`${apiBase}/geojson`);
        if (res.ok) {
          const json = await res.json();
          setGeoData(json);
        }
      } catch (err) {
        console.error("GeoJSON error:", err);
      }
    }
    fetchGeo();
  }, [apiBase]);

  useEffect(() => {
    if (!state) return;
    async function fetchDrilldown() {
      setLoading(true);
      try {
        const res = await fetch(`${apiBase}/state-drilldown?state=${encodeURIComponent(state)}`);
        if (!res.ok) throw new Error();
        const json = await res.json();
        setDrilldownData(json);
        setError(false);
      } catch (err) {
        console.error("Drilldown error:", err);
        setError(true);
      } finally {
        setLoading(false);
      }
    }
    fetchDrilldown();
  }, [state, apiBase]);

  if (loading) return <Skeleton style={{ height: '400px', marginTop: '16px' }} />;
  if (!drilldownData) return <div className="text-secondary p-4">No drilldown data for state.</div>;

  let mapFigure = null;
  if (geoData?.geojson && geoData?.rates) {
    const locations = Object.keys(geoData.rates);
    const z = Object.values(geoData.rates);

    mapFigure = (
      <Plot
        data={[{
          type: 'choroplethmapbox',
          geojson: geoData.geojson,
          locations: locations,
          z: z,
          featureidkey: "properties.NAME_1",
          colorscale: [
            [0, '#06B6D4'],    // Cyan for low risk
            [0.5, '#3B82F6'],  // Blue for medium
            [1, '#F87171']     // Red for high
          ],
          marker: { line: { color: "rgba(255,255,255,0.1)", width: 1 } },
          colorbar: { 
            title: { text: "Base Rate '23", font: { color: '#94A3B8' } },
            tickfont: { color: '#94A3B8' }
          }
        }]}
        layout={{
          mapbox: {
            style: "carto-darkmatter", // Dark mode map style
            center: { lat: 22.5937, lon: 80.9629 },
            zoom: 3.5
          },
          margin: { r: 0, t: 0, l: 0, b: 0 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          autosize: true
        }}
        useResizeHandler={true}
        style={{ width: "100%", height: "450px" }}
      />
    );
  }

  return (
    <div className="mt-4 flex-col gap-4" style={{ animation: 'fadeIn var(--transition-smooth)' }}>
      {error && <ErrorBanner />}
      {mapFigure && (
        <Card className="mb-6 border-subtle" style={{ padding: '0', overflow: 'hidden' }}>
          <div style={{ padding: 'var(--space-3) var(--space-4)', borderBottom: '1px solid var(--border-subtle)', background: 'rgba(255,255,255,0.02)' }}>
            <h3 className="text-lg font-bold">Geospatial Intelligence Heatmap</h3>
            <p className="text-sm text-secondary mt-1">Live geographic distribution of state-level risk projections.</p>
          </div>
          {mapFigure}
        </Card>
      )}

      <div className="grid grid-cols-2 gap-4">
        <div className="card" style={{ padding: '0' }}>
          <div style={{ padding: 'var(--space-4)', borderBottom: '1px solid var(--border-subtle)', background: 'rgba(255,255,255,0.01)' }}>
            <h3 className="text-lg font-bold">Top Districts by Gross Volume</h3>
          </div>
          <div className="custom-scroll" style={{ maxHeight: '250px', overflowY: 'auto' }}>
            <ul style={{ listStyle: 'none', padding: '0 var(--space-4)' }}>
            {drilldownData.districts_2023?.map((d, i) => (
              <li key={d.district} className="flex justify-between items-center py-4" style={{ borderBottom: i !== drilldownData.districts_2023.length - 1 ? '1px solid var(--border-subtle)' : 'none' }}>
                <span className="font-bold text-sm tracking-wide">{d.district}</span>
                <span className="text-accent text-sm font-bold bg-[rgba(59,130,246,0.1)] px-2 py-1 rounded-md">{d.count?.toLocaleString()}</span>
              </li>
            ))}
            </ul>
          </div>
        </div>
        
        <div className="card" style={{ padding: '0' }}>
          <div style={{ padding: 'var(--space-4)', borderBottom: '1px solid var(--border-subtle)', background: 'rgba(255,255,255,0.01)' }}>
            <h3 className="text-lg font-bold">Crime Type Breakdown</h3>
          </div>
          <div className="custom-scroll" style={{ maxHeight: '250px', overflowY: 'auto' }}>
            <ul style={{ listStyle: 'none', padding: '0 var(--space-4)' }}>
            {drilldownData.types_2023?.map((d, i) => (
              <li key={d.type} className="flex justify-between items-center py-4" style={{ borderBottom: i !== drilldownData.types_2023.length - 1 ? '1px solid var(--border-subtle)' : 'none' }}>
                <span className="font-medium text-sm text-secondary">{d.type}</span>
                <span className="text-main text-sm font-bold">{d.count?.toLocaleString()}</span>
              </li>
            ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
