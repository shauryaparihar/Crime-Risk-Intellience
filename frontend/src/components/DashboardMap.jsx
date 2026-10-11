import React from 'react';
import { useApi } from '../hooks/useApi';
import { Skeleton, Card, ErrorBanner } from './ui';
import PlotModule from 'react-plotly.js';

// react-plotly.js is CommonJS; depending on the bundler the component arrives as `.default`.
const Plot = PlotModule.default ?? PlotModule;

export default function DashboardMap({ state, apiBase }) {
  const geo = useApi(`${apiBase}/geojson`);
  const drill = useApi(state ? `${apiBase}/state-drilldown?state=${encodeURIComponent(state)}` : null);
  const geoData = geo.data;
  const drilldownData = drill.data;

  if (geo.loading || drill.loading) return <Skeleton style={{ height: '400px', marginTop: '16px' }} />;

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
            [0, '#06B6D4'],    // lower rate
            [0.5, '#3B82F6'],  // middle
            [1, '#F87171']     // higher rate
          ],
          marker: { line: { color: "rgba(255,255,255,0.1)", width: 1 } },
          colorbar: { 
            title: { text: "2023 rate per 100k", font: { color: '#94A3B8' } },
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
      {geo.error && <ErrorBanner message={`Map data: ${geo.error}`} />}
      {drill.error && <ErrorBanner message={`State details: ${drill.error}`} />}
      {!state && !geo.error && <div className="text-secondary p-4">Select a state in the sidebar to see its district and crime-type breakdown.</div>}
      {mapFigure && (
        <Card className="mb-6 border-subtle" style={{ padding: '0', overflow: 'hidden' }}>
          <div style={{ padding: 'var(--space-3) var(--space-4)', borderBottom: '1px solid var(--border-subtle)', background: 'rgba(255,255,255,0.02)' }}>
            <h3 className="text-lg font-bold">State Crime-Rate Map (Observed 2023)</h3>
            <p className="text-sm text-secondary mt-1">Observed 2023 crime rate per 100k by state, from the unverified NCRB-style sample dataset. Not a projection.</p>
          </div>
          {mapFigure}
        </Card>
      )}

      {drilldownData && <div className="grid grid-cols-2 gap-4">
        <div className="card" style={{ padding: '0' }}>
          <div style={{ padding: 'var(--space-4)', borderBottom: '1px solid var(--border-subtle)', background: 'rgba(255,255,255,0.01)' }}>
            <h3 className="text-lg font-bold">Top Districts by Reported Cases (2023)</h3>
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
            <h3 className="text-lg font-bold">Reported Cases by Crime Type (2023)</h3>
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
      </div>}
    </div>
  );
}
