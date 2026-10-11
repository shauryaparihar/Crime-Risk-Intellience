import { useEffect, useState } from 'react';

async function describeFailure(res) {
  let detail = '';
  try {
    detail = (await res.json()).detail || '';
  } catch {
    // body was not JSON; the status code alone is enough
  }
  return `Request failed (HTTP ${res.status})${detail ? `: ${detail}` : ''}`;
}

/**
 * GET `url` and return { data, error, loading }.
 * - A new url immediately discards the previous url's data (no stale state).
 * - The previous request is aborted, and a response is only accepted if it
 *   belongs to the url currently requested, so rapid changes can't race.
 * - Pass null/'' to skip fetching (e.g. no state selected yet).
 */
export function useApi(url) {
  const [result, setResult] = useState({ url: null, data: null, error: null });

  useEffect(() => {
    if (!url) return undefined;
    const controller = new AbortController();
    (async () => {
      try {
        const res = await fetch(url, { signal: controller.signal });
        if (!res.ok) throw new Error(await describeFailure(res));
        const data = await res.json();
        setResult({ url, data, error: null });
      } catch (err) {
        if (err.name === 'AbortError') return;
        const message = err instanceof TypeError
          ? 'Cannot reach the API. Is the backend running?'
          : err.message;
        setResult({ url, data: null, error: message });
      }
    })();
    return () => controller.abort();
  }, [url]);

  if (!url) return { data: null, error: null, loading: false };
  if (result.url !== url) return { data: null, error: null, loading: true };
  return { data: result.data, error: result.error, loading: false };
}
