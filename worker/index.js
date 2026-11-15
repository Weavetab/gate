export default {
  async fetch(req, env) {
    const inUrl = new URL(req.url);
    const upstream = new URL(inUrl.pathname + inUrl.search, env.UPSTREAM_ORIGIN);
    const headers = new Headers(req.headers);
    headers.set("Host", upstream.host);
    return fetch(new Request(upstream, {
      method: req.method,
      headers,
      body: req.method === "GET" || req.method === "HEAD" ? undefined : req.body,
      redirect: "follow",
    }));
  },
};
