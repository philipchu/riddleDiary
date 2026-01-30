"""Tools for the brainstorming agent."""

import os

import httpx


async def web_search(query: str) -> str:
    """Search the web for current information on a topic.

    Args:
        query: The search query to look up.

    Returns:
        Search results as formatted text.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Web search unavailable: TAVILY_API_KEY not set"

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "query": query,
                    "api_key": api_key,
                    "search_depth": "basic",
                    "max_results": 5,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for r in data.get("results", []):
                results.append(f"**{r.get('title', 'No title')}**\n{r.get('content', '')}\n")

            return "\n---\n".join(results) if results else "No results found."
        except httpx.HTTPError as e:
            return f"Search error: {e}"


async def web_fetch(url: str) -> str:
    """Fetch and extract content from a webpage.

    Args:
        url: The URL to fetch content from.

    Returns:
        Extracted text content from the page.
    """
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()

            try:
                from trafilatura import extract

                content = extract(resp.text)
                if content:
                    return content[:10000]  # Limit content size
            except ImportError:
                pass

            # Fallback: return raw text truncated
            return resp.text[:5000]
        except httpx.HTTPError as e:
            return f"Fetch error: {e}"
